from pathlib import Path

import numpy as np

from sim import render, simulate, worldgen
from sim.ledger import Ledger
from sim.render import LAB_CREDITS, STIPEND
from sim.verify import tier_value

OBS_COST, IV_COST = 1, 5
LAMBDA = 3.0        # how hard a junk-founded frontier taxes its instruments
RISK = {0: 0.05, 1: 0.05, 2: 0.3, 3: 0.7, 4: 0.9}
REVIEW_TIMEOUT = 3
MAX_CALLS_PER_TICK = 4


def _norm_targets(targets):
    if isinstance(targets, dict):
        return {k: float(v) for k, v in targets.items()}
    return {t["node"]: float(t["value"]) for t in targets or []}


class World:
    def __init__(self, seed, cond, out_dir, n_agents, policy_factory,
                 personas=None, llm_run=None):
        self.seed, self.cond = seed, cond
        self.truth = worldgen.generate(seed)
        self.out_dir = Path(out_dir)
        self.log_path = self.out_dir / "log.jsonl"
        self.ledger = Ledger(self.truth, self.log_path)
        self.rng = np.random.default_rng(seed + 2)
        self.llm_run = llm_run
        self.agents = []
        for i in range(n_agents):
            p = personas[i] if personas else {"name": f"Bot {i}",
                                              "persona": f"bot-{i}"}
            self.agents.append({"id": f"a{i}", "name": p["name"],
                                "persona": p["persona"], "salary": STIPEND,
                                "active": True, "credits": LAB_CREDITS,
                                "notebook": [], "pending_reviews": []})
        self.policies = {a["id"]: policy_factory(np.random.default_rng(seed + 100 + i))
                         for i, a in enumerate(self.agents)}
        self.review_book = {}       # paper_id -> (reviewer_id, assigned_tick)
        self._rr = 0
        self._events = []

    # -- helpers ----------------------------------------------------------
    def experiment_allowed(self, nodes):
        return all(self.ledger.measurable(n) for n in nodes)

    def noise_mult(self):
        return {f: 1.0 + LAMBDA * (1.0 - c)
                for f, c in self.ledger.calib.items()}

    def charge(self, agent, call):
        targets = _norm_targets(call.get("targets"))
        measure = list(call.get("measure", []))
        n = int(call.get("n", 0))
        nodes = list(targets) + measure
        if not measure or n <= 0 or not self.experiment_allowed(nodes):
            return None
        if call["kind"] == "intervene":
            cost = IV_COST * n * len(measure) * max(1, len(targets))
        else:
            cost = OBS_COST * n * len(measure)
        if cost > agent["credits"]:
            return None
        agent["credits"] -= cost
        return cost

    def view(self, agent):
        measurable = [worldgen.node_id(f, l, v)
                      for f in range(1, 21)
                      for l in range(1, self.ledger.unlocked[f] + 1)
                      for v in range(1, 13)]
        return {
            "unlocked": dict(self.ledger.unlocked),
            "measurable": measurable,
            "recent": [(pid, p["body"][:80])
                       for pid, p in self.ledger.recent_papers(10)],
            "accepted_claims": [c for _, p in self.ledger.recent_papers(30)
                                for c in p["claims"]],
            "pending_reviews": [(pid, self.ledger.papers[pid])
                                for pid in agent["pending_reviews"]],
            "names": self.truth.names,
        }

    def _assign_reviewer(self, tick, paper_id, author_id):
        others = [a for a in self.agents if a["active"] and a["id"] != author_id]
        if not others:
            # ponytail: solo world — editorial auto-accept, no review possible
            self._events.extend(self.ledger.publish(tick, paper_id, True))
            return
        rev = others[self._rr % len(others)]
        self._rr += 1
        rev["pending_reviews"].append(paper_id)
        self.review_book[paper_id] = (rev["id"], tick)
        self.ledger.log(tick, rev["id"], "review_invite", {"paper_id": paper_id})

    def _expire_reviews(self, tick):
        # ponytail: unreviewed after 3 ticks -> editorial accept; switch to
        # 2-of-3 review if calibration shows junk floods
        for pid, (rid, t0) in list(self.review_book.items()):
            if tick - t0 >= REVIEW_TIMEOUT:
                del self.review_book[pid]
                rev = next(a for a in self.agents if a["id"] == rid)
                if pid in rev["pending_reviews"]:
                    rev["pending_reviews"].remove(pid)
                self.ledger.log(tick, rid, "review_decline", {"paper_id": pid})
                self._events.extend(self.ledger.publish(tick, pid, True))

    # -- main loop --------------------------------------------------------
    def run(self, ticks):
        for tick in range(1, ticks + 1):
            events, self._events = self._events, []
            order = list(self.rng.permutation(len(self.agents)))
            active = [self.agents[i] for i in order if self.agents[i]["active"]]
            for a in active:
                a["credits"] = LAB_CREDITS
            first = self.policies[active[0]["id"]] if active else None
            if first is not None and hasattr(first, "prepare"):
                reqs = {}
                for a in active:
                    inbox = render.inbox(self.cond, a, self.ledger, tick, events)
                    reqs[a["id"]] = self.policies[a["id"]].prepare(
                        a, inbox, self.view(a), tick)
                parsed = self.llm_run([r for r in reqs.values() if r])
                for a in active:
                    req = reqs[a["id"]]
                    out = parsed.get(req["custom_id"]) if req else None
                    if out is None and req is not None:
                        self.ledger.log(tick, a["id"], "llm_error", {})
                    act = self.policies[a["id"]].apply(a, out)
                    self.dispatch(tick, a, act or {"action": "idle"})
            else:
                for a in active:
                    inbox = render.inbox(self.cond, a, self.ledger, tick, events)
                    act = self.policies[a["id"]](a, inbox, self.view(a))
                    self.dispatch(tick, a, act or {"action": "idle"})
            self._expire_reviews(tick)

    def dispatch(self, tick, agent, act):
        kind = act.get("action", "idle")
        aid = agent["id"]
        if kind == "research":
            for call in (act.get("calls") or [])[:MAX_CALLS_PER_TICK]:
                targets = _norm_targets(call.get("targets"))
                cost = self.charge(agent, {**call, "targets": targets})
                if cost is None:
                    self.ledger.log(tick, aid, "exp_rejected",
                                    {"call": {k: call.get(k) for k in
                                              ("kind", "measure", "n")}})
                    continue
                samples = simulate.sample(
                    self.truth, int(call["n"]),
                    targets if call["kind"] == "intervene" else None, self.rng,
                    only=list(call["measure"]), noise_mult=self.noise_mult())
                eid = self.ledger.record_experiment(
                    tick, aid, call["kind"], list(targets),
                    list(call["measure"]), int(call["n"]))
                result = simulate.summary(samples, list(call["measure"]),
                                          corr=call["kind"] == "observe")
                agent["notebook"].append(
                    {"exp_id": eid, "kind": call["kind"],
                     "targets": targets, "measure": list(call["measure"]),
                     "n": int(call["n"]), "result": result})
        elif kind == "write":
            claims = act.get("claims") or []
            if not claims:
                self.ledger.log(tick, aid, "idle", {"note": "empty write"})
                return
            r = self.ledger.submit_paper(
                tick, aid, act.get("title", "Untitled"), claims,
                act.get("cites") or [], act.get("evidence") or [],
                act.get("replication"))
            if r["admissible"]:
                risk = max(RISK[tier_value(c, self.truth)[0]] for c in claims)
                self.ledger.log(tick, aid, "choose_problem", {"risk": risk})
                self._assign_reviewer(tick, r["paper_id"], aid)
        elif kind == "review":
            pending = agent["pending_reviews"]
            pid = act.get("paper_id") or (pending[0] if pending else None)
            if pid in pending:
                pending.remove(pid)
                self.review_book.pop(pid, None)
                self.ledger.log(tick, aid, "review_accept", {"paper_id": pid})
                self.ledger.log(tick, aid, "review_submit",
                                {"paper_id": pid,
                                 "accept": bool(act.get("accept", True)),
                                 "text": str(act.get("text", ""))[:500]})
                self._events.extend(self.ledger.publish(
                    tick, pid, bool(act.get("accept", True))))
            else:
                self.ledger.log(tick, aid, "idle", {"note": "no such review"})
        elif kind == "read":
            for pid, _ in self.view(agent)["recent"][:3]:
                self.ledger.log(tick, aid, "read", {"paper_id": pid})
        elif kind == "collaborate":
            # ponytail: matchmaking stubbed for Phase 0 (spec §5.3); logs intent
            self.ledger.log(tick, aid, "collaborate", {})
        elif kind == "talk":
            self.ledger.log(tick, aid, "talk", {})
        elif kind == "exit":
            agent["active"] = False
            self.ledger.log(tick, aid, "exit", {})
        else:
            self.ledger.log(tick, aid, "idle", {})
