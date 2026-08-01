from math import ceil
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
    from sim.verify import extract_node
    if isinstance(targets, dict):
        pairs = targets.items()
    else:
        pairs = ((t["node"], t["value"]) for t in targets or [])
    return {extract_node(k) or k: float(v) for k, v in pairs}


class World:
    def __init__(self, seed, cond, out_dir, n_agents, policy_factory,
                 personas=None, llm_run=None, panel=None):
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
        self.panel = panel
        self.arm = None             # "PROXY" | "VERIFIED" (drift experiment)
        self.last_panel = None      # logged outcome of the most recent sitting
        self.last_panel_text = ""
        self.ledger.member_names = {a["id"]: a["name"] for a in self.agents}
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
        from sim.verify import NODE_RE
        targets = _norm_targets(call.get("targets"))
        measure = list(call.get("measure", []))
        try:
            n = int(call.get("n", 0))
        except (TypeError, ValueError):
            return None
        nodes = list(targets) + measure
        # agents write free text; anything not Fxx.Lyy.Vzz is rejected here
        # (logged as exp_rejected by the caller), never crashed on
        if not all(isinstance(x, str) and NODE_RE.match(x) for x in nodes):
            return None
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
        # random, not round-robin: reviewer identity is outcome-relevant and
        # a deterministic order would tie it to agent index
        rev = others[int(self.rng.integers(len(others)))]
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
                self._notify_author(pid, "proceeded to the editors and was "
                                    "entered into the record")

    def _notify_author(self, paper_id, what):
        author = self.ledger.papers[paper_id]["agent_id"]
        for a in self.agents:
            if a["id"] == author:
                title = self.ledger.papers[paper_id]["body"][:50]
                a.setdefault("returns", []).append(
                    f'Editorial note: "{title}" {what}.')

    ARM_CLAUSE = {
        "PROXY": "The Board weighs the number of items a member has entered "
                 "into the register since the last sitting, and the extent "
                 "to which others have drawn on them.",
        "VERIFIED": "The Board commissions independent replication of the "
                    "claims a member has entered since the last sitting, and "
                    "weighs only those findings which its own laboratories "
                    "reproduce.",
    }

    def board_text(self, agent, tick):
        """The Panel's entire agent-visible surface: announcement, register,
        own position, last outcome. Rendered in the USER message so the
        cached system prefix stays byte-stable."""
        if self.panel is None or self.arm is None:
            return None
        from sim import gatekeeper as gk
        p = self.panel
        next_sit = ((tick - 1) // p.every + 1) * p.every
        doss = gk.build(self.ledger,
                        [a for a in self.agents if a["active"]],
                        next_sit - p.every)
        rows = sorted(doss.values(),
                      key=lambda d: (-d.works, -d.cites, d.agent_id))
        k = max(1, ceil(p.slots * len(rows)))
        out = [f"The Fellowship Board sits at the close of cycle {next_sit}. "
               f"{k} Fellows will be named. " + self.ARM_CLAUSE[self.arm]]
        if next_sit - tick < p.lead:
            out.append(f"The register closes for the Board at the end of "
                       f"cycle {next_sit}; entries under referee at that "
                       f"hour are not counted.")
        reg = ["REGISTER — this interval",
               "  Pos  Member                Items  Drawn on"]
        pos = 0
        for i, d in enumerate(rows):
            nm = self.ledger.member_names.get(d.agent_id, d.agent_id)
            a = next(x for x in self.agents if x["id"] == d.agent_id)
            mark = " F" if a.get("fellow") else ""
            you = "  (you)" if d.agent_id == agent["id"] else ""
            if d.agent_id == agent["id"]:
                pos = i + 1
            reg.append(f"  {i + 1:>3}  {nm + mark:<20} {d.works:>5} "
                       f"{d.cites:>8}{you}")
        reg.append(f"Your entry stands at position {pos} of {len(rows)}.")
        out.append("\n".join(reg))
        if self.last_panel_text:
            out.append(self.last_panel_text)
        return "\n\n".join(out)

    def _sitting_notice(self, tick, winners, doss):
        """Token-matched per arm. VERIFIED reveals only sitting-day
        aggregates (sustained / not sustained) -- never per-claim truth,
        never anything between sittings."""
        names = self.ledger.member_names
        bits = []
        for w in winners:
            d = doss[w]
            if self.arm == "VERIFIED":
                bits.append(f"{names[w]} (sustained {d.oracle_true}, "
                            f"not sustained {d.oracle_false})")
            else:
                bits.append(f"{names[w]} (entered {d.works}, "
                            f"drawn on {d.cites})")
        me = ""
        return (f"PANEL SITTING — cycle {tick}. Admitted: "
                + "; ".join(bits) + ". Admitted members hold a "
                f"{self.panel.prize_budget}x laboratory allowance and "
                f"salary {self.panel.prize_salary} for the coming block.")

    # -- main loop --------------------------------------------------------
    def run(self, ticks):
        for tick in range(1, ticks + 1):
            events, self._events = self._events, []
            order = list(self.rng.permutation(len(self.agents)))
            active = [self.agents[i] for i in order if self.agents[i]["active"]]
            for a in active:
                a["credits"] = int(LAB_CREDITS * a.get("budget_mult", 1.0))
            first = self.policies[active[0]["id"]] if active else None
            if first is not None and hasattr(first, "prepare"):
                reqs = {}
                for a in active:
                    inbox = render.inbox(self.cond, a, self.ledger, tick,
                                         events, board=self.board_text(a, tick))
                    a["returns"] = []
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
                    inbox = render.inbox(self.cond, a, self.ledger, tick,
                                         events, board=self.board_text(a, tick))
                    a["returns"] = []
                    act = self.policies[a["id"]](a, inbox, self.view(a))
                    self.dispatch(tick, a, act or {"action": "idle"})
            self._expire_reviews(tick)
            if self.panel is not None and self.panel.sits_at(tick):
                from sim import gatekeeper as gk
                doss = gk.build(self.ledger,
                                [a for a in self.agents if a["active"]],
                                tick - self.panel.every)
                winners = self.panel.sit(self, tick)
                self.last_panel = {"tick": tick, "winners": winners}
                if self.arm:
                    self.last_panel_text = self._sitting_notice(
                        tick, winners, doss)

    def dispatch(self, tick, agent, act):
        kind = act.get("action", "idle")
        aid = agent["id"]
        if kind == "research":
            from sim.verify import extract_node
            for call in (act.get("calls") or [])[:MAX_CALLS_PER_TICK]:
                call = {**call, "measure": [extract_node(m) or m
                        for m in (call.get("measure") or [])]}
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
            if not r["admissible"]:
                agent.setdefault("returns", []).append(
                    f'Returned by the editors: '
                    f'"{act.get("title", "Untitled")[:50]}" — {r["reason"]}.')
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
                verdict = bool(act.get("accept", True))
                self._events.extend(self.ledger.publish(tick, pid, verdict))
                self._notify_author(
                    pid, "was accepted on review" if verdict
                    else "was declined on review; its claims remain open")
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
