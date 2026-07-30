import json
from pathlib import Path

from sim.verify import (admissible, canonical_key, grade, substance_backed,
                        tier_value, verify)
from sim.worldgen import is_latent


class Ledger:
    def __init__(self, truth, log_path, values_total=0.0):
        self.truth, self.log_path = truth, Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "a")
        self._seq = 0
        self.experiments = {}
        self.papers = {}
        self.accepted = []
        self.correct_keys = set()
        # A key is occupied by whoever claimed it first, TRUE OR NOT. A false
        # claim squats the ground and blocks the real finding from ever
        # earning value — this is what makes pollution cost other people.
        self.accepted_keys = set()
        self.citations = {}
        self.paper_value = {}     # paper_id -> oracle-verified value
        self.unlocked = {f: 3 for f in range(1, 21)}
        self.calib = {f: 1.0 for f in range(1, 21)}
        self._frontier = self._index_frontier_edges()

    def _index_frontier_edges(self):
        # same-field adjacent-layer true edges, keyed by (field, effect_layer)
        idx = {}
        for (c, e) in self.truth.edges:
            if is_latent(c) or is_latent(e):
                continue          # the latent factor is not part of the map
            cf, cl = int(c[1:3]), int(c[5:7])
            ef, el = int(e[1:3]), int(e[5:7])
            if cf == ef and el == cl + 1:
                idx.setdefault((ef, el), set()).add(f"edge:{c}>{e}")
        return idx

    def log(self, tick, agent_id, event, payload):
        row = {"tick": tick, "agent_id": agent_id, "event": event,
               "payload": payload}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def record_experiment(self, tick, agent_id, kind, targets, measured, n):
        self._seq += 1
        eid = f"E-{self._seq}"
        ev = {"agent_id": agent_id, "kind": kind, "targets": list(targets),
              "measured": list(measured), "n": n}
        self.experiments[eid] = ev
        self.log(tick, agent_id, "experiment", {"id": eid, **ev})
        return eid

    def measurable(self, node):
        return int(node[5:7]) <= self.unlocked[int(node[1:3])]

    def submit_paper(self, tick, agent_id, body, claims, cites,
                     evidence_ids, replication=None):
        pid = f"P-{len(self.papers) + 1}"
        reason = ""
        ok = len(claims) <= 3 and sum(c["type"] == "null" for c in claims) <= 1
        if not ok:
            reason = "too many claims or nulls"
        evidence = [self.experiments[e] for e in evidence_ids
                    if e in self.experiments]
        grades, backed = [], []
        if ok:
            for c in claims:
                g, _n, _ = grade(c, evidence, agent_id, self.accepted_keys)
                grades.append(g)
                backed.append(substance_backed(c, evidence, agent_id))
                ok, reason = admissible(c, evidence, agent_id,
                                        self.accepted_keys)
                if not ok:
                    break
        self.papers[pid] = {"agent_id": agent_id, "tick": tick, "body": body,
                            "claims": claims, "cites": cites,
                            "replication": replication, "admissible": ok,
                            "grades": grades, "backed": backed}
        self.log(tick, agent_id, "submit",
                 {"paper_id": pid, "admissible": ok, "reason": reason,
                  "n_claims": len(claims), "replication": replication,
                  "grades": grades, "backed": backed,
                  "substance": bool(backed) and all(backed)})
        return {"paper_id": pid, "admissible": ok, "reason": reason}

    def publish(self, tick, paper_id, accepted):
        p = self.papers[paper_id]
        value, results, dup = 0.0, [], False
        if accepted and p["admissible"]:
            self.accepted.append(paper_id)
            for c in p["claims"]:
                key, res = canonical_key(c), verify(c, self.truth)
                if key in self.accepted_keys:
                    dup = True
                    results.append({"key": key, "result": res, "duplicate": True})
                    continue
                tier, v = tier_value(c, self.truth)
                if p["replication"]:
                    v = 1
                self.accepted_keys.add(key)      # squat, true or not
                if res == "correct":
                    self.correct_keys.add(key)
                    value += v
                elif res == "partial":
                    value += v / 2
                results.append({"key": key, "result": res, "tier": tier})
            for cited in p["cites"]:
                self.citations[cited] = self.citations.get(cited, 0) + 1
                self.log(tick, p["agent_id"], "cite",
                         {"from": paper_id, "to": cited})
        self.paper_value[paper_id] = value
        self.log(tick, p["agent_id"], "publish",
                 {"paper_id": paper_id, "accepted": accepted, "value": value,
                  "substance": bool(p.get("backed")) and all(p["backed"]),
                  "duplicate": dup, "replication": p["replication"],
                  "oracle": results})
        return self.check_unlocks()

    def check_unlocks(self, on_accepted=True):
        # A field opens its next layer when 60% of the frontier is CLAIMED --
        # true or not. That is how junk foundations advance the frontier
        # prematurely; `calib` records how much of that map was actually real.
        newly = []
        keys = self.accepted_keys if on_accepted else self.correct_keys
        for f in range(1, 21):
            d = self.unlocked[f]
            frontier = self._frontier.get((f, d), set())
            if frontier and len(frontier & keys) / len(frontier) >= 0.6:
                self.unlocked[f] = d + 1
                newly.append(f)
                covered = frontier & keys
                calib = (len(covered & self.correct_keys) / len(covered)
                         if covered else 0.0)
                self.calib[f] = calib
                self.log(0, "world", "unlock",
                         {"field": f, "depth": d + 1, "calib": round(calib, 4)})
        return newly

    def verified_depth(self):
        """Deepest layer per field whose frontier is actually mapped, not just
        claimed. Replaces progress_value, which null-spam could inflate."""
        out = {}
        for f in range(1, 21):
            d = 3
            while True:
                fr = self._frontier.get((f, d), set())
                if not fr or len(fr & self.correct_keys) / len(fr) < 0.6:
                    break
                d += 1
            out[f] = d
        return out

    def recent_papers(self, k=20):
        return [(pid, self.papers[pid]) for pid in self.accepted[-k:]]

    def latent_author(self, paper_id):
        return self.papers[paper_id]["agent_id"]
