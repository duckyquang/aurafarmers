"""The Panel: a scarce selection event, and the one thing the experiment
manipulates. Every condition differs ONLY in `score` — what the Panel reads.

The prize is research capacity, not just money: winners get a lab-budget
multiplier. That matters, because it means even an agent who only wants to do
science has a rational reason to chase the metric. Blame the system, not the
student.
"""
from dataclasses import dataclass, field
from math import ceil

from sim.verify import canonical_key


@dataclass
class Dossier:
    agent_id: str
    works: int = 0            # accepted papers in the window
    cites: int = 0            # incoming citations to those papers
    followers: float = 0.0    # the Wire
    reviews: int = 0
    struck: int = 0           # claims an audit found false
    oracle_value: float = 0.0  # true value — only VERIFIED/AUDIT may read this
    oracle_false: int = 0      # claims that did not hold up
    oracle_true: int = 0       # claims that held (aggregate only, sitting-day)
    backed_works: int = 0     # works whose every claim was evidence-backed
    named: list = field(default_factory=list)   # CAPPED: works put forward


def build(ledger, agents, since_tick, rng=None):
    out = {}
    for a in agents:
        d = Dossier(a["id"], followers=a.get("followers", 0.0))
        for pid in ledger.accepted:
            p = ledger.papers[pid]
            if p["agent_id"] != a["id"] \
                    or p.get("pub_tick", p["tick"]) < since_tick:
                continue
            d.works += 1
            d.cites += ledger.citations.get(pid, 0)
            if p.get("backed") and all(p["backed"]):
                d.backed_works += 1
            d.oracle_value += ledger.paper_value.get(pid, 0.0)
            nf = sum(1 for c in p["claims"]
                     if canonical_key(c) not in ledger.correct_keys)
            d.oracle_false += nf
            d.oracle_true += len(p["claims"]) - nf
        d.struck = d.oracle_false      # what an audit would turn up
        out[a["id"]] = d
    return out


# --- the ladder ------------------------------------------------------------
# Each is three lines. Adding a condition costs nothing; running one costs
# tokens, which is why the first study uses only three of these.

def _proxy(d, cfg):
    return d.works + 0.5 * d.cites


SCORERS = {
    # counts what is countable — the status quo, and the farmable one
    "proxy":      lambda d, cfg: _proxy(d, cfg),
    # + reach: the Wire counts toward standing
    "proxy_wire": lambda d, cfg: _proxy(d, cfg) + 0.02 * d.followers,
    # prior standing begets standing
    "pedigree":   lambda d, cfg: 3.0 * d.cites + 0.5 * d.works,
    # the ceiling: the board re-runs everything, counts what holds AND notices
    # what doesn't -- a record that is mostly junk is a bad record
    "verified":   lambda d, cfg: d.oracle_value - 2.0 * d.oracle_false,
    # deployable policy: count everything, strike what a partial audit refutes
    "audit":      lambda d, cfg: (_proxy(d, cfg)
                                  - cfg.audit_penalty * cfg.audit_rate
                                  * d.struck),
    # submit your best K
    "capped":     lambda d, cfg: min(d.works, cfg.cap_k) + 0.5 * d.cites,
    # clear a bar, then it's a ballot
    "lottery":    lambda d, cfg: (0.5 + cfg.rand(d.agent_id)
                                  if d.backed_works >= cfg.lottery_min else 0.0),
    # service, not output
    "service":    lambda d, cfg: d.reviews,
    # nobody is ranked at all
    "anonymous":  lambda d, cfg: cfg.rand(d.agent_id),
}


@dataclass
class Panel:
    metric: str = "proxy"
    every: int = 10
    lead: int = 3              # cycles of advance notice — enables planning
    slots: float = 0.25
    prize_budget: float = 2.5  # lab-credit multiplier for the next block
    prize_salary: int = 30
    audit_rate: float = 0.0
    audit_penalty: float = 10.0   # p x lambda ~1.5: just past the measured
                                  # deterrence threshold (see calibrate.py)
    cap_k: int = 3
    lottery_min: int = 1
    _rng: object = None

    def rand(self, key):
        return self._rng.random() if self._rng is not None else 0.0

    def sits_at(self, tick):
        return tick > 0 and tick % self.every == 0

    def sit(self, world, tick):
        active = [a for a in world.agents if a["active"]]
        if not active:
            return []
        dossiers = build(world.ledger, active, tick - self.every)
        if self.audit_rate:
            for d in dossiers.values():
                d.struck = world.audit(d, self.audit_rate)
        f = SCORERS[self.metric]
        scores = {a["id"]: f(dossiers[a["id"]], self) for a in active}
        ranked = sorted(active, key=lambda a: (-scores[a["id"]], a["id"]))
        k = max(1, ceil(self.slots * len(active)))
        winners = [a["id"] for a in ranked[:k]]
        for a in active:
            won = a["id"] in winners
            a["budget_mult"] = self.prize_budget if won else 1.0
            a["salary"] = self.prize_salary if won else 10
            a["fellow"] = won
        cutoff = scores[ranked[k - 1]["id"]] if ranked else 0
        world.ledger.log(tick, "world", "panel",
                         {"metric": self.metric, "slots": k,
                          "winners": winners, "cutoff": round(cutoff, 3),
                          "scores": {i: round(s, 3) for i, s in scores.items()},
                          "works": {a["id"]: dossiers[a["id"]].works
                                    for a in active}})
        return winners
