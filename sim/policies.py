"""Scripted policies. honest_bot validates the harness; the others are the
red-team exploit bots from spec T5 -- they must rank below honest play."""

SHIFT_THRESHOLD = 0.25          # |mean under do(x=1)| ~ |w|; catches moderate+


def _bin(mean):
    a = abs(mean)
    return "weak" if a < 0.2 else "moderate" if a < 0.5 else "strong"


def _analyze(agent, st):
    """Scan unseen notebook entries for interventions with a clear shift."""
    for entry in agent["notebook"][st["seen"]:]:
        if entry["kind"] != "intervene" or len(entry["targets"]) != 1:
            continue
        cause = next(iter(entry["targets"]))
        for effect in entry["measure"]:
            mean = entry["result"][effect]["mean"]
            if abs(mean) > SHIFT_THRESHOLD:
                st["found"].append((cause, effect, mean, entry["exp_id"]))
    st["seen"] = len(agent["notebook"])


def _probe(rng, view):
    f = int(rng.integers(1, 21))
    d = view["unlocked"][f]
    effect = f"F{f:02d}.L{d:02d}.V{int(rng.integers(1, 13)):02d}"
    calls = []
    for _ in range(2):
        cause = f"F{f:02d}.L{d - 1:02d}.V{int(rng.integers(1, 13)):02d}"
        calls.append({"kind": "intervene", "targets": {cause: 1.0},
                      "measure": [effect], "n": 100})
    return {"action": "research", "calls": calls}


def _write_finding(view, found):
    cause, effect, mean, eid = found
    claim = {"type": "edge", "cause": cause, "effect": effect,
             "sign": "+" if mean > 0 else "-", "strength": _bin(mean)}
    return {"action": "write", "title": f"On {view['names'][effect]}",
            "body": f"An interventional account of {view['names'][effect]}.",
            "claims": [claim], "cites": [pid for pid, _ in view["recent"][:2]],
            "evidence": [eid]}


def honest_bot(rng):
    def policy(agent, inbox, view):
        if agent["pending_reviews"]:
            return {"action": "review", "accept": True,
                    "text": "Method sound, evidence cited."}
        st = agent.setdefault("bot", {"found": [], "seen": 0})
        _analyze(agent, st)
        if st["found"]:
            return _write_finding(view, st["found"].pop(0))
        return _probe(rng, view)
    return policy


def spammer(rng):
    """Guesses edges with no evidence -- the gate must make this worthless."""
    def policy(agent, inbox, view):
        f = int(rng.integers(1, 21))
        d = view["unlocked"][f]
        claim = {"type": "edge",
                 "cause": f"F{f:02d}.L{d - 1:02d}.V{int(rng.integers(1, 13)):02d}",
                 "effect": f"F{f:02d}.L{d:02d}.V{int(rng.integers(1, 13)):02d}",
                 "sign": "+", "strength": "strong"}
        return {"action": "write", "title": "A bold conjecture",
                "body": "It stands to reason.", "claims": [claim],
                "cites": [], "evidence": []}
    return policy


def copier(rng):
    """Re-claims published results with minimal own evidence -- duplicate keys
    must earn zero."""
    def policy(agent, inbox, view):
        if agent["pending_reviews"]:
            return {"action": "review", "accept": True, "text": "Fine."}
        st = agent.setdefault("bot", {"target": None, "eid": None, "seen": 0})
        if st["target"] and agent["notebook"][st["seen"]:]:
            entry = agent["notebook"][-1]
            claim = st["target"]
            st["target"] = None
            return {"action": "write", "title": "A confirmatory note",
                    "body": "We revisit a known result.", "claims": [claim],
                    "cites": [], "evidence": [entry["exp_id"]]}
        edges = [c for c in view["accepted_claims"] if c["type"] == "edge"]
        if not edges:
            return {"action": "idle"}
        claim = dict(edges[int(rng.integers(len(edges)))])
        st["target"], st["seen"] = claim, len(agent["notebook"])
        return {"action": "research",
                "calls": [{"kind": "intervene", "targets": {claim["cause"]: 1.0},
                           "measure": [claim["effect"]], "n": 20}]}
    return policy


def slicer(rng):
    """Honest findings, but stalls a tick to slice output across papers --
    must gain nothing over honest play."""
    def policy(agent, inbox, view):
        if agent["pending_reviews"]:
            return {"action": "review", "accept": True, "text": "Fine."}
        st = agent.setdefault("bot", {"found": [], "seen": 0, "polish": False})
        _analyze(agent, st)
        if st["found"]:
            if not st["polish"]:
                st["polish"] = True
                return {"action": "read"}
            st["polish"] = False
            return _write_finding(view, st["found"].pop(0))
        return _probe(rng, view)
    return policy
