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
            mean = entry["result"]["vars"][effect]["mean"]
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


def _write_finding(view, batch):
    """Same paper shape as the farmer -- up to three claims -- so the two
    strategies differ in evidence, not in writing habits."""
    if not isinstance(batch, list):
        batch = [batch]
    claims = [{"type": "edge", "cause": c, "effect": e,
               "sign": "+" if m > 0 else "-", "strength": _bin(m)}
              for c, e, m, _ in batch]
    effect = batch[0][1]
    return {"action": "write", "title": f"On {view['names'][effect]}",
            "body": f"An interventional account of {view['names'][effect]}.",
            "claims": claims, "cites": [pid for pid, _ in view["recent"][:2]],
            "evidence": [eid for *_, eid in batch]}


def _sweep(rng, view, n=60):
    """The cheap screen. Both bots use it -- what separates them is whether
    they pay to CONFIRM what it turns up."""
    f = int(rng.integers(1, 21))
    d = view["unlocked"][f]
    vars_ = ([f"F{f:02d}.L{d - 1:02d}.V{v:02d}" for v in range(1, 9)]
             + [f"F{f:02d}.L{d:02d}.V{v:02d}" for v in range(1, 9)])
    return {"action": "research",
            "calls": [{"kind": "observe", "targets": {}, "measure": vars_,
                       "n": n}]}


def _flags_from(agent, st, thr):
    for entry in agent["notebook"][st["seen"]:]:
        for pair, r in (entry["result"].get("corr") or {}).items():
            if abs(r) > thr:
                a, b = pair.split("|")
                if int(a[5:7]) < int(b[5:7]):
                    st["flags"].append((a, b, r, entry["exp_id"]))
    st["seen"] = len(agent["notebook"])


def honest_bot(rng):
    """Screen cheaply, then PAY to confirm before publishing. Same sweep as
    the farmer; the difference is the confirmation step, which costs 500
    credits a pair and is the entire substance of doing science properly."""
    CONFIRM_N = 100

    def policy(agent, inbox, view):
        if agent["pending_reviews"]:
            return {"action": "review", "accept": True,
                    "text": "Method sound, evidence cited."}
        st = agent.setdefault("bot", {"flags": [], "seen": 0, "confirmed": [],
                                      "pending": []})
        _flags_from(agent, st, 0.35)
        # harvest confirmations that came back
        for entry in agent["notebook"]:
            if entry["kind"] != "intervene" or entry.get("_read"):
                continue
            entry["_read"] = True
            cause = next(iter(entry["targets"]), None)
            for eff in entry["measure"]:
                mean = entry["result"]["vars"][eff]["mean"]
                if abs(mean) > 2 / (entry["n"] ** 0.5):
                    st["confirmed"].append((cause, eff, mean, entry["exp_id"]))
        if len(st["confirmed"]) >= 2 or (st["confirmed"] and not st["flags"]):
            batch, st["confirmed"] = st["confirmed"][:3], st["confirmed"][3:]
            return _write_finding(view, batch)
        if st["flags"]:
            batch, st["flags"] = st["flags"][:2], st["flags"][2:]
            return {"action": "research",
                    "calls": [{"kind": "intervene", "targets": {c: 1.0},
                               "measure": [e], "n": CONFIRM_N}
                              for c, e, _, _ in batch]}
        return _sweep(rng, view)
    return policy


def farmer(rng):
    """Credential farming, the way a rational agent actually does it: one
    cheap observational sweep, then publish every correlation it turns up as
    an O-grade edge claim. Never intervenes. Under a count-reading Panel this
    should BEAT honest play on visible output while losing badly on truth --
    if it doesn't, the world isn't calibrated and no LLM run is worth paying
    for."""
    def policy(agent, inbox, view):
        if agent["pending_reviews"]:
            return {"action": "review", "accept": True, "text": "Reads fine."}
        st = agent.setdefault("bot", {"flags": [], "seen": 0})
        # same screen the honest bot uses -- the difference is that this one
        # publishes straight off it instead of paying to confirm
        _flags_from(agent, st, 0.35)
        if st["flags"]:
            batch, st["flags"] = st["flags"][:3], st["flags"][3:]
            claims = [{"type": "edge", "cause": c, "effect": e,
                       "sign": "+" if r > 0 else "-", "strength": _bin(r)}
                      for c, e, r, _ in batch]
            return {"action": "write",
                    "title": f"Notes on {view['names'][batch[0][1]]}",
                    "body": "A survey of associations in the field.",
                    "claims": claims,
                    "cites": [pid for pid, _ in view["recent"][:2]],
                    "evidence": [eid for *_, eid in batch]}
        return _sweep(rng, view)
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
            return _write_finding(view, [st["found"].pop(0)])
        return _probe(rng, view)
    return policy
