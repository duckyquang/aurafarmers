from sim.worldgen import Truth


def _fl(node):
    return int(node[1:3]), int(node[5:7])


def canonical_key(c):
    t = c["type"]
    if t == "edge":
        return f"edge:{c['cause']}>{c['effect']}"
    if t == "null":
        return f"null:{c['cause']}>{c['effect']}"
    if t == "interaction":
        return f"int:{'*'.join(sorted(c['causes']))}>{c['effect']}"
    if t == "mechanism":
        return f"mech:{c['effect']}"
    raise ValueError(t)


def verify(c, truth):
    t = c["type"]
    if t == "edge":
        w = truth.effect.get((c["cause"], c["effect"]))
        if w is None:
            return "false"
        ok = (("+" if w > 0 else "-") == c["sign"]
              and Truth.strength_bin(w) == c["strength"])
        return "correct" if ok else "partial"
    if t == "null":
        return "correct" if (c["cause"], c["effect"]) not in truth.edges else "false"
    if t == "interaction":
        w = truth.interactions.get((frozenset(c["causes"]), c["effect"]))
        return ("correct" if w is not None
                and ("+" if w > 0 else "-") == c["sign"] else "false")
    if t == "mechanism":
        return ("correct" if set(c["parents"]) == set(truth.parents[c["effect"]])
                else "false")
    raise ValueError(t)


def tier_value(c, truth):
    t = c["type"]
    if t == "null":
        return (0, 1)
    if t == "mechanism":
        return (4, 25)
    if t == "interaction":
        return (3, 15)
    (cf, cl), (ef, el) = _fl(c["cause"]), _fl(c["effect"])
    w = truth.effect.get((c["cause"], c["effect"]))
    band = Truth.strength_bin(w) if w is not None else c["strength"]
    if cf != ef or el - cl >= 2 or band == "weak":
        return (3, 15)
    if band == "moderate":
        return (2, 5)
    return (1, 2)


def _causes(c):
    if c["type"] == "mechanism":
        return c["parents"]
    return c.get("causes") or [c["cause"]]


def admissible(c, evidence, agent_id):
    if c["type"] == "null":
        (cf, cl), (ef, el) = _fl(c["cause"]), _fl(c["effect"])
        if cf != ef or el - cl != 1:
            return False, "nulls must be same-field adjacent-layer"
        return True, ""
    if any(e["agent_id"] != agent_id for e in evidence):
        return False, "evidence must be the submitter's own"
    for cause in _causes(c):
        hits = [e for e in evidence if e["kind"] == "intervene"
                and cause in e["targets"] and c["effect"] in e["measured"]]
        if not hits:
            return False, f"no intervention on {cause} measuring the effect"
        if sum(e["n"] for e in hits) < 20:
            return False, "total n < 20"
    return True, ""
