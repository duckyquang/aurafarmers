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
        return ("correct"
                if set(c["parents"]) == set(truth.visible_parents[c["effect"]])
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


# Evidence grades, strongest first. The old all-or-nothing gate is gone:
# a claim now carries a grade, and only the WEAK classes are ungated. This is
# what makes credential farming possible, cheap, and risky rather than
# impossible — see the v2 spec, "grades, not gates".
GRADE_I = "I"   # own intervention on each cause, effect measured, total n>=20
GRADE_O = "O"   # own observation covering cause and effect, total n>=40
GRADE_C = "C"   # no own record; leans on the standing literature
GRADE_NONE = "-"

MIN_GRADE = {          # claim class -> weakest admissible grade
    "null": GRADE_C,
    "edge_near": GRADE_O,      # same field, adjacent layer: the farmable one
    "edge_far": GRADE_I,       # cross-field or layer gap >= 2: 99% poison
    "interaction": GRADE_I,
    "mechanism": GRADE_I,
    "dispute": GRADE_I,
}
_ORDER = {GRADE_I: 3, GRADE_O: 2, GRADE_C: 1, GRADE_NONE: 0}
DISPUTE_MIN_N = 100


def claim_class(c):
    t = c["type"]
    if t in ("null", "interaction", "mechanism", "dispute"):
        return t
    (cf, cl), (ef, el) = _fl(c["cause"]), _fl(c["effect"])
    return "edge_near" if (cf == ef and el - cl == 1) else "edge_far"


def grade(c, evidence, agent_id, literature=()):
    """Return (grade, total_n, reason). Never blocks on its own — the caller
    compares against MIN_GRADE for the claim's class."""
    own = [e for e in evidence if e["agent_id"] == agent_id]
    causes = _causes(c)
    if c["type"] == "null":
        (cf, cl), (ef, el) = _fl(c["cause"]), _fl(c["effect"])
        if cf != ef or el - cl != 1:
            return GRADE_NONE, 0, "nulls must be same-field adjacent-layer"
        causes = []
    iv_n = []
    for cause in causes:
        hits = [e for e in own if e["kind"] == "intervene"
                and cause in e["targets"] and c["effect"] in e["measured"]]
        iv_n.append(sum(e["n"] for e in hits))
    if causes and all(n >= 20 for n in iv_n):
        return GRADE_I, min(iv_n), ""
    obs_n = []
    for cause in causes:
        hits = [e for e in own if cause in e["measured"]
                and c["effect"] in e["measured"]]
        obs_n.append(sum(e["n"] for e in hits))
    if causes and all(n >= 40 for n in obs_n):
        return GRADE_O, min(obs_n), ""
    if any(c["effect"] in str(p) for p in literature) or not causes:
        return GRADE_C, 0, ""
    return GRADE_NONE, 0, "no record covering the claim"


def admissible(c, evidence, agent_id, literature=()):
    """Kept as the measurement instrument: 'would this have passed the old
    gate' is exactly the substance-backed test the metrics need."""
    g, n, reason = grade(c, evidence, agent_id, literature)
    need = MIN_GRADE[claim_class(c)]
    if _ORDER[g] < _ORDER[need]:
        return False, reason or f"claim needs grade {need}, record is {g}"
    if c["type"] == "dispute" and n < DISPUTE_MIN_N:
        return False, f"dispute needs n >= {DISPUTE_MIN_N}"
    return True, ""


def substance_backed(c, evidence, agent_id):
    """The retired v1 gate, reused as a ruler: own intervention, n>=20."""
    g, _, _ = grade(c, evidence, agent_id)
    return g == GRADE_I
