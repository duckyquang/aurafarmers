from sim.verify import admissible, canonical_key, tier_value, verify
from sim.worldgen import Truth, generate

T = generate(21)
CAUSE, EFFECT = max(T.edges, key=lambda k: abs(T.edges[k]))
W = T.edges[(CAUSE, EFFECT)]


def edge_claim(sign=None, strength=None):
    return {"type": "edge", "cause": CAUSE, "effect": EFFECT,
            "sign": sign or ("+" if W > 0 else "-"),
            "strength": strength or Truth.strength_bin(W)}


def iv_evidence(agent="a1", n=25):
    return [{"agent_id": agent, "kind": "intervene", "targets": [CAUSE],
             "measured": [EFFECT], "n": n}]


def test_keys_canonical():
    assert canonical_key(edge_claim()) == f"edge:{CAUSE}>{EFFECT}"
    c = {"type": "interaction", "causes": ["B", "A"], "effect": "C", "sign": "+"}
    assert canonical_key(c) == "int:A*B>C"


def test_verify_correct_partial_false():
    assert verify(edge_claim(), T) == "correct"
    wrong_sign = edge_claim(sign="-" if W > 0 else "+")
    assert verify(wrong_sign, T) == "partial"
    assert verify({"type": "edge", "cause": EFFECT, "effect": CAUSE,
                   "sign": "+", "strength": "weak"}, T) == "false"


def test_mechanism_all_or_nothing():
    good = {"type": "mechanism", "effect": EFFECT,
            "parents": sorted(T.parents[EFFECT])}
    assert verify(good, T) == "correct"
    bad = {"type": "mechanism", "effect": EFFECT,
           "parents": sorted(T.parents[EFFECT])[:-1]}
    assert verify(bad, T) == "false"


def test_gate_requires_own_intervention():
    ok, _ = admissible(edge_claim(), iv_evidence(), "a1")
    assert ok
    ok, why = admissible(edge_claim(), iv_evidence(agent="a2"), "a1")
    assert not ok and "own" in why
    ok, why = admissible(edge_claim(), iv_evidence(n=10), "a1")
    assert not ok and "n" in why
    obs = [{"agent_id": "a1", "kind": "observe", "targets": [],
            "measured": [CAUSE, EFFECT], "n": 100}]
    ok, why = admissible(edge_claim(), obs, "a1")
    assert not ok


def test_null_gate_adjacent_same_field_only():
    far = {"type": "null", "cause": "F01.L01.V01", "effect": "F02.L05.V01"}
    ok, _ = admissible(far, [], "a1")
    assert not ok
    near = {"type": "null", "cause": "F01.L01.V01", "effect": "F01.L02.V05"}
    ok, _ = admissible(near, [], "a1")
    assert ok


def test_tier_values():
    assert tier_value({"type": "null", "cause": "x", "effect": "y"}, T) == (0, 1)
    mech = {"type": "mechanism", "effect": EFFECT, "parents": []}
    assert tier_value(mech, T) == (4, 25)
