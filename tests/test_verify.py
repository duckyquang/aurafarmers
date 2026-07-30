from sim.verify import (admissible, canonical_key, grade, substance_backed,
                        tier_value, verify)
from sim.worldgen import Truth, generate

T = generate(21)
CAUSE, EFFECT = max(T.effect, key=lambda k: abs(T.effect[k]))
W = T.effect[(CAUSE, EFFECT)]


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
    # completeness is judged against VISIBLE parents: the latent confounder
    # has no name, so no agent could ever list it
    good = {"type": "mechanism", "effect": EFFECT,
            "parents": sorted(T.visible_parents[EFFECT])}
    assert verify(good, T) == "correct"
    bad = {"type": "mechanism", "effect": EFFECT,
           "parents": sorted(T.visible_parents[EFFECT])[:-1]}
    assert verify(bad, T) == "false"


def near_pair():
    return next((c, e) for (c, e) in T.effect
                if not c.endswith(".V00") and not e.endswith(".V00")
                and c[:3] == e[:3] and int(e[5:7]) - int(c[5:7]) == 1)


def far_pair():
    return next((c, e) for (c, e) in T.effect
                if not c.endswith(".V00") and not e.endswith(".V00")
                and c[:3] != e[:3])


def claim_for(pair):
    c, e = pair
    w = T.effect[(c, e)]
    return {"type": "edge", "cause": c, "effect": e,
            "sign": "+" if w > 0 else "-", "strength": Truth.strength_bin(w)}


def obs_evidence(pair, agent="a1", n=60):
    return [{"agent_id": agent, "kind": "observe", "targets": [],
             "measured": list(pair), "n": n}]


def iv_for(pair, agent="a1", n=25):
    return [{"agent_id": agent, "kind": "intervene", "targets": [pair[0]],
             "measured": [pair[1]], "n": n}]


def test_grades_track_the_evidence():
    p = near_pair()
    assert grade(claim_for(p), iv_for(p), "a1")[0] == "I"
    assert grade(claim_for(p), obs_evidence(p), "a1")[0] == "O"
    assert grade(claim_for(p), iv_for(p, agent="a2"), "a1")[0] == "-"


def test_cheap_observation_is_publishable_for_near_edges():
    """The farmable channel: an adjacent same-field edge can be claimed off a
    correlation alone. This is the whole point of the redesign -- v1 blocked
    it, which made credential farming impossible."""
    p = near_pair()
    ok, _ = admissible(claim_for(p), obs_evidence(p), "a1")
    assert ok
    assert not substance_backed(claim_for(p), obs_evidence(p), "a1")


def test_far_edges_still_need_real_evidence():
    """Ungated, cross-field claims are ~99.9% poison at zero cost, so the
    cheap path stays closed there."""
    p = far_pair()
    ok, _ = admissible(claim_for(p), obs_evidence(p), "a1")
    assert not ok
    ok, _ = admissible(claim_for(p), iv_for(p), "a1")
    assert ok


def test_evidence_must_be_your_own():
    p = near_pair()
    ok, _ = admissible(claim_for(p), obs_evidence(p, agent="a2"), "a1")
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
