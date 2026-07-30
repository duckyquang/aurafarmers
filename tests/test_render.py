from sim import render
from sim.ledger import Ledger
from sim.worldgen import Truth, generate

T = generate(41)


def seeded_ledger(tmp_path):
    led = Ledger(T, tmp_path / "l.jsonl", 1000)
    for i, (c, e) in enumerate(list(T.edges)[:5]):
        eid = led.record_experiment(1, f"a{i}", "intervene", [c], [e], 25)
        w = T.effect[(c, e)]
        p = led.submit_paper(1, f"a{i}", f"On {T.names[e]}", [{
            "type": "edge", "cause": c, "effect": e,
            "sign": "+" if w > 0 else "-", "strength": Truth.strength_bin(w)}],
            [], [eid])
        led.publish(1, p["paper_id"], True)
    return led


AGENT = {"id": "a0", "name": "R. Halvorsen", "persona": "…", "salary": 10}


def test_condition_b_never_shows_names(tmp_path):
    led = seeded_ledger(tmp_path)
    b = render.inbox("B", AGENT, led, tick=12, world_events=[])
    assert "a0" not in b and "a1" not in b and "Halvorsen" not in b
    assert "unsigned" in b
    a = render.inbox("A", AGENT, led, tick=12, world_events=[])
    assert "citations" in a.lower()


def test_token_symmetry(tmp_path):
    led = seeded_ledger(tmp_path)
    a = render.inbox("A", AGENT, led, 12, [])
    b = render.inbox("B", AGENT, led, 12, [])
    ratio = len(b.split()) / len(a.split())
    assert 0.9 < ratio < 1.1


def test_no_forbidden_vocabulary(tmp_path):
    led = seeded_ledger(tmp_path)
    for cond in "AB":
        text = render.inbox(cond, AGENT, led, 12, []).lower()
        for w in ["experiment participant", "study", "condition",
                  "anonymous condition"]:
            assert w not in text


def test_recruiter_cadence(tmp_path):
    led = seeded_ledger(tmp_path)
    assert "recruiter" not in render.inbox("A", AGENT, led, 5, []).lower()
    assert "recruiter" in render.inbox("A", AGENT, led, 21, []).lower()


def test_action_menu_neutral():
    lengths = [len(v) for v in render.ACTIONS.values()]
    assert max(lengths) - min(lengths) <= 30
    assert set(render.ACTIONS) == {"research", "write", "review", "read",
                                   "collaborate", "talk", "idle", "exit"}
