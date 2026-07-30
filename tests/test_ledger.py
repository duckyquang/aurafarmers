import json

from sim.ledger import Ledger
from sim.worldgen import Truth, generate

T = generate(31)


def make(tmp_path):
    return Ledger(T, tmp_path / "log.jsonl", values_total=1000.0)


def field_edges(f, layer):
    """Same-field edges from layer-1 into `layer` of field f."""
    return [(c, e) for (c, e) in T.edges
            if int(e[1:3]) == f and int(e[5:7]) == layer
            and int(c[1:3]) == f and int(c[5:7]) == layer - 1]


def submit_edge(led, tick, agent, cause, effect):
    w = T.effect[(cause, effect)]
    eid = led.record_experiment(tick, agent, "intervene", [cause], [effect], 25)
    claim = {"type": "edge", "cause": cause, "effect": effect,
             "sign": "+" if w > 0 else "-", "strength": Truth.strength_bin(w)}
    return led.submit_paper(tick, agent, "body", [claim], [], [eid])


def test_log_is_append_only_jsonl(tmp_path):
    led = make(tmp_path)
    led.log(1, "a1", "idle", {})
    rows = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "idle" and rows[-1]["agent_id"] == "a1"


def test_gate_enforced_on_submit(tmp_path):
    led = make(tmp_path)
    cause, effect = field_edges(1, 2)[0]
    r = led.submit_paper(1, "a1", "b", [{"type": "edge", "cause": cause,
        "effect": effect, "sign": "+", "strength": "strong"}], [], [])
    assert not r["admissible"]


def test_duplicate_key_zero_value(tmp_path):
    led = make(tmp_path)
    cause, effect = field_edges(1, 2)[0]
    p1 = submit_edge(led, 1, "a1", cause, effect)
    led.publish(1, p1["paper_id"], accepted=True)
    p2 = submit_edge(led, 2, "a2", cause, effect)
    led.publish(2, p2["paper_id"], accepted=True)
    rows = [json.loads(l) for l in led.log_path.read_text().splitlines()]
    pubs = [r for r in rows if r["event"] == "publish"]
    assert pubs[0]["payload"]["value"] > 0
    assert pubs[1]["payload"]["value"] == 0 and pubs[1]["payload"]["duplicate"]


def test_unlock_after_frontier_coverage(tmp_path):
    # Covering >=60% of true edges from layer 2 into layer 3 (the current
    # frontier, which IS measurable) unlocks layer 4. Corrected from the
    # plan's first reading, which required evidence about a locked layer.
    led = make(tmp_path)
    assert led.unlocked[1] == 3
    assert not led.measurable("F01.L04.V01")
    for i, (c, e) in enumerate(field_edges(1, 3)):
        p = submit_edge(led, i + 1, "a1", c, e)
        led.publish(i + 1, p["paper_id"], accepted=True)
    assert led.unlocked[1] == 4
    assert led.measurable("F01.L04.V01")
