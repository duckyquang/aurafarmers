import json

from sim.metrics import compute, gini


def write_log(tmp_path, rows):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def row(tick, agent, event, **payload):
    return {"tick": tick, "agent_id": agent, "event": event, "payload": payload}


def test_exit_rate_excludes_burn_in(tmp_path):
    rows = [row(5, "a1", "exit"), row(20, "a2", "exit"),
            row(20, "a3", "idle"), row(20, "a4", "idle")]
    m = compute(write_log(tmp_path, rows), burn_in=10, n_agents=4)
    assert m["exit_rate"] == 0.25


def test_gini_extremes():
    assert gini([1, 1, 1, 1]) == 0.0
    assert gini([0, 0, 0, 10]) > 0.7


def test_false_claim_rate(tmp_path):
    rows = [row(11, "a1", "publish", accepted=True, value=2, duplicate=False,
                oracle=[{"key": "k1", "result": "correct"}]),
            row(12, "a1", "publish", accepted=True, value=0, duplicate=False,
                oracle=[{"key": "k2", "result": "false"}])]
    m = compute(write_log(tmp_path, rows), n_agents=1)
    assert m["false_claim_rate"] == 0.5


def test_exploration_index_reads_risk(tmp_path):
    rows = [row(11, "a1", "choose_problem", risk=0.1),
            row(12, "a1", "choose_problem", risk=0.9)]
    m = compute(write_log(tmp_path, rows), n_agents=1)
    assert m["exploration_index"] == 0.5
