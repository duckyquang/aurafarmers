import json

from sim.trace import Run


def read(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def test_run_records_provenance(tmp_path):
    r = Run("unit", config={"seed": 7, "cond": "A"}, root=tmp_path)
    r.close()
    m = json.loads((r.dir / "manifest.json").read_text())
    # a result must be traceable to the exact code and config that made it
    for k in ("git_sha", "code_sha256", "config", "started_utc", "ended_utc",
              "tokens", "cost_usd", "wall_seconds"):
        assert k in m
    assert m["config"]["seed"] == 7
    assert len(m["code_sha256"]) == 16


def test_events_and_llm_calls_are_separate_streams(tmp_path):
    r = Run("unit", root=tmp_path)
    r.event(3, "a1", "submit", {"paper_id": "P-1", "grades": ["O"]})
    r.llm_call(custom_id="a1-t3", model="m", system="sys", user="usr",
               schema=True, response={"action": "write"},
               usage={"in": 10, "out": 5, "reasoning": 3}, latency=0.5,
               finish_reason="stop", cost=0.0001)
    r.close()
    ev = read(r.dir / "events.jsonl")
    calls = read(r.dir / "llm.jsonl")
    assert ev[0]["event"] == "submit" and ev[0]["payload"]["grades"] == ["O"]
    assert calls[0]["usage"]["reasoning"] == 3
    # prompts are kept in full: without them a surprising action cannot be
    # investigated after the fact
    assert calls[0]["system"] == "sys" and calls[0]["user"] == "usr"


def test_counts_and_human_trace(tmp_path):
    r = Run("unit", root=tmp_path)
    for _ in range(3):
        r.event(1, "a1", "experiment", {})
    r.note("something happened")
    m = r.close()
    assert m["event_counts"]["experiment"] == 3
    assert "something happened" in (r.dir / "trace.log").read_text()


def test_errors_are_recorded_not_swallowed(tmp_path):
    r = Run("unit", root=tmp_path)
    r.llm_call(custom_id="x", model="m", system="s", user="u", schema=False,
               response=None, usage=None, latency=0.1, error="RateLimit")
    m = r.close()
    assert m["event_counts"]["llm_error"] == 1
