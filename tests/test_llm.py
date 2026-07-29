import pytest

from sim.llm import ACTION_SCHEMA, ROUTINE_MODEL, build_request


def test_request_shape_and_cache_marker():
    r = build_request("a1-t3", ROUTINE_MODEL,
                      system_blocks=["world rules...", "persona..."],
                      user_text="inbox...", schema=ACTION_SCHEMA,
                      max_tokens=800)
    assert r["custom_id"] == "a1-t3"
    p = r["params"]
    assert p["model"] == "claude-haiku-4-5"
    assert p["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert p["output_config"]["format"]["type"] == "json_schema"


def test_action_schema_rejects_extra_keys():
    for variant in ACTION_SCHEMA["anyOf"]:
        assert variant["additionalProperties"] is False
        assert "action" in variant["required"]


@pytest.mark.live
def test_one_real_call():
    from sim.llm import complete
    out = complete(ROUTINE_MODEL, "Answer with one word.", "Say OK.", 16)
    assert out.strip()
