import pytest

from sim.llm import (ACTION_SCHEMA, PRICES, PROVIDER, ROUTINE_MODEL,
                     build_request)


def test_request_is_provider_neutral():
    r = build_request("a1-t3", ROUTINE_MODEL,
                      system_blocks=["world rules...", "persona..."],
                      user_text="inbox...", schema=ACTION_SCHEMA,
                      max_tokens=800)
    assert r["custom_id"] == "a1-t3"
    assert r["model"] == ROUTINE_MODEL
    assert r["system_blocks"][-1] == "persona..."
    assert r["schema"] is ACTION_SCHEMA


def test_action_schema_is_strict_and_flat():
    # OpenAI strict mode rejects a root-level anyOf and requires every
    # property listed in `required`; Anthropic accepts the same shape.
    assert ACTION_SCHEMA["type"] == "object"
    assert ACTION_SCHEMA["additionalProperties"] is False
    assert set(ACTION_SCHEMA["required"]) == set(ACTION_SCHEMA["properties"])


def test_every_configured_model_has_a_price():
    assert ROUTINE_MODEL in PRICES
    assert PROVIDER in ("openai", "anthropic")


@pytest.mark.live
def test_one_real_call():
    from sim.llm import complete
    assert complete(ROUTINE_MODEL, "Answer with one word.", "Say OK.", 16)
