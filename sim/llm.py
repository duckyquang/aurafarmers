"""Anthropic driver. Credentials resolve like any SDK app: ANTHROPIC_API_KEY,
ANTHROPIC_AUTH_TOKEN, or the active `ant auth login` profile — so a pilot can
run on an existing Claude subscription with no API key set."""
import json
import time

import anthropic

ROUTINE_MODEL = "claude-haiku-4-5"
HEAVY_MODEL = "claude-sonnet-5"

IN_TOKENS = 0
OUT_TOKENS = 0
CACHE_READ_TOKENS = 0

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _tally(usage):
    global IN_TOKENS, OUT_TOKENS, CACHE_READ_TOKENS
    IN_TOKENS += usage.input_tokens
    OUT_TOKENS += usage.output_tokens
    CACHE_READ_TOKENS += getattr(usage, "cache_read_input_tokens", 0) or 0


# -- action schema (strict: additionalProperties false everywhere) ---------
_STR = {"type": "string"}
_CALL = {"type": "object", "additionalProperties": False,
         "properties": {
             "kind": {"type": "string", "enum": ["observe", "intervene"]},
             "targets": {"type": "array", "items": {
                 "type": "object", "additionalProperties": False,
                 "properties": {"node": _STR, "value": {"type": "number"}},
                 "required": ["node", "value"]}},
             "measure": {"type": "array", "items": _STR},
             "n": {"type": "integer"}},
         "required": ["kind", "targets", "measure", "n"]}
_CLAIM = {"type": "object", "additionalProperties": False,
          "properties": {
              "type": {"type": "string",
                       "enum": ["edge", "null", "interaction", "mechanism"]},
              "cause": _STR,
              "effect": _STR,
              "causes": {"type": "array", "items": _STR},
              "parents": {"type": "array", "items": _STR},
              "sign": {"type": "string", "enum": ["+", "-"]},
              "strength": {"type": "string",
                           "enum": ["weak", "moderate", "strong"]}},
          "required": ["type", "effect"]}


def _variant(action, props=None, req=None):
    p = {"action": {"type": "string", "const": action}}
    p.update(props or {})
    return {"type": "object", "properties": p,
            "required": ["action"] + (req or []),
            "additionalProperties": False}


ACTION_SCHEMA = {"anyOf": [
    _variant("research", {"calls": {"type": "array", "items": _CALL}}, ["calls"]),
    _variant("write", {"title": _STR, "body": _STR,
                       "claims": {"type": "array", "items": _CLAIM},
                       "cites": {"type": "array", "items": _STR},
                       "evidence": {"type": "array", "items": _STR}},
             ["title", "body", "claims", "evidence"]),
    _variant("review", {"paper_id": _STR, "accept": {"type": "boolean"},
                        "text": _STR}, ["accept", "text"]),
    _variant("read"),
    _variant("collaborate"),
    _variant("talk"),
    _variant("idle"),
    _variant("exit"),
]}


def build_request(custom_id, model, system_blocks, user_text, schema,
                  max_tokens):
    system = [{"type": "text", "text": b} for b in system_blocks]
    system[-1]["cache_control"] = {"type": "ephemeral"}
    params = {"model": model, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user_text}]}
    if schema:
        params["output_config"] = {"format": {"type": "json_schema",
                                              "schema": schema}}
    return {"custom_id": custom_id, "params": params}


def _parse(msg):
    _tally(msg.usage)
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def run_batch(requests):
    """Batch API path (50% discount) — the full-study driver."""
    if not requests:
        return {}
    batch = client().messages.batches.create(requests=requests)
    while True:
        st = client().messages.batches.retrieve(batch.id)
        if st.processing_status == "ended":
            break
        time.sleep(15)
    out = {}
    for res in client().messages.batches.results(batch.id):
        out[res.custom_id] = (_parse(res.result.message)
                              if res.result.type == "succeeded" else None)
    return out


def run_direct(requests):
    """Sequential path for small pilots and debugging.
    # ponytail: sequential; thread pool if pilot latency ever hurts"""
    out = {}
    for r in requests:
        try:
            out[r["custom_id"]] = _parse(client().messages.create(**r["params"]))
        except anthropic.APIError:
            out[r["custom_id"]] = None
    return out


def complete(model, system, user, max_tokens):
    msg = client().messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}])
    _tally(msg.usage)
    return next((b.text for b in msg.content if b.type == "text"), "")
