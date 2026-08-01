"""Model driver. Two providers, one interface.

Credentials come from the environment; nothing is ever hardcoded or logged.
    OpenAI     -> OPENAI_API_KEY
    Anthropic  -> ANTHROPIC_API_KEY, or an `ant auth login` profile

Pick a provider with AURAFARMERS_PROVIDER=openai|anthropic (default openai,
because it is currently much cheaper per token -- see PRICES).

NOTE ON THE EXPERIMENT: the study's claim boundary is scoped to whichever
model family actually ran. Mixing families inside one comparison confounds
it; running a second family as a separate arm is a designed-for robustness
check (see the v2 spec, threat T10). Pin the model for a whole study.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_env():
    """Pick up .env so no script needs the key exported in the shell."""
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

# $ per million tokens (input, output). Verified 2026-08-01 against each
# provider's pricing page; re-check before quoting costs.
PRICES = {
    "gpt-5-nano":       (0.05, 0.40),
    "gpt-4o-mini":      (0.15, 0.60),
    "gpt-4.1-nano":     (0.10, 0.40),
    "gpt-4.1-mini":     (0.40, 1.60),
    "gpt-4.1":          (2.00, 8.00),
    "gpt-5-mini":       (0.25, 2.00),
    "gpt-5.1":          (1.25, 10.00),
    "gpt-4o":           (2.50, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5":  (3.00, 15.00),
}

PROVIDER = os.environ.get("AURAFARMERS_PROVIDER", "openai").lower()
MODELS = {
    "openai":    {"routine": "gpt-5-nano",       "heavy": "gpt-4o-mini"},
    "anthropic": {"routine": "claude-haiku-4-5", "heavy": "claude-sonnet-5"},
}
ROUTINE_MODEL = MODELS[PROVIDER]["routine"]
HEAVY_MODEL = MODELS[PROVIDER]["heavy"]

# gpt-5* are REASONING models: they spend output tokens on hidden reasoning
# before writing anything. Starve the budget and you get finish_reason
# 'length' with empty content -- a silent failure that looks exactly like an
# incompetent model. Reasoning tokens are billed as output, so they are real
# money and are tracked separately below.
REASONING_MODELS = ("gpt-5",)
REASONING_EFFORT = "low"
REASONING_HEADROOM = 1400     # tokens reserved for reasoning, on top of output

IN_TOKENS = 0
OUT_TOKENS = 0
CACHE_READ_TOKENS = 0
REASONING_TOKENS = 0
COST_USD = 0.0          # accumulated per call at each model's own rate

_client = None
_lock = threading.Lock()


def client():
    global _client
    with _lock:
        if _client is not None:
            return _client
        if PROVIDER == "openai":
            import openai
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Run scripts/setup_key.py")
            _client = openai.OpenAI()
        else:
            import anthropic
            _client = anthropic.Anthropic()
        return _client


def spend():
    """Dollars burned so far, summed per call at each model's own rate."""
    return COST_USD


# --- the action schema ------------------------------------------------------
# Flat and fully-required, with nullable payloads. A root-level anyOf is
# rejected by OpenAI strict mode, so one shape serves both providers.
_STR = {"type": "string"}
_NSTR = {"type": ["string", "null"]}
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
                       "enum": ["association", "edge", "null", "interaction",
                                "mechanism"]},
              "cause": _NSTR, "effect": _STR,
              "causes": {"type": ["array", "null"], "items": _STR},
              "parents": {"type": ["array", "null"], "items": _STR},
              "sign": {"type": ["string", "null"], "enum": ["+", "-", None]},
              "strength": {"type": ["string", "null"],
                           "enum": ["weak", "moderate", "strong", None]}},
          "required": ["type", "cause", "effect", "causes", "parents",
                       "sign", "strength"]}

ACTION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "action": {"type": "string",
                   "enum": ["research", "write", "review", "read",
                            "collaborate", "talk", "idle", "exit"]},
        "calls": {"type": ["array", "null"], "items": _CALL},
        "title": _NSTR, "body": _NSTR,
        "claims": {"type": ["array", "null"], "items": _CLAIM},
        "cites": {"type": ["array", "null"], "items": _STR},
        "evidence": {"type": ["array", "null"], "items": _STR},
        "paper_id": _NSTR,
        "accept": {"type": ["boolean", "null"]},
        "text": _NSTR,
    },
    "required": ["action", "calls", "title", "body", "claims", "cites",
                 "evidence", "paper_id", "accept", "text"],
}


def build_request(custom_id, model, system_blocks, user_text, schema,
                  max_tokens):
    """Provider-neutral request envelope; translated at call time."""
    return {"custom_id": custom_id, "model": model,
            "system_blocks": list(system_blocks), "user": user_text,
            "schema": schema, "max_tokens": max_tokens}


def is_reasoning(model):
    return any(model.startswith(p) for p in REASONING_MODELS)


def _tally(usage, provider):
    global IN_TOKENS, OUT_TOKENS, CACHE_READ_TOKENS, REASONING_TOKENS
    with _lock:
        if provider == "openai":
            IN_TOKENS += usage.prompt_tokens
            OUT_TOKENS += usage.completion_tokens
            det = getattr(usage, "prompt_tokens_details", None)
            CACHE_READ_TOKENS += getattr(det, "cached_tokens", 0) or 0
            cd = getattr(usage, "completion_tokens_details", None)
            REASONING_TOKENS += getattr(cd, "reasoning_tokens", 0) or 0
        else:
            IN_TOKENS += usage.input_tokens
            OUT_TOKENS += usage.output_tokens
            CACHE_READ_TOKENS += getattr(usage, "cache_read_input_tokens",
                                         0) or 0


def _call(req):
    """One request, provider-appropriate. Returns parsed JSON or raw text.
    Every call is recorded in full if a trace Run is active. Cost comes from
    the call's own usage dict — global deltas are racy under threads."""
    from sim import trace
    run, t0 = trace.current(), time.time()
    try:
        out, usage, finish = _dispatch(req)
    except Exception as e:
        if run:
            run.llm_call(custom_id=req["custom_id"], model=req["model"],
                         system="\n\n".join(req["system_blocks"]),
                         user=req["user"], schema=req["schema"],
                         response=None, usage=None,
                         latency=time.time() - t0, error=repr(e)[:400])
        raise
    global COST_USD
    pin, pout = PRICES.get(req["model"], (0, 0))
    cost = (usage["in"] * pin + usage["out"] * pout) / 1e6
    with _lock:
        COST_USD += cost
    if run:
        run.llm_call(custom_id=req["custom_id"], model=req["model"],
                     system="\n\n".join(req["system_blocks"]),
                     user=req["user"], schema=req["schema"],
                     response=out, usage=usage, latency=time.time() - t0,
                     finish_reason=finish, cost=round(cost, 8))
    return out


def _dispatch(req):
    """Returns (parsed_or_text, usage_dict, finish_reason)."""
    if PROVIDER == "openai":
        cap = req["max_tokens"]
        kw = {"model": req["model"],
              "messages": [{"role": "system",
                            "content": "\n\n".join(req["system_blocks"])},
                           {"role": "user", "content": req["user"]}]}
        if is_reasoning(req["model"]):
            kw["reasoning_effort"] = REASONING_EFFORT
            cap += REASONING_HEADROOM
        kw["max_completion_tokens"] = cap
        if req["schema"]:
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "strict": True,
                                "schema": req["schema"]}}
        r = client().chat.completions.create(**kw)
        _tally(r.usage, "openai")
        text = r.choices[0].message.content or ""
        finish = r.choices[0].finish_reason
        cd = getattr(r.usage, "completion_tokens_details", None)
        usage = {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens,
                 "reasoning": getattr(cd, "reasoning_tokens", 0) or 0}
        if not text and finish == "length":
            raise RuntimeError(
                f"{req['model']} spent its whole budget on reasoning and "
                f"returned nothing; raise max_tokens (cap was {cap})")
    else:
        system = [{"type": "text", "text": b} for b in req["system_blocks"]]
        system[-1]["cache_control"] = {"type": "ephemeral"}
        kw = {"model": req["model"], "max_tokens": req["max_tokens"],
              "system": system,
              "messages": [{"role": "user", "content": req["user"]}]}
        if req["schema"]:
            kw["output_config"] = {"format": {"type": "json_schema",
                                              "schema": req["schema"]}}
        r = client().messages.create(**kw)
        _tally(r.usage, "anthropic")
        text = next((b.text for b in r.content if b.type == "text"), "")
        finish = r.stop_reason
        usage = {"in": r.usage.input_tokens, "out": r.usage.output_tokens,
                 "reasoning": 0}
    if not req["schema"]:
        return text, usage, finish
    try:
        return (json.loads(text) if text else None), usage, finish
    except json.JSONDecodeError:
        return None, usage, finish


def run_direct(requests):
    """Sequential. Fine for pilots; use run_batch for a real study.
    # ponytail: no thread pool -- add one only if pilot latency actually hurts
    """
    out = {}
    for r in requests:
        for attempt in range(3):
            try:
                out[r["custom_id"]] = _call(r)
                break
            except Exception:
                if attempt == 2:
                    out[r["custom_id"]] = None
                else:
                    time.sleep(2 ** attempt)
    return out


def run_parallel(requests, workers=8):
    """Threaded direct calls: one tick's agent turns run concurrently.
    Counters are ints under the GIL and trace.Run is lock-guarded."""
    if not requests:
        return {}
    def one(r):
        for attempt in range(3):
            try:
                return r["custom_id"], _call(r)
            except Exception:
                if attempt == 2:
                    return r["custom_id"], None
                time.sleep(2 ** attempt)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(one, requests))


def run_batch(requests):
    """Anthropic batch API (50% off). OpenAI batching has a different
    submit/poll shape; falls back to sequential there for now.
    # ponytail: add OpenAI /v1/batches when a full study is actually funded
    """
    if PROVIDER != "anthropic" or not requests:
        return run_direct(requests)
    payload = []
    for r in requests:
        system = [{"type": "text", "text": b} for b in r["system_blocks"]]
        system[-1]["cache_control"] = {"type": "ephemeral"}
        params = {"model": r["model"], "max_tokens": r["max_tokens"],
                  "system": system,
                  "messages": [{"role": "user", "content": r["user"]}]}
        if r["schema"]:
            params["output_config"] = {"format": {"type": "json_schema",
                                                  "schema": r["schema"]}}
        payload.append({"custom_id": r["custom_id"], "params": params})
    batch = client().messages.batches.create(requests=payload)
    while client().messages.batches.retrieve(batch.id).processing_status \
            != "ended":
        time.sleep(15)
    out = {}
    for res in client().messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            out[res.custom_id] = None
            continue
        msg = res.result.message
        _tally(msg.usage, "anthropic")
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            out[res.custom_id] = json.loads(text) if text else None
        except json.JSONDecodeError:
            out[res.custom_id] = None
    return out


def complete(model, system, user, max_tokens):
    return _call(build_request("x", model, [system], user, None, max_tokens))
