"""Run provenance and logging.

A result you cannot trace back to exact conditions is not a result. Every run
gets its own directory holding everything needed to audit or reproduce it:

    runs/<utc-timestamp>-<name>/
        manifest.json   config, seed, model, git SHA, dirty flag, code hash,
                        wall clock, and final token/cost totals
        events.jsonl    every world event -- one JSON object per line, the
                        source of truth every metric folds over
        llm.jsonl       every model call: full prompt, full response, tokens
                        (including hidden reasoning tokens), cost, latency,
                        finish reason, and any error
        trace.log       the same story in human-readable form

Nothing here interprets anything. It records. Analysis reads the files back.
"""
import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_run = None


def current():
    return _run


def set_run(run):
    global _run
    _run = run


def _git(*args):
    try:
        return subprocess.run(("git",) + args, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _code_hash():
    """Fingerprint of the simulation source, so a run can be tied to the exact
    logic that produced it even if the tree was dirty."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


class Run:
    def __init__(self, name, config=None, root="runs", echo=False):
        self.name = name
        self.started = time.time()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(root) / f"{stamp}-{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.counts = {}
        self._lock = threading.Lock()      # agents run in worker threads
        self._events = open(self.dir / "events.jsonl", "a")
        self._llm = open(self.dir / "llm.jsonl", "a")
        self._trace = open(self.dir / "trace.log", "a")
        self.manifest = {
            "name": name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config": config or {},
            "git_sha": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "code_sha256": _code_hash(),
            "python": os.sys.version.split()[0],
        }
        self._write_manifest()
        self.note(f"run {name} -> {self.dir}")
        if self.manifest["git_dirty"]:
            self.note("WARNING: working tree is dirty; this run is not "
                      "reproducible from the commit alone")

    # -- writers ----------------------------------------------------------
    def _write_manifest(self):
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str) + "\n")

    def event(self, tick, agent_id, event, payload=None):
        """One world event. This is what every metric folds over."""
        row = {"t": round(time.time() - self.started, 3), "tick": tick,
               "agent_id": agent_id, "event": event, "payload": payload or {}}
        with self._lock:
            self._events.write(json.dumps(row, default=str) + "\n")
            self._events.flush()
            self.counts[event] = self.counts.get(event, 0) + 1
        return row

    def llm_call(self, *, custom_id, model, system, user, schema, response,
                 usage, latency, finish_reason=None, error=None, cost=None):
        """Every model call, in full. Prompts included: without them a
        surprising behaviour cannot be investigated after the fact."""
        row = {"t": round(time.time() - self.started, 3),
               "custom_id": custom_id, "model": model,
               "latency_s": round(latency, 3),
               "finish_reason": finish_reason, "error": error,
               "usage": usage, "cost_usd": cost,
               "schema": bool(schema),
               "system": system, "user": user, "response": response}
        with self._lock:
            self._llm.write(json.dumps(row, default=str) + "\n")
            self._llm.flush()
            self.counts["llm_call"] = self.counts.get("llm_call", 0) + 1
            err = bool(error)
            if err:
                self.counts["llm_error"] = self.counts.get("llm_error", 0) + 1
        if err:
            self.note(f"  llm ERROR {custom_id}: {error}")
        return row

    def note(self, msg):
        """Human-readable narration. Read this first when something looks
        wrong, then go to the JSONL for detail."""
        line = f"[{time.time() - self.started:8.2f}s] {msg}"
        with self._lock:
            self._trace.write(line + "\n")
            self._trace.flush()
        if self.echo:
            print(line, flush=True)

    def section(self, title):
        self.note("")
        self.note(f"=== {title} ===")

    # -- lifecycle --------------------------------------------------------
    def close(self, result=None):
        from sim import llm
        self.manifest.update({
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(time.time() - self.started, 2),
            "event_counts": self.counts,
            "tokens": {"in": llm.IN_TOKENS, "out": llm.OUT_TOKENS,
                       "reasoning": llm.REASONING_TOKENS,
                       "cache_read": llm.CACHE_READ_TOKENS},
            "cost_usd": round(llm.spend(), 6),
            "model": {"provider": llm.PROVIDER, "routine": llm.ROUTINE_MODEL,
                      "heavy": llm.HEAVY_MODEL},
            "result": result,
        })
        self._write_manifest()
        self.note(f"done in {self.manifest['wall_seconds']}s, "
                  f"${self.manifest['cost_usd']:.5f}, "
                  f"{self.counts.get('llm_call', 0)} model calls")
        for f in (self._events, self._llm, self._trace):
            f.close()
        return self.manifest
