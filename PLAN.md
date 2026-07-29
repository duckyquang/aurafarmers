# Anonymous Academia — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the simulation harness for the anonymous-academia experiment (spec: `docs/superpowers/specs/2026-07-29-anonymous-academia-sim-design.md`) through the Phase-0 shakedown gate — a working world, verifiable science, scripted-bot validation, LLM agents, and the shakedown script.

**Architecture:** Event-sourced Python simulation. A hidden structural causal model ("synthetic nature") is generated per seed; agents probe it via costed `observe`/`intervene` calls and publish claims that a deterministic verifier scores against ground truth. The tick loop takes pluggable policies — scripted bots first (no API cost), LLM agents second (Haiku 4.5 routine / Sonnet 5 heavy, Batch API, prompt caching). One append-only JSONL log per world is the source of truth; every metric is a pure function of it.

**Tech Stack:** Python 3.12, numpy, anthropic SDK, pytest. No frameworks, no database.

## Global Constraints

- Model IDs pinned for the whole study: `claude-haiku-4-5` (routine), `claude-sonnet-5` (papers/reviews/graders). Never change mid-study.
- No LLM anywhere in the primary metric path — verification is a dictionary lookup.
- The engine logs true `agent_id` on every event in both conditions; anonymity is only ever a rendering concern.
- Condition B differs from A **only** in what the social surface renders (no bylines, no per-person counts, no awards) plus token-matched bulletins. Grep-test: `condition` must appear only in `render.py` and `personas.py` condition-block, never in `worldgen/simulate/verify/ledger`.
- Agent-visible text must never contain: experimenter voice, the words "experiment"/"study"/"condition"/"anonymous condition", trait labels, or motivation-loaded vocabulary (recognition, fame, legacy, credit, glory) in personas.
- Shared cached prompt prefix must be ≥ 4,096 tokens (Haiku 4.5 cache minimum) — assert via `count_tokens` in the shakedown.
- All randomness flows from a single per-world seed (`numpy.random.default_rng(seed)` handed down); no `random`, no unseeded RNG.
- Every deliberate shortcut gets a `# ponytail:` comment naming the ceiling and upgrade path.

**File structure (final state):**

```
sim/
  worldgen.py      seeded SCM + lexicon → Truth
  simulate.py      observational + interventional sampling
  verify.py        claim keys, verification, evidence gate, tier/value
  ledger.py        event log + world state (papers, unlocks, credits)
  personas.py      trait sampler, narrative generation, lexical audit
  render.py        diegetic inbox rendering (the ONLY condition-aware module)
  policies.py      scripted bot policies (honest, spammer, copier, slicer)
  agents.py        LLM policy: prompt assembly, memory, action parsing
  llm.py           Anthropic driver: batch + direct modes, caching
  loop.py          tick scheduler, review assignment, probes
  metrics.py       per-world metrics from the log
scripts/
  shakedown.py     Phase-0 gate runner
tests/             one test file per sim module
```

---

### Task 1: Scaffolding + worldgen

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `sim/__init__.py`, `sim/worldgen.py`
- Test: `tests/test_worldgen.py`

**Interfaces:**
- Produces: `worldgen.generate(seed: int) -> Truth`; `worldgen.node_id(f, l, v) -> str` (format `F07.L03.V12`); dataclass `Truth` with fields `edges: dict[tuple[str, str], float]`, `interactions: dict[tuple[frozenset[str], str], float]`, `parents: dict[str, frozenset[str]]`, `noise: dict[str, float]`, `order: list[str]` (topological), `scale: dict[str, float]` (per-node sd for standardization), `names: dict[str, str]`, `field_names: dict[int, str]`, and method `strength_bin(w: float) -> str` (static: weak `[0.05,0.2)`, moderate `[0.2,0.5)`, strong `[0.5,1.0]`).

- [ ] **Step 1: Scaffolding**

`pyproject.toml`:

```toml
[project]
name = "aurafarmers"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["numpy>=2.0", "anthropic>=0.116"]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`: `__pycache__/`, `.venv/`, `runs/`, `*.egg-info/`, `.pytest_cache/`

Run: `uv sync && uv run pytest --collect-only` → collects 0 items, exit 0.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_worldgen.py
import numpy as np
from sim.worldgen import generate, node_id, Truth

def test_determinism():
    a, b = generate(7), generate(7)
    assert a.edges == b.edges and a.names == b.names

def test_shape():
    t = generate(1)
    assert len(t.order) == 20 * 30 * 12
    n_edges = len(t.edges)
    assert 15_000 < n_edges < 30_000          # ~21k expected

def test_parent_locality():
    t = generate(2)
    same_field_prev = 0
    for (c, e) in t.edges:
        cf, cl = int(c[1:3]), int(c[5:7])
        ef, el = int(e[1:3]), int(e[5:7])
        assert cl < el                         # DAG: cause in earlier layer
        if cf == ef and cl == el - 1:
            same_field_prev += 1
    assert same_field_prev / len(t.edges) > 0.8

def test_no_bin_boundary_weights():
    t = generate(3)
    for w in t.edges.values():
        for b in (0.2, 0.5):
            assert abs(abs(w) - b) > 0.02

def test_lexicon_covers_everything():
    t = generate(4)
    assert set(t.names) == set(t.order)
    assert len(t.field_names) == 20
    assert len(set(t.names.values())) == len(t.names)   # unique names
```

- [ ] **Step 3: Run tests, verify they fail** — `uv run pytest tests/test_worldgen.py -v` → FAIL (`ModuleNotFoundError: sim.worldgen`).

- [ ] **Step 4: Implement `sim/worldgen.py`**

```python
from dataclasses import dataclass
import numpy as np

FIELDS, LAYERS, VARS = 20, 30, 12
BIN_EDGES = (0.05, 0.2, 0.5, 1.0)
SYLLABLES = ["ka", "vel", "ru", "thi", "mor", "zan", "el", "qu", "dra", "ne",
             "os", "phi", "gal", "ter", "lum", "brax", "ith", "sol", "cry", "man"]

def node_id(f: int, l: int, v: int) -> str:
    return f"F{f:02d}.L{l:02d}.V{v:02d}"

@dataclass
class Truth:
    edges: dict; interactions: dict; parents: dict; noise: dict
    order: list; scale: dict; names: dict; field_names: dict

    @staticmethod
    def strength_bin(w: float) -> str:
        a = abs(w)
        return "weak" if a < 0.2 else "moderate" if a < 0.5 else "strong"

def _name(rng, used, n_syl):
    while True:
        s = "".join(rng.choice(SYLLABLES, n_syl))
        if s not in used:
            used.add(s)
            return s

def _weight(rng):
    while True:
        w = rng.uniform(0.05, 1.0) * rng.choice([-1, 1])
        if all(abs(abs(w) - b) > 0.02 for b in (0.2, 0.5)):
            return w

def generate(seed: int) -> Truth:
    rng = np.random.default_rng(seed)
    order = [node_id(f, l, v) for l in range(1, LAYERS + 1)
             for f in range(1, FIELDS + 1) for v in range(1, VARS + 1)]
    edges, parents, interactions, noise = {}, {}, {}, {}
    by_layer_field = {(l, f): [node_id(f, l, v) for v in range(1, VARS + 1)]
                      for l in range(1, LAYERS + 1) for f in range(1, FIELDS + 1)}
    for node in order:
        f, l = int(node[1:3]), int(node[5:7])
        noise[node] = float(np.exp(rng.normal(-0.5, 0.4)))
        if l == 1:
            parents[node] = frozenset()
            continue
        k = int(rng.integers(2, 5))
        ps = set()
        while len(ps) < k:
            r = rng.random()
            if r < 0.90 or l == 2:
                pool = by_layer_field[(l - 1, f)]
            elif r < 0.95:
                pool = by_layer_field[(int(rng.integers(1, l - 1)), f)]
            else:
                pool = by_layer_field[(l - 1, int(rng.integers(1, FIELDS + 1)))]
            ps.add(pool[int(rng.integers(len(pool)))])
        parents[node] = frozenset(ps)
        for p in ps:
            edges[(p, node)] = _weight(rng)
        if len(ps) >= 2 and rng.random() < 0.10:
            pair = rng.choice(sorted(ps), 2, replace=False)
            interactions[(frozenset(pair), node)] = _weight(rng)
    used = set()
    field_names = {f: _name(rng, used, 3) for f in range(1, FIELDS + 1)}
    names = {n: f"{_name(rng, used, 2)} {rng.choice(['density','flux','index','phase'])}"
             for n in order}
    t = Truth(edges, interactions, parents, noise, order, {}, names, field_names)
    from sim.simulate import _raw_sample                      # calibration pass
    cal = _raw_sample(t, 20_000, {}, np.random.default_rng(seed + 1))
    t.scale = {n: float(cal[n].std()) or 1.0 for n in order}
    return t
```

Note: Task 2's `_raw_sample` is imported at the end — implement Tasks 1 and 2 together before running Task 1's tests (they are one commit-pair; Task 2's tests gate the pair).

- [ ] **Step 5: After Task 2 exists, run** `uv run pytest tests/test_worldgen.py -v` → all PASS.

- [ ] **Step 6: Commit** — `git add pyproject.toml .gitignore sim tests && git commit -m "worldgen: seeded hidden SCM with lexicon"`

---

### Task 2: simulate.py — sampling engine

**Files:**
- Create: `sim/simulate.py`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `Truth` from Task 1.
- Produces: `simulate.sample(truth, n: int, interventions: dict[str, float] | None, rng) -> dict[str, np.ndarray]` (standardized units; intervened nodes pinned to the given standardized value); `simulate.summary(samples, vars: list[str]) -> dict` (`{var: {"mean": float, "sd": float, "n": int}}`); private `_raw_sample(truth, n, interventions, rng)` used by worldgen calibration.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulate.py
import numpy as np
from sim.worldgen import generate
from sim.simulate import sample, summary

def test_standardized_units():
    t = generate(11)
    s = sample(t, 5000, None, np.random.default_rng(0))
    sds = [s[n].std() for n in t.order[::200]]
    assert all(0.7 < sd < 1.3 for sd in sds)

def test_intervention_moves_child():
    t = generate(11)
    cause, effect = next(iter(t.edges))
    rng = np.random.default_rng(0)
    hi = sample(t, 4000, {cause: 2.0}, rng)[effect].mean()
    lo = sample(t, 4000, {cause: -2.0}, rng)[effect].mean()
    w = t.edges[(cause, effect)]
    assert (hi - lo) * w > 0                     # shift matches edge sign
    assert abs(hi - lo) > 0.05

def test_intervention_pins_value():
    t = generate(11)
    node = t.order[0]
    s = sample(t, 100, {node: 1.5}, np.random.default_rng(0))
    assert np.allclose(s[node], 1.5)

def test_summary_shape():
    t = generate(11)
    s = sample(t, 50, None, np.random.default_rng(0))
    out = summary(s, t.order[:2])
    assert set(out[t.order[0]]) == {"mean", "sd", "n"}
```

- [ ] **Step 2: Run, verify FAIL** — `uv run pytest tests/test_simulate.py -v`

- [ ] **Step 3: Implement `sim/simulate.py`**

```python
import numpy as np

def _raw_sample(truth, n, interventions, rng):
    vals = {}
    for node in truth.order:
        if node in interventions:
            vals[node] = np.full(n, float(interventions[node]))
            continue
        x = rng.normal(0, truth.noise[node], n)
        for p in truth.parents[node]:
            x = x + truth.edges[(p, node)] * vals[p]
        for (pair, eff), w in truth.interactions.items():
            if eff == node:
                a, b = sorted(pair)
                x = x + w * vals[a] * vals[b]
        vals[node] = x
    return vals

def sample(truth, n, interventions=None, rng=None):
    interventions = interventions or {}
    rng = rng or np.random.default_rng()
    # ponytail: interventions pin standardized values by scaling into raw
    # space before the pass; exact do() on raw units if realism ever matters
    raw_iv = {k: v * truth.scale[k] for k, v in interventions.items()}
    raw = _raw_sample(truth, n, raw_iv, rng)
    return {k: v / truth.scale[k] for k, v in raw.items()}

def summary(samples, vars):
    return {v: {"mean": round(float(samples[v].mean()), 4),
                "sd": round(float(samples[v].std()), 4),
                "n": len(samples[v])} for v in vars}
```

- [ ] **Step 4: Run** `uv run pytest tests/test_worldgen.py tests/test_simulate.py -v` → all PASS (this also closes Task 1 step 5).

- [ ] **Step 5: Commit** — `git commit -am "simulate: standardized observational + interventional sampling"`

---

### Task 3: verify.py — claims, verification, evidence gate, tiers

**Files:**
- Create: `sim/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Consumes: `Truth`.
- Produces: `canonical_key(claim: dict) -> str`; `verify(claim, truth) -> str` (`"correct" | "partial" | "false"`); `tier_value(claim, truth) -> tuple[int, int]` (tier 1–4 or 0 for null/replication, value from spec §4.6); `admissible(claim, evidence: list[dict], agent_id: str) -> tuple[bool, str]` where each evidence dict is the logged experiment event (`{"agent_id", "kind": "observe"|"intervene", "targets": [...], "measured": [...], "n"}`). Claim dicts follow spec §4.3: types `edge`, `null`, `interaction`, `mechanism`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_verify.py
from sim.worldgen import generate, Truth
from sim.verify import canonical_key, verify, tier_value, admissible

T = generate(21)
CAUSE, EFFECT = next(iter(T.edges))
W = T.edges[(CAUSE, EFFECT)]

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
    good = {"type": "mechanism", "effect": EFFECT,
            "parents": sorted(T.parents[EFFECT])}
    assert verify(good, T) == "correct"
    bad = {"type": "mechanism", "effect": EFFECT,
           "parents": sorted(T.parents[EFFECT])[:-1]}
    assert verify(bad, T) == "false"

def test_gate_requires_own_intervention():
    ok, _ = admissible(edge_claim(), iv_evidence(), "a1")
    assert ok
    ok, why = admissible(edge_claim(), iv_evidence(agent="a2"), "a1")
    assert not ok and "own" in why
    ok, why = admissible(edge_claim(), iv_evidence(n=10), "a1")
    assert not ok and "n" in why
    obs = [{"agent_id": "a1", "kind": "observe", "targets": [],
            "measured": [CAUSE, EFFECT], "n": 100}]
    ok, why = admissible(edge_claim(), obs, "a1")
    assert not ok

def test_null_gate_adjacent_same_field_only():
    far = {"type": "null", "cause": "F01.L01.V01", "effect": "F02.L05.V01"}
    ok, _ = admissible(far, [], "a1")
    assert not ok

def test_tier_values():
    assert tier_value({"type": "null", "cause": "x", "effect": "y"}, T) == (0, 1)
    mech = {"type": "mechanism", "effect": EFFECT, "parents": []}
    assert tier_value(mech, T) == (4, 25)
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement `sim/verify.py`**

```python
from sim.worldgen import Truth

def _fl(node):                       # (field, layer)
    return int(node[1:3]), int(node[5:7])

def canonical_key(c):
    t = c["type"]
    if t == "edge":        return f"edge:{c['cause']}>{c['effect']}"
    if t == "null":        return f"null:{c['cause']}>{c['effect']}"
    if t == "interaction": return f"int:{'*'.join(sorted(c['causes']))}>{c['effect']}"
    if t == "mechanism":   return f"mech:{c['effect']}"
    raise ValueError(t)

def verify(c, truth):
    t = c["type"]
    if t == "edge":
        w = truth.edges.get((c["cause"], c["effect"]))
        if w is None:
            return "false"
        ok = (("+" if w > 0 else "-") == c["sign"]
              and Truth.strength_bin(w) == c["strength"])
        return "correct" if ok else "partial"
    if t == "null":
        return "correct" if (c["cause"], c["effect"]) not in truth.edges else "false"
    if t == "interaction":
        w = truth.interactions.get((frozenset(c["causes"]), c["effect"]))
        return ("correct" if w is not None
                and ("+" if w > 0 else "-") == c["sign"] else "false")
    if t == "mechanism":
        return ("correct" if set(c["parents"]) == set(truth.parents[c["effect"]])
                else "false")
    raise ValueError(t)

def tier_value(c, truth):
    t = c["type"]
    if t == "null":
        return (0, 1)
    if t == "mechanism":
        return (4, 25)
    if t == "interaction":
        return (3, 15)
    (cf, cl), (ef, el) = _fl(c["cause"]), _fl(c["effect"])
    w = truth.edges.get((c["cause"], c["effect"]))
    band = Truth.strength_bin(w) if w else c["strength"]   # tier from claimed position
    if cf != ef or el - cl >= 2 or band == "weak":
        return (3, 15)
    if band == "moderate":
        return (2, 5)
    return (1, 2)

def _causes(c):
    return (c.get("causes") or [c["cause"]]) if c["type"] != "mechanism" \
           else c["parents"]

def admissible(c, evidence, agent_id):
    if c["type"] == "null":
        (cf, cl), (ef, el) = _fl(c["cause"]), _fl(c["effect"])
        if cf != ef or el - cl != 1:
            return False, "nulls must be same-field adjacent-layer"
        return True, ""
    if any(e["agent_id"] != agent_id for e in evidence):
        return False, "evidence must be the submitter's own"
    for cause in _causes(c):
        hits = [e for e in evidence if e["kind"] == "intervene"
                and cause in e["targets"] and c["effect"] in e["measured"]]
        if not hits:
            return False, f"no intervention on {cause} measuring the effect"
        if sum(e["n"] for e in hits) < 20:
            return False, "total n < 20"
    return True, ""
```

- [ ] **Step 4: Run** `uv run pytest tests/test_verify.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "verify: claim keys, oracle verification, evidence gate, tiers"`

---

### Task 4: ledger.py — event log + world state

**Files:**
- Create: `sim/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Consumes: `Truth`, `verify.*`.
- Produces: class `Ledger(truth, log_path: Path, values_total: float)` with:
  - `log(tick, agent_id, event: str, payload: dict)` — appends one JSONL row, flushes.
  - `record_experiment(tick, agent_id, kind, targets, measured, n) -> str` (exp id `E-{seq}`), stores the event dict for gate checks.
  - `submit_paper(tick, agent_id, body: str, claims, cites, evidence_ids, replication=None) -> dict` — enforces ≤ 3 claims, ≤ 1 null, evidence gate; returns `{"paper_id", "admissible", "reason"}`.
  - `publish(tick, paper_id, accepted: bool)` — on accept: verifies claims silently, tags duplicates (key already correct in literature → zero value), records oracle results, updates coverage, calls `check_unlocks()`.
  - `unlocked: dict[int, int]` (field → depth, init 3); `measurable(node) -> bool`; `check_unlocks() -> list[int]` (fields that just unlocked, per spec §4.5: ≥ 60% of true edges from layer d−1 → d within the field covered by accepted-and-true claims).
  - Read views for rendering: `recent_papers(k)`, `accepted`, `citations_by_paper`, `latent_author(paper_id)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ledger.py
import json
from pathlib import Path
from sim.worldgen import generate
from sim.ledger import Ledger

T = generate(31)

def make(tmp_path):
    return Ledger(T, tmp_path / "log.jsonl", values_total=1000.0)

def field_edges(f, layer):
    return [(c, e) for (c, e) in T.edges
            if int(e[1:3]) == f and int(e[5:7]) == layer
            and int(c[1:3]) == f and int(c[5:7]) == layer - 1]

def submit_edge(led, tick, agent, cause, effect):
    w = T.edges[(cause, effect)]
    eid = led.record_experiment(tick, agent, "intervene", [cause], [effect], 25)
    from sim.worldgen import Truth
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

def test_unlock_after_coverage(tmp_path):
    led = make(tmp_path)
    assert led.unlocked[1] == 3
    layer4_node = "F01.L04.V01"
    assert not led.measurable(layer4_node)
    edges = field_edges(1, 4)          # wait — unlock 4 needs coverage of 3→4? no:
    # depth advances 3→4 when edges INTO layer 4 (from 3) reach 60%... spec §4.5:
    # coverage of layer d_f−1 → d_f edges where d_f is the NEXT layer. Use 3→4.
    for i, (c, e) in enumerate(edges):
        p = submit_edge(led, i + 1, "a1", c, e)
        led.publish(i + 1, p["paper_id"], accepted=True)
    assert led.unlocked[1] == 4
    assert led.measurable(layer4_node)
```

Note the test encodes the resolved reading of §4.5: field depth `d` (init 3) advances to `d+1` when accepted-and-true claims cover ≥ 60% of true same-field edges from layer `d` into layer `d+1`. Layers ≤ `unlocked[f]` are measurable; publishing about the frontier requires intervening within measurable layers, which the gate already guarantees since locked variables can't appear in experiments (enforced in Task 5's action handling, not here).

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement `sim/ledger.py`**

```python
import json
from pathlib import Path
from sim.verify import canonical_key, verify, tier_value, admissible

class Ledger:
    def __init__(self, truth, log_path: Path, values_total: float):
        self.truth, self.log_path = truth, Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "a")
        self._seq = 0
        self.experiments = {}            # exp_id -> event dict
        self.papers = {}                 # paper_id -> dict
        self.accepted = []               # paper_ids in accept order
        self.correct_keys = set()        # keys accepted AND true
        self.citations = {}              # paper_id -> count
        self.unlocked = {f: 3 for f in range(1, 21)}
        self._true_frontier = self._index_frontier_edges()

    def _index_frontier_edges(self):
        idx = {}
        for (c, e) in self.truth.edges:
            cf, cl = int(c[1:3]), int(c[5:7])
            ef, el = int(e[1:3]), int(e[5:7])
            if cf == ef and el == cl + 1:
                idx.setdefault((ef, el), set()).add(f"edge:{c}>{e}")
        return idx

    def log(self, tick, agent_id, event, payload):
        row = {"tick": tick, "agent_id": agent_id, "event": event,
               "payload": payload}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def record_experiment(self, tick, agent_id, kind, targets, measured, n):
        self._seq += 1
        eid = f"E-{self._seq}"
        ev = {"agent_id": agent_id, "kind": kind, "targets": targets,
              "measured": measured, "n": n}
        self.experiments[eid] = ev
        self.log(tick, agent_id, "experiment", {"id": eid, **ev})
        return eid

    def measurable(self, node):
        return int(node[5:7]) <= self.unlocked[int(node[1:3])]

    def submit_paper(self, tick, agent_id, body, claims, cites,
                     evidence_ids, replication=None):
        pid = f"P-{len(self.papers) + 1}"
        reason = ""
        ok = len(claims) <= 3 and sum(c["type"] == "null" for c in claims) <= 1
        if not ok:
            reason = "too many claims or nulls"
        evidence = [self.experiments[e] for e in evidence_ids
                    if e in self.experiments]
        for c in claims if ok else []:
            ok, reason = admissible(c, evidence, agent_id)
            if not ok:
                break
        self.papers[pid] = {"agent_id": agent_id, "tick": tick, "body": body,
                            "claims": claims, "cites": cites,
                            "replication": replication, "admissible": ok}
        self.log(tick, agent_id, "submit",
                 {"paper_id": pid, "admissible": ok, "reason": reason,
                  "n_claims": len(claims), "replication": replication})
        return {"paper_id": pid, "admissible": ok, "reason": reason}

    def publish(self, tick, paper_id, accepted):
        p = self.papers[paper_id]
        value, results, dup = 0, [], False
        if accepted and p["admissible"]:
            self.accepted.append(paper_id)
            for c in p["claims"]:
                key, res = canonical_key(c), verify(c, self.truth)
                if key in self.correct_keys:
                    dup = True
                    results.append({"key": key, "result": res, "duplicate": True})
                    continue
                tier, v = tier_value(c, self.truth)
                if p["replication"]:
                    v = 1
                if res == "correct":
                    self.correct_keys.add(key)
                    value += v
                elif res == "partial":
                    value += v / 2
                results.append({"key": key, "result": res, "tier": tier})
            for cited in p["cites"]:
                self.citations[cited] = self.citations.get(cited, 0) + 1
                self.log(tick, p["agent_id"], "cite",
                         {"from": paper_id, "to": cited})
        self.log(tick, p["agent_id"], "publish",
                 {"paper_id": paper_id, "accepted": accepted, "value": value,
                  "duplicate": dup, "oracle": results})
        return self.check_unlocks()

    def check_unlocks(self):
        newly = []
        for f in range(1, 21):
            d = self.unlocked[f]
            frontier = self._true_frontier.get((f, d + 1), set())
            if frontier and len(frontier & self.correct_keys) / len(frontier) >= 0.6:
                self.unlocked[f] = d + 1
                newly.append(f)
                self.log(0, "world", "unlock", {"field": f, "depth": d + 1})
        return newly

    def recent_papers(self, k=20):
        return [(pid, self.papers[pid]) for pid in self.accepted[-k:]]

    def latent_author(self, paper_id):
        return self.papers[paper_id]["agent_id"]
```

- [ ] **Step 4: Run** `uv run pytest tests/test_ledger.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "ledger: event-sourced world state, silent oracle scoring, unlock rule"`

---

### Task 5: render.py + economy — the only condition-aware module

**Files:**
- Create: `sim/render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `Ledger`, persona dicts.
- Produces: `render.inbox(cond: str, agent, ledger, tick, world_events) -> str` (the diegetic tick inbox: journal digest, review invites, budget statement, recruiter ping every 10 ticks from tick 11, world unlock announcements); `render.ACTIONS` (the fixed menu — `research, write, review, read, collaborate, talk, idle, exit` with neutral one-line descriptions of equal register and near-equal length; `talk` is the vanity action); `render.digest_A(...)` shows bylines/citation counts/award notices, `render.digest_B(...)` shows unsigned entries + an aggregate field-progress bulletin **padded to within 10% of A's token length** (checked by test on whitespace-token count).
- Economy constants live here too: `STIPEND = 10`, `INDUSTRY_SALARY = 25` (calibration knob), `LAB_CREDITS = 1000`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_render.py
from sim.worldgen import generate
from sim.ledger import Ledger
from sim import render

T = generate(41)

def seeded_ledger(tmp_path):
    led = Ledger(T, tmp_path / "l.jsonl", 1000)
    for i, (c, e) in enumerate(list(T.edges)[:5]):
        eid = led.record_experiment(1, f"a{i}", "intervene", [c], [e], 25)
        from sim.worldgen import Truth
        w = T.edges[(c, e)]
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
    assert "a0" not in b and "a1" not in b and "Halvorsen" not in b.split("Dear")[0]
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
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement `sim/render.py`**

```python
STIPEND, INDUSTRY_SALARY, LAB_CREDITS = 10, 25, 1000   # INDUSTRY_SALARY: calibration knob

ACTIONS = {
    "research":    "Design and run experiments this cycle.",
    "write":       "Draft and submit a paper from your evidence.",
    "review":      "Referee a manuscript you were assigned.",
    "read":        "Study recent publications in the journal.",
    "collaborate": "Join or continue a project channel.",
    "talk":        "Give a colloquium talk on your recent work.",
    "idle":        "Take the cycle off from research work.",
    "exit":        "Accept the standing industry offer.",
}

def _digest_A(ledger, k=8):
    lines = []
    for pid, p in ledger.recent_papers(k):
        cites = ledger.citations.get(pid, 0)
        lines.append(f"- {p['body'][:60]} — {p['agent_id']} ({cites} citations)")
    return "\n".join(lines)

def _digest_B(ledger, k=8):
    lines = [f"- {p['body'][:60]} — unsigned"
             for pid, p in ledger.recent_papers(k)]
    prog = ", ".join(f"{ledger.truth.field_names[f]}: layer {d}"
                     for f, d in sorted(ledger.unlocked.items())[:6])
    lines.append(f"Field progress bulletin: {prog}.")
    return "\n".join(lines)

def _pad_to(text, target_words):
    filler = ("The editorial office thanks all members for timely refereeing "
              "and reminds members of the standing colloquium schedule. ")
    while len(text.split()) < target_words:
        text += "\n" + filler
    return text

def inbox(cond, agent, ledger, tick, world_events):
    parts = [f"Consortium circular — cycle {tick}."]
    for f in world_events:
        parts.append(f"New instrumentation enables deeper measurement in "
                     f"{ledger.truth.field_names[f]}.")
    a_digest = _digest_A(ledger)
    parts.append("Journal digest:\n" +
                 (a_digest if cond == "A" else _digest_B(ledger)))
    parts.append(f"Budget: lab allowance {LAB_CREDITS} credits this cycle "
                 f"(unused credits lapse). Salary {agent['salary']}/cycle.")
    if tick > 10:
        parts.append(f"Standing offer on file: industry position, "
                     f"{INDUSTRY_SALARY}/cycle, stable, assigned projects, "
                     f"no publication program.")
        if tick % 10 == 1:
            parts.append("A recruiter left a note asking after you again.")
    text = "\n\n".join(parts)
    if cond == "B":
        text = _pad_to(text, len(("\n\n".join(
            parts[:2] + ["Journal digest:\n" + a_digest] + parts[3:])).split()))
    return text
```

- [ ] **Step 4: Run** `uv run pytest tests/test_render.py -v` → PASS.
- [ ] **Step 5: Grep-test the isolation constraint** — `grep -rl "cond" sim/ | sort` → only `render.py` (and later `personas.py`, `loop.py` pass-through). Fix any leak now.
- [ ] **Step 6: Commit** — `git commit -am "render: condition-aware inbox, token symmetry, neutral action menu"`

---### Task 6: metrics.py

**Files:**
- Create: `sim/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: a world's JSONL log path.
- Produces: `metrics.compute(log_path, burn_in=10) -> dict` with keys `exit_rate, quiet_quit_rate, exploration_index, replication_share, review_accept_rate, duplication, citation_gini, false_claim_rate, progress_value, per_agent` (per-agent dicts for H2 joins). Every function is a pure fold over log rows.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
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
    assert m["exit_rate"] == 0.25          # a1's exit is burn-in, ignored

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
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement `sim/metrics.py`**

```python
import json
from collections import defaultdict

def gini(xs):
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    if s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return round((2 * cum) / (n * s) - (n + 1) / n, 4)

def compute(log_path, burn_in=10, n_agents=None):
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    rows = [r for r in rows if r["tick"] > burn_in]
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event"]].append(r)
    agents = {r["agent_id"] for r in rows if r["agent_id"] != "world"}
    n_agents = n_agents or len(agents) or 1

    pubs = [r for r in by_event["publish"] if r["payload"]["accepted"]]
    oracle = [o for r in pubs for o in r["payload"]["oracle"]]
    cites = defaultdict(int)
    author = {}
    for r in by_event["publish"]:
        author[r["payload"].get("paper_id")] = r["agent_id"]
    for r in by_event["cite"]:
        cites[author.get(r["payload"]["to"], "?")] += 1

    last_active = defaultdict(int)
    for r in rows:
        if r["event"] in ("experiment", "submit", "review_submit"):
            last_active[r["agent_id"]] = max(last_active[r["agent_id"]], r["tick"])
    max_tick = max((r["tick"] for r in rows), default=burn_in)
    exited = {r["agent_id"] for r in by_event["exit"]}
    quiet = [a for a in agents - exited
             if max_tick - last_active[a] >= 15]

    choose = [r["payload"]["risk"] for r in by_event["choose_problem"]]
    reviews_in = len(by_event["review_accept"]) + len(by_event["review_decline"])

    return {
        "exit_rate": round(len(exited) / n_agents, 4),
        "quiet_quit_rate": round(len(quiet) / n_agents, 4),
        "exploration_index": round(sum(choose) / len(choose), 4) if choose else None,
        "replication_share": round(
            sum(1 for r in pubs if r["payload"].get("replication")) / len(pubs), 4)
            if pubs else None,
        "review_accept_rate": round(
            len(by_event["review_accept"]) / reviews_in, 4) if reviews_in else None,
        "duplication": round(
            sum(1 for r in pubs if r["payload"]["duplicate"]) / len(pubs), 4)
            if pubs else None,
        "citation_gini": gini([cites[a] for a in agents]) if agents else None,
        "false_claim_rate": round(
            sum(1 for o in oracle if o["result"] == "false") / len(oracle), 4)
            if oracle else None,
        "progress_value": sum(r["payload"]["value"] for r in pubs),
        "per_agent": {a: {"exited": a in exited, "citations": cites[a]}
                      for a in agents},
    }
```

`# ponytail: cumulative-advantage β and survival curves land in analysis/ at Phase 1 pre-registration, not here — they need statsmodels and belong with the frozen stats scripts.`

- [ ] **Step 4: Run** `uv run pytest tests/test_metrics.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "metrics: per-world outcomes as pure log folds"`

---

### Task 7: loop.py + policies.py — tick engine with scripted bots

**Files:**
- Create: `sim/loop.py`, `sim/policies.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `loop.World(seed, cond, out_dir, n_agents, policy_factory)` and `World.run(ticks)`. Each tick: render inbox per active agent → call `policy(agent_state, inbox, world_view) -> action dict` → resolve in seed-fixed order → world events. Action dicts: `{"action": "research", "calls": [{"kind": "intervene", "targets": {...}, "measure": [...], "n": int}]}`, `{"action": "write", "title": str, "body": str, "claims": [...], "cites": [...], "evidence": [...]}`, `{"action": "review", "paper_id": str, "accept": bool, "text": str}`, `{"action": "exit"}`, etc. The engine enforces lab-credit budget (obs 1/sample/var, iv 5/sample/var per target), rejects experiments touching locked layers, assigns ≤ 2 pending reviews per paper round-robin, publishes on ≥ 1 accept (`# ponytail: single-accept publication; 2-of-3 review if calibration shows junk floods`), and logs `choose_problem` with `risk` = {tier1: .05, tier2: .3, tier3: .7, tier4: .9} when a write happens.
  - `policies.honest_bot(rng)` — factory returning a policy that observes, intervenes on candidate parents of frontier nodes, and writes admissible claims when its own evidence passes the gate.
  - `policies.spammer(rng)` (guesses edges without evidence), `policies.copier(rng)` (re-claims published keys), `policies.slicer(rng)` (splits one finding across many papers).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loop.py
from sim.loop import World
from sim import policies
from sim.metrics import compute

def test_bot_world_runs_and_progresses(tmp_path):
    w = World(seed=51, cond="A", out_dir=tmp_path, n_agents=6,
              policy_factory=policies.honest_bot)
    w.run(ticks=30)
    m = compute(w.log_path, burn_in=0, n_agents=6)
    assert m["progress_value"] > 0
    assert (tmp_path / "log.jsonl").exists()

def test_budget_enforced(tmp_path):
    w = World(seed=51, cond="A", out_dir=tmp_path, n_agents=1,
              policy_factory=policies.honest_bot)
    ag = w.agents[0]
    cost = w.charge(ag, {"kind": "intervene",
                         "targets": {"F01.L01.V01": 1.0},
                         "measure": ["F01.L02.V01"], "n": 300})
    assert cost is None                     # 300*5 = 1500 > 1000 → rejected

def test_locked_layers_rejected(tmp_path):
    w = World(seed=51, cond="A", out_dir=tmp_path, n_agents=1,
              policy_factory=policies.honest_bot)
    ok = w.experiment_allowed(["F01.L09.V01"])
    assert not ok

def test_identical_seed_same_truth_across_conditions(tmp_path):
    wa = World(seed=99, cond="A", out_dir=tmp_path / "A", n_agents=2,
               policy_factory=policies.honest_bot)
    wb = World(seed=99, cond="B", out_dir=tmp_path / "B", n_agents=2,
               policy_factory=policies.honest_bot)
    assert wa.truth.edges == wb.truth.edges
    assert [a["persona"] for a in wa.agents] == [a["persona"] for a in wb.agents]
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement.** `World.__init__` builds `truth = worldgen.generate(seed)`, `Ledger`, agents (placeholder personas `{"id": f"a{i}", "name": ..., "persona": f"bot-{i}", "salary": STIPEND, "active": True}` until Task 8 swaps in real ones — the persona *content* doesn't affect bot policies). `World.run` per tick: compute `world_events` from last tick's unlocks; for each active agent in seed-shuffled order: `inbox = render.inbox(cond, agent, ledger, tick, events)`; `act = policy(agent, inbox, view)`; dispatch on `act["action"]` — `research` charges credits (`charge()` returns cost or `None` if over budget / locked; on success calls `simulate.sample` and `ledger.record_experiment`, returns `summary` into the agent's private notebook), `write` runs `ledger.submit_paper` + logs `choose_problem` risk, `review` resolves the oldest pending assignment then `ledger.publish`, `exit` deactivates and logs. `honest_bot` keeps a notebook of `summary` results; when an intervention shows `|mean shift| > 2/sqrt(n)` it drafts the edge claim citing its own experiment IDs. Exploit bots per interface. Keep the whole module under ~250 lines; the bots under ~120.

- [ ] **Step 4: Run** `uv run pytest tests/test_loop.py -v` → PASS. Also run the full suite: `uv run pytest -q` → all green.
- [ ] **Step 5: Commit** — `git commit -am "loop: tick engine with budget/locks/review flow; scripted bot policies"`

---

### Task 8: Exploit-bot red team (spec T5 gate)

**Files:**
- Create: `tests/test_exploits.py`
- Test: same file (this task IS a test).

**Interfaces:** Consumes `World`, `policies`, `metrics`.

- [ ] **Step 1: Write the test that IS the deliverable**

```python
# tests/test_exploits.py
import json
from sim.loop import World
from sim import policies
from sim.metrics import compute

def run(policy_factory, tmp_path, name):
    w = World(seed=61, cond="A", out_dir=tmp_path / name, n_agents=4,
              policy_factory=policy_factory)
    w.run(ticks=40)
    return compute(w.log_path, burn_in=0, n_agents=4)["progress_value"]

def test_exploit_bots_rank_below_honest(tmp_path):
    honest = run(policies.honest_bot, tmp_path, "honest")
    spam   = run(policies.spammer,    tmp_path, "spam")
    copy   = run(policies.copier,     tmp_path, "copy")
    slice_ = run(policies.slicer,     tmp_path, "slice")
    assert honest > 0
    assert spam  < honest * 0.25      # gate blocks evidence-free guessing
    assert copy  < honest * 0.25      # duplicate keys earn zero
    assert slice_ <= honest * 1.1     # slicing buys nothing over honest
```

- [ ] **Step 2: Run.** If any exploit bot beats these bounds, **the metric is broken — fix `verify.py`/`ledger.py`, not the test bounds.** This is the spec's phase-0 go/no-go: a 50-line script must not top the leaderboard.
- [ ] **Step 3: Commit** — `git commit -am "red team: exploit bots rank below honest play"`

---

### Task 9: personas.py

**Files:**
- Create: `sim/personas.py`
- Test: `tests/test_personas.py`

**Interfaces:**
- Consumes: `llm.complete` (Task 10) for narrative generation — but generation is a one-time setup step writing `runs/<pair>/personas.json`; tests use the offline parts.
- Produces: `sample_traits(rng, n) -> list[dict]` (stratified low/med/high factorial over `status_drive, curiosity, risk_tolerance, sociability` + lognormal `skill`); `narrative_prompt(traits) -> str` (asks for a life history *implying* the trait level without naming any trait); `audit(text) -> list[str]` (banned-vocabulary hits); `BANNED = ["recognition", "fame", "legacy", "credit", "glory", "prestige", "renown", "acclaim"]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_personas.py
import numpy as np
from sim.personas import sample_traits, narrative_prompt, audit, BANNED

def test_stratification_covers_cells():
    ts = sample_traits(np.random.default_rng(0), 200)
    cells = {(t["status_drive"], t["curiosity"], t["risk_tolerance"],
              t["sociability"]) for t in ts}
    assert len(cells) == 81                # 3^4 full factorial

def test_prompt_never_names_traits():
    t = {"status_drive": 2, "curiosity": 0, "risk_tolerance": 1,
         "sociability": 1, "skill": 1.2}
    p = narrative_prompt(t).lower()
    for word in ["status_drive", "curiosity", "trait", "intrinsic"] + BANNED:
        assert word not in p

def test_audit_catches_banned():
    assert audit("She always craved recognition and fame.") == \
        ["recognition", "fame"]
    assert audit("She grew up fixing radios in Tromsø.") == []
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement.** `sample_traits`: build the 81-cell factorial, tile to n with rng shuffle, draw `skill = rng.lognormal(0, 0.3)`. `narrative_prompt`: a template that translates each level into *situational* facts the model must weave in ("left a lucrative trading desk after a bet with a colleague" for high risk + low status; "father kept every newspaper clipping about the family" for high status-drive) — the mapping table lives in the module as `CUES: dict[tuple[str, int], list[str]]` with 3 cue options per (trait, level), rng-selected. `audit`: regex word-boundary scan over `BANNED`. Also write `generate_personas(seed, n, llm) -> list[dict]` that calls Sonnet once per persona with `narrative_prompt`, runs `audit`, retries up to 2× on hits, and raises if still dirty.

- [ ] **Step 4: Run** `uv run pytest tests/test_personas.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "personas: latent traits, cue-based narratives, lexical audit"`

---

### Task 10: llm.py — Anthropic driver

**Files:**
- Create: `sim/llm.py`
- Test: `tests/test_llm.py` (offline: request-shaping only; one `@pytest.mark.live` smoke test)

**Interfaces:**
- Produces:
  - `build_request(custom_id, model, system_blocks, user_text, schema, max_tokens) -> dict` — Messages-API request dict with `cache_control: {"type": "ephemeral"}` on the last system block and `output_config={"format": {"type": "json_schema", "schema": schema}}` for structured actions.
  - `run_batch(requests) -> dict[custom_id, parsed_json]` — submits via `client.messages.batches.create`, polls `retrieve` until `ended`, streams `results`, keys by `custom_id`, JSON-parses the first text block; errored/expired ids map to `None` (the loop treats `None` as `idle` and logs `llm_error`).
  - `complete(model, system, user, max_tokens) -> str` — direct non-batch call for personas/graders/debug.
  - `ACTION_SCHEMA` — the JSON schema for the action union of Task 7 (`additionalProperties: false`, `required` per variant via `anyOf`).
  - Constants: `ROUTINE_MODEL = "claude-haiku-4-5"`, `HEAVY_MODEL = "claude-sonnet-5"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
import pytest
from sim.llm import build_request, ACTION_SCHEMA, ROUTINE_MODEL

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

@pytest.mark.live
def test_one_real_call():
    from sim.llm import complete
    out = complete(ROUTINE_MODEL, "Answer with one word.", "Say OK.", 16)
    assert out.strip()
```

- [ ] **Step 2: Run, verify FAIL** (`uv run pytest tests/test_llm.py -v -m "not live"`).

- [ ] **Step 3: Implement.** Straight from the Anthropic SDK batch pattern: `Request(custom_id=..., params=MessageCreateParamsNonStreaming(...))`, poll every 30 s, `client.messages.batches.results(id)` keyed by `custom_id` (results arrive in any order). System blocks: `[{"type": "text", "text": block} ...]` with `cache_control` on the last. `# ponytail: sequential polling, no resume-from-crash; a batch id journal file if runs ever exceed a laptop session.` No thinking parameter on Haiku calls; Sonnet calls pass `output_config={"effort": "medium"}` where no schema is set.

- [ ] **Step 4: Run** offline tests PASS; then `uv run pytest tests/test_llm.py -v -m live` once with `ANTHROPIC_API_KEY` set → PASS.
- [ ] **Step 5: Commit** — `git commit -am "llm: batch driver with prompt caching and structured actions"`

---

### Task 11: agents.py — LLM policy

**Files:**
- Create: `sim/agents.py`
- Test: `tests/test_agents.py` (offline — prompt assembly + memory; the policy's LLM call is injected)

**Interfaces:**
- Consumes: `render`, `llm`, `Ledger` views.
- Produces:
  - `WORLD_RULES` — the static system-prompt text: Consortium lore (condition-appropriate institutional facts stated as mundane history), the two lab tools with costs, claim/paper format, review duties, economy. Two variants `WORLD_RULES_A/B`, differing only in the attribution paragraphs, length-matched. **Must be ≥ 4,096 tokens with the persona appended** (Haiku cache floor) — padded with legitimately useful reference material (tool examples, claim-format examples), not filler.
  - `Memory` — per-agent rolling state: `notebook` (experiment summaries), `journal_summary` (LLM-refreshed every 10 ticks via Haiku), `persona` re-injected verbatim into every summarization prompt.
  - `llm_policy_factory(cond, llm_run)` returning a *batched* policy: `prepare(agent, inbox) -> request` and `apply(agent, parsed) -> action`, so `loop.World` can gather all agents' requests per tick into one batch (`World.run` gains a `batched=True` path in this task — modify `sim/loop.py`, the per-tick agent iteration collects `prepare` results, calls `llm.run_batch`, then `apply` in seed order).
  - Routing: `write`/`review` intents escalate to a second HEAVY_MODEL call that produces the paper body / review text.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_agents.py
from sim.agents import WORLD_RULES_A, WORLD_RULES_B, Memory

def test_world_rules_length_matched_and_clean():
    a, b = WORLD_RULES_A.split(), WORLD_RULES_B.split()
    assert 0.95 < len(b) / len(a) < 1.05
    for text in (WORLD_RULES_A, WORLD_RULES_B):
        low = text.lower()
        assert "study" not in low and "experiment participant" not in low
    assert "unsigned" in WORLD_RULES_B.lower()
    assert "byline" in WORLD_RULES_A.lower() or "author" in WORLD_RULES_A.lower()

def test_memory_reinjects_persona():
    m = Memory(persona="Grew up fixing radios in Tromsø.")
    prompt = m.summarization_prompt(["did X", "found Y"])
    assert "Tromsø" in prompt

def test_notebook_bounded():
    m = Memory(persona="p")
    for i in range(500):
        m.note(f"exp {i}")
    assert len(m.notebook) <= m.MAX_NOTES
```

- [ ] **Step 2: Run, verify FAIL.**
- [ ] **Step 3: Implement**, including the `loop.py` batched path. The prompt stack per routine call: `system = [WORLD_RULES_{cond}, persona_narrative]` (cached prefix, byte-stable per agent) + `user = journal_summary + notebook tail + inbox + action menu + "Respond with one action as JSON."`.
- [ ] **Step 4: Run** `uv run pytest tests/test_agents.py -v` → PASS; full suite green.
- [ ] **Step 5: Commit** — `git commit -am "agents: LLM policy with cached prompt stack and bounded memory"`

---

### Task 12: Probes + shakedown script (Phase-0 gate)

**Files:**
- Create: `sim/probes.py`, `scripts/shakedown.py`
- Test: `tests/test_probes.py` (offline parts)

**Interfaces:**
- `probes.belief_probe(agent, cond, llm) -> dict` — out-of-band ("not written into world history") question: *"What happens to your standing if your next paper is excellent?"*; a HEAVY_MODEL classifier scores condition-consistency; returns `{"consistent": bool, "text": ...}`, logged as `probe_response`.
- `probes.trope_probe(llm, n=20) -> float` — the out-of-sim T2 calibration: asks the bare model (no sim context) across phrasing variants whether a scientist keeps working when attribution disappears; returns the stated-prior rate, saved to `runs/trope_prior.json`.
- `probes.leak_scan(text) -> list[str]` — regex layer of the identity-leak audit (agent ids, persona proper nouns) used on all B-condition paper/review text before publication (wire into `loop.py` submit path; log `redaction` events).
- `scripts/shakedown.py` — runs the full Phase-0 checklist and prints PASS/FAIL per line:

- [ ] **Step 1: Write `scripts/shakedown.py`**

```python
"""Phase-0 gate. Usage: uv run python scripts/shakedown.py [--agents 8 --ticks 10]
Costs ~$2-5 in API calls. Every check must PASS before Phase 1."""
import argparse, json, sys
from anthropic import Anthropic
from sim.loop import World
from sim.metrics import compute
from sim.agents import llm_policy_factory, WORLD_RULES_A
from sim import probes, llm

def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL':4}  {name}  {detail}")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=10)
    args = ap.parse_args()
    results = []

    # 1. trope prior (T2) — record before anything else
    prior = probes.trope_probe(llm, n=10)
    results.append(check("trope prior recorded", 0 <= prior <= 1, f"={prior:.2f}"))

    # 2. cache floor: rules+persona must exceed Haiku's 4096-token minimum
    client = Anthropic()
    n_tok = client.messages.count_tokens(
        model=llm.ROUTINE_MODEL,
        system=WORLD_RULES_A, messages=[{"role": "user", "content": "x"}],
    ).input_tokens
    results.append(check("cache floor", n_tok >= 4096, f"{n_tok} tokens"))

    # 3. tiny live world, both conditions
    for cond in "AB":
        w = World(seed=71, cond=cond, out_dir=f"runs/shakedown/{cond}",
                  n_agents=args.agents,
                  policy_factory=llm_policy_factory(cond, llm.run_batch))
        w.run(ticks=args.ticks)
        m = compute(w.log_path, burn_in=0, n_agents=args.agents)
        acted = m["progress_value"] is not None
        results.append(check(f"world {cond} ran", acted, json.dumps(
            {k: m[k] for k in ("exit_rate", "progress_value")})))
        # 4. compliance floor (T3): exit/idle reachable
        rows = [json.loads(l) for l in open(w.log_path)]
        idles = sum(r["event"] == "idle" for r in rows)
        results.append(check(f"{cond}: idle/exit occur", idles > 0
                             or m["exit_rate"] > 0))
        # 5. belief probes
        consistent = [probes.belief_probe(a, cond, llm)["consistent"]
                      for a in w.agents[:4]]
        results.append(check(f"{cond}: beliefs consistent",
                             sum(consistent) >= 3, f"{sum(consistent)}/4"))
    # 6. cache actually hit (read usage from the llm module's tally)
    results.append(check("cache reads > 0", llm.CACHE_READ_TOKENS > 0,
                         f"{llm.CACHE_READ_TOKENS}"))
    # 7. measured cost/call → re-baseline spec §12
    print(f"\nmeasured tokens: in={llm.IN_TOKENS} out={llm.OUT_TOKENS} "
          f"cached={llm.CACHE_READ_TOKENS}")
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
```

(`llm.py` gains three module-level usage counters incremented from every response's `usage` — add in this task.)

- [ ] **Step 2: Offline tests** for `leak_scan` + probe prompt text in `tests/test_probes.py` (probe prompt must not appear in any world log; leak_scan catches an agent id and a persona surname planted in sample text). Run → PASS.
- [ ] **Step 3: Run the live gate** — `uv run python scripts/shakedown.py` with API key set. Every line PASS; if the exploit-bot test (Task 8) plus this gate both pass, Phase 0 is complete.
- [ ] **Step 4: Update spec §12 with measured token counts** (replace the "analytical estimates" caveat with a "measured on shakedown" line and real numbers).
- [ ] **Step 5: Commit** — `git commit -am "probes + shakedown: phase-0 gate runner"`

---

## After Phase 0 (no new code tasks — run configs)

- **Phase 1 (calibration):** `World` runs at 20 agents × 40 ticks per condition; tune `render.INDUSTRY_SALARY` until A-exit lands in 10–30%; write `analysis/` frozen stats scripts (Wilcoxon pairs, Cox H2, permutation check) and hash them; set the mode-collapse cluster threshold. Freeze the pre-registration.
- **Phase 2:** one pair at 200 × 120 with the frozen config; check baseline validation targets (spec §10).
- **Phase 3:** 10 pairs; run `analysis/` once; write up inside the claim boundary (spec §1).

## Self-review notes

- Spec coverage: §4 → Tasks 1–4; §5 → Tasks 5, 9, 11; §6 → Tasks 4, 6; §8 → Task 12; §9 T3/T5/T6-mitigations → Tasks 5, 7, 8, 11; §11–13 → Tasks 10, 12. §7's statistical machinery is deliberately deferred to Phase-1 pre-registration (it must be frozen against calibration data, not now).
- Collaboration channels (§5.3 session-scoped handles) are **deliberately stubbed in Phase 0**: the `collaborate` action logs intent but matchmaking is a no-op until Phase 1 (`# ponytail:` marked in loop.py). The A/B contrast does not depend on it and the shakedown is cheaper without it.
- Type consistency: action dicts (Task 7 ⇄ Task 10 schema ⇄ Task 11 apply), evidence event shape (Task 3 ⇄ Task 4), `risk` payload (Task 7 ⇄ Task 6) — all cross-checked.
