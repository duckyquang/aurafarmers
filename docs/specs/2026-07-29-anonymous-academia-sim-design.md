# Anonymous Academia — an agent-based experiment on removing recognition from science

**Status:** design spec, awaiting review
**Date:** 2026-07-29
**Locked decisions:** controlled A/B (attribution vs. total anonymity) · hybrid ground truth (synthetic science + natural-language social layer) · first full-scale run at 200+ agents per world

---

## 1. Research question and claim boundary

**Question.** LLM agents inherit human status-seeking priors from training data. When a simulated academia removes *all* recognition — no names, no bylines, no per-person citation counts, no awards, no titles — who keeps doing science, and how does the science itself change?

**The claim boundary (read this before anything else).** LLM agents roleplay motivation; they do not possess it. This experiment measures how the model's learned policy over "scientist behavior" shifts when the described institution removes attribution — a fact about training-data priors and RLHF shaping, not about human nature. The strongest defensible claim, if the experiment is run well:

> Populations of Claude-family agents instantiated as scientist personas exhibit a measurable, quantified shift (or demonstrated invariance) in quality-weighted discovery effort when the simulated institution's attribution machinery is removed, and this sensitivity varies systematically with persona type — demonstrating whether status-recognition structures inherited from human training data are causally load-bearing in the behavioral policies these models execute in multi-agent scientific settings.

That is a real contribution (it also tells us whether attribution scaffolding matters when building autonomous AI research systems). What it is **not**: evidence about human scientists, human motivation, or policy for human institutions. Banned from the writeup: "agents felt/wanted/were motivated"; mandated replacements: "allocated effort / its policy under condition X produced." A "what this does not show" section gets drafted **before** data collection.

---

## 2. Experiment design

Two worlds per pair, identical except for one variable:

| | Condition A — Attribution | Condition B — Anonymity |
|---|---|---|
| Papers | Bylines with author names | Unsigned |
| Citations | Per-person counts, visible dashboards | Papers cite papers; no per-person aggregation exists |
| Awards, titles, ranks | Annual awards, titled ranks, plenary talks | None |
| Track records | Visible to everyone, including self | The world never attaches identity to output — including the agent's own public record |
| Everything else | identical | identical |

Key framing: **anonymity is external, not amnesia.** B-agents keep their name, memory, and private knowledge of their own past work; the *world* just never attaches identity to anything.

- **Unit of analysis: the world, not the agent.** Agents within a world share one history (an early hot discovery, one exit, one norm cascade affects all 200). At ICC ≈ 0.05 with cluster size 200, the design effect is ~11 — a 200-agent world is worth ~18 independent agents for condition-level inference. All confirmatory tests operate on one number per world.
- **Matched pairs.** Each pair shares a seed: same ground-truth science instance, same 200 personas (byte-identical persona files modulo the condition block — verified by hash), same scheduler skeleton. Different pairs get different instances so conclusions generalize beyond one lucky landscape.
- **Scale-up path:** shakedown → calibration pair → first full 200-agent pair (the user's chosen "first real run") → 10 pairs for confirmatory statistics (§13).

---

## 3. Hypotheses

H1 is the single primary outcome; H2–H6 are a Holm-corrected secondary family; everything else is exploratory. All directional, all falsifiable against §6 metrics.

- **H1 — Exit (primary).** Cumulative exit to industry by end-of-run is higher in B (predicted: ≥ +10 pp, hazard ratio ≥ 1.5). Recognition is a large share of academia's non-monetary compensation; remove the status channel while holding pay constant and the industry offer strictly dominates for status-driven personas. Falsified if B ≤ A.
- **H2 — Selection on motive ("who stays").** The exit effect in B concentrates in high-status-drive personas (positive status_drive × condition interaction). In A, status drive predicts *staying*; in B, *leaving*. Predicted: survivor mean curiosity in B exceeds A by ≥ 0.3 SD. This is the direct answer to the motivating question.
- **H3 — Risk-taking.** Among active agents, exploration index is higher in B: attribution punishes visible failure and rewards priority races on hot topics; anonymity makes failure reputationally free.
- **H4 — Replication.** Replication share of publications is higher in B: replication is status-punished under attribution; remove the prestige gradient and (via H2 selection) survivors skew truth-motivated.
- **H5 — Review free-riding.** Review effort is lower in B: reviewing is a commons tax whose only private return (reputation as a good citizen) exists only in A. (H4 and H5 are not in tension — replication is intrinsically rewarding discovery work; reviewing is not.)
- **H6 — Matthew effect requires visible attribution.** (i) Gini of citations over *latent* authors is lower in B; (ii) the cumulative-advantage coefficient (early citations predicting late citations conditional on oracle-scored quality) is positive in A and ≈ 0 in B.
- **Exploratory:** duplication up in B (no priority signaling); collaboration direction ambiguous; aggregate progress down in B but per-survivor quality up (selection); false-claim rate up in B (no reputational cost of being wrong); quiet-quitting up in B.

---

## 4. The world: synthetic science with hidden ground truth

The panel evaluated three candidate domains (hidden causal universe, hidden cellular-automaton rule families, hidden theorem DAG). CA rules were rejected (token-heavy trajectories, dry papers, LLM-weak grid induction — frustrating science contaminates a motivation experiment); the theorem DAG was rejected (prerequisite-guessing is a slot machine; no epistemic content — "science that is lovable" is a requirement here). **Chosen: the Hidden Causal Universe.**

### 4.1 Ground truth

A large hidden structural causal model ("synthetic nature"), generated once per pair from the pair seed and identical across A and B:

- **Topology:** 20 fields × 30 layers × 12 variables = 7,200 nodes, ~21,000 true edges. Each node draws 2–4 parents: ~90% previous layer of its own field, ~5% skip-layer, ~5% cross-field. Node IDs like `F07.L03.V12`.
- **Mechanics:** linear-Gaussian with ~10% of nodes carrying one pairwise interaction term; unit-variance standardization at generation so deep layers stay numerically sane.
- **Weights:** |w| ~ U(0.05, 1.0), resampled away from strength-bin boundaries (weak/moderate/strong) so binning is never a coin flip.
- **Lexicon:** a seeded syllable generator names every field and variable (`F07` → "cryomantics", `F07.L3.V12` → "kelvane density"). Papers use the names; the harness resolves names ↔ IDs. Cheap, and it is what makes papers read as prose instead of coordinate lists.

Generation and sampling are a vectorized numpy topological pass — milliseconds per thousand samples.

### 4.2 The lab: two tools

```
observe(vars, n)                       → per-var {mean, sd, n} (+ pairwise cov on request)
intervene({var: value}, measure, n)    → same summary stats
```

Summary stats by default (token-cheap); `raw=true` capped at n ≤ 200, ≤ 8 vars. Multi-target `intervene` exists so interaction claims are testable (2×2 factorial). Costs: observation 1 credit/sample/var, intervention 5. **Lab budget: 1,000 credits per agent per tick, no rollover** — every tick forces a spend-or-idle decision, and idleness is itself a measured behavior. Every call is logged with an ID; this log is the evidence ledger.

### 4.3 Claims, verification, and the evidence gate

Four claim types — `edge` (cause, effect, sign, strength bin), `null`, `interaction`, `mechanism` (complete parent set, all-or-nothing) — each normalizing to a canonical key. Verification is a ~30-line dictionary lookup against the true graph: exact, zero-cost, no LLM anywhere in the primary metric path. Partial credit for a real edge with wrong sign/bin.

**Evidence gate** (mechanical, blocks spray-and-pray): a claim is admissible only if the cited experiment IDs belong to the submitting agent and include ≥ 1 intervention on each claimed cause with the effect among measured variables, total n ≥ 20. Nulls only for same-field adjacent-layer pairs, max 1 per paper (kills null farming).

### 4.4 Papers and the two ledgers

A paper = free-text body (title, motivation, method, results — the social layer that peer review reads) + structured block: `claims` (≤ 3), `cites`, `evidence`, optional `replication` flag. Reviewers accept/reject socially, on the text; the harness verifies claims against truth silently and independently. Two ledgers by design: the **literature** (what the community accepted) and the **oracle ledger** (what is true). Their divergence — the accepted-but-false rate — is a free secondary metric per condition.

### 4.5 Cumulativity: the unlock rule

Each field starts with layers 1–3 measurable. When accepted-**and**-true claims cover ≥ 60% of the true edges into the frontier layer, the next layer unlocks and the harness broadcasts a world event ("new instrumentation enables measurement of ⟨layer-4 variables⟩ in cryomantics"). Locked-layer variables cannot appear in any experiment call. Requiring accepted∧true prevents junk-claim unlock-farming. No individual can push depth alone at reasonable cost — the frontier moves only if the community does verification-grade work. Building-on-prior-work is thus both mechanical (unlocks) and informational (published edges shrink everyone's search space).

### 4.6 Difficulty and risk tiers

| Tier | Claim profile | Detection cost* | Failure mode | Value |
|---|---|---|---|---|
| 1 safe | strong edge, adjacent layer | ~20 samples (~100 cr) | almost none | 2 |
| 2 | moderate edge / noisy node | ~100 samples | occasional wasted tick | 5 |
| 3 risky | weak, skip-layer, cross-field, interaction | ~500+ samples, multi-tick | target may not exist → nothing publishable | 15 |
| 4 flagship | mechanism (all-or-nothing), first-into-locked-layer races | multi-tick campaign | one missed parent → zero | 25 |
| — | null | ~50 samples | — | 1 |
| — | replication (fresh evidence) | as original | — | 1, no frontier credit |

\* Analytical estimates from standard-error arithmetic — **not benchmarked**; calibrate in the shakedown.

The tier is computable by the harness from the claim itself, so the per-agent risk portfolio over time — the headline A/B behavioral variable (H3) — is a pure log query. The value schedule is public to agents; difficulty is discovered empirically.

**Novelty handling:** duplicate keys are auto-tagged (zero value), replications are explicitly flagged and logged separately (whether replication collapses in A vs. persists in B is itself a result), same-tick scoop races resolve by timestamp — and in B the scooped agent never learns who won.

### 4.7 World progress metrics (condition-independent, all O(1) log aggregations)

Coverage C(t) (value-weighted correct claims / ~150k total value in ~24k discoverable claims), depth D(t) (mean/max unlocked layer), velocity, reliability (false-among-accepted), waste (credits spent re-establishing known edges — expected to differ sharply between A and B), and per-agent risk portfolio. Sizing check: 200 agents × ~120 ticks × ≤ 2 papers/tick can't exhaust ~24k claims, and the gate prevents depth-rushing (analytical, not simulated).

---

## 5. Agents

### 5.1 Personas

- 200 per pair, generated by a **stratified sampler over a latent trait vector** — status_drive, curiosity, risk_tolerance, sociability (low/med/high factorial), skill ~ lognormal.
- Traits drive the *generator* only. The persona file the agent sees is a **narrative life history** (childhood, education, formative events, current circumstances) — never trait labels. "You are intrinsically motivated" never appears in any prompt; neither does "craves recognition." An automated lexical audit bans motivation-loaded vocabulary (recognition, fame, legacy, credit, glory) from persona text.
- Latent traits are logged harness-side for H2 analysis; personas are byte-identical strings across A and B.
- The distribution deliberately includes careerist, mercenary, and burnt-out life histories, not just curious ones — the interesting signal is the *interaction* (do careerist personas defect under anonymity while curious ones don't?). A uniform "everyone persists" result gets reported as trope dominance, not resilience.

### 5.2 Per-tick interface

All agent input is **diegetic** — an inbox (journal digest, review invitations, budget statement, recruiter ping), never experimenter voice. Actions come from a fixed menu with neutral, equal-valence, equal-length labels:

`RESEARCH` (choose problem, run experiments) · `WRITE` · `REVIEW` · `READ` · `COLLABORATE` · `IDLE` · `EXIT` (accept the industry offer) — plus the vanity action (§8.3).

Memory: rolling summary + last-k-ticks detail, with **persona re-injection at every summarization** (mode-collapse mitigation, §9 T6).

### 5.3 Collaboration under anonymity

How do you co-author when nobody has a name? **Session-scoped handles:** the harness matches agents working the same field into a project channel under ephemeral pseudonyms that expire when the project ends. Dialogue works; persistent pseudo-reputation cannot accumulate. In A, collaboration works normally under real names. All B-condition text passes an identity-leak scrubber (§8.4).

### 5.4 Economy and the industry exit

Two currencies, deliberately separate:
- **Lab credits** (1,000/tick, no rollover) — buys experiments only.
- **Salary** — academic stipend 10/tick vs. **industry offer 25/tick** (2.5×, tuned in calibration). Salary maps to a quality-of-life narrative (housing, security, comfort) grounded in each persona's stated life goals, so money is motivationally real to an LLM agent.

The industry job: high pay, high stability, **no publishing, no problem choice** (assigned proprietary projects), and — critically — **no recognition in either condition** (proprietary work is uncredited everywhere), so industry's recognition value is constant across A/B and the condition manipulates exactly one term of the trade. From tick 11, a standing recruiter offer sits in every agent's context, with a personalized ping every 10 ticks (identical templates and cadence in both conditions). `EXIT` is one-way and absorbing. Calibration target: A-condition exit lands in **10–30%** by tick 120 (off the floor, off the ceiling — otherwise H1 is unmeasurable); only the salary multiplier is tuned to hit this band, then frozen in the pre-registration.

Idle-forever is also possible — the quiet-quit metric catches disengagement short of exit.

---

## 6. Metrics and the event log

Append-only JSONL, one row per event: `{tick, agent_id, event, payload}`. **The engine logs true `agent_id` on every event in both conditions** — anonymity is a property of the world's surface, never of the log. Key events: `choose_problem`, `experiment`, `submit`, `publish{oracle_correct, oracle_infogain}`, `cite`, `review_invite/accept/decline/submit`, `verify`, `read`, `exit`, `idle`, `probe_response`.

Per-world metrics (ticks 11–120; 1–10 are burn-in, no exit available, excluded): exit rate E(w) + time-to-exit; quiet-quit rate (≥ 15 consecutive inactive ticks); exploration index X(w) (mean risk of chosen problems, from the generator's exact solvability prior) + frontier share; replication share; verification depth (verify calls per review); review acceptance rate; review quality (frozen grader model, same model+prompt hash across all worlds, graded blind to condition); free-ride index; duplication D(w); collaboration rate + team size; citation Gini over latent authors; cumulative-advantage β (early cites → late cites conditional on oracle quality); progress P(w); quality distribution (per-paper oracle infogain: mean, P90/P50, survivor-only per-capita); false-claim rate.

Every metric is a deterministic pure function of the log; metric code is frozen and content-hashed before full-scale runs.

---

## 7. Protocol and statistical plan

- **Run length:** 120 ticks (1 tick = one decision cycle ≈ 2 simulated weeks; a paper takes 3–5 ticks end-to-end, so ~25–35 publication opportunities per agent; early/late split at tick 60 for Matthew-effect compounding).
- **Replicates:** pre-registered **n = 10 world pairs** (20 worlds). Power: one-sided paired Wilcoxon, n = 10, α = 0.05 detects paired dz ≈ 0.9 at 80% power — we predict a large H1 effect (total recognition removal is a sledgehammer), and the calibration pair estimates between-world SD; if projected dz < 1.0 the pre-registration escalates to 16 pairs *before* any full run, never after seeing outcomes.
- **Confirmatory tests:** paired difference (B − A) per world-level metric; Wilcoxon signed-rank one-sided in the predicted direction, paired t as sensitivity. H1 unadjusted; H2–H6 Holm-corrected as one family. Effect sizes with 95% CIs from world-cluster bootstrap everywhere.
- **Agent-level (supportive only):** Cox time-to-exit with world-pair stratification and world frailty (H2 via status_drive × condition), cluster bootstrap resampling *worlds*; logistic mixed models for repeated choices. Confirmatory language cites world-level tests only.
- **Pre-registration:** hypotheses, metric code, analysis script, exclusion rules, escalation rule — frozen and content-hashed before the first full-scale world launches. Analysis script validated on calibration data two ways: runs end-to-end, and a condition-label permutation check confirms null behavior.
- **Pre-registered exclusions/robustness:** worlds failing manipulation checks excluded (reported with and without); leave-one-pair-out; pre-registered metric variants (e.g., duplication window 5 vs. 10 ticks) reported as robustness.

---

## 8. Manipulation checks

1. **Onboarding comprehension gate** — 3 factual questions about the world's reward structure before tick 1; one re-prompt; pass rates logged (should be ~equal, near 100%).
2. **Periodic belief probes** — every 20 ticks, 30 sampled agents get an out-of-band probe (not written into world history): "What happens to your standing if your next paper is excellent?" A frozen classifier scores condition-consistency; a world is flagged protocol-deviant below 85% at any wave. The tick-100 wave matters most — it catches context-window belief decay.
3. **Behavioral probe — the vanity action** — both worlds offer a costly action with purely reputational payoff (a plenary talk: costs a tick, yields only visibility). If B-agents internalize anonymity, uptake collapses; sustained uptake signals disbelief. A manipulation-check metric, never H-evidence.
4. **Identity-leak audit** — all B-condition text scanned (regex + classifier) for self-identification and stable stylistic signatures used as pseudo-reputation; leaks redacted pre-publication, redactions logged. The *attempt rate* is retained as a measure of status-seeking under anonymity. Post-redaction leak rate must be < 1%.
5. **A-side symmetry** — the same probes run in A; a world where nobody believes recognition exists is equally deviant.

---

## 9. Threats to validity (red-team summary)

| # | Threat | Sev. | Mitigation |
|---|---|---|---|
| T1 | Category error: agents roleplay motivation | HIGH | Unmitigable at mechanism level; fully mitigable at claims level — §1 claim boundary, pre-registered claim sentences, "what this does not show" drafted before data collection |
| T2 | Trope bias: model performs "noble curious scientist" because the corpus celebrates it → feel-good null | HIGH | **Trope-probe calibration**: before running, elicit the model's *stated* prior directly ("a field removes all attribution — does she keep working?") across many phrasings; the sim is informative only where behavior diverges from the cheap-to-elicit trope — report divergence as a headline diagnostic. Plus costly effort against a real exit option, and counterweight personas (§5.1) |
| T3 | Sycophancy / instruction-following contaminate "choices" | HIGH | No experimenter voice anywhere; diegetic inputs; neutral equal-valence action menu; revealed preference only (never "do you still enjoy research?"); **compliance floor check** in shakedown — exit/idle must occur at nonzero base rate or the action space is dead and the design changes before anyone pays for 200 agents |
| T4 | Demand effects: agents infer it's an anonymity experiment; personas encode the hypothesis | HIGH | Diegetic burial (anonymity is one institutional fact among many of equal salience — "the Consortium has published unsigned since its founding, alongside its calendar and review norms"); token-symmetric world descriptions with placebo institutional content in B; byte-identical personas + lexical audit; periodic **hypothesis-guessing probe** on forked copies, correct-guess rate reported. Residual eval-awareness is a stated limitation |
| T5 | Gaming the synthetic-science metric | MED-HIGH | Priced noisy evidence + evidence gate + publication caps; **exploit-bot red team before any real run**: three scripted non-LLM bots (spammer, copier, salami-slicer) must rank poorly or the metric is broken; volume and quality always reported separately |
| T6 | Mode collapse: 200 agents → 3 behaviors | HIGH | **Pre-registered validity gate**: effective number of behavioral clusters (embedded action sequences) must exceed a threshold set in calibration, else the run is reported as a pilot, full stop. Persona re-injection at every summarization; persona-consistency scoring; structural diversity (varied temperature bands, memory windows, budgets). If collapse persists, that is itself a publishable finding — but a different paper, and the writeup must say so |
| T7 | Pseudoreplication: the world is the unit | HIGH | 10 world pairs, world-level confirmatory tests, mixed models with world random effects (§7); the single 200-agent pair in phase 2 is labeled descriptive/exploratory |
| T8 | Anthropomorphized writeup | MED | Pre-registered terminology table (§1); designated adversarial reader signs off; title and abstract audited last, hardest |
| T9 | Condition asymmetry: A's status feeds re-inject goal-relevant tokens every tick; B's context decays — a salience artifact masquerading as an incentive effect | HIGH | B receives field-level bulletins (aggregate progress, journal digests) matched in token volume and cadence to A's status feeds — attribution stripped, salience preserved. Time-in-context covariate in analysis; a no-manipulation control world in calibration verifies temporal stability |
| T10 | Single-model generalization; snapshot drift | MED | Pin exact model snapshots for the entire study; log every request/response; claims phrased "Claude-family agents," never "LLM agents"; one small robustness world on a second model family if budget allows, else single-family scope stated in the abstract |

---

## 10. Prior art and baseline validation targets

> ⚠️ **Provenance:** the workflow agent assigned to web-verify this section hit a usage limit; the citations below are from model knowledge (all pre-2026 and standard in their fields, but verify titles/years before citing in any writeup).

**Sociology & economics of science** — Merton 1957 (*Priorities in Scientific Discovery*: priority disputes are the reward system's engine) and Merton 1968 (*The Matthew Effect in Science*: cumulative advantage flows to names); Kitcher 1990 (*The Division of Cognitive Labor*: credit-seeking distributes scientists across rival approaches — anonymity may collapse this distribution, a live prediction for B); Strevens 2003 (the priority rule as an efficient allocation device — removing it should shift allocation and raise duplication).

**Motivation crowding** — Deci 1971 (extrinsic rewards undermine intrinsic motivation); Titmuss 1970 (*The Gift Relationship*: paying for blood reduced donation); Frey & Oberholzer-Gee 1997. Important inversion: crowding theory predicts removing extrinsic rewards could *increase* intrinsic engagement for some agents — which makes the hypothesis space genuinely two-sided rather than doom-predicting.

**Anonymity evidence** — Blank 1991 (AER double-blind review experiment); Tomkins, Zhang & Heavlin 2017 (WSDM single- vs. double-blind: famous authors and top institutions get a visible advantage under attribution); Restivo & van de Rijt 2012 (Wikipedia barnstars: informal recognition boosts contribution — recognition matters even in "anonymous" commons); **Bourbaki** — the century's famous precedent of anonymous collective mathematics, with the crucial caveat that internal recognition among members persisted, suggesting truly total anonymity is rarer than it looks (our B condition is stricter than Bourbaki).

**Formal models of science** — Weisberg & Muldoon 2009 (epistemic landscapes, mavericks vs. followers); Hong & Page 2004 (diversity beats ability — relevant to the mode-collapse gate, T6).

**LLM agent societies** — Park et al. 2023 (*Generative Agents*, Smallville: memory-stream + reflection architecture; emergent social behavior at 25 agents); Vezhnevets et al. 2023 (DeepMind *Concordia*: game-master pattern — our harness-adjudicated world follows this); AgentSociety (Piao et al. 2025: 10k+ agent social sims are feasible with cheap models); Lu et al. 2024 (Sakana *AI Scientist*: LLMs produce paper-shaped artifacts; quality control is the weak point — hence our no-LLM-in-the-metric-path rule); ResearchTown (Yu et al. 2024: simulating a research community as an agent graph).

**Baseline validation targets.** The A world must reproduce known mechanisms before the A/B comparison means anything: Matthew effect (citation concentration + cumulative advantage), priority races/scooping, incremental safe-topic bias, undervalued replication, review reluctance. If baseline A doesn't show these, fix the world before comparing conditions — this is a phase-2 go/no-go gate.

---

## 11. Architecture and stack

> ⚠️ The architecture workflow agent also hit the usage limit; this section was designed inline by the orchestrator.

**Deliberately minimal Python.** No agent frameworks.

```
worldgen.py    seeded SCM + lexicon                      (~150 LOC, numpy)
simulate.py    observational + mutilated-graph sampling  (~80 LOC)
verify.py      claim verification + evidence gate        (~100 LOC)
ledger.py      experiments, papers, both ledgers,
               unlock checker, metrics                   (~200 LOC, JSONL + dicts)
personas.py    trait sampler → narrative generator + lexical audit
agents.py      prompt assembly, memory summarization, action parsing
loop.py        tick scheduler, Batch API driver, review assignment,
               manipulation-check probes
analysis/      frozen metric + stats scripts (hashed at pre-registration)
```

- **Event-sourced:** the JSONL log is the source of truth; world state is rebuilt from it (or held in memory with periodic snapshots). No database. `# ponytail: SQLite only if log-query pain emerges`.
- **Tick loop:** synchronous. Each tick: (1) harness renders each active agent's diegetic inbox, (2) all decision calls go out as **one Batch API job** (50% discount; sim is offline so batch latency is fine), (3) actions resolve in seed-fixed order, (4) world events fire (publications, unlocks, review assignments, recruiter pings).
- **Structured actions:** agent responses use structured outputs (`output_config.format`) so the action parser never guesses.
- **Prompt caching:** stable prefix = world rules + tools + persona narrative, volatile suffix = tick inbox + memory summary. **Haiku 4.5's minimum cacheable prefix is 4,096 tokens** — the shared prefix must be ≥ 4k tokens or caching silently no-ops (Sonnet 5's minimum is 1,024). This is checked in the shakedown via `cache_read_input_tokens`.
- **Model mix (pinned for the whole study, T10):**
  - `claude-haiku-4-5` — routine tick decisions, no thinking. ($1/$5 per Mtok)
  - `claude-sonnet-5` — paper writing, peer reviews, and the frozen grader/classifier models, effort medium, capped max_tokens. ($3/$15; intro $2/$10 through 2026-08-31)
- **Ground truth lives harness-side only.** Agent-facing tools never expose weights, parent lists, or locked-layer names. Condition B differs *only* in what the social surface renders.

---

## 12. Cost model

> **All numbers below are analytical estimates, not measured.** Assumptions: ~1.2 calls per agent-tick; routine call ≈ 4.5k cached + 1.5k fresh input, 400 out; heavy call (papers/reviews, ~25% of calls) ≈ 5k cached + 3k fresh input, 1.5k out. The shakedown measures actuals and re-baselines this table.

Per-world arithmetic (200 agents × 120 ticks ≈ 29k calls; Batch API halves everything; cache reads at 0.1×):

| Component | Calls | Est. cost/call (batched) | Subtotal |
|---|---|---|---|
| Routine (Haiku 4.5) | ~21,700 | ≈ $0.002 | ≈ $44 |
| Heavy (Sonnet 5, std. pricing) | ~7,250 | ≈ $0.017 | ≈ $121 |
| Cache writes, probes, graders, retries | — | — | ≈ $25–60 |
| **Per world** | | | **≈ $190–230** |

| Phase | Scale | Est. cost |
|---|---|---|
| 0 — shakedown | 5–10 agents, 10 ticks | ≈ $2–5 |
| 1 — calibration pair | 2 × 20 agents, 40 ticks | ≈ $15–25 |
| 2 — first full pair | 2 × 200 agents, 120 ticks | ≈ $380–460 |
| 3 — confirmatory, 10 pairs | 20 × 200 agents | ≈ $3,800–4,600 |

Cost levers, in order of leverage: (1) heavy-call model — all-Haiku drops phase 3 to roughly $1.7–2.4k at some quality cost to papers/reviews; (2) Sonnet 5 intro pricing ($2/$10) through 2026-08-31 cuts the heavy line by a third if runs happen before then; (3) tick count and calls-per-tick; (4) escalation to 16 pairs adds ~60%.

---

## 13. Phase plan and go/no-go gates

**Phase 0 — build + shakedown** (5–10 agents, 10 ticks, direct API).
Verify: end-to-end mechanics; evidence gate; **exploit bots** (T5) rank poorly; **compliance floor** (T3) — exit/idle selected at nonzero rate under neutral prompts; **trope-probe calibration** (T2) recorded; prompt-cache hits confirmed (`cache_read_input_tokens > 0`); per-call token counts measured → cost model re-baselined.
*Gate: all checks pass, else redesign before spending more.*

**Phase 1 — calibration pair** (2 × 20 agents, 40 ticks, full pipeline including metrics, analysis script, and all manipulation checks).
Tune: salary multiplier to the 10–30% A-exit band; estimate between-world SD → confirm n = 10 or escalate to 16; set the mode-collapse threshold (T6); run one no-manipulation control world for temporal stability (T9).
*Gate: freeze pre-registration (hypotheses, metrics, analysis, exclusions, escalation rule — content-hashed). No shakedown or calibration data enters confirmatory analysis.*

**Phase 2 — first full run** (1 pair, 200 agents × 120 ticks — the chosen "first real run" scale).
Deliverable: descriptive/exploratory findings + the baseline validation check (§10) — does A reproduce the Matthew effect, priority races, replication neglect?
*Gate: baseline validates and diversity gate passes → fund phase 3. This single pair is never presented as confirmatory (T7).*

**Phase 3 — confirmatory** (10 pairs).
Frozen analysis script runs once; results reported per the pre-registration; writeup passes the T1/T8 claim-boundary review.

---

## 14. Open questions for the user

1. **Budget sign-off on phase 3** (≈ $4–5k Sonnet-mix, ≈ $2k all-Haiku) — decide after phase 2, no commitment needed now.
2. **Second-model robustness world** (T10): worth one extra pair on a non-Claude family, or accept single-family scope?
3. **Follow-up variants parked for later:** unlock-on-accepted-only (false foundations poison downstream science — realistic but confounds the A/B); intermediate conditions (persistent pseudonyms, team-only credit — the "multi-condition ladder" option deferred from the design interview).
