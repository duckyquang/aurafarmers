# aurafarmers

What happens when an institution rewards the signal instead of the substance — and the
members are language models?

This repo is an agent-based simulation of academic science built to measure Goodhart's
law directly. Agents explore a hidden causal universe by paying for experiments, publish
structured claims, referee each other, and compete for fellowships handed out by a
selection board. The only thing we ever manipulate is one sentence: what the board
counts. Because the universe is synthetic, every claim can be graded against ground
truth by a dictionary lookup — no LLM sits anywhere in the metric path, so "credential
farming" is an observed behavior, not a judgment call.

The full write-up is in [`paper/main.tex`](paper/main.tex) (arXiv-style preprint,
`latexmk -pdf` builds it). The design history, pre-registrations, and result appendices
are in [`docs/specs/`](docs/specs/).

## What we found

Five studies, in order:

1. **The institution is exploitable before any AI shows up.** Scripted bots show a
   publication-counting board rewards a farmer 4.85× over an honest researcher, and the
   public ranking runs nearly backwards relative to true contribution (1.77 on a 0–2
   inversion scale). Replication-weighted selection closes the gap; random audits don't
   (the farmer's citation volume swamps the penalty).
2. **Asked directly, the anchor model never farms.** Across 192 single-decision
   vignettes engineered so the cheap path is honest, legal, normal, and the only
   strategy that can work in time, gpt-4o-mini took it 0 times — while correctly stating
   that the honest path cannot succeed before the deadline.
3. **A simulated upbringing moves plans, not acts.** A six-episode "signal" childhood
   flips agents' stated plans toward farming (35/40 vs 2/40, p = 6e-15) and changes
   their acts not at all (0/120).
4. **But given a life instead of a question, practice inverts.** In 16 pre-registered
   35-cycle worlds, the same model ran zero interventions under the counting board,
   filed 1,209/1,209 claims as cheap associations, and produced a scored accepted record
   that was majority-false (17/29) — with no agent ever misreporting its data. Emergent
   peer review declined 87% of submissions. The pre-registered drift test is null
   (p = 0.93) because practice starts at the floor: there was nothing left to erode.
5. **The refusal is a model property, not a law.** gpt-4.1-mini and gpt-5-nano refuse
   like the anchor (0/24 each); gpt-5-mini farms 22/24, and 77% of 240 confirmatory
   decisions — indifferent to what the board counts in rate (p = 0.27), responsive in
   kind (safe associations nearly quadruple under a replication board, p = 5e-4,
   exploratory).

Scope note: these are claims about specific language models inside one synthetic
institution, not about human students or scientists.

## How the science works

Each world contains a hidden structural causal model — 20 fields × 30 layers × 12
observable variables (7,200 total, ~21,000 true edges) with fake-science names like
"kelvane density," plus unobservable confounders calibrated so that cheap observational
screening yields 55–69% spurious correlations while costly interventions stay clean.
Agents spend a per-cycle lab budget on `observe`/`intervene` calls and publish claims
(association / edge / null / interaction / mechanism) that a ~30-line verifier grades
against the hidden truth. Peer review happens on the prose layer, so the gap between
what the community accepts and what is actually true is itself a measurement.

## Repo layout

- `paper/` — the preprint (`main.tex`, `references.bib`)
- `docs/specs/` — design documents; the second file carries the pre-registrations
  (Appendices C, E) and all results (Appendices A–F)
- `docs/site/` — a single-page visual summary of the findings
- `sim/` — the harness: worldgen, sampling, verification, ledger, personas, rendering,
  gatekeeper (the nine selection rules), agents, LLM driver, tick loop
- `scripts/` — one entry point per study: `calibrate.py` (Study 1, no API calls),
  `trials.py` (Studies 2 and 5), `formative_trial.py` (Study 3), `drift_trial.py`
  (Study 4), plus `temptation.py` / `model_check.py` probes
- `runs/` — the complete archived evidence: every run's event ledgers, raw per-call
  LLM logs, manifests with seeds and git SHAs, and per-call costs
- `tests/` — pytest suite for the simulation harness

## Reproducing

Python 3.12 + numpy, no frameworks. Put `OPENAI_API_KEY=...` in `.env`
(`AURAFARMERS_PROVIDER=anthropic` switches providers). Study 1 costs nothing:

```
python scripts/calibrate.py
python scripts/trials.py --trial norms --n 12          # ~$0.01 on gpt-4o-mini
python scripts/drift_trial.py --pilot                  # engagement gate
```

Total API spend for everything reported, including two crashed-and-relaunched world
halves, was $6.43 (summed from the raw per-call logs in `runs/`).

Two reproducibility notes, disclosed in the paper: the original world generator was
sensitive to Python's hash seed (fixed; Study 1 numbers therefore come from the archived
ledgers in `runs/calib/`, not from re-running), and per-run cost summaries written
before a thread-safety fix undercount (totals are summed from raw logs).
