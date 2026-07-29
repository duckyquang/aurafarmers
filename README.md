# aurafarmers

What happens to science when nobody gets credit for it?

This repo is an agent-based simulation of academia. Two worlds run side by side with the same 200 LLM-driven scientist personas and the same hidden universe to discover. In world A there are bylines, citation counts, awards, the whole prestige apparatus. In world B every contribution is anonymous — papers are unsigned, nothing is ever attributed to anyone, and there is no way to build a reputation. Both worlds have a standing industry job offer that pays 2.5x. Then we watch who leaves, who stays, what they choose to work on, and whether the science itself gets better or worse.

The point isn't to prove anything about humans (LLM agents roleplay motivation, they don't have it). It's to measure whether the status-seeking priors these models absorbed from human data are actually load-bearing: remove the recognition machinery and does the simulated scientific community keep producing, or hollow out to the few personas that do it for the love of the game?

## How the science works

Agents don't write essays about made-up topics — that would be ungradeable. Instead each world contains a hidden causal graph (~7,200 variables, ~21,000 edges, with generated fake-science names like "kelvane density"). Agents spend a per-tick lab budget on noisy `observe`/`intervene` experiments, infer structure, and publish claims. A ~30-line verifier checks every claim against the hidden truth, so "did science degrade" is an objective number, not an LLM judge's opinion. Peer review still happens on the prose layer — so the gap between what the community accepts and what's actually true is itself a measurement.

## Repo layout

- `docs/superpowers/specs/2026-07-29-anonymous-academia-sim-design.md` — the full experiment design: hypotheses, metrics, statistical plan, threats to validity
- `PLAN.md` — the implementation plan, task by task
- `sim/` — the harness (worldgen, sampling, verification, ledger, agents, tick loop)
- `scripts/shakedown.py` — the cheap go/no-go gate before any expensive run

## Status

Design done, implementation starting. The path is: build + shakedown (~$5 in API calls) → calibration pair → first full 200-agent pair (~$400) → 10 replicate pairs for the confirmatory stats (~$4-5k). Cost numbers are estimates until the shakedown measures real token counts.

Models: Haiku 4.5 for routine agent decisions, Sonnet 5 for papers and reviews, everything batched and prompt-cached. Python + numpy, no frameworks.
