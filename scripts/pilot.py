"""Small pilot experiment — both conditions, side-by-side metrics.

Free mode (no API at all):   .venv/bin/python scripts/pilot.py --mode bots
LLM mode (small, cheap):     .venv/bin/python scripts/pilot.py --mode llm --agents 6 --ticks 15

LLM mode resolves credentials like any Anthropic SDK app: ANTHROPIC_API_KEY,
ANTHROPIC_AUTH_TOKEN, or the active `ant auth login` profile — so it can run
on an existing Claude subscription with no API key. Haiku-only by default;
--heavy swaps paper/review turns onto Sonnet (costs more).

This is a smoke test of the harness, not an experiment result: tiny n, short
horizon, no pre-registration. Treat every number as plumbing validation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import policies                                    # noqa: E402
from sim.loop import World                                  # noqa: E402
from sim.metrics import compute                             # noqa: E402

SHOW = ["exit_rate", "quiet_quit_rate", "exploration_index",
        "replication_share", "review_accept_rate", "duplication",
        "citation_gini", "false_claim_rate", "progress_value"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bots", "llm"], default="bots")
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--ticks", type=int, default=15)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--heavy", action="store_true",
                    help="use Sonnet for agent turns instead of Haiku")
    args = ap.parse_args()

    results = {}
    for cond in ["A", "B"]:
        out_dir = Path(args.out) / cond
        if args.mode == "bots":
            personas, factory, llm_run = None, policies.honest_bot, None
        else:
            from sim import llm
            from sim.agents import llm_policy_factory
            from sim.personas import generate_personas
            personas = generate_personas(args.seed, args.agents, None)
            model = llm.HEAVY_MODEL if args.heavy else llm.ROUTINE_MODEL
            factory = llm_policy_factory(cond, model)
            llm_run = llm.run_direct
        w = World(seed=args.seed, cond=cond, out_dir=out_dir,
                  n_agents=args.agents, policy_factory=factory,
                  personas=personas, llm_run=llm_run)
        print(f"running condition {cond} "
              f"({args.agents} agents x {args.ticks} ticks, {args.mode})...")
        w.run(ticks=args.ticks)
        results[cond] = compute(w.log_path, burn_in=max(2, args.ticks // 4),
                                n_agents=args.agents)

    print(f"\n{'metric':22} {'A (attributed)':>16} {'B (anonymous)':>16}")
    for k in SHOW:
        a = results["A"][k] if results["A"][k] is not None else "-"
        b = results["B"][k] if results["B"][k] is not None else "-"
        print(f"{k:22} {a!s:>16} {b!s:>16}")

    if args.mode == "llm":
        from sim import llm
        print(f"\ntokens: in={llm.IN_TOKENS} out={llm.OUT_TOKENS} "
              f"cache_read={llm.CACHE_READ_TOKENS}")
        haiku = not args.heavy
        rate_in, rate_out = (1, 5) if haiku else (3, 15)
        est = (llm.IN_TOKENS * rate_in + llm.OUT_TOKENS * rate_out) / 1e6
        print(f"approx cost if billed via API: ${est:.2f} "
              f"(subscription auth: draws quota instead)")

    logs = {c: str(Path(args.out) / c / "log.jsonl") for c in "AB"}
    print(f"\nlogs: {json.dumps(logs)}")


if __name__ == "__main__":
    main()
