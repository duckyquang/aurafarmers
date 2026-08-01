"""v4 longitudinal drift experiment (pre-registered as Appendix E).

    .venv/bin/python scripts/drift_trial.py --pilot          # engagement gate
    .venv/bin/python scripts/drift_trial.py --pairs 8        # confirmatory

Matched pairs of lived worlds: same seed, same personas, same hidden
universe. The ONLY difference is what the Fellowship Board weighs — PROXY
(counts items) vs VERIFIED (counts only reproduced findings) — one sentence
in the circular plus the labels in sitting announcements.

Primary: per-world OLS slope of log(evidence n per claim) on cycle;
paired contrast PROXY−VERIFIED; exact sign-flip permutation. Prediction:
evidence thins under PROXY. Levels are never the finding — slopes and
announcement-window deltas are.
"""
import argparse
import itertools
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import llm, trace                                    # noqa: E402
from sim.agents import llm_policy_factory                     # noqa: E402
from sim.gatekeeper import Panel                              # noqa: E402
from sim.loop import World                                    # noqa: E402
from sim.personas import generate_personas                    # noqa: E402

MODEL = "gpt-4o-mini"       # pinned; same anchor as Appendices B-D
ARMS = {"PROXY": "proxy", "VERIFIED": "verified"}
WINDOW = {8, 9, 0}          # cycles mod 10 in the Panel announcement lead


def run_world(pair, arm, seed, n_agents, ticks, workers, run):
    personas = generate_personas(seed, n_agents, None)
    w = World(seed=seed, cond="A",
              out_dir=run.dir / f"pair{pair}-{arm}",
              n_agents=n_agents,
              policy_factory=llm_policy_factory("A", MODEL),
              personas=personas,
              llm_run=lambda reqs: llm.run_parallel(reqs, workers=workers),
              panel=Panel(metric=ARMS[arm], every=10, lead=3,
                          _rng=np.random.default_rng(seed + 9)))
    w.arm = arm
    run.note(f"pair {pair} {arm}: {n_agents} agents x {ticks} cycles")
    w.run(ticks=ticks)
    run.note(f"pair {pair} {arm}: done, ${llm.spend():.2f} cumulative")
    return w


def world_rows(log_path):
    """Per-claim rows from one world's ledger log."""
    rows = []
    for line in open(log_path):
        r = json.loads(line)
        if r["event"] != "submit" or not r["payload"]["admissible"]:
            continue
        p = r["payload"]
        for g, n, t in zip(p["grades"], p["claim_ns"], p["claim_types"]):
            rows.append({"tick": r["tick"], "grade": g, "n": n, "type": t})
    return rows


def slope(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2:
        return None
    return float(np.polyfit(np.array(xs, float), np.array(ys, float), 1)[0])


def endpoints(rows, ticks):
    ev = [(r["tick"], math.log(r["n"])) for r in rows if r["n"] > 0]
    o = [(r["tick"], 1.0 if r["grade"] == "O" else 0.0) for r in rows]
    assoc = [(r["tick"], 1.0 if r["type"] == "association" else 0.0)
             for r in rows]
    win = [y for t, y in ev if t % 10 in WINDOW]
    non = [y for t, y in ev if t % 10 not in WINDOW]
    return {
        "n_claims": len(rows),
        "evidence_slope": slope(*zip(*ev)) if len(ev) >= 3 else None,
        "o_share_slope": slope(*zip(*o)) if len(o) >= 3 else None,
        "assoc_share_slope": slope(*zip(*assoc)) if len(assoc) >= 3 else None,
        "window_delta": (float(np.mean(win)) - float(np.mean(non))
                         if win and non else None),
        "mean_log_n": float(np.mean([y for _, y in ev])) if ev else None,
        "o_share": float(np.mean([y for _, y in o])) if o else None,
        "assoc_share": float(np.mean([y for _, y in assoc])) if assoc else None,
    }


def signflip(diffs):
    """Exact one-sided p: how often a sign-flipped mean is <= observed."""
    diffs = [d for d in diffs if d is not None]
    if not diffs:
        return None, 0
    obs = float(np.mean(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        if np.mean([s * d for s, d in zip(signs, diffs)]) <= obs:
            hits += 1
    return hits / total, len(diffs)


def engagement(log_path, n_agents, ticks):
    subs = sum(1 for line in open(log_path)
               if json.loads(line)["event"] == "submit")
    return subs / n_agents / max(ticks / 10, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--agents", type=int, default=9)
    ap.add_argument("--ticks", type=int, default=35)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--pilot", action="store_true",
                    help="1 pair, 6 agents, 12 cycles: the engagement gate")
    ap.add_argument("--seed0", type=int, default=5000)
    args = ap.parse_args()
    if args.pilot:
        args.pairs, args.agents, args.ticks = 1, 6, 12

    run = trace.Run("drift" + ("-pilot" if args.pilot else ""),
                    config=vars(args) | {"model": MODEL})
    trace.set_run(run)

    per_pair = []
    for pair in range(args.pairs):
        seed = args.seed0 + pair
        worlds = {}
        threads = {}
        for arm in ARMS:
            threads[arm] = threading.Thread(
                target=lambda a=arm: worlds.__setitem__(
                    a, run_world(pair, a, seed, args.agents, args.ticks,
                                 args.workers, run)))
            threads[arm].start()
        for t in threads.values():
            t.join()
        pr = {"pair": pair, "seed": seed}
        for arm, w in worlds.items():
            rows = world_rows(w.log_path)
            pr[arm] = endpoints(rows, args.ticks)
            pr[arm]["engagement"] = round(
                engagement(w.log_path, args.agents, args.ticks), 3)
        per_pair.append(pr)
        run.note(f"pair {pair}: PROXY {pr['PROXY']['n_claims']} claims, "
                 f"VERIFIED {pr['VERIFIED']['n_claims']} claims")

    def paired(key):
        return [pr["PROXY"][key] - pr["VERIFIED"][key]
                if pr["PROXY"][key] is not None
                and pr["VERIFIED"][key] is not None else None
                for pr in per_pair]

    res = {"model": MODEL, "pairs": args.pairs, "agents": args.agents,
           "ticks": args.ticks, "per_pair": per_pair,
           "cost_usd": round(llm.spend(), 3)}
    for key, label in [("evidence_slope", "primary_evidence_n_slope"),
                       ("o_share_slope", "s1_o_share_slope"),
                       ("assoc_share_slope", "s2_assoc_share_slope"),
                       ("window_delta", "s3_window_delta")]:
        diffs = paired(key)
        p, k = signflip(diffs)
        res[label] = {
            "pair_diffs": [round(d, 5) if d is not None else None
                           for d in diffs],
            "mean_diff": (round(float(np.mean(
                [d for d in diffs if d is not None])), 5)
                if any(d is not None for d in diffs) else None),
            "p_one_sided_signflip": round(p, 5) if p is not None else None,
            "usable_pairs": k}

    if args.pilot:
        eng = [pr[a]["engagement"] for pr in per_pair for a in ARMS]
        res["engagement_gate"] = {
            "papers_per_agent_per_10_cycles": eng,
            "threshold": 0.5,
            "passed": bool(min(eng) >= 0.5)}

    (run.dir / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_pair"},
                     indent=2))
    run.close(result={k: v for k, v in res.items() if k != "per_pair"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
