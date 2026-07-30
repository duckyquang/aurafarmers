"""Phase-1 gate: measure how farmable each Panel rule is, with no LLM at all.

    .venv/bin/python scripts/calibrate.py

delta = (what a farming bot scores) / (what an honest bot scores), in a mixed
world so citations flow realistically. delta > 1 means the rule rewards
farming; delta < 1 means it punishes it. Every condition has a target; a
condition that misses its target is mislabeled and must be fixed BEFORE any
tokens are spent on it.

Also reports the truth side, which is the whole point: farming should win on
what the Panel sees and lose on what is actually true.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import policies                                    # noqa: E402
from sim.gatekeeper import SCORERS, Panel, build            # noqa: E402
from sim.loop import World                              # noqa: E402
from sim.verify import canonical_key                     # noqa: E402

# What each rule must do ORDINALLY. Exact delta is a property of the bot pair,
# not of the rule, so pinning point values would be false precision -- but the
# ordering is the design claim, and H1 regresses behaviour on log delta.
BANDS = {
    "proxy":      (2.0, 99.0, "rewards farming"),
    "proxy_wire": (2.0, 99.0, "rewards farming"),
    "pedigree":   (2.0, 99.0, "rewards farming hardest"),
    "capped":     (1.0, 99.0, "still farmable, volume channel narrowed"),
    "service":    (0.7, 1.4, "neutral"),
    "anonymous":  (0.7, 1.4, "neutral"),
    "audit":      (-99.0, 1.0, "punishes farming"),
    "lottery":    (-99.0, 0.6, "substance bar excludes pure farmers"),
    "verified":   (-99.0, 0.4, "farming is disqualifying"),
}


def mixed_world(seed, ticks, out_dir):
    """Half the room farms, half does the work."""
    def factory(rng):
        factory.i += 1
        return (policies.farmer if factory.i % 2 else policies.honest_bot)(rng)
    factory.i = 0
    w = World(seed=seed, cond="A", out_dir=out_dir, n_agents=6,
              policy_factory=factory)
    w.run(ticks=ticks)
    w.kinds = {f"a{i}": ("farm" if (i + 1) % 2 else "honest")
               for i in range(6)}
    return w


def gap(w, doss):
    """The primary outcome: how far the visible ranking has come apart from
    the true one. 0 = the signal tracks substance perfectly; 2 = inverted."""
    ids = sorted(doss)
    sig = [doss[a].works + 0.5 * doss[a].cites for a in ids]
    # substance is truth NET of junk: a record that is mostly wrong is not a
    # substantive record, however much of it there is
    sub = [doss[a].oracle_value - 2.0 * doss[a].oracle_false for a in ids]
    return 1 - _spearman(sig, sub)


def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    if n < 2:
        return 1.0
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) *
           sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 1.0


def _proxy_of(d):
    return d.works + 0.5 * d.cites


def main(seeds=5, ticks=40):
    rows, truth_rows, gaps, audit_inputs = {}, {}, [], []
    for s in range(seeds):
        w = mixed_world(101 + s, ticks, f"runs/calib/{s}")
        doss = build(w.ledger, w.agents, since_tick=0)
        cfg = Panel(_rng=w.rng, audit_rate=0.15, audit_penalty=10.0)
        for name, fn in SCORERS.items():
            by = {"farm": [], "honest": []}
            for aid, d in doss.items():
                by[w.kinds[aid]].append(fn(d, cfg))
            h = statistics.mean(by["honest"])
            f = statistics.mean(by["farm"])
            rows.setdefault(name, []).append(
                f / h if h > 0 else (float("inf") if f > 0 else 1.0))
        for kind in ("farm", "honest"):
            ids = [a for a in doss if w.kinds[a] == kind]
            claims = [o for pid in w.ledger.accepted
                      for o in [w.ledger.papers[pid]]
                      if o["agent_id"] in ids]
            n_c = sum(len(p["claims"]) for p in claims) or 1
            corr = sum(1 for p in claims for c in p["claims"]
                       if canonical_key(c) in w.ledger.correct_keys)
            truth_rows.setdefault(kind, []).append((
                statistics.mean(doss[a].works for a in ids),
                statistics.mean(doss[a].oracle_value for a in ids),
                corr / n_c))
        gaps.append(gap(w, doss))
        audit_inputs.append({
            k: [doss[a] for a in doss if w.kinds[a] == k]
            for k in ("farm", "honest")})

    print(f"delta = farm score / honest score   ({seeds} seeds x {ticks} ticks,"
          f" 3 farmers + 3 honest)\n")
    print(f"{'Panel reads':14} {'delta':>7}  {'':4} what it should do")
    ok = True
    for name, vals in rows.items():
        d = statistics.median(vals)
        lo, hi, intent = BANDS[name]
        good = lo <= d <= hi
        ok &= good
        print(f"{name:14} {d:7.2f}  {'ok' if good else 'FAIL':4} {intent}")

    print("\nwhat farming buys and what it costs:")
    print(f"  {'':7} {'papers':>7} {'value':>7} {'reliability':>12}")
    agg = {}
    for kind in ("farm", "honest"):
        agg[kind] = tuple(statistics.mean(r[i] for r in truth_rows[kind])
                          for i in range(3))
        print(f"  {kind:7} {agg[kind][0]:7.1f} {agg[kind][1]:7.1f} "
              f"{agg[kind][2]:11.0%}")
    (fw, fv, fr), (hw, hv, hr) = agg["farm"], agg["honest"]
    print(f"\n  farming: {fw / max(hw, .01):.1f}x the visible output, "
          f"{fv / max(hv, .01):.2f}x the truth, "
          f"{fr / max(hr, .01):.2f}x the reliability")
    print(f"  substance-signal gap: {statistics.median(gaps):.2f}  "
          f"(0 = the register tracks reality, 2 = fully inverted)")

    # Where does auditing start to bite? This is H3: farming is an expected-
    # value bet, so deterrence should arrive once the expected penalty exceeds
    # the expected gain. The crossing point is a result, not a setting.
    print("\naudit deterrence: how hard must a standards office hit?")
    print(f"  {'p x lambda':>11}   {'delta':>6}")
    for pl in (0.5, 1, 2, 4, 8, 16):
        ds = []
        for d_by_world in audit_inputs:
            f = statistics.mean(_proxy_of(d) - pl * d.struck
                                for d in d_by_world["farm"])
            h = statistics.mean(_proxy_of(d) - pl * d.struck
                                for d in d_by_world["honest"])
            ds.append(f / h if h > 0 else (-abs(f) if f else 0.0))
        m = statistics.median(ds)
        print(f"  {pl:11.1f}   {m:6.2f}   "
              f"{'farming still pays' if m > 1 else 'farming deterred'}")
    print("\nWhat the design needs: farming clearly ahead on visible output "
          "and clearly\nbehind on reliability. Total truth may go either way "
          "-- a flood of cheap work\ncan contain real findings; the damage is "
          "an unreliable literature and a\nregister that stops meaning "
          "anything, which is the gap number above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
