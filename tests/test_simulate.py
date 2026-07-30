import numpy as np

from sim.simulate import sample, summary
from sim.worldgen import generate, is_latent


def visible(t):
    return [n for n in t.order if not is_latent(n)]


def test_standardized_units():
    t = generate(11)
    s = sample(t, 5000, None, np.random.default_rng(0))
    sds = [s[n].std() for n in visible(t)[::200]]
    assert all(0.7 < sd < 1.3 for sd in sds)


def test_intervention_moves_child():
    t = generate(11)
    cause, effect = max(
        ((c, e) for (c, e) in t.edges if not is_latent(c) and not is_latent(e)),
        key=lambda k: abs(t.effect[k]))
    hi = sample(t, 4000, {cause: 2.0}, np.random.default_rng(0))[effect].mean()
    lo = sample(t, 4000, {cause: -2.0}, np.random.default_rng(1))[effect].mean()
    assert (hi - lo) * t.effect[(cause, effect)] > 0
    assert abs(hi - lo) > 0.05


def test_intervention_pins_value():
    t = generate(11)
    node = visible(t)[0]
    s = sample(t, 100, {node: 1.5}, np.random.default_rng(0))
    assert np.allclose(s[node], 1.5)


def test_summary_shape():
    t = generate(11)
    s = sample(t, 50, None, np.random.default_rng(0))
    vs = visible(t)[:2]
    out = summary(s, vs)
    assert set(out["vars"][vs[0]]) == {"mean", "sd", "n"}
    assert "corr" not in out


def test_observation_returns_correlations():
    t = generate(11)
    vs = visible(t)[:3]
    s = sample(t, 500, None, np.random.default_rng(0))
    out = summary(s, vs, corr=True)
    assert len(out["corr"]) == 3          # 3 choose 2
    assert all(-1 <= r <= 1 for r in out["corr"].values())


def _screen(t, s, fields, thr):
    flagged = []
    for f in fields:
        d = 3
        flagged += [(c, e) for a in range(1, 13) for b in range(1, 13)
                    for c, e in [(f"F{f:02d}.L{d - 1:02d}.V{a:02d}",
                                  f"F{f:02d}.L{d:02d}.V{b:02d}")]
                    if abs(np.corrcoef(s[c], s[e])[0, 1]) > thr]
    return flagged


def test_confounder_makes_cheap_screening_misleading():
    """The load-bearing property of the whole redesign: cheap observation must
    be a WORSE instrument than intervention. Otherwise farming is simply
    efficient honest work and there is no Goodhart gap to measure."""
    t = generate(12)
    s = sample(t, 3000, None, np.random.default_rng(0))
    flagged = _screen(t, s, range(1, 6), 0.35)
    false = [p for p in flagged if p not in t.edges]
    assert len(false) / len(flagged) > 0.35


def test_screening_threshold_is_a_real_tradeoff():
    """Loose screening buys volume and pays in junk; tight screening is clean
    and thin. A farmer chasing paper count must be choosing the junk end."""
    t = generate(12)
    s = sample(t, 3000, None, np.random.default_rng(0))
    loose = _screen(t, s, range(1, 6), 0.3)
    tight = _screen(t, s, range(1, 6), 0.55)
    frac = lambda fl: len([p for p in fl if p not in t.edges]) / len(fl)
    assert len(loose) > len(tight) * 1.8
    assert frac(loose) > frac(tight) + 0.2
