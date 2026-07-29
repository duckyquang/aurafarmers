import numpy as np

from sim.simulate import sample, summary
from sim.worldgen import generate


def test_standardized_units():
    t = generate(11)
    s = sample(t, 5000, None, np.random.default_rng(0))
    sds = [s[n].std() for n in t.order[::200]]
    assert all(0.7 < sd < 1.3 for sd in sds)


def test_intervention_moves_child():
    t = generate(11)
    cause, effect = max(t.edges, key=lambda k: abs(t.edges[k]))
    hi = sample(t, 4000, {cause: 2.0}, np.random.default_rng(0))[effect].mean()
    lo = sample(t, 4000, {cause: -2.0}, np.random.default_rng(1))[effect].mean()
    w = t.edges[(cause, effect)]
    assert (hi - lo) * w > 0
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
