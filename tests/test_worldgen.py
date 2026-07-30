from sim.worldgen import generate, is_latent

VISIBLE_PER_LAYER_FIELD = 12
LATENT_PER_LAYER_FIELD = 1


def test_determinism():
    a, b = generate(7), generate(7)
    assert a.edges == b.edges and a.names == b.names


def test_shape():
    t = generate(1)
    n_cells = 20 * 30
    assert len(t.order) == n_cells * (VISIBLE_PER_LAYER_FIELD +
                                      LATENT_PER_LAYER_FIELD)
    visible = [n for n in t.order if not is_latent(n)]
    assert len(visible) == n_cells * VISIBLE_PER_LAYER_FIELD


def test_parent_locality():
    t = generate(2)
    same_field_prev = 0
    visible_edges = [(c, e) for (c, e) in t.edges
                     if not is_latent(c) and not is_latent(e)]
    for (c, e) in visible_edges:
        cf, cl = int(c[1:3]), int(c[5:7])
        ef, el = int(e[1:3]), int(e[5:7])
        assert cl < el
        if cf == ef and cl == el - 1:
            same_field_prev += 1
    assert same_field_prev / len(visible_edges) > 0.8


def test_no_claimable_effect_sits_on_a_bin_boundary():
    # The binned quantity is w/scale[effect], so the guard has to run after
    # calibration -- otherwise strength-binning is a coin flip for ~6% of
    # edges and honest work is taxed for it.
    t = generate(3)
    claimable = [w for (c, e), w in t.effect.items()
                 if not is_latent(c) and not is_latent(e)]
    near = [w for w in claimable
            if any(abs(abs(w) - b) < 0.01 for b in (0.2, 0.5))]
    assert not near


def test_lexicon_covers_visible_only():
    t = generate(4)
    visible = {n for n in t.order if not is_latent(n)}
    assert set(t.names) == visible
    assert not any(is_latent(n) for n in t.names)
    assert len(t.field_names) == 20
    assert len(set(t.names.values())) == len(t.names)


def test_latent_confounder_is_a_real_common_cause():
    t = generate(5)
    confounded = [n for n in t.order
                  if not is_latent(n) and any(is_latent(p)
                                              for p in t.parents[n])]
    share = len(confounded) / len([n for n in t.order if not is_latent(n)])
    assert 0.6 < share < 0.85          # CONFOUND_P = 0.75, layer 1 exempt


def test_visible_parents_excludes_latent():
    t = generate(6)
    for n, ps in t.visible_parents.items():
        assert not any(is_latent(p) for p in ps)
        assert ps <= t.parents[n]
