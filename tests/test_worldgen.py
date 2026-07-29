from sim.worldgen import generate


def test_determinism():
    a, b = generate(7), generate(7)
    assert a.edges == b.edges and a.names == b.names


def test_shape():
    t = generate(1)
    assert len(t.order) == 20 * 30 * 12
    assert 15_000 < len(t.edges) < 30_000


def test_parent_locality():
    t = generate(2)
    same_field_prev = 0
    for (c, e) in t.edges:
        cf, cl = int(c[1:3]), int(c[5:7])
        ef, el = int(e[1:3]), int(e[5:7])
        assert cl < el
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
    assert len(set(t.names.values())) == len(t.names)
