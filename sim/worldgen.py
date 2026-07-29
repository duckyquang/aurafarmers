import functools
from dataclasses import dataclass

import numpy as np

FIELDS, LAYERS, VARS = 20, 30, 12
SYLLABLES = ["ka", "vel", "ru", "thi", "mor", "zan", "el", "qu", "dra", "ne",
             "os", "phi", "gal", "ter", "lum", "brax", "ith", "sol", "cry", "man"]
SUFFIXES = ["density", "flux", "index", "phase"]


def node_id(f: int, l: int, v: int) -> str:
    return f"F{f:02d}.L{l:02d}.V{v:02d}"


@dataclass
class Truth:
    edges: dict
    interactions: dict
    parents: dict
    noise: dict
    order: list
    scale: dict
    names: dict
    field_names: dict

    @staticmethod
    def strength_bin(w: float) -> str:
        a = abs(w)
        return "weak" if a < 0.2 else "moderate" if a < 0.5 else "strong"


def _name(rng, used, n_syl):
    # 20 syllables -> only 400 two-syllable combos for 7,200 nodes, so after
    # a few tries fall back to a counter suffix (unique by construction)
    for _ in range(8):
        s = "".join(rng.choice(SYLLABLES, n_syl))
        if s not in used:
            used.add(s)
            return s
    s = "".join(rng.choice(SYLLABLES, n_syl)) + str(len(used))
    used.add(s)
    return s


def _weight(rng):
    while True:
        w = rng.uniform(0.05, 1.0) * rng.choice([-1, 1])
        if all(abs(abs(w) - b) > 0.02 for b in (0.2, 0.5)):
            return float(w)


@functools.cache          # Truth is treated as immutable everywhere
def generate(seed: int) -> Truth:
    rng = np.random.default_rng(seed)
    order = [node_id(f, l, v) for l in range(1, LAYERS + 1)
             for f in range(1, FIELDS + 1) for v in range(1, VARS + 1)]
    edges, parents, interactions, noise = {}, {}, {}, {}
    by_layer_field = {(l, f): [node_id(f, l, v) for v in range(1, VARS + 1)]
                      for l in range(1, LAYERS + 1) for f in range(1, FIELDS + 1)}
    for node in order:
        f, l = int(node[1:3]), int(node[5:7])
        noise[node] = float(np.exp(rng.normal(-0.5, 0.4)))
        if l == 1:
            parents[node] = frozenset()
            continue
        k = int(rng.integers(2, 5))
        ps = set()
        while len(ps) < k:
            r = rng.random()
            if r < 0.90 or l == 2:
                pool = by_layer_field[(l - 1, f)]
            elif r < 0.95:
                pool = by_layer_field[(int(rng.integers(1, l - 1)), f)]
            else:
                pool = by_layer_field[(l - 1, int(rng.integers(1, FIELDS + 1)))]
            ps.add(pool[int(rng.integers(len(pool)))])
        parents[node] = frozenset(ps)
        for p in ps:
            edges[(p, node)] = _weight(rng)
        if len(ps) >= 2 and rng.random() < 0.10:
            pair = rng.choice(sorted(ps), 2, replace=False)
            interactions[(frozenset(pair), node)] = _weight(rng)
    used = set()
    field_names = {f: _name(rng, used, 3) for f in range(1, FIELDS + 1)}
    names = {n: f"{_name(rng, used, 2)} {rng.choice(SUFFIXES)}" for n in order}
    # Calibration pass: fit each node's scale so its STANDARDIZED value has
    # unit variance, standardizing inline so children always consume bounded
    # inputs (raw values explode double-exponentially via interaction terms).
    scale = {}
    rng2 = np.random.default_rng(seed + 1)
    inter_by_effect = {}
    for (pair, eff), w in interactions.items():
        inter_by_effect.setdefault(eff, []).append((sorted(pair), w))
    vals = {}
    for node in order:
        x = rng2.normal(0, noise[node], 4_000)
        for p in parents[node]:
            x = x + edges[(p, node)] * vals[p]
        for (a, b), w in inter_by_effect.get(node, []):
            x = x + w * vals[a] * vals[b]
        scale[node] = float(x.std()) or 1.0
        vals[node] = x / scale[node]
    return Truth(edges, interactions, parents, noise, order, scale, names,
                 field_names)
