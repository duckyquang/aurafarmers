import numpy as np


def _closure(truth, wanted, interventions):
    """Ancestral closure of `wanted`; intervened nodes are pinned, so their
    ancestry is irrelevant and traversal stops there."""
    needed, stack = set(), list(wanted)
    while stack:
        node = stack.pop()
        if node in needed:
            continue
        needed.add(node)
        if node in interventions:
            continue
        stack.extend(truth.parents[node])
    return needed


def sample(truth, n, interventions=None, rng=None, only=None):
    """Standardized-space sampling: every node is divided by its calibration
    scale before children consume it, mirroring the generation pass. do()
    pins a node's standardized value directly."""
    interventions = interventions or {}
    rng = rng or np.random.default_rng()
    needed = None
    if only is not None:
        needed = _closure(truth, set(only) | set(interventions), interventions)
    inter_by_effect = {}
    for (pair, eff), w in truth.interactions.items():
        if needed is None or eff in needed:
            inter_by_effect.setdefault(eff, []).append((sorted(pair), w))
    vals = {}
    order = truth.order if needed is None \
        else [nd for nd in truth.order if nd in needed]
    for node in order:
        if node in interventions:
            vals[node] = np.full(n, float(interventions[node]))
            continue
        x = rng.normal(0, truth.noise[node], n)
        for p in truth.parents[node]:
            x = x + truth.edges[(p, node)] * vals[p]
        for (a, b), w in inter_by_effect.get(node, []):
            x = x + w * vals[a] * vals[b]
        vals[node] = x / truth.scale[node]
    return vals


def summary(samples, vars):
    return {v: {"mean": round(float(samples[v].mean()), 4),
                "sd": round(float(samples[v].std()), 4),
                "n": len(samples[v])} for v in vars}
