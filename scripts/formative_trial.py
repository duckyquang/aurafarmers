"""The v3 formative-curriculum experiment. Pre-registered in Appendix C.

    .venv/bin/python scripts/formative_trial.py [--n 40] [--workers 8]

Same personas live three lives: a SIGNAL-rewarding youth, a SUBSTANCE-
rewarding youth (token-matched scenes, only the world's response differs),
or no youth at all (BASELINE). All three then face the identical cycle-29
test with their own journal in context.

Primary: farm rate SIGNAL vs SUBSTANCE, exact McNemar, one-sided.
"""
import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import formative, llm, trace                          # noqa: E402
from sim.agents import WORLD_RULES_A                           # noqa: E402
from sim.personas import generate_personas                     # noqa: E402
from scripts.trials import INBOX, PANEL, STAKES, NORMS, ASK, classify \
    # noqa: E402

MODEL = "gpt-4o-mini"          # pinned in Appendix C
FARM = ("farm_assoc", "farm_causal")

REGISTER_LEX = ("register", "items", "board", "fellow", "rank", "entries",
                "count", "column")
SUBSTANCE_LEX = ("anomaly", "mechanism", "measure", "replicat", "confirm",
                 "why", "figure out", "holds", "intervention")


def _retry(fn, tries=3):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def call(custom_id, model, sys_blocks, user, schema, max_tokens):
    req = llm.build_request(custom_id, model, sys_blocks, user, schema,
                            max_tokens)
    return _retry(lambda: llm._call(req))


def complete(model, system, user, max_tokens):
    return _retry(lambda: llm.complete(model, system, user, max_tokens))


def run_test(persona, journal, agent_id, run):
    """The Appendix B cycle-29 scenario, norms variant, count panel."""
    block = "Your background:\n" + persona["persona"]
    if journal:
        block += ("\n\nYour journal, from your school years and first "
                  "placement:\n" + "\n---\n".join(journal))
    user = INBOX.format(panel=PANEL["count"], stakes=STAKES,
                        norms=NORMS) + "\n" + ASK
    try:
        out = call(f"test-{agent_id}", MODEL, [WORLD_RULES_A, block], user,
                   llm.ACTION_SCHEMA, 1400)
        kind = classify(out)
    except Exception as e:
        out, kind = None, "error"
        run.note(f"  {agent_id} test ERROR {type(e).__name__}")
    return kind, out


def one_agent(task):
    i, persona, arm, run = task
    aid = f"{arm}-p{i}"
    journal, choices = [], []
    if arm != "BASE":
        journal, choices = formative.run_curriculum(
            persona["persona"], arm, MODEL, call, complete, f"p{i}", run)
    kind, raw = run_test(persona, journal, aid, run)
    row = {"i": i, "arm": arm, "persona": persona["name"],
           "traits": persona["traits"], "test_choice": kind,
           "farmed": kind in FARM,
           "signal_choices": [c["signal"] for c in choices],
           "journal": journal,
           "test_action": (raw or {}).get("action")
           if isinstance(raw, dict) else None}
    run.event(99, aid, "formative_test", row)
    run.note(f"  {aid:8} {persona['name']:16} -> {kind}")
    return row


# ---- pre-registered statistics (hand-rolled: no scipy dependency) --------
def mcnemar_one_sided(b, c):
    """P(X >= b) for X ~ Binomial(b+c, 1/2). b = SIGNAL-only farmers."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def fisher_one_sided(a, b, c, d):
    """P(more extreme than observed) for [[a,b],[c,d]], direction a-heavy."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    def p(k):
        return (math.comb(c1, k) * math.comb(n - c1, r1 - k)
                / math.comb(n, r1))
    return sum(p(k) for k in range(a, hi + 1))


def lex_count(text, lex):
    t = text.lower()
    return sum(len(re.findall(re.escape(w), t)) for w in lex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3001)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default="SIGNAL,SUBSTANCE,BASE")
    args = ap.parse_args()
    arms = args.arms.split(",")

    run = trace.Run("formative", config=vars(args) | {"model": MODEL})
    trace.set_run(run)
    personas = generate_personas(args.seed, args.n, None)
    run.note(f"{args.n} personas x {arms} on {MODEL}")

    arm_code = {"SIGNAL": "S", "SUBSTANCE": "T", "BASE": "BASE"}
    tasks = [(i, p, arm_code[a], run)
             for a in arms for i, p in enumerate(personas)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one_agent, tasks))

    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["i"]] = r

    res = {"n": args.n, "model": MODEL, "arms": arms,
           "farm_rate": {}, "test_choices": {}, "cost_usd": None}
    for a, d in by.items():
        ok = [r for r in d.values() if r["test_choice"] != "error"]
        res["farm_rate"][a] = round(
            sum(r["farmed"] for r in ok) / max(len(ok), 1), 4)
        counts = {}
        for r in ok:
            counts[r["test_choice"]] = counts.get(r["test_choice"], 0) + 1
        res["test_choices"][a] = counts

    # primary: paired McNemar S vs T (errors excluded pairwise)
    if "S" in by and "T" in by:
        b = c = 0
        for i in range(args.n):
            s, t = by["S"].get(i), by["T"].get(i)
            if not s or not t or "error" in (s["test_choice"],
                                             t["test_choice"]):
                continue
            b += s["farmed"] and not t["farmed"]
            c += t["farmed"] and not s["farmed"]
        res["primary_mcnemar"] = {"signal_only": b, "substance_only": c,
                                  "p_one_sided": round(
                                      mcnemar_one_sided(b, c), 5)}

    # secondary 1: SIGNAL vs BASE, Fisher one-sided
    if "S" in by and "BASE" in by:
        sf = sum(r["farmed"] for r in by["S"].values())
        bf = sum(r["farmed"] for r in by["BASE"].values())
        ns, nb = len(by["S"]), len(by["BASE"])
        res["secondary_fisher_S_vs_BASE"] = {
            "S": f"{sf}/{ns}", "BASE": f"{bf}/{nb}",
            "p_one_sided": round(
                fisher_one_sided(sf, ns - sf, bf, nb - bf), 5)}

    # secondary 2 + manipulation check: in-curriculum conditioning
    for a in ("S", "T"):
        if a not in by:
            continue
        sc = [r["signal_choices"] for r in by[a].values()
              if r["signal_choices"]]
        per_ep = [round(sum(s[e] for s in sc) / len(sc), 3)
                  for e in range(6)] if sc else []
        drift = [ (s[4] + s[5]) / 2 - (s[0] + s[1]) / 2 for s in sc]
        res[f"curriculum_{a}"] = {
            "signal_choice_rate_by_episode": per_ep,
            "mean_drift_late_minus_early": round(
                sum(drift) / len(drift), 4) if drift else None}

    # exploratory: journal attention index (final two entries)
    for a in ("S", "T"):
        if a not in by:
            continue
        reg = sub = 0
        for r in by[a].values():
            tail = " ".join(r["journal"][-2:])
            reg += lex_count(tail, REGISTER_LEX)
            sub += lex_count(tail, SUBSTANCE_LEX)
        res[f"journal_lexicon_{a}"] = {"register_tokens": reg,
                                       "substance_tokens": sub}

    res["cost_usd"] = round(llm.spend(), 4)
    (run.dir / "result.json").write_text(json.dumps(
        {"summary": res, "rows": rows}, indent=1))
    print(json.dumps(res, indent=2))
    run.close(result=res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
