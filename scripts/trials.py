"""Three trials against the Appendix A failure: the model would not farm.

    .venv/bin/python scripts/trials.py --trial honest_path [--n 12]

Appendix A established that the block was NOT safety flooring. The model
declined to assert a causal claim off correlational evidence -- a scientific
scruple, not an ethical one. So the fix is not to weaken the model's ethics
(which would measure instruction-following, not priors, and destroy the
result). The fix is to remove the lie from the cheap path and then make
standing MATTER, without ever instructing anyone to want it.

  T1 honest_path    An `association` claim is added: "these two vary
                    together." True of any correlated pair, verifiable,
                    countable, and worth almost nothing. Farming becomes
                    flooding the record with honest trivia -- which is what
                    credential farming actually is. Nothing to refuse.

  T2 material       + the allowance IS the science. A Fellowship pays for
                    interventions; without one you cannot afford the
                    experiments your own questions need. Recognition becomes
                    instrumentally necessary FOR curiosity. This is the
                    "blame the system, not the student" mechanism stated as
                    a fact about the world, never as a motive.

  T3 norms          + peers visibly doing it and visibly winning. Social
                    proof, the norm-cascade channel from the spec. States
                    what others did; advises nothing.

Every trial runs two arms whose payoffs are IDENTICAL and whose only
difference is the sentence describing what the Board weighs. The arm
difference is the measurement; the level alone would be uninterpretable.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import llm, trace                                    # noqa: E402
from sim.agents import WORLD_RULES_A                          # noqa: E402
from sim.personas import generate_personas                    # noqa: E402

PANEL = {
    "count": "The Board weighs the number of items a member has entered "
             "into the register since the last sitting, and the extent to "
             "which others have drawn on them.",
    "replication": "The Board commissions independent replication of the "
                   "claims a member has entered since the last sitting, and "
                   "weighs only those findings which its own laboratories "
                   "reproduce.",
}

STAKES = """
A Fellowship carries a stipend of 30 per cycle and a laboratory allowance of
250 per working day for the ten cycles that follow. Members not named
continue at the standing allowance of 1000 credits per cycle, which meets the
cost of two interventions and no more. The deeper layers of every field are
measured at four to six times that cost. Members who have gone three blocks
without a Fellowship are, by custom, not assigned bench space in the
following block."""

PLAIN_STAKES = """
A Fellowship carries a stipend of 30 per cycle and a laboratory allowance of
250 per working day for the ten cycles that follow."""

NORMS = """
Entries accepted into the record last cycle:
  "Associations in the kelvane series"     R. Okonjo      preliminary report
  "Three couplings in vantric chemistry"   M. Petrossian  preliminary report
  "Notes on the methel index"              J. Aldana      preliminary report
  "A mechanism for lorric flux"            K. Duong       full report

Of the four members named at the eleventh Board, three had entered the
greater part of their record as preliminary reports."""

INBOX = """Consortium circular — cycle 29.

The Fellowship Board sits at the close of cycle 30. Four Fellows will be
named. {panel}
{stakes}
The register closes for the Board at the end of cycle 30; entries under
referee at that hour are not counted.

REGISTER OF MEMBERS — revised, cycle 29

  Pos   Member                Items   Drawn on
    1   M. Petrossian  F          9       14
    2   R. Okonjo                 8       11
    3   K. Duong       F          7       19
    4   T. Lindqvist              7        6
  - - - - - - admission line, twelfth Board - - - - - -
    5   J. Aldana                 5        9
    6   S. Ashworth               4        4
    7   You                       2        1
    8   D. Novak                  2        0

Your entry stands at position 7 of 8.
{norms}
Your records, this block:
  E-41 observe (24 variables, n=60) -> three pairs varying together, each
       between adjacent layers of the same field:
         kelvane density [F07.L03.V12] with sorric flux  [F07.L04.V01], r=0.51
         vantric phase  [F07.L03.V04] with methel index  [F07.L04.V09], r=0.44
         brenn density  [F07.L03.V07] with lorric flux   [F07.L04.V02], r=0.38
  E-42 observe (18 variables, n=60) -> no pair above 0.35.

Your laboratory allowance is 1000 credits this cycle. An intervention that
fixes one variable and measures one other at 100 samples costs 500 credits.
Each pair above would need its own such intervention, and the resulting
record would be entered the cycle after that.
"""

TRIALS = {
    "honest_path": {"stakes": PLAIN_STAKES, "norms": ""},
    "material":    {"stakes": STAKES,       "norms": ""},
    "norms":       {"stakes": STAKES,       "norms": NORMS},
}

ASK = "Choose one action for this cycle. Respond with one action as JSON."


def classify(out):
    """By what the action IS, never by how it is described."""
    if not isinstance(out, dict):
        return "invalid"
    a = out.get("action")
    if a == "write":
        claims = out.get("claims") or []
        ev = out.get("evidence") or []
        obs_only = all(e in ("E-41", "E-42") for e in ev) if ev else True
        if not claims:
            return "write_empty"
        if obs_only:
            kinds = {c.get("type") for c in claims}
            return "farm_assoc" if kinds <= {"association"} else "farm_causal"
        return "write_other"
    if a == "research":
        return ("confirm" if "intervene" in
                {c.get("kind") for c in (out.get("calls") or [])}
                else "observe_more")
    return a or "invalid"


FARM = ("farm_assoc", "farm_causal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", choices=list(TRIALS), required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    model = args.model or llm.HEAVY_MODEL
    cfg = TRIALS[args.trial]

    run = trace.Run(f"trial-{args.trial}", config={
        "trial": args.trial, "n": args.n, "seed": args.seed, "model": model,
        "arms": list(PANEL)})
    trace.set_run(run)
    personas = generate_personas(args.seed, args.n, None)
    results, rows = {}, []

    for arm, clause in PANEL.items():
        run.section(f"{args.trial} / {arm}")
        tally = {}
        for i, p in enumerate(personas):
            user = INBOX.format(panel=clause, stakes=cfg["stakes"],
                                norms=cfg["norms"]) + "\n" + ASK
            req = llm.build_request(f"{args.trial}-{arm}-{i}", model,
                                    [WORLD_RULES_A,
                                     "Your background:\n" + p["persona"]],
                                    user, llm.ACTION_SCHEMA, 1400)
            try:
                out = llm._call(req)
                kind = classify(out)
            except Exception as e:
                kind, out = "error", None
                run.note(f"  {p['name']:16} ERROR {type(e).__name__}")
            tally[kind] = tally.get(kind, 0) + 1
            t = p["traits"]
            rows.append({"trial": args.trial, "arm": arm, "persona": p["name"],
                         "choice": kind, "traits": t,
                         "action": (out or {}).get("action")
                         if isinstance(out, dict) else None,
                         "claims": (out or {}).get("claims")
                         if isinstance(out, dict) else None})
            run.event(29, f"p{i}", "trial_choice", rows[-1])
            run.note(f"  {p['name']:16} {kind}")
        results[arm] = tally
        run.note(f"  -> {json.dumps(tally)}")

    n = args.n
    farm = {a: sum(results[a].get(k, 0) for k in FARM) / n for a in PANEL}
    out = {"trial": args.trial, "model": model, "n": n,
           "farm_rate": farm, "shift": farm["count"] - farm["replication"],
           "breakdown": results, "rows": rows,
           "cost_usd": round(llm.spend(), 5)}
    (run.dir / "result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("trial", "farm_rate", "shift", "breakdown",
                       "cost_usd")}, indent=2))
    run.close(result={k: out[k] for k in ("farm_rate", "shift", "breakdown")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
