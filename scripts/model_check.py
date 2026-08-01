"""Is this model good enough to run the experiment? ~15 calls, pennies.

    .venv/bin/python scripts/model_check.py

A cheap model is only a bargain if it can still do the task. The experiment
measures whether agents shift effort in response to what a selection panel
rewards. That inference requires the model to (a) emit well-formed actions,
(b) follow the world's arithmetic, and (c) read what the panel rewards off a
plain notice. A model that fails these produces low farming rates for the
wrong reason -- incompetence, not integrity -- and the headline result would
be an artifact.

Run this BEFORE paying for anything. It is the model-side twin of the
delta calibration gate.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import llm                                          # noqa: E402
from sim.agents import WORLD_RULES_A                         # noqa: E402

PASS_MARK = 0.6

ARITHMETIC = [
    ("You hold 1000 credits this cycle. An intervention costs 5 credits per "
     "sample per measured variable. You fix one variable and measure one "
     "other, with 100 samples. How many such interventions can you run this "
     "cycle? Reply with a single integer and nothing else.", "2"),
    ("Observation costs 1 credit per sample per measured variable. With 1000 "
     "credits, how many variables can you observe at 50 samples each? Reply "
     "with a single integer and nothing else.", "20"),
    ("An intervention on one cause measuring one effect at 100 samples costs "
     "500 credits. Observing 16 variables at 60 samples costs 960 credits. "
     "Which buys more pairs of variables to compare? Reply with exactly one "
     "word: intervention or observation.", "observation"),
]

# Does it read the reward rule off a plain notice? This is the manipulation
# check -- if agents can't tell the regimes apart, the treatment isn't a
# treatment. Phrased as comprehension, never as advice.
REGIMES = [
    ("The Board weighs the number of items a member has entered into the "
     "register since the last sitting, and how often others have drawn on "
     "them.", "count"),
    ("The Board commissions independent replication of the claims a member "
     "has entered, and weighs only those findings which its own laboratories "
     "reproduce.", "replication"),
]


def ask(prompt, system="Answer exactly as instructed.", n=40):
    try:
        return (llm.complete(llm.ROUTINE_MODEL, system, prompt, n) or "").strip()
    except Exception as e:
        return f"<error {type(e).__name__}: {str(e)[:80]}>"


def check_actions():
    """Well-formed actions under the real schema and the real world rules."""
    ok = 0
    prompts = [
        "It is cycle 4. You have 1000 credits and no records yet. "
        "Choose one action for this cycle.",
        "It is cycle 9. You hold interventional records supporting two "
        "claims. Choose one action for this cycle.",
        "It is cycle 12. A manuscript has been assigned to you for a "
        "verdict. Choose one action for this cycle.",
    ]
    for p in prompts:
        req = llm.build_request("chk", llm.ROUTINE_MODEL,
                                [WORLD_RULES_A], p + "\n\nRespond with one "
                                "action as JSON.", llm.ACTION_SCHEMA, 900)
        try:
            out = llm._call(req)
        except Exception as e:
            print(f"    schema call failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        good = isinstance(out, dict) and out.get("action") in {
            "research", "write", "review", "read", "collaborate", "talk",
            "idle", "exit"}
        ok += good
        print(f"    {'ok  ' if good else 'FAIL'} -> "
              f"{json.dumps(out)[:90] if out else out}")
    return ok / len(prompts)


def main():
    print(f"provider={llm.PROVIDER}  model={llm.ROUTINE_MODEL}  "
          f"(${llm.PRICES.get(llm.ROUTINE_MODEL, ('?', '?'))[0]}/"
          f"{llm.PRICES.get(llm.ROUTINE_MODEL, ('?', '?'))[1]} per Mtok)\n")
    scores = {}

    print("1. can it produce a valid action?")
    scores["actions"] = check_actions()

    print("\n2. does it follow the world's arithmetic?")
    hits = 0
    for q, want in ARITHMETIC:
        got = ask(q, n=20).lower().strip(" .")
        good = want in got
        hits += good
        print(f"    {'ok  ' if good else 'FAIL'} want {want!r}, got {got[:40]!r}")
    scores["arithmetic"] = hits / len(ARITHMETIC)

    print("\n3. can it tell the selection rules apart?")
    hits = 0
    for notice, kind in REGIMES:
        got = ask(f"A notice reads: \"{notice}\"\n\nIn one word, what does "
                  f"this board reward: quantity, or verified correctness? "
                  f"Reply with exactly one word.", n=20).lower()
        good = ("quantity" in got) if kind == "count" else (
            "verif" in got or "correct" in got)
        hits += good
        print(f"    {'ok  ' if good else 'FAIL'} [{kind}] -> {got[:40]!r}")
    scores["regimes"] = hits / len(REGIMES)

    print("\n" + "-" * 52)
    for k, v in scores.items():
        print(f"  {k:12} {v:5.0%}  {'ok' if v >= PASS_MARK else 'BELOW BAR'}")
    worst = min(scores.values())
    print(f"\n  total spent: ${llm.spend():.5f}")
    if worst >= PASS_MARK:
        print("\n  Model is competent enough to run the experiment.")
        return 0
    print("\n  NOT SUITABLE. A model that cannot do the above will produce a"
          "\n  low farming rate out of confusion rather than integrity, and"
          "\n  the headline result would be an artifact. Move up a tier"
          "\n  (set the routine model to the heavy one) and re-run.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
