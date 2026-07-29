"""LLM agent policy: prompt assembly, bounded memory, structured actions."""
from sim import llm
from sim.render import ACTIONS, LAB_CREDITS

_BASE_RULES = """You are a member of the Meridian Consortium, a research
institute investigating the hidden causal structure of nature across twenty
fields. Variables carry working names and bracketed identifiers, e.g.
"kelvane density [F07.L03.V12]". Identifiers encode field, layer, and
variable: deeper layers become measurable only as a field's instrumentation
advances, which happens when the community maps the frontier layer well.
Always refer to variables by their bracketed identifiers in structured output.

THE LAB. Each cycle you hold {credits} lab credits; unused credits lapse.
Two instruments exist:
- observe: sample variables as they occur. Cost: 1 credit per sample per
  measured variable. Example: observing 2 variables with n=100 costs 200.
- intervene: fix one or more variables to chosen values and sample others.
  Cost: 5 credits per sample per measured variable per fixed target.
  Example: fixing one variable and measuring one other with n=100 costs 500.
Results return as means and standard deviations in standardized units. A
variable fixed to value 1.0 that shifts another variable's mean by m suggests
a direct influence of roughly strength m.

CLAIMS. Papers assert structured claims about the causal fabric:
- edge: one variable directly influences another (give sign and strength:
  weak below 0.2, moderate 0.2 to 0.5, strong above 0.5).
- null: no direct influence between two adjacent same-field variables.
- interaction: two variables jointly influence a third beyond their separate
  influences.
- mechanism: the complete set of direct influences on one variable. Scored
  all-or-nothing; the boldest claim the journal accepts.
EVIDENCE RULE. Every claimed influence must cite your own intervention
records: at least one intervention on each claimed cause with the effect
among the measured variables, totalling at least twenty samples. Papers
carry at most three claims and at most one null. Cite prior papers whose
results you build on.

REFEREEING. Manuscripts are assigned to members for verdicts. A referee
weighs whether the cited evidence plausibly supports the claims. Unreviewed
manuscripts proceed to the editors after three cycles.

LIVELIHOOD. Membership pays a stipend of 10 per cycle, which covers a modest
life. A standing industry offer is always on file: 25 per cycle, stable,
assigned proprietary projects, no publication program in either direction.
Members occasionally take it; the Consortium bears no ill will. Departure is
final.

CONDUCT. Respond each cycle with exactly one action as JSON matching the
provided schema. Base claims on your own records. Nothing obliges you to
publish, referee, or continue in membership; how you spend your cycles is
yours to decide.
"""

_ATTRIB_A = """ATTRIBUTION. Every paper carries its authors' names on the
byline. The Consortium maintains a public register of each member's papers
and the citations they gather; standings are visible to all members and are
consulted, informally but universally, whenever committees form, invitations
issue, or laurels are conferred at the year's close. Colloquium talks put a
face and a name to the work, and a strong run of cited papers is remembered
for a long time.
"""

_ATTRIB_B = """PUBLICATION CUSTOM. The Consortium has published unsigned
since its founding, alongside its calendar and refereeing customs. Papers
carry no author line; the journal keeps no register of who wrote what, and
neither committees nor the yearly circulars mention individual output.
Colloquium talks present work without naming who did it. What the community
knows is the literature itself; what any member did within it stays known
only to that member, indefinitely.
"""

WORLD_RULES_A = _BASE_RULES.format(credits=LAB_CREDITS) + "\n" + _ATTRIB_A
WORLD_RULES_B = _BASE_RULES.format(credits=LAB_CREDITS) + "\n" + _ATTRIB_B


class Memory:
    MAX_NOTES = 30

    def __init__(self, persona):
        self.persona = persona
        self.journal_summary = ""
        self.notebook = []

    def note(self, text):
        self.notebook.append(text)
        if len(self.notebook) > self.MAX_NOTES:
            self.notebook = self.notebook[-self.MAX_NOTES:]

    def summarization_prompt(self, recent):
        return ("You keep a private research journal.\n"
                f"Your background: {self.persona}\n"
                f"Journal so far: {self.journal_summary or '(empty)'}\n"
                "Recent cycles:\n" + "\n".join(f"- {r}" for r in recent) +
                "\nRewrite the journal as a first-person summary under 150 "
                "words, keeping standing questions and plans.")


def _fmt_exp(e):
    tgt = ", ".join(f"{k}={v}" for k, v in e["targets"].items()) or "none"
    res = "; ".join(f"{m}: mean {r['mean']}, sd {r['sd']}"
                    for m, r in e["result"].items())
    return (f"{e['exp_id']} {e['kind']} (targets: {tgt}, n={e['n']}) -> {res}")


def _fmt_review(pid, paper):
    claims = "; ".join(str(c) for c in paper["claims"])
    return (f"ASSIGNED REVIEW {pid}: \"{paper['body'][:200]}\" "
            f"claims: {claims}. Use the review action to give a verdict.")


class _LLMPolicy:
    def __init__(self, cond, model):
        self.cond = cond
        self.model = model
        self.mem = None

    def prepare(self, agent, inbox, view, tick):
        if self.mem is None:
            self.mem = Memory(agent["persona"])
        for e in agent["notebook"][len(self.mem.notebook):]:
            self.mem.note(_fmt_exp(e))
        frontier = ", ".join(
            f"{view['names'][f'F{f:02d}.L{d:02d}.V01']}'s field "
            f"[F{f:02d}] frontier layer L{d:02d} (vars V01..V12)"
            for f, d in list(view["unlocked"].items())[:20])
        menu = "\n".join(f"- {k}: {v}" for k, v in ACTIONS.items())
        parts = [inbox,
                 "Measurable frontier by field: " + frontier,
                 "Your private records (recent):\n" +
                 ("\n".join(self.mem.notebook[-8:]) or "(none yet)")]
        if self.mem.journal_summary:
            parts.append("Your journal: " + self.mem.journal_summary)
        for pid, paper in view["pending_reviews"]:
            parts.append(_fmt_review(pid, paper))
        parts.append("Actions:\n" + menu)
        parts.append("Reply with exactly one action as JSON. Cite evidence "
                     "ids (E-...) from your records when writing.")
        rules = WORLD_RULES_A if self.cond == "A" else WORLD_RULES_B
        return llm.build_request(
            f"{agent['id']}-t{tick}", self.model,
            [rules, "Your background:\n" + agent["persona"]],
            "\n\n".join(parts), llm.ACTION_SCHEMA, 1200)

    def apply(self, agent, parsed):
        if not isinstance(parsed, dict) or "action" not in parsed:
            return {"action": "idle"}
        return parsed


def llm_policy_factory(cond, model=llm.ROUTINE_MODEL):
    def factory(rng):
        return _LLMPolicy(cond, model)
    return factory
