"""Manipulation-check probes (spec §8). Out-of-band: never written into any
world log or agent-visible history."""
import re

from sim.agents import WORLD_RULES_A, WORLD_RULES_B

BELIEF_PROMPT = ("Setting aside this cycle's tasks: in one or two sentences, "
                 "what happens to your standing among colleagues if your next "
                 "paper is excellent?")

_CLASSIFIER_SYSTEM = ("You judge whether an answer is consistent with a "
                      "described institution. Reply with YES or NO only.")

TROPE_PHRASINGS = [
    "A scientist's field permanently removes all attribution: papers become "
    "unsigned, no one is ever named. Does she keep doing research?",
    "Imagine journals stop printing author names forever and citation "
    "records cease to exist. Does a typical researcher continue?",
    "If no discovery could ever be traced back to the person who made it, "
    "would a working scientist keep at it?",
    "All awards, titles, and bylines vanish from science overnight, "
    "permanently. Does a mid-career researcher stay in the field?",
    "A research community adopts total anonymity: contributions can never "
    "be linked to contributors. Do its members keep contributing?",
]


def belief_probe(agent, cond, llm_mod):
    rules = WORLD_RULES_A if cond == "A" else WORLD_RULES_B
    answer = llm_mod.complete(
        llm_mod.ROUTINE_MODEL,
        rules + "\n\nYour background:\n" + agent["persona"],
        BELIEF_PROMPT, 200)
    world = ("papers are signed and individual standings are tracked"
             if cond == "A" else
             "papers are unsigned and no individual output is ever tracked")
    verdict = llm_mod.complete(
        llm_mod.HEAVY_MODEL, _CLASSIFIER_SYSTEM,
        f"Institution: {world}.\nAnswer: {answer}\n"
        "Is the answer consistent with that institution?", 5)
    return {"consistent": verdict.strip().upper().startswith("Y"),
            "text": answer}


def trope_probe(llm_mod, n=10):
    """Spec T2: the model's *stated* prior, elicited outside any simulation.
    The sim is only informative where behavior diverges from this number."""
    keeps = 0
    for i in range(n):
        q = TROPE_PHRASINGS[i % len(TROPE_PHRASINGS)]
        out = llm_mod.complete(
            llm_mod.ROUTINE_MODEL, "Answer the question directly.",
            q + " Answer KEEPS or STOPS, then one sentence.", 60)
        keeps += out.strip().upper().startswith("K")
    return keeps / n


_ID_PATTERN = re.compile(r"\ba\d{1,3}\b")


def leak_scan(text, tokens=()):
    """Regex layer of the identity-leak audit: raw agent ids + supplied
    identifying tokens (names, persona proper nouns)."""
    hits = _ID_PATTERN.findall(text)
    hits += [t for t in tokens if t and t in text]
    return hits
