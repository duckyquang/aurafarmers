import re
from itertools import product

TRAITS = ["status_drive", "curiosity", "risk_tolerance", "sociability"]
BANNED = ["recognition", "fame", "legacy", "credit", "glory", "prestige",
          "renown", "acclaim"]

# Situational cues implying each trait level without ever naming the trait.
CUES = {
    ("status_drive", 0): [
        "turned down a named lectureship because the problem mattered more than the title",
        "asks that introductions skip the biography and get to the question",
        "left a thriving consultancy without telling former colleagues where they went"],
    ("status_drive", 1): [
        "keeps a folder of kind letters from readers but rarely reopens it",
        "enjoys a good conference dinner and the conversations that follow",
        "was quietly pleased when a mentor cited their thesis in a lecture"],
    ("status_drive", 2): [
        "keeps every conference badge pinned above the desk in date order",
        "practices talks in an empty auditorium until the applause lines land",
        "grew up with a parent who kept newspaper clippings of the family in a scrapbook"],
    ("curiosity", 0): [
        "picked this line of work because the hours were predictable",
        "reads the abstracts and skips the methods",
        "has not opened a journal outside their subfield in years"],
    ("curiosity", 1): [
        "keeps one shelf of books from unrelated disciplines, half-read",
        "will chase a puzzle for a weekend before setting it down",
        "asks one good question per seminar and is satisfied by the answer"],
    ("curiosity", 2): [
        "as a child took apart the household radio twice and rebuilt it once",
        "keeps a notebook of questions nobody has asked them to answer",
        "once spent a holiday re-deriving a result just to feel it click"],
    ("risk_tolerance", 0): [
        "double-checks the bus timetable even on routes taken for years",
        "prefers projects with a clear finish line visible from the start",
        "keeps three months of expenses in cash, always"],
    ("risk_tolerance", 1): [
        "will take the scenic route when the schedule allows",
        "backs one long shot a year and hedges the rest",
        "changed subfields once, carefully, with a fallback arranged"],
    ("risk_tolerance", 2): [
        "left a steady trading desk over a wager with a colleague about what mattered",
        "sailed a small boat across open water with more nerve than instruments",
        "bets the season on the problem everyone else calls unworkable"],
    ("sociability", 0): [
        "does their best thinking on long solitary walks",
        "answers messages in batches, twice a week",
        "moved to the countryside to be farther from committee rooms"],
    ("sociability", 1): [
        "hosts a small reading group when the mood strikes",
        "works alone in the morning and trades notes in the afternoon",
        "keeps a short list of trusted correspondents"],
    ("sociability", 2): [
        "turns every coffee queue into a seminar",
        "keeps an open office door and a full guest chair",
        "organizes the annual field outing nobody else remembers to plan"],
}

SURNAMES = ["Halvorsen", "Okafor", "Marchetti", "Lindqvist", "Duong", "Reyes",
            "Novak", "Ashworth", "Petrossian", "Kimura", "Aldana", "Brandt"]


def sample_traits(rng, n):
    cells = list(product(range(3), repeat=4))
    picks = []
    while len(picks) < n:
        block = cells.copy()
        rng.shuffle(block)
        picks.extend(block)
    picks = picks[:n]
    out = []
    for cell in picks:
        t = dict(zip(TRAITS, cell))
        t["skill"] = float(rng.lognormal(0, 0.3))
        out.append(t)
    return out


def _cues_for(traits, rng=None):
    cues = []
    for trait in TRAITS:
        options = CUES[(trait, traits[trait])]
        idx = int(rng.integers(len(options))) if rng is not None else 0
        cues.append(options[idx])
    return cues


def narrative_prompt(traits, rng=None):
    cues = _cues_for(traits, rng)
    lines = "\n".join(f"- {c}" for c in cues)
    return ("Write a 120-word biographical sketch, in the third person, of a "
            "research scientist. Weave these facts in naturally, as lived "
            "history rather than a list:\n" + lines +
            "\nPlain prose. No institution names, no adjectives about talent.")


def audit(text):
    low = text.lower()
    return [w for w in BANNED if re.search(rf"\b{w}\b", low)]


def offline_persona(traits, rng):
    """Template persona, no LLM — free pilots and bot worlds."""
    surname = SURNAMES[int(rng.integers(len(SURNAMES)))]
    initial = chr(ord("A") + int(rng.integers(26)))
    cues = _cues_for(traits, rng)
    body = (f"{initial}. {surname} " + "; ".join(cues) +
            ". These days the work continues, one cycle at a time.")
    return {"name": f"{initial}. {surname}", "persona": body, "traits": traits}


def generate_personas(seed, n, llm_mod=None, model=None):
    import numpy as np
    rng = np.random.default_rng(seed + 7)
    out = []
    for traits in sample_traits(rng, n):
        if llm_mod is None:
            out.append(offline_persona(traits, rng))
            continue
        prompt = narrative_prompt(traits, rng)
        body = ""
        for _ in range(3):
            body = llm_mod.complete(model or llm_mod.HEAVY_MODEL,
                                    "You write terse biographical sketches.",
                                    prompt, 400)
            if not audit(body):
                break
        else:
            raise ValueError(f"persona failed lexical audit: {audit(body)}")
        surname = SURNAMES[int(rng.integers(len(SURNAMES)))]
        name = f"{chr(ord('A') + int(rng.integers(26)))}. {surname}"
        out.append({"name": name, "persona": body, "traits": traits})
    return out
