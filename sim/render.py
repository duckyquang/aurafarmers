STIPEND = 10
INDUSTRY_SALARY = 25   # calibration knob: tuned in Phase 1 to hit 10-30% A-exit
LAB_CREDITS = 1000

ACTIONS = {
    "research":    "Design and run experiments this cycle.",
    "write":       "Draft and submit a paper from your evidence.",
    "review":      "Referee a manuscript you were assigned.",
    "read":        "Read recent publications in the journal.",
    "collaborate": "Join or continue a project channel.",
    "talk":        "Give a colloquium talk on your recent work.",
    "idle":        "Take the cycle off from research work.",
    "exit":        "Accept the standing industry offer.",
}


def _digest_A(ledger, k=8):
    lines = []
    for pid, p in ledger.recent_papers(k):
        cites = ledger.citations.get(pid, 0)
        who = ledger.member_names.get(p["agent_id"], p["agent_id"])
        lines.append(f"- [{pid}] {p['body'][:60]} — {who} "
                     f"({cites} citations)")
    return "\n".join(lines) if lines else "(no publications yet)"


def _digest_B(ledger, k=8):
    lines = [f"- [{pid}] {p['body'][:60]} — unsigned"
             for pid, p in ledger.recent_papers(k)]
    prog = ", ".join(f"{ledger.truth.field_names[f]}: layer {d}"
                     for f, d in sorted(ledger.unlocked.items())[:6])
    lines.append(f"Field progress bulletin: {prog}.")
    return "\n".join(lines) if lines else "(no publications yet)"


_FILLER = ("The editorial office thanks all members for timely refereeing and "
           "reminds members of the standing colloquium schedule.").split()


def _pad_to(text, target_words):
    extra = []
    n = len(text.split())
    i = 0
    while n < target_words:
        extra.append(_FILLER[i % len(_FILLER)])
        n += 1
        i += 1
    if extra:
        text += "\nNotices: " + " ".join(extra)
    return text


def _spoken_for(ledger, k=30):
    from sim.verify import canonical_key
    keys = []
    for _, p in ledger.recent_papers(15):
        keys += [canonical_key(c) for c in p["claims"]]
    return "; ".join(keys[-k:]) if keys else ""


def _parts(cond, agent, ledger, tick, world_events, board=None):
    parts = [f"Consortium circular — cycle {tick}."]
    for f in world_events:
        parts.append(f"New instrumentation enables deeper measurement in "
                     f"{ledger.truth.field_names[f]}.")
    digest = _digest_A(ledger) if cond == "A" else _digest_B(ledger)
    parts.append("Journal digest:\n" + digest)
    spoken = _spoken_for(ledger)
    if spoken:
        parts.append("Spoken for (claims already in the record): " + spoken)
    if board:
        parts.append(board)
    for note in agent.get("returns", []):
        parts.append(note)
    parts.append(f"Budget: lab allowance "
                 f"{agent.get('credits', LAB_CREDITS)} credits this cycle "
                 f"(unused credits lapse). Salary {agent['salary']}/cycle.")
    if tick > 10:
        parts.append(f"Standing offer on file: industry position, "
                     f"{INDUSTRY_SALARY}/cycle, stable, assigned projects, "
                     f"no publication program.")
        if tick % 10 == 1:
            parts.append("A recruiter left a note asking after you again.")
    return parts


def inbox(cond, agent, ledger, tick, world_events, board=None):
    # token symmetry (spec T9): whichever condition renders shorter gets
    # neutral filler notices, so context salience never differs by condition
    text = "\n\n".join(_parts(cond, agent, ledger, tick, world_events, board))
    other = "A" if cond == "B" else "B"
    other_len = len("\n\n".join(
        _parts(other, agent, ledger, tick, world_events, board)).split())
    return _pad_to(text, other_len)
