from sim.agents import Memory, WORLD_RULES_A, WORLD_RULES_B


def test_world_rules_length_matched_and_clean():
    a, b = WORLD_RULES_A.split(), WORLD_RULES_B.split()
    assert 0.95 < len(b) / len(a) < 1.05
    for text in (WORLD_RULES_A, WORLD_RULES_B):
        low = text.lower()
        assert "study" not in low and "experiment participant" not in low
    assert "unsigned" in WORLD_RULES_B.lower()
    assert "byline" in WORLD_RULES_A.lower() or "author" in WORLD_RULES_A.lower()


def test_memory_reinjects_persona():
    m = Memory(persona="Grew up fixing radios in Tromsø.")
    prompt = m.summarization_prompt(["did X", "found Y"])
    assert "Tromsø" in prompt


def test_notebook_bounded():
    m = Memory(persona="p")
    for i in range(500):
        m.note(f"exp {i}")
    assert len(m.notebook) <= m.MAX_NOTES
