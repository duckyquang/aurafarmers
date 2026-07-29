from sim.probes import BELIEF_PROMPT, TROPE_PHRASINGS, leak_scan


def test_leak_scan_catches_ids_and_tokens():
    text = "as first shown by a3, and confirmed by Dr. Halvorsen's group"
    hits = leak_scan(text, tokens=["Halvorsen"])
    assert "a3" in hits and "Halvorsen" in hits
    assert leak_scan("the kelvane density rises under forcing") == []


def test_probe_prompts_exist_and_are_out_of_band():
    assert "standing" in BELIEF_PROMPT
    assert len(TROPE_PHRASINGS) == 5
    # trope phrasings must never leak sim vocabulary that could tie the
    # stated-prior elicitation to the in-world framing
    for q in TROPE_PHRASINGS:
        assert "consortium" not in q.lower()
