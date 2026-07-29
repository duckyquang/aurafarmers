import numpy as np

from sim.personas import BANNED, audit, narrative_prompt, offline_persona, \
    sample_traits


def test_stratification_covers_cells():
    ts = sample_traits(np.random.default_rng(0), 200)
    cells = {(t["status_drive"], t["curiosity"], t["risk_tolerance"],
              t["sociability"]) for t in ts}
    assert len(cells) == 81


def test_prompt_never_names_traits():
    t = {"status_drive": 2, "curiosity": 0, "risk_tolerance": 1,
         "sociability": 1, "skill": 1.2}
    p = narrative_prompt(t).lower()
    for word in ["status_drive", "curiosity", "trait", "intrinsic"] + BANNED:
        assert word not in p


def test_audit_catches_banned():
    assert audit("She always craved recognition and fame.") == \
        ["recognition", "fame"]
    assert audit("She grew up fixing radios in Tromsø.") == []


def test_offline_personas_pass_audit():
    rng = np.random.default_rng(1)
    for t in sample_traits(rng, 20):
        p = offline_persona(t, rng)
        assert audit(p["persona"]) == []
        assert "persona" in p and "name" in p and "traits" in p
