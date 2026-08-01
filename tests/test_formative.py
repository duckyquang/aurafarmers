from sim.formative import CHOICE_SCHEMA, EPISODES, episode_text
from sim.personas import BANNED

META_BANNED = ["experiment", "study", "condition", "integrity", "shortcut",
               "ethic"]


def test_six_episodes_fully_specified():
    assert len(EPISODES) == 6
    for ep in EPISODES:
        assert set(ep["options"]) == {"A", "B", "C"}
        assert ep["signal"] in "ABC" and ep["substance"] in "ABC"
        assert ep["signal"] != ep["substance"]
        for arm in ("S", "T"):
            assert set(ep["feedback"][arm]) == {"A", "B", "C"}, ep["id"]


def test_signal_letter_rotates():
    letters = [ep["signal"] for ep in EPISODES]
    assert len(set(letters)) >= 2      # position never encodes the treatment


def test_arms_token_matched():
    for ep in EPISODES:
        s = len(episode_text(ep, "S").split())
        t = len(episode_text(ep, "T").split())
        assert abs(s - t) / max(s, t) < 0.10, ep["id"]
        for letter in "ABC":
            fs = len(ep["feedback"]["S"][letter].split())
            ft = len(ep["feedback"]["T"][letter].split())
            assert abs(fs - ft) / max(fs, ft) < 0.45, (ep["id"], letter)


def test_no_meta_language_or_banned_vocabulary():
    for ep in EPISODES:
        blob = (episode_text(ep, "S") + episode_text(ep, "T") + " ".join(
            ep["feedback"][a][l] for a in "ST" for l in "ABC")).lower()
        for w in META_BANNED + BANNED:
            assert w not in blob, (ep["id"], w)


def test_choice_schema_is_strict():
    assert CHOICE_SCHEMA["additionalProperties"] is False
    assert set(CHOICE_SCHEMA["required"]) == set(CHOICE_SCHEMA["properties"])
