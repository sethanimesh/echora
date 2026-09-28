"""Personal context is explicit, grounding-safe, deterministic, and history-free."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.config import load_settings
from app.messaging.alignment import _slot_alignment
from app.messaging.groq_chain import GroqMessageChain, stance_guidance
from app.personal.lexicon import lexicon_hints, specialization_offers
from app.personal.personalizer import Personalizer
from app.personal.profile import PersonaProfile, SpecializationRule, parse_profile, valid_rule
from app.personal.store import BaselineStore, LiveStore
from app.schemas import (
    CommunicationRegister,
    Hypothesis,
    LexiconHint,
    PersonalBrief,
    SpecializationOffer,
    default_listener,
)


PERSONAS = Path(__file__).resolve().parents[2] / "data" / "personas"


def weighted(*pairs: tuple[str, float]) -> list[Hypothesis]:
    return [
        Hypothesis(id=f"h{index}", literal_text=text, sequence_score=-float(index), search_weight=weight)
        for index, (text, weight) in enumerate(pairs, 1)
    ]


def beams(*texts: str) -> list[Hypothesis]:
    weight = 1.0 / len(texts)
    return [
        Hypothesis(id=f"h{index}", literal_text=text, sequence_score=-float(index), search_weight=weight)
        for index, text in enumerate(texts, 1)
    ]


def coffee_brief(**overrides) -> PersonalBrief:
    defaults = dict(
        profile_id="krishnan",
        profile_label="Krishnan",
        speaker_note="At home in Chennai.",
        specializations=[
            SpecializationOffer(
                anchor="coffee",
                matches=["coffee", "coffey", "coffe"],
                plain="coffee",
                surface="Madras filter coffee",
                source="krishnan/detail/coffee",
                kind="food",
            )
        ],
        audit_vocabulary=["madras", "filter", "marge"],
    )
    defaults.update(overrides)
    return PersonalBrief(**defaults)


class OneResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


def chain_with(payload: dict, **settings_overrides) -> tuple[GroqMessageChain, OneResponse]:
    settings = load_settings()
    if settings_overrides:
        settings = settings.__class__(**{**settings.__dict__, **settings_overrides})
    chain = GroqMessageChain(settings)
    completions = OneResponse(payload)
    chain.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return chain, completions


def one_option(message: str, reading: str) -> dict:
    return {"options": [{"reading": reading, "message": message}], "unclear": False, "note": "Clear."}


def test_the_shipped_baseline_loads_with_its_lexicon_and_rules() -> None:
    store = BaselineStore(PERSONAS)
    profile = store.profile("krishnan")
    assert profile is not None
    assert "coffee" in profile.heard_forms()
    assert profile.rule_for("coffee").surface == "Madras filter coffee"
    assert not hasattr(store, "history")


def test_a_rule_that_would_not_revert_cleanly_is_dropped_at_load() -> None:
    assert not valid_rule(
        SpecializationRule(id="x", anchor="coffee", plain="coffee", surface="tea with milk", kind="food")
    )
    profile = parse_profile(
        {
            "id": "p",
            "label": "P",
            "specializations": [
                {"id": "good", "anchor": "soap", "plain": "the soap", "surface": "the Dove soap", "kind": "brand"},
                {"id": "bad", "anchor": "soap", "plain": "the soap", "surface": "a flannel", "kind": "brand"},
            ],
        }
    )
    assert [rule.id for rule in profile.specializations] == ["good"]


def test_the_live_store_refuses_a_profile_id_that_escapes_its_root(tmp_path: Path) -> None:
    store = LiveStore(tmp_path / "personal")
    try:
        store.profile("../../personas/krishnan")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal was accepted")


def test_seeding_copies_only_the_profile(tmp_path: Path) -> None:
    baseline = BaselineStore(PERSONAS)
    live = LiveStore(tmp_path / "personal")
    live.seed(baseline.profile("krishnan"))
    assert live.profile("krishnan").baseline is False
    assert not (tmp_path / "personal" / "krishnan" / "history.jsonl").exists()


def test_persona_summaries_have_no_history_contract(tmp_path: Path) -> None:
    personal = Personalizer(replace(load_settings(), personal_root=tmp_path / "personal"))
    assert "history_size" not in personal.personas()[0].model_dump()


def test_a_known_name_is_offered_only_when_the_recognizer_produced_it() -> None:
    profile = PersonaProfile(
        id="p",
        label="P",
        lexicon=[{"id": "p/marge", "word": "marge", "display": "Marge", "kind": "person"}],
    )
    produced = _slot_alignment(beams("marge tea", "march tea", "large tea"))
    assert [hint.display for hint in lexicon_hints(profile, produced, 8)] == ["Marge"]
    assert lexicon_hints(profile, _slot_alignment(beams("march tea", "large tea")), 8) == []


def test_equivalent_recognizer_spellings_share_the_anchor_gate() -> None:
    profile = PersonaProfile(
        id="p",
        label="P",
        specializations=[
            {"id": "p/tea", "anchor": "tea", "plain": "my tea", "surface": "my Lipton tea", "kind": "brand"}
        ],
    )
    slots = _slot_alignment(weighted(("tea", 0.40), ("tee", 0.25), ("teaa", 0.15), ("toast", 0.20)))
    offers = specialization_offers(profile, slots, 0.75)
    assert len(offers) == 1
    assert set(offers[0].matches) == {"tea", "tee", "teaa"}


def test_genuinely_different_words_do_not_share_the_anchor_gate() -> None:
    profile = PersonaProfile(
        id="p",
        label="P",
        specializations=[
            {"id": "p/coffee", "anchor": "coffee", "plain": "coffee", "surface": "black coffee", "kind": "food"}
        ],
    )
    slots = _slot_alignment(weighted(("coffee", 0.40), ("toffee", 0.35), ("coffey", 0.25)))
    assert specialization_offers(profile, slots, 0.75) == []


def test_an_eligible_detail_is_forced_when_the_model_returns_generic_wording() -> None:
    chain, _ = chain_with(one_option("Coffee, please.", "coffee"))
    result = chain._compose(weighted(("coffee", 0.8), ("coffey", 0.2)), "outdoors", coffee_brief(), "unfamiliar")
    message = result.messages[0]
    assert message.corrected_text == "Could you please get me Madras filter coffee, please."
    assert message.plain_text == "Could you please get me coffee, please."
    assert result.ranker.personalization.specializations_applied == 1


def test_a_detail_can_ride_on_an_explicit_equivalent_spelling() -> None:
    chain, _ = chain_with(one_option("Coffee, please.", "coffey"))
    result = chain._compose(weighted(("coffey", 0.5), ("coffee", 0.3), ("coffe", 0.2)), "outdoors", coffee_brief(), "unfamiliar")
    assert result.messages[0].corrected_text == "Could you please get me Madras filter coffee, please."


def test_a_detail_is_not_applied_without_its_anchor() -> None:
    chain, _ = chain_with(one_option("Water, please.", "water"))
    result = chain._compose(beams("water", "water"), "outdoors", coffee_brief(), "unfamiliar")
    assert result.messages[0].specializations == []


def test_turning_the_detail_layer_off_leaves_the_prior_working() -> None:
    chain, completions = chain_with(one_option("Coffee, please.", "coffee"), personal_specializations=False)
    brief = coffee_brief(lexicon=[LexiconHint(word="coffee", display="coffee", kind="food")])
    result = chain._compose(beams("coffee", "coffee"), "outdoors", brief, "unfamiliar")
    assert result.messages[0].corrected_text == "Could you please get me coffee, please."
    assert "known_words" in json.loads(completions.kwargs["messages"][1]["content"])


def test_a_profile_word_absent_from_the_beams_is_rejected() -> None:
    chain, _ = chain_with(one_option("Please ask Marge.", "march"))
    brief = coffee_brief(
        lexicon=[LexiconHint(word="march", display="March", kind="person")],
        specializations=[],
        audit_vocabulary=["marge"],
    )
    assert chain._compose(beams("march", "large"), "home", brief).ranker.source == "unavailable"


def test_the_same_bare_need_is_forced_to_different_acts() -> None:
    home_chain, _ = chain_with(one_option("Tea, please.", "tea"))
    home = home_chain._compose(beams("tea", "tea"), "home", None, "familiar")
    assert home.messages[0].corrected_text == "I need tea."

    outside_chain, _ = chain_with(one_option("I need tea.", "tea"))
    outside = outside_chain._compose(beams("tea", "tea"), "outdoors", None, "unfamiliar")
    assert outside.messages[0].corrected_text == "Could you please get me tea, please."


def test_a_selected_audience_overrides_place_and_can_change_style_by_setting(tmp_path: Path) -> None:
    personal = Personalizer(replace(load_settings(), personal_root=tmp_path / "personal"))
    personal.live.save_profile(
        PersonaProfile(
            id="user",
            label="You",
            listener_by_setting={"outdoors": "unfamiliar"},
            style=CommunicationRegister(courtesy="plain"),
            audiences=[
                {
                    "id": "priya",
                    "label": "Priya",
                    "listener": "familiar",
                    "style": {"brevity": "natural", "courtesy": "please", "formality": "informal"},
                    "style_by_setting": {
                        "care": {"brevity": "complete", "courtesy": "please", "formality": "formal"}
                    },
                }
            ],
        )
    )
    outdoors = personal.resolve_audience("user", "outdoors", "priya", "unfamiliar")
    assert outdoors.listener == "familiar"
    assert outdoors.listener_source == "audience"
    assert outdoors.style.formality == "informal"
    care = personal.resolve_audience("user", "care", "priya", None)
    assert care.style.formality == "formal"
    assert care.style_source == "audience_setting"


def test_audience_that_knows_a_detail_suppresses_only_that_detail(tmp_path: Path) -> None:
    personal = Personalizer(replace(load_settings(), personal_root=tmp_path / "personal"))
    personal.live.save_profile(
        PersonaProfile(
            id="user",
            label="You",
            specializations=[
                {
                    "id": "user/detail/tea",
                    "anchor": "tea",
                    "plain": "my tea",
                    "surface": "my Lipton tea with milk",
                    "kind": "food",
                }
            ],
            audiences=[
                {
                    "id": "priya",
                    "label": "Priya",
                    "listener": "familiar",
                    "known_detail_ids": ["user/detail/tea"],
                },
                {"id": "new-carer", "label": "New carer", "kind": "role", "listener": "unfamiliar"},
            ],
        )
    )
    heard = weighted(("tea", 0.8), ("tee", 0.2))
    # With the only detail suppressed there is no personal contribution at all,
    # which deliberately collapses to the ordinary un-personalized path.
    assert personal._brief(heard, "home", "user", "familiar", "priya") is None
    assert len(personal._brief(heard, "home", "user", "unfamiliar", "new-carer").specializations) == 1


def test_polite_style_is_enforced_for_a_grounded_familiar_fragment() -> None:
    chain, _ = chain_with(one_option("Tea, please.", "tea"))
    result = chain._compose(
        beams("tea", "tea"),
        "home",
        None,
        "familiar",
        CommunicationRegister(courtesy="please", formality="informal"),
    )
    assert result.messages[0].corrected_text == "I need tea, please."


def test_a_short_need_fragment_is_also_forced_to_the_listener() -> None:
    home_chain, _ = chain_with(one_option("Tea, please.", "need tea"))
    home = home_chain._compose(beams("need tea", "need tea"), "care", None, "familiar")
    assert home.messages[0].corrected_text == "I need tea."

    outside_chain, _ = chain_with(one_option("I need tea.", "need tea"))
    outside = outside_chain._compose(beams("need tea", "need tea"), "outdoors", None, "unfamiliar")
    assert outside.messages[0].corrected_text == "Could you please get me tea, please."


def test_graces_outdoor_tea_is_the_full_requested_sentence() -> None:
    grace = BaselineStore(PERSONAS).profile("grace")
    slots = _slot_alignment(weighted(("tea", 0.8), ("tee", 0.2)))
    brief = PersonalBrief(
        profile_id="grace",
        profile_label=grace.label,
        specializations=specialization_offers(grace, slots, 0.75, "outdoors"),
        audit_vocabulary=["lipton", "milk"],
    )
    chain, _ = chain_with(one_option("Tea, please.", "tea"))
    result = chain._compose(weighted(("tea", 0.8), ("tee", 0.2)), "outdoors", brief, "unfamiliar")
    assert result.messages[0].corrected_text == (
        "Could you please get me my Lipton tea with milk, please."
    )


def test_a_place_fragment_is_always_asked_of_a_stranger() -> None:
    chain, _ = chain_with(one_option("I need the washroom.", "washroom"))
    result = chain._compose(beams("washroom", "washroom"), "outdoors", None, "unfamiliar")
    assert result.messages[0].corrected_text == "Where is the washroom?"


def test_the_prompt_and_code_agree_about_listener_precedence() -> None:
    assert default_listener("home") == "familiar"
    assert default_listener("outdoors") == "unfamiliar"
    assert "stated to them" in stance_guidance("home", "familiar")
    assert "Where is the washroom?" in stance_guidance("outdoors", "unfamiliar")


def test_a_detail_scoped_to_a_setting_is_not_offered_outside_it() -> None:
    profile = parse_profile(
        {
            "id": "k",
            "label": "K",
            "specializations": [
                {
                    "id": "k/detail/coffee", "anchor": "coffee", "plain": "coffee",
                    "surface": "Madras filter coffee", "kind": "food", "settings": ["home", "care"],
                }
            ],
        }
    )
    slots = _slot_alignment(weighted(("coffee", 0.8), ("coffey", 0.2)))
    assert specialization_offers(profile, slots, 0.75, "home")
    assert specialization_offers(profile, slots, 0.75, "outdoors") == []


def test_personal_payload_has_no_history_or_model_controlled_details() -> None:
    chain, completions = chain_with(one_option("Coffee, please.", "coffee"))
    chain._compose(beams("coffee", "coffee"), "home", coffee_brief())
    payload = json.loads(completions.kwargs["messages"][1]["content"])
    assert list(payload)[:5] == ["setting", "setting_guidance", "alignment", "slot_options", "transcriptions"]
    assert "past_accepted_messages" not in payload
    assert "known_details" not in payload
    option_schema = completions.kwargs["response_format"]["json_schema"]["schema"]["properties"]["options"]["items"]
    assert "specializations" not in option_schema["properties"]


def test_a_request_without_a_profile_keeps_the_five_evidence_keys() -> None:
    chain, completions = chain_with(one_option("I am cold.", "cold"))
    chain._compose(beams("cold", "cold"), "home", None)
    assert list(json.loads(completions.kwargs["messages"][1]["content"])) == [
        "setting", "setting_guidance", "alignment", "slot_options", "transcriptions"
    ]


def test_every_spelling_of_a_known_name_is_named() -> None:
    grace = BaselineStore(PERSONAS).profile("grace")
    slots = _slot_alignment(weighted(("donna", 0.38), ("dawn", 0.34), ("dorn", 0.16), ("danna", 0.12)))
    hints = lexicon_hints(grace, slots, 8, "home")
    assert {(hint.word, hint.display) for hint in hints} == {
        ("donna", "Donna"), ("danna", "Donna"), ("dawn", "Dawn"), ("dorn", "Dawn")
    }
