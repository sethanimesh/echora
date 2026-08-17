"""Personal context: a prior that only ever chooses, and a detail layer that is audited."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import load_settings
from app.messaging.alignment import _slot_alignment
from app.messaging.groq_chain import GroqMessageChain
from app.personal.lexicon import lexicon_hints, specialization_offers
from app.personal.profile import PersonaProfile, SpecializationRule, parse_profile, valid_rule
from app.personal.store import BaselineStore, LiveStore
from app.schemas import (
    AcceptedMessage,
    Hypothesis,
    LexiconHint,
    PersonalBrief,
    SpecializationOffer,
)


PERSONAS = Path(__file__).resolve().parents[2] / "data" / "personas"


def weighted(*pairs: tuple[str, float]) -> list[Hypothesis]:
    """Beams with real search weights, as the recognizer actually returns them."""
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


def one_option(message: str, reading: str, specializations: list[dict] | None = None) -> dict:
    option = {"reading": reading, "message": message}
    if specializations is not None:
        option["specializations"] = specializations
    return {"options": [option], "unclear": False, "note": "Clear."}


# ---------------------------------------------------------------- profiles and stores


def test_the_shipped_baseline_loads_with_its_lexicon_and_rules() -> None:
    store = BaselineStore(PERSONAS)
    profile = store.profile("krishnan")
    assert profile is not None
    assert "coffee" in profile.heard_forms()
    assert profile.rule_for("coffee").surface == "Madras filter coffee"
    assert len(store.history("krishnan")) >= 15


def test_a_rule_that_would_not_revert_cleanly_is_dropped_at_load() -> None:
    """`plain` must survive inside `surface`, or reverting cannot be a pure deletion."""
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


def test_a_baseline_has_no_way_to_be_written_into() -> None:
    """Not a writer that refuses -- no writer at all, so a mistake fails at author time."""
    assert not hasattr(BaselineStore, "save_history")
    assert not hasattr(BaselineStore, "save_profile")
    assert not hasattr(BaselineStore, "seed")


def test_the_live_store_refuses_a_profile_id_that_escapes_its_root(tmp_path: Path) -> None:
    """The id arrives from a form field, so `../` in it would otherwise write anywhere."""
    store = LiveStore(tmp_path / "personal")
    with pytest.raises(ValueError):
        store.history("../../personas/krishnan")


def test_seeding_copies_a_baseline_without_touching_it(tmp_path: Path) -> None:
    baseline = BaselineStore(PERSONAS)
    before = (PERSONAS / "history" / "krishnan.jsonl").read_bytes()
    live = LiveStore(tmp_path / "personal")
    live.seed(baseline.profile("krishnan"), baseline.history("krishnan"))
    assert len(live.history("krishnan")) == len(baseline.history("krishnan"))
    assert live.profile("krishnan").baseline is False
    assert (PERSONAS / "history" / "krishnan.jsonl").read_bytes() == before


# ---------------------------------------------------------------- the disambiguation prior


def test_a_known_name_is_offered_only_when_the_recognizer_produced_it() -> None:
    profile = PersonaProfile(
        id="p",
        label="P",
        lexicon=[{"id": "p/marge", "word": "marge", "display": "Marge", "kind": "person"}],
    )
    produced = _slot_alignment(beams("marge tea", "march tea", "large tea"))
    assert [hint.display for hint in lexicon_hints(profile, produced, 8)] == ["Marge"]

    absent = _slot_alignment(beams("march tea", "large tea"))
    assert lexicon_hints(profile, absent, 8) == []


def test_a_detail_is_offered_only_where_the_evidence_settled_the_word() -> None:
    """Share, not perfect agreement: beam search splits one word across spellings.

    A real utterance of "coffee" comes back as coffee 0.88 / coffey 0.06 /
    coffeey 0.06, so demanding a stable slot would mean the detail layer never
    fired at all. Two genuinely competing words split the weight far more evenly
    and stay unsettled.
    """
    profile = PersonaProfile(
        id="p",
        label="P",
        specializations=[
            {"id": "p/coffee", "anchor": "coffee", "plain": "coffee", "surface": "Madras filter coffee", "kind": "food"}
        ],
    )
    settled = _slot_alignment(weighted(("coffee", 0.88), ("coffey", 0.06), ("coffeey", 0.06)))
    assert [offer.surface for offer in specialization_offers(profile, settled, 0.75)] == [
        "Madras filter coffee"
    ]

    split = _slot_alignment(weighted(("coffee", 0.34), ("toffee", 0.33), ("coffey", 0.33)))
    assert specialization_offers(profile, split, 0.75) == []


# ---------------------------------------------------------------- the detail layer


def test_a_declared_detail_is_applied_and_recorded() -> None:
    chain, _ = chain_with(
        one_option(
            "I would like my Madras filter coffee.",
            "coffee",
            [{"anchor": "coffee", "surface": "Madras filter coffee", "source": "krishnan/detail/coffee"}],
        )
    )
    result = chain._compose(beams("coffee", "coffee"), "home", coffee_brief())
    message = result.messages[0]
    assert message.corrected_text == "I would like my Madras filter coffee."
    assert [item.source for item in message.specializations] == ["krishnan/detail/coffee"]
    assert result.ranker.personalization.specializations_applied == 1


def test_the_plain_wording_is_derived_in_code_rather_than_asked_for() -> None:
    chain, _ = chain_with(
        one_option(
            "I would like my Madras filter coffee.",
            "coffee",
            [{"anchor": "coffee", "surface": "Madras filter coffee", "source": "krishnan/detail/coffee"}],
        )
    )
    result = chain._compose(beams("coffee", "coffee"), "home", coffee_brief())
    assert result.messages[0].plain_text == "I would like my coffee."


def test_a_detail_whose_anchor_is_not_in_the_reading_is_refused() -> None:
    chain, _ = chain_with(
        one_option(
            "Please bring my Madras filter coffee.",
            "water",
            [{"anchor": "coffee", "surface": "Madras filter coffee", "source": "krishnan/detail/coffee"}],
        )
    )
    result = chain._compose(beams("water", "water"), "home", coffee_brief())
    assert result.messages[0].specializations == []
    assert result.ranker.personalization.specializations_refused == 1


def test_wording_the_profile_never_declared_is_refused_and_stripped() -> None:
    """The model marked a span as personal knowledge; it was not, so it comes back out."""
    chain, _ = chain_with(
        one_option(
            "I would like my Colombian filter coffee.",
            "coffee",
            [{"anchor": "coffee", "surface": "Colombian filter coffee", "source": "krishnan/detail/coffee"}],
        )
    )
    result = chain._compose(beams("coffee", "coffee"), "home", coffee_brief())
    message = result.messages[0]
    assert "Colombian" not in message.corrected_text
    assert message.corrected_text == "I would like my coffee."
    assert message.specializations == []


def test_a_detail_riding_on_a_contested_sound_is_refused() -> None:
    """A single message speaks itself on arrival, so a misheard anchor must stay plain."""
    chain, _ = chain_with(
        one_option(
            "I would like my Madras filter coffee.",
            "coffee",
            [{"anchor": "coffee", "surface": "Madras filter coffee", "source": "krishnan/detail/coffee"}],
        )
    )
    result = chain._compose(beams("coffee", "toffee", "coffey"), "home", coffee_brief())
    assert result.messages[0].specializations == []
    assert "Madras" not in result.messages[0].corrected_text


def test_a_declaration_with_no_profile_at_all_is_ignored() -> None:
    chain, _ = chain_with(
        one_option(
            "I would like my Madras filter coffee.",
            "coffee",
            [{"anchor": "coffee", "surface": "Madras filter coffee", "source": "krishnan/detail/coffee"}],
        )
    )
    result = chain._compose(beams("coffee", "coffee"), "home", None)
    assert result.messages[0].specializations == []
    assert result.ranker.personalization is None


def test_a_profile_name_the_recognizer_never_produced_cannot_reach_the_message() -> None:
    """`march` was heard, `Marge` was not; free prose would otherwise let it through."""
    chain, _ = chain_with(one_option("Please ask Marge for tea.", "march tea", []))
    brief = coffee_brief(
        lexicon=[LexiconHint(word="march", display="March", kind="person")],
        specializations=[],
        audit_vocabulary=["marge"],
    )
    result = chain._compose(beams("march tea", "large tea"), "home", brief)
    assert result.ranker.source == "unavailable"


def test_turning_the_detail_layer_off_leaves_the_prior_working() -> None:
    chain, completions = chain_with(
        one_option(
            "I would like my Madras filter coffee.",
            "coffee",
            [{"anchor": "coffee", "surface": "Madras filter coffee", "source": "krishnan/detail/coffee"}],
        ),
        personal_specializations=False,
    )
    brief = coffee_brief(lexicon=[LexiconHint(word="coffee", display="coffee", kind="food")])
    result = chain._compose(beams("coffee", "coffee"), "home", brief)
    assert result.messages[0].specializations == []
    assert "Madras" not in result.messages[0].corrected_text
    assert "known_words" in json.loads(completions.kwargs["messages"][1]["content"])


# ---------------------------------------------------------------- the un-personalized floor


def test_a_request_without_a_profile_sends_exactly_what_it_always_sent() -> None:
    chain, completions = chain_with(one_option("I am cold.", "cold"))
    chain._compose(beams("cold", "cold"), "home", None)
    payload = json.loads(completions.kwargs["messages"][1]["content"])
    assert list(payload) == [
        "setting",
        "setting_guidance",
        "alignment",
        "slot_options",
        "transcriptions",
    ]
    assert completions.kwargs["response_format"]["json_schema"]["name"] == "echora_message"
    assert "known_details" not in completions.kwargs["messages"][0]["content"]


def test_an_empty_brief_is_treated_as_no_brief() -> None:
    chain, completions = chain_with(one_option("I am cold.", "cold"))
    empty = PersonalBrief(profile_id="krishnan", profile_label="Krishnan")
    chain._compose(beams("cold", "cold"), "home", empty)
    assert list(json.loads(completions.kwargs["messages"][1]["content"]))[-1] == "transcriptions"


def test_the_grounding_rule_still_applies_when_a_profile_is_present() -> None:
    chain, _ = chain_with(one_option("I have a dog.", "dog", []))
    result = chain._compose(beams("cold", "cold"), "home", coffee_brief())
    assert result.ranker.source == "unavailable"


def test_the_personal_payload_keeps_the_evidence_keys_first() -> None:
    chain, completions = chain_with(
        one_option("I would like my Madras filter coffee.", "coffee", [])
    )
    brief = coffee_brief(lexicon=[LexiconHint(word="coffee", display="coffee", kind="food")])
    chain._compose(beams("coffee", "coffee"), "home", brief)
    keys = list(json.loads(completions.kwargs["messages"][1]["content"]))
    assert keys[:5] == ["setting", "setting_guidance", "alignment", "slot_options", "transcriptions"]
    assert completions.kwargs["response_format"]["json_schema"]["name"] == "echora_personal_message"


# ---------------------------------------------------------------- retrieval and consolidation

BUNDLE = Path(__file__).resolve().parents[2] / "models" / "echora-minilm-l6-v2" / "encoder"
needs_encoder = pytest.mark.skipif(not BUNDLE.is_dir(), reason="embedding bundle not fetched")


def message(text: str, *, heard: str = "", days: int = 1, hour: int = 12, context: str = "home", uses: int = 1):
    from datetime import datetime, timedelta, timezone

    when = (datetime.now(timezone.utc) - timedelta(days=days)).replace(hour=hour)
    return AcceptedMessage(
        id=f"x-{text[:8]}-{days}-{hour}",
        profile_id="p",
        heard=heard or text.lower(),
        message=text,
        context=context,
        hour=hour,
        accepted_at=when.isoformat().replace("+00:00", "Z"),
        uses=uses,
    )


def test_the_query_weights_a_contested_sound_down_to_one_words_worth() -> None:
    """Five spellings of one sound must not outvote the word every beam agreed on."""
    from app.personal.retrieval import query_terms

    slots = _slot_alignment(beams("leg pain", "link pain", "league pain", "lleg pain", "ling pain"))
    text, weights = query_terms(slots)
    assert weights["pain"] == 1.0
    assert all(weights[word] < 0.5 for word in ("leg", "link", "league", "lleg", "ling"))
    assert "pain" in text


@needs_encoder
def test_the_weighted_query_lands_nearer_the_true_utterance_than_the_raw_beams() -> None:
    from app.personal.embedder import Embedder
    from app.personal.retrieval import query_terms

    encoder = Embedder(BUNDLE.parent)
    slots = _slot_alignment(beams("leg pain", "link pain", "league pain", "lleg pain", "ling pain"))
    text, weights = query_terms(slots)
    truth = encoder.encode("leg pain")
    assert float(encoder.encode(text, weights) @ truth) > float(encoder.encode(text) @ truth)


@needs_encoder
def test_a_store_below_the_cold_start_floor_offers_nothing() -> None:
    from datetime import datetime, timezone

    from app.personal.embedder import Embedder
    from app.personal.retrieval import ExampleIndex, index_text

    encoder = Embedder(BUNDLE.parent)
    settings = load_settings()
    index = ExampleIndex(encoder.dimension)
    for text in ["I am cold.", "I need a blanket.", "My leg hurts."]:
        record = message(text)
        index.add(record, encoder.encode(index_text(record)))
    query = encoder.encode("cold")
    assert index.search(query, "home", datetime.now(timezone.utc), settings) == []


@needs_encoder
def test_recency_separates_two_equally_similar_messages() -> None:
    from datetime import datetime, timezone

    from app.personal.embedder import Embedder
    from app.personal.retrieval import ExampleIndex, index_text

    encoder = Embedder(BUNDLE.parent)
    settings = load_settings()
    index = ExampleIndex(encoder.dimension)
    rows = [
        message("I would like my Madras filter coffee.", heard="coffee", days=2),
        message("I would like my Narasu's filter coffee.", heard="coffee", days=400),
        message("I am cold.", days=5),
        message("I need the toilet.", days=6),
        message("Please call Priya.", days=7),
    ]
    for record in rows:
        index.add(record, encoder.encode(index_text(record)))
    found = index.search(encoder.encode("coffee"), "home", datetime.now(timezone.utc), settings)
    assert "Madras" in found[0].message


@needs_encoder
def test_saying_the_same_thing_again_merges_instead_of_growing_the_store() -> None:
    from datetime import datetime, timezone

    from app.personal.consolidation import merge_into, prune
    from app.personal.embedder import Embedder
    from app.personal.retrieval import ExampleIndex, index_text

    encoder = Embedder(BUNDLE.parent)
    index = ExampleIndex(encoder.dimension)
    first = message("I would like some water.", heard="water")
    merged, _ = merge_into(index, first, encoder.encode(index_text(first)), 0.92)
    assert merged is False

    again = message("I would like some water.", heard="water", days=0)
    merged, record = merge_into(index, again, encoder.encode(index_text(again)), 0.92)
    assert merged is True
    assert len(index) == 1
    assert record.uses == 2

    other = message("My leg hurts.", heard="leg pain")
    merged, _ = merge_into(index, other, encoder.encode(index_text(other)), 0.92)
    assert merged is False
    assert len(index) == 2

    settings = load_settings()
    capped = settings.__class__(**{**settings.__dict__, "personal_store_cap": 1})
    dropped = prune(index, datetime.now(timezone.utc), "home", capped)
    assert dropped == 1
    # The frequent, recent message is the one that survives.
    assert index.records[0].message == "I would like some water."
