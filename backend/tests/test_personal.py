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


def message(
    text: str,
    *,
    heard: str = "",
    days: int = 1,
    hour: int = 12,
    context: str = "home",
    listener: str = "familiar",
    uses: int = 1,
):
    from datetime import datetime, timedelta, timezone

    when = (datetime.now(timezone.utc) - timedelta(days=days)).replace(hour=hour)
    return AcceptedMessage(
        id=f"x-{text[:8]}-{days}-{hour}",
        profile_id="p",
        heard=heard or text.lower(),
        message=text,
        context=context,
        listener=listener,
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
    assert index.search(query, "home", "familiar", datetime.now(timezone.utc), settings) == []


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
    found = index.search(
        encoder.encode("coffee"), "home", "familiar", datetime.now(timezone.utc), settings
    )
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
    dropped = prune(index, datetime.now(timezone.utc), "home", "familiar", capped)
    assert dropped == 1
    # The frequent, recent message is the one that survives.
    assert index.records[0].message == "I would like some water."


# --------------------------------------------------------------- the stance

def test_a_setting_alone_decides_who_is_listening() -> None:
    """No profile, no place: outdoors still means strangers and home still does not."""
    from app.schemas import default_listener

    assert default_listener("home") == "familiar"
    assert default_listener("care") == "familiar"
    assert default_listener("general") == "familiar"
    assert default_listener("outdoors") == "unfamiliar"


def test_the_same_word_asks_outdoors_and_states_it_at_home() -> None:
    """The one sentence the model reads about the setting differs by listener."""
    from app.messaging.groq_chain import stance_guidance

    at_home = stance_guidance("home", "familiar")
    among_strangers = stance_guidance("outdoors", "unfamiliar")
    assert at_home != among_strangers
    # The home reading tells the model to state a need; the stranger reading
    # tells it to ask, and says the question word is form rather than content --
    # without which the prompt's own "supply nothing they did not say" rule
    # would rule the question form out.
    assert "stated to them" in at_home
    assert "Where is the washroom?" in among_strangers
    assert "not new content" in among_strangers
    # Shorter with a stranger, not longer. Politeness scaffolding is the wrong
    # direction here.
    assert "SHORTER" in among_strangers


def test_a_familiar_listener_outdoors_keeps_the_home_register() -> None:
    """Out with your daughter is not out among strangers."""
    from app.messaging.groq_chain import stance_guidance

    with_family = stance_guidance("outdoors", "familiar")
    assert "stated to them exactly as it would be at home" in with_family
    assert "Where is the washroom?" not in with_family


def test_an_unknown_stance_falls_back_to_the_setting_rather_than_failing() -> None:
    from app.messaging.groq_chain import stance_guidance

    assert stance_guidance("home", "nonsense") == stance_guidance("home", "familiar")  # type: ignore[arg-type]


def test_the_profile_says_who_it_is_usually_with_only_where_it_differs() -> None:
    profile = parse_profile(
        {
            "id": "k",
            "label": "K",
            "context_default": "home",
            "listener_by_setting": {"outdoors": "familiar"},
        }
    )
    assert profile is not None
    assert profile.listener_by_setting.get("outdoors") == "familiar"
    # Silence everywhere else, so the setting's own default stands.
    assert profile.listener_by_setting.get("home") is None


# ------------------------------------------------------- details, by setting

def test_a_detail_scoped_to_a_setting_is_not_offered_outside_it() -> None:
    """"Madras filter coffee" is his words at home and unhelpful across a counter."""
    profile = parse_profile(
        {
            "id": "k",
            "label": "K",
            "specializations": [
                {
                    "id": "k/detail/coffee",
                    "anchor": "coffee",
                    "plain": "coffee",
                    "surface": "Madras filter coffee",
                    "kind": "food",
                    "settings": ["home", "care"],
                }
            ],
        }
    )
    assert profile is not None
    slots = _slot_alignment(weighted(("coffee please", 0.9), ("coffey please", 0.1)))
    assert specialization_offers(profile, slots, 0.75, "home")
    assert specialization_offers(profile, slots, 0.75, "outdoors") == []
    # An unscoped rule is unchanged: it applies everywhere, as every rule did
    # before scoping existed.
    profile.specializations[0].settings = []
    assert specialization_offers(profile, slots, 0.75, "outdoors")


def test_a_scoped_out_detail_still_cannot_be_written_by_the_model() -> None:
    """Scoping narrows what is offered. It must never widen what is accepted."""
    from app.personal.personalizer import Personalizer

    profile = parse_profile(
        {
            "id": "k",
            "label": "K",
            "specializations": [
                {
                    "id": "k/detail/coffee",
                    "anchor": "coffee",
                    "plain": "coffee",
                    "surface": "Madras filter coffee",
                    "kind": "food",
                    "settings": ["home"],
                }
            ],
        }
    )
    assert profile is not None
    # The audit vocabulary is a rejection list, not an offer list, so the words
    # a scoped-out detail would have licensed stay in it.
    vocabulary = Personalizer._audit_vocabulary(None, profile)  # type: ignore[arg-type]
    assert "madras" in vocabulary


# ------------------------------------- retrieval and consolidation, by stance

def test_a_home_example_is_penalized_harder_among_strangers_than_across_settings() -> None:
    """The two mismatches cost different things, so they cannot share a penalty.

    A care example read at home has the right shape and the wrong furniture. A
    home example read among strangers has the wrong shape, and shape is what a
    few-shot example teaches -- "I need the toilet." is what a carer is told and
    exactly not what a stranger is asked.
    """
    from datetime import datetime, timezone

    from app.personal.retrieval import score

    settings = load_settings()
    now = datetime.now(timezone.utc)
    at_home = message("I need the toilet.", context="home", listener="familiar", hour=now.hour)

    same = score(at_home, 0.9, now, "home", "familiar", settings)
    other_setting = score(at_home, 0.9, now, "care", "familiar", settings)
    other_listener = score(at_home, 0.9, now, "home", "unfamiliar", settings)
    both = score(at_home, 0.9, now, "outdoors", "unfamiliar", settings)

    assert same > other_setting > other_listener > both
    # Still a boost and never a filter: the example survives, it just stops
    # outranking a same-stance one.
    assert both > 0.0


def test_an_unfamiliar_acceptance_never_absorbs_the_familiar_one() -> None:
    """The bug this guard exists for, with the merge forced to fire.

    A merge takes the longer wording and stamps the newcomer's setting over the
    old one. Without the guard, accepting "Where is the washroom?" outdoors
    could swallow "I want to use the washroom.", keep the home wording because
    it is longer, and relabel it outdoors -- leaving the speaker with neither
    phrasing intact.
    """
    import numpy as np

    from app.personal.consolidation import merge_into
    from app.personal.retrieval import ExampleIndex

    index = ExampleIndex(3)
    # Identical vectors, so cosine is 1.0 and only the guard can stop the merge.
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    at_home = message("I want to use the washroom.", heard="washroom", listener="familiar")
    index.add(at_home, vector)

    outside = message("Where is the washroom?", heard="washroom", context="outdoors", listener="unfamiliar")
    merged, _ = merge_into(index, outside, vector, 0.92)

    assert merged is False
    assert len(index.records) == 2
    kept = {record.message: record for record in index.records}
    assert kept["I want to use the washroom."].context == "home"
    assert kept["I want to use the washroom."].listener == "familiar"
    assert kept["Where is the washroom?"].listener == "unfamiliar"


def test_two_familiar_settings_still_fold_together() -> None:
    """Gating on the listener, not the setting, so the store splits in two at worst."""
    import numpy as np

    from app.personal.consolidation import merge_into
    from app.personal.retrieval import ExampleIndex

    index = ExampleIndex(3)
    vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    index.add(message("I need the toilet.", context="home", listener="familiar"), vector)
    on_the_ward = message("I need the toilet.", context="care", listener="familiar")

    merged, record = merge_into(index, on_the_ward, vector, 0.92)
    assert merged is True
    assert len(index.records) == 1
    assert record.uses == 2
