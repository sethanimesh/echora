from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.config import load_settings
from app.messaging.groq_chain import (
    GroqMessageChain,
    _alignment_template,
    _grounded_reading,
    _is_rate_limited,
    _quota_exhausted,
    _slot_alignment,
    _slot_options,
    _word_alternatives,
)
from app.schemas import Hypothesis


def leg_pain_beams() -> list[Hypothesis]:
    return [
        Hypothesis(id="h1", literal_text="leg pain", sequence_score=-0.2, search_weight=0.32),
        Hypothesis(id="h2", literal_text="link pain", sequence_score=-0.3, search_weight=0.25),
        Hypothesis(id="h3", literal_text="league pain", sequence_score=-0.4, search_weight=0.18),
        Hypothesis(id="h4", literal_text="lleg pain", sequence_score=-0.5, search_weight=0.14),
        Hypothesis(id="h5", literal_text="ling pain", sequence_score=-0.6, search_weight=0.11),
    ]


def water_beams() -> list[Hypothesis]:
    return [
        Hypothesis(id="h1", literal_text="I water", sequence_score=-0.2, search_weight=0.55),
        Hypothesis(id="h2", literal_text="high water", sequence_score=-0.4, search_weight=0.45),
    ]


class OneResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


def chain_with(payload: dict) -> tuple[GroqMessageChain, OneResponse]:
    chain = GroqMessageChain(load_settings())
    completions = OneResponse(payload)
    chain.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return chain, completions


# --------------------------------------------------------------------------
# The behaviour this redesign exists for.
# --------------------------------------------------------------------------


def test_one_sound_heard_five_ways_yields_one_natural_message() -> None:
    chain, _ = chain_with(
        {"options": [{"reading": "leg pain", "message": "My leg hurts."}], "note": "Only leg is a body part.", "unclear": False}
    )
    result = chain._compose(leg_pain_beams(), "care")
    assert len(result.messages) == 1
    assert result.messages[0].corrected_text == "My leg hurts."
    assert result.ranker.decision == "selected"


def test_natural_rephrasing_is_no_longer_rejected() -> None:
    """The old allowlist replaced "My leg hurts." with a blunter sentence."""
    chain, _ = chain_with(
        {"options": [{"reading": "leg pain", "message": "My leg is really hurting me."}], "note": "", "unclear": False}
    )
    result = chain._compose(leg_pain_beams(), "care")
    assert result.messages[0].corrected_text == "My leg is really hurting me."


def test_the_reading_is_shown_beside_the_message() -> None:
    chain, _ = chain_with(
        {"options": [{"reading": "leg pain", "message": "My leg hurts."}], "note": "", "unclear": False}
    )
    message = chain._compose(leg_pain_beams(), "care").messages[0]
    assert message.interpreted_intent == "leg pain"
    assert "leg pain" in message.repair_note
    assert set(message.word_alternatives["leg"]) == {"link", "league", "lleg", "ling"}


def test_genuinely_different_words_still_ask() -> None:
    beams = [
        Hypothesis(id="h1", literal_text="my leg hurts", sequence_score=-0.1, search_weight=0.51),
        Hypothesis(id="h2", literal_text="my arm hurts", sequence_score=-0.2, search_weight=0.49),
    ]
    chain, _ = chain_with(
        {
            "options": [
                {"reading": "my leg hurts", "message": "My leg hurts."},
                {"reading": "my arm hurts", "message": "My arm hurts."},
            ],
            "note": "Leg and arm are both real.", "unclear": False,
        }
    )
    result = chain._compose(beams, "care")
    assert result.ranker.decision == "ambiguous"
    assert [item.corrected_text for item in result.messages] == ["My leg hurts.", "My arm hurts."]


# --------------------------------------------------------------------------
# The one property enforced in code.
# --------------------------------------------------------------------------


def test_a_word_no_beam_produced_is_rejected() -> None:
    slots = _slot_alignment(leg_pain_beams())
    assert _grounded_reading("leg pain", slots) == ["leg", "pain"]
    assert _grounded_reading("knee pain", slots) is None
    assert _grounded_reading("pain leg", slots) is None


def test_an_ungrounded_option_is_dropped_not_shown() -> None:
    chain, _ = chain_with(
        {
            "options": [
                {"reading": "leg pain", "message": "My leg hurts."},
                {"reading": "knee pain", "message": "My knee hurts."},
            ],
            "note": "",
            "unclear": False,
        }
    )
    result = chain._compose(leg_pain_beams(), "care")
    assert [item.corrected_text for item in result.messages] == ["My leg hurts."]


def test_every_ungrounded_option_falls_back_to_raw_beams() -> None:
    chain, _ = chain_with(
        {"options": [{"reading": "knee ache", "message": "My knee aches."}], "note": "", "unclear": False}
    )
    result = chain._compose(leg_pain_beams(), "care")
    assert all(item.repair_status == "unavailable" for item in result.messages)
    assert [item.corrected_text for item in result.messages][0] == "leg pain"


def test_distinct_readings_are_not_discarded_by_word_set_containment() -> None:
    """Containment cannot prove two proposed readings mean the same thing."""
    chain, _ = chain_with(
        {
            "options": [
                {"reading": "leg pain", "message": "My leg hurts."},
                {"reading": "pain", "message": "I have pain."},
            ],
            "note": "",
            "unclear": False,
        }
    )
    result = chain._compose(leg_pain_beams(), "care")
    assert [item.corrected_text for item in result.messages] == ["My leg hurts.", "I have pain."]
    assert result.ranker.decision == "ambiguous"


def test_options_are_capped_and_deduped() -> None:
    beams = [
        Hypothesis(id=f"h{i}", literal_text=text, sequence_score=-0.1 * i, search_weight=0.25)
        for i, text in enumerate(["want tea", "want toast", "want soup", "want juice"], 1)
    ]
    chain, _ = chain_with(
        {
            "options": [
                {"reading": "want tea", "message": "I want tea."},
                {"reading": "want tea", "message": "I would like tea."},
                {"reading": "want toast", "message": "I want toast."},
                {"reading": "want soup", "message": "I want soup."},
            ],
            "note": "",
            "unclear": False,
        }
    )
    result = chain._compose(beams, "home")
    texts = [item.corrected_text for item in result.messages]
    assert len(texts) <= 3
    assert texts == ["I want tea.", "I want toast.", "I want soup."]


@pytest.mark.asyncio
async def test_missing_groq_shows_raw_beams_labelled() -> None:
    settings = load_settings()
    settings = settings.__class__(**{**settings.__dict__, "groq_api_key": None})
    result = await GroqMessageChain(settings).run(water_beams(), "home")
    assert result.ranker.decision == "ambiguous"
    assert [item.corrected_text for item in result.messages] == ["I water", "high water"]
    assert all(item.repair_status == "unavailable" for item in result.messages)


def _rate_limit_error(body: str) -> Exception:
    error = Exception("429")
    error.response = SimpleNamespace(status_code=429, text=body, headers={})
    return error


TPD_BODY = "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199000"
TPM_BODY = "Rate limit reached ... on tokens per minute (TPM): Limit 8000"


class ScriptedModels:
    """Fails for named models, answers for the rest."""

    def __init__(self, failures: dict[str, Exception], payload: dict) -> None:
        self.failures = failures
        self.payload = payload
        self.tried: list[str] = []

    def create(self, **kwargs):
        model = kwargs["model"]
        self.tried.append(model)
        if model in self.failures:
            raise self.failures[model]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


def test_a_rate_limited_model_falls_through_to_the_next_one() -> None:
    """Groq meters each model separately, so a 429 means switch, not give up."""
    answer = {"options": [{"reading": "leg pain", "message": "My leg hurts."}], "note": "", "unclear": False}
    settings = load_settings()
    primary = settings.groq_models[0]
    scripted = ScriptedModels({primary: _rate_limit_error(TPM_BODY)}, answer)
    chain = GroqMessageChain(settings)
    chain.client = SimpleNamespace(chat=SimpleNamespace(completions=scripted))
    result = chain._compose(leg_pain_beams(), "care")
    assert result.messages[0].corrected_text == "My leg hurts."
    assert result.ranker.assistant_model == settings.groq_models[1]
    assert scripted.tried[0] == primary


@pytest.mark.asyncio
async def test_every_model_exhausted_says_so_and_shows_raw_beams() -> None:
    settings = load_settings()
    scripted = ScriptedModels(
        {model: _rate_limit_error(TPD_BODY) for model in settings.groq_models}, {}
    )
    chain = GroqMessageChain(settings)
    chain.client = SimpleNamespace(chat=SimpleNamespace(completions=scripted))
    result = await chain.run(leg_pain_beams(), "care")
    assert result.ranker.source == "unavailable"
    assert "daily usage limit" in result.ranker.reason
    assert set(scripted.tried) == set(settings.groq_models)
    assert [item.repair_status for item in result.messages] == ["unavailable"] * len(leg_pain_beams())


def test_a_daily_cap_is_distinguished_from_a_per_minute_throttle() -> None:
    assert _quota_exhausted(_rate_limit_error(TPD_BODY)) is True
    assert _quota_exhausted(_rate_limit_error(TPM_BODY)) is False
    assert _is_rate_limited(_rate_limit_error(TPM_BODY)) is True


# --------------------------------------------------------------------------
# Alignment primitives.
# --------------------------------------------------------------------------


def test_alignment_localizes_the_contested_sound() -> None:
    assert _alignment_template(_slot_alignment(leg_pain_beams())) == (
        "[leg|link|league|lleg|ling] pain"
    )
    assert _slot_options(_slot_alignment(leg_pain_beams())) == [
        ["leg", "link", "league", "lleg", "ling"],
        ["pain"],
    ]


def test_alignment_handles_a_dropped_word() -> None:
    beams = [
        Hypothesis(id="h1", literal_text="dict sheet uncomfortable", sequence_score=-0.3, search_weight=0.5),
        Hypothesis(id="h2", literal_text="dictate uncomfortable", sequence_score=-0.5, search_weight=0.5),
    ]
    slots = _slot_alignment(beams)
    assert _alignment_template(slots) == "[dict|dictate] [sheet|—] uncomfortable"
    assert _slot_options(slots)[1] == ["sheet"]


def test_word_alternatives_only_cover_contested_positions() -> None:
    slots = _slot_alignment(leg_pain_beams())
    assert _word_alternatives(slots, ["leg", "pain"]) == {
        "leg": ["link", "league", "lleg", "ling"]
    }


def test_prompt_carries_the_alignment_and_the_setting() -> None:
    chain, completions = chain_with(
        {"options": [{"reading": "leg pain", "message": "My leg hurts."}], "note": "", "unclear": False}
    )
    chain._compose(leg_pain_beams(), "care")
    sent = json.loads(completions.kwargs["messages"][1]["content"])
    assert sent["alignment"] == "[leg|link|league|lleg|ling] pain"
    assert sent["setting"] == "care"
    system = completions.kwargs["messages"][0]["content"]
    assert "never a word absent from that position's options" in system
    assert "Never make a message vaguer to be safe" in system
    assert "NO PLAUSIBLE READING" in system


def test_an_unclear_verdict_shows_raw_beams_instead_of_inventing() -> None:
    """Noise must not be assembled into a fluent sentence."""
    beams = [
        Hypothesis(id="h1", literal_text="fen tog", sequence_score=-0.1, search_weight=0.28),
        Hypothesis(id="h2", literal_text="thin tog", sequence_score=-0.3, search_weight=0.24),
    ]
    chain, _ = chain_with(
        {
            "options": [{"reading": "thin tog", "message": "I have a thin dog."}],
            "unclear": True,
            "note": "Nothing here forms a message.",
        }
    )
    result = chain._compose(beams, "general")
    assert result.ranker.source == "unavailable"
    assert all(item.repair_status == "unavailable" for item in result.messages)
    assert "thin dog" not in " ".join(item.corrected_text for item in result.messages)
