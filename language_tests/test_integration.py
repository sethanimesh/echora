"""End-to-end behaviour across the categories the pipeline must handle."""
import pytest

from echora.classify.port import (
    FakeSpanClassifier, HeuristicSpanClassifier, SpanDecision,
)
from echora.core.model import Label
from echora.pipeline import Pipeline
from echora.tts.normalize import HinglishTtsNormalizer, NullTtsNormalizer


def test_the_canonical_worked_example():
    fake = FakeSpanClassifier({"Main": Label.HI, "office": Label.EN,
                               "meeting": Label.EN, "attend": Label.EN})
    result = Pipeline(classifier=fake).run(
        "Main kal office mein important meeting attend karunga."
    )
    assert result.text == "मैं कल office में important meeting attend करूंगा."


def test_labels_for_the_worked_example():
    fake = FakeSpanClassifier({"Main": Label.HI, "office": Label.EN,
                               "meeting": Label.EN, "attend": Label.EN})
    labels = dict(Pipeline(classifier=fake).run(
        "Main kal office mein important meeting attend karunga."
    ).labels())
    assert labels == {
        "Main": "HI", "kal": "HI", "office": "EN", "mein": "HI",
        "important": "EN", "meeting": "EN", "attend": "EN", "karunga": "HI",
    }


def test_context_selects_mai_as_in_without_allowing_model_text():
    class _ClosedSenseChoice:
        def classify(self, sentence, spans):
            labels = {
                "gale": Label.HI, "mai": Label.HI, "pain": Label.EN,
                "ho": Label.HI, "hai": Label.HI,
            }
            return [SpanDecision(
                span.span_id,
                labels.get(span.text.lower(), Label.EN),
                "certain",
                span.candidates.index("में") if span.text.lower() == "mai" else 0,
            ) for span in spans]

    text = "Mujhe gale mai pain ho raha hai"
    result = Pipeline(classifier=_ClosedSenseChoice()).run(text)

    assert result.text == "मुझे गले में pain हो रहा है"
    mai = next(seg for seg in result.utterance.words() if seg.text == "mai")
    pain = next(seg for seg in result.utterance.words() if seg.text == "pain")
    assert mai.candidates[mai.candidate_index] == "में"
    assert pain.output is None and pain.rendered == "pain"
    for seg in result.utterance.segments:
        assert seg.text == text[seg.start:seg.end]


def test_urls_and_emails_never_transliterate():
    p = Pipeline(classifier=HeuristicSpanClassifier())
    text = "Mail aman@example.co.in ya https://example.com/do/bar dekho"
    out = p.run(text).text
    assert "aman@example.co.in" in out
    assert "https://example.com/do/bar" in out


def test_acronyms_are_not_transliterated_by_the_linguistic_stage():
    p = Pipeline(classifier=HeuristicSpanClassifier())
    out = p.run("API aur AWS ka LLM setup karunga").text
    for acronym in ("API", "AWS", "LLM"):
        assert acronym in out


def test_numbers_currency_and_time_are_preserved():
    p = Pipeline(classifier=HeuristicSpanClassifier())
    text = "10:30am pe ₹1,00,000 ka 5kg order 12/03/2026 ko"
    out = p.run(text).text
    for token in ("10:30am", "₹1,00,000", "5kg", "12/03/2026"):
        assert token in out


def test_a_fully_english_sentence_is_untouched():
    text = "The quick brown fox jumps over the lazy dog"
    assert Pipeline(classifier=HeuristicSpanClassifier()).run(text).text == text


def test_a_devanagari_sentence_is_untouched():
    text = "मैं कल घर जाऊंगा।"
    assert Pipeline(classifier=HeuristicSpanClassifier()).run(text).text == text


def test_informal_spellings_convert_via_the_strong_hindi_set():
    p = Pipeline(classifier=HeuristicSpanClassifier())
    out = p.run("nahi yaar bahut accha hai").text
    assert any(0x0900 <= ord(c) <= 0x097F for c in out), out


def test_technical_vocabulary_stays_english():
    fake = FakeSpanClassifier({}, default=Label.EN)
    out = Pipeline(classifier=fake).run("laptop ka password reset karunga").text
    assert "laptop" in out and "password" in out and "reset" in out


def test_llm_is_only_asked_about_ambiguous_spans():
    """The cost guarantee: unambiguous tokens must never reach the model."""
    fake = FakeSpanClassifier({}, default=Label.EN)
    result = Pipeline(classifier=fake).run(
        "karunga nahi important computer mein"
    )
    asked = {s.text for _, spans in fake.calls for s in spans}
    assert "karunga" not in asked, "resolved HI token was sent to the LLM"
    assert "important" not in asked, "resolved EN token was sent to the LLM"
    assert result.llm_spans == len(asked)


def test_no_llm_call_at_all_when_nothing_is_ambiguous():
    fake = FakeSpanClassifier({})
    result = Pipeline(classifier=fake).run("karunga nahi jaldi")
    assert fake.calls == []
    assert result.llm_spans == 0


def test_the_llm_receives_the_whole_sentence_as_context():
    fake = FakeSpanClassifier({"main": Label.HI})
    sentence = "Main ghar ja raha hoon"
    Pipeline(classifier=fake).run(sentence)
    assert fake.calls[0][0] == sentence


def test_pipeline_still_produces_output_with_every_model_down():
    """Degraded mode: no model anywhere, and the device still speaks."""
    class _Boom:
        def classify(self, sentence, spans):
            raise RuntimeError("no model")

    from echora.classify.port import FallbackChain
    chain = FallbackChain([_Boom(), HeuristicSpanClassifier()])
    result = Pipeline(classifier=chain).run("Main kal office jaunga")
    assert result.text
    assert "office" in result.text


# --- TTS normalisation is a separate stage --------------------------------

def test_tts_stage_expands_acronyms_and_units():
    n = HinglishTtsNormalizer()
    assert "A P I" in n.normalize("API call")
    assert "kilogram" in n.normalize("5kg")
    assert "rupees" in n.normalize("₹500")


def test_tts_stage_does_not_touch_devanagari():
    n = HinglishTtsNormalizer()
    assert n.normalize("मैं कल जाऊंगा") == "मैं कल जाऊंगा"


def test_tts_stage_is_separable():
    """Linguistic output and speech output are distinct fields."""
    p = Pipeline(classifier=HeuristicSpanClassifier(), tts=NullTtsNormalizer())
    result = p.run("API test")
    assert result.speech == result.text


def test_speech_differs_from_text_when_normalisation_applies():
    result = Pipeline(classifier=HeuristicSpanClassifier()).run("API kal")
    assert result.speech != result.text
    assert "A P I" in result.speech
