"""The load-bearing property: text we did not deliberately change is preserved.

These tests are the reason the total-partition model exists. If any of them
fail, the device is corrupting its user's speech.
"""
import pytest

from communication.backend.hinglish_core.core.model import Label, Segment, Utterance
from communication.backend.hinglish_core.detect.segment import segment
from communication.backend.hinglish_core.pipeline import Pipeline
from communication.backend.hinglish_core.classify.port import FakeSpanClassifier
from communication.backend.hinglish_core.translit.transliterator import IdentityTransliterator

ROUNDTRIP_CASES = [
    "Main kal office mein important meeting attend karunga.",
    "Hello world",
    "नमस्ते दोस्त",
    "Check https://example.com/a.b?q=1&z=2 abhi",
    "Mail me at aman.k@example.co.in kal",
    "Main   double  spaced\tand\ttabbed",
    "Emoji 😊 beech mein 🎉 hai",
    "Price ₹1,00,000 hai bhai",
    "Call at 10:30am on 12/03/2026",
    "Run run_test.py --flag v1.2.3 abhi",
    "Nested @handle #hashtag ~/dir/file.py done",
    "Trailing whitespace   ",
    "   Leading whitespace",
    "",
    "!!!",
    "kya??? nahi!!! ...",
    "Line one\nline two\r\nline three",
    "IIT-B se hoon 😊 https://x.com/a.b?q=1",
    "नमस्ते! Main aa raha hoon। Theek hai?",
    "a",
    "Ek do teen 1 2 3 ek-do",
]


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_segmentation_is_a_total_partition(text):
    """Every character lands in exactly one segment, and render() rebuilds it."""
    u = segment(text)
    u.validate()
    assert u.render() == text


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_pipeline_preserves_text_when_nothing_transliterates(text):
    """With an identity transliterator the output must be byte-identical."""
    p = Pipeline(transliterator=IdentityTransliterator())
    assert p.run(text).text == text


def test_offsets_always_index_the_original():
    u = segment("Main kal office mein karunga.")
    for seg in u.segments:
        assert seg.text == u.raw[seg.start:seg.end]


def test_partition_rejects_a_gap():
    """A partition that stops covering the string must fail loudly."""
    with pytest.raises(ValueError, match="partition"):
        Utterance.from_segments("abcdef", [Segment("abc", 0, 3)])


def test_partition_rejects_an_overlap():
    with pytest.raises(ValueError, match="partition"):
        Utterance.from_segments(
            "abcdef", [Segment("abcd", 0, 4), Segment("cdef", 2, 6)]
        )


def test_segment_rejects_text_not_matching_its_span():
    with pytest.raises(ValueError, match="length"):
        Segment("abc", 0, 5)


def test_length_changing_replacement_does_not_drift_later_offsets():
    """The core offset-drift guard.

    'karunga' (7 chars) becomes 'करूंगा' (6). If offsets were rewritten as text
    changed, every later segment would shift. They are not, so they do not.
    """
    text = "Main karunga aur office jaunga kal"
    fake = FakeSpanClassifier({"Main": Label.HI, "aur": Label.HI,
                               "office": Label.EN, "jaunga": Label.HI})
    result = Pipeline(classifier=fake).run(text)
    for seg in result.utterance.segments:
        assert seg.text == text[seg.start:seg.end], (
            f"offset drift at {seg.start}..{seg.end}"
        )
    assert "office" in result.text


def test_emoji_adjacent_to_words_survives():
    p = Pipeline(transliterator=IdentityTransliterator())
    for text in ["Main😊kal", "😊", "🇮🇳 India", "👨‍👩‍👧 family hai", "5️⃣ number"]:
        assert p.run(text).text == text
