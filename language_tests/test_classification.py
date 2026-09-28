"""Lexical fast path, the arbitration rule, and the fallback chain."""
import pytest

from echora.classify.lexical import (
    HI_MARGIN_VETO, LexicalClassifier, ambiguous_spans,
)
from echora.classify.port import (
    CachingSpanClassifier, FakeSpanClassifier, FallbackChain,
    HeuristicSpanClassifier, SpanDecision, SpanQuery,
)
from echora.classify.prompt import (
    SYSTEM_PROMPT, build_user_message, parse_response, response_schema,
    system_prompt,
)
from echora.core.model import Label, Source
from echora.detect.segment import segment


@pytest.fixture(scope="module")
def lex():
    return LexicalClassifier()


# --- The arbitration rule -------------------------------------------------

@pytest.mark.parametrize("token", [
    "karunga", "mein", "kal", "nahi", "kya", "ghar", "dost", "jaldi",
])
def test_hindi_only_evidence_gives_HI(lex, token):
    label, _, _ = lex.classify_token(token)
    assert label is Label.HI


@pytest.mark.parametrize("token", [
    "important", "yesterday", "understand", "necessary", "probably",
])
def test_english_only_evidence_gives_EN(lex, token):
    label, _, _ = lex.classify_token(token)
    assert label is Label.EN


@pytest.mark.parametrize("token", [
    "main", "is", "to", "me", "par", "or", "do", "bar", "so", "the", "hum",
])
def test_collision_words_are_never_decided_lexically(lex, token):
    """The heart of the design: both-evidence must refuse to guess.

    Every one of these is simultaneously a frequent English word and a common
    Roman-Hindi word. Deciding either way without context would be wrong about
    half the time, so the only correct lexical answer is AMBIGUOUS.
    """
    label, _, _ = lex.classify_token(token)
    assert label is Label.AMBIGUOUS, f"{token!r} must defer to context"


@pytest.mark.parametrize("token", ["zzzblah", "qwertyx", "flurble"])
def test_unknown_tokens_are_ambiguous_not_english(lex, token):
    """OOV must not default to EN, or informal Hinglish would never convert."""
    label, _, _ = lex.classify_token(token)
    assert label is Label.AMBIGUOUS


# --- The decisive-EN gate -------------------------------------------------

@pytest.mark.parametrize("token", [
    "office", "meeting", "attend", "computer", "beautiful", "laptop",
    "password", "network", "release", "delivery", "connection",
])
def test_frequent_english_loanwords_are_decided_without_the_llm(lex, token):
    """Loanwords exist in Dakshina as Devanagari (office -> ऑफिस), so the
    both-evidence rule alone leaves them AMBIGUOUS.

    That is correct in principle and ruinous in practice: thousands of ordinary
    English words end up on the LLM's hot path, and the model is measurably
    worse at them than a threshold is -- qwen3:4b labels "attend" HI. Long,
    frequent, and English-dominant is decisive enough to skip the call.
    """
    label, source, _ = lex.classify_token(token)
    assert label is Label.EN, f"{token!r} should not need an LLM call"
    assert source is Source.LEXICON


@pytest.mark.parametrize("token", ["maine", "karen", "banana", "thick"])
def test_gate_never_fires_on_frequent_native_hindi_words(lex, token):
    """The gate's central risk, and the reason for the Hindi-margin veto.

    maine (मैंने), karen (करें), banana (बनाना) and thick (ठीक) are lexicon keys
    with English Zipf above the decisive bar, so length and English frequency
    alone would force them to EN. They are among the most frequent words in
    spoken Hinglish, and the shipped model classifies them correctly today --
    the gate must not take that decision away from it.
    """
    label, _, _ = lex.classify_token(token)
    assert label is Label.AMBIGUOUS, f"{token!r} must still reach the LLM"


def test_hindi_margin_veto_is_what_protects_them(lex):
    """Pin the mechanism, not just the outcome.

    If a wordfreq refresh moves these words, this test should fail here with a
    readable margin rather than silently at some downstream benchmark.
    """
    for token in ("maine", "karen", "banana", "thick"):
        candidates = lex.hindi_candidates(token)
        margin = lex.english_zipf(token) - lex.hindi_zipf(candidates)
        assert margin < HI_MARGIN_VETO, (
            f"{token!r} margin {margin:.2f} no longer trips the veto "
            f"({HI_MARGIN_VETO}); it now depends on DECISIVE_EN_EXCEPTIONS alone"
        )


@pytest.mark.parametrize("token", ["chain", "cheese", "niche"])
def test_exception_list_covers_what_the_margin_cannot(lex, token):
    """Hindi and English frequency are too close here for the veto to separate
    them (chain/चैन "peace", cheese/चीज़ "thing"), so they are listed by hand."""
    assert lex.classify_token(token)[0] is Label.AMBIGUOUS


@pytest.mark.parametrize("token", ["school", "doctor", "hospital", "college"])
def test_loanwords_the_policy_preserves_are_decided_english(lex, token):
    """These are the loanwords the gate is *supposed* to catch. Echora's stated
    policy preserves English loanwords as English, so deciding them here matches
    what the LLM should have said anyway."""
    assert lex.classify_token(token)[0] is Label.EN


def test_gate_cannot_overturn_a_hindi_decision(lex):
    """Structural guarantee: the gate lives inside the both-evidence branch, so
    a token with no English evidence never reaches it. It can only ever convert
    AMBIGUOUS to EN -- never HI to EN, and never anything to HI."""
    for token in ("karunga", "mein", "nahi", "ghar", "chahiye", "samajh"):
        assert lex.classify_token(token)[0] is Label.HI


def test_gate_does_not_fire_on_short_tokens(lex):
    """Length is the collision-risk proxy: every genuine short collision is
    <= 4 characters, so the gate must not reach them however frequent."""
    for token in ("main", "sale", "chal", "bar", "din"):
        assert lex.classify_token(token)[0] is not Label.EN


def test_midsentence_proper_noun_still_beats_the_gate(lex):
    """Placement check. "Delhi" is a lexicon key with English Zipf 4.29, so the
    gate would decide it EN if it ran before the capitalisation deferral."""
    tagged = lex.run(segment("Mujhe kal Delhi jana hai"))
    labels = {s.text: s.label for s in tagged.words()}
    assert labels["Delhi"] is Label.AMBIGUOUS


def test_protected_terms_win_outright():
    lex = LexicalClassifier(protected_terms={"Zomato", "Paytm", "main"})
    assert lex.classify_token("Zomato")[0] is Label.NAME
    assert lex.classify_token("zomato")[0] is Label.NAME
    # An allowlist entry overrides even a strong lexical signal.
    assert lex.classify_token("main")[0] is Label.NAME


def test_devanagari_input_is_already_hindi(lex):
    assert lex.classify_token("नमस्ते")[0] is Label.HI


def test_mai_can_reach_the_contextually_correct_preposition(lex):
    """The closed candidate set must contain the informal mai -> में reading."""
    assert "में" in lex.hindi_candidates("mai")


def test_structural_spans_are_never_reclassified(lex):
    u = lex.run(segment("Visit https://example.com kal"))
    url = next(s for s in u.segments if s.text.startswith("https"))
    assert url.source is Source.STRUCTURAL
    assert url.label is Label.CODE_URL


def test_ambiguous_spans_helper(lex):
    # "office" deliberately does NOT appear here: the decisive-EN gate resolves
    # it without a call. "Main" and "par" are real collisions and still defer.
    u = lex.run(segment("Main par baitha hoon"))
    assert {s.text for s in ambiguous_spans(u)} >= {"Main", "par"}


# --- Fallback chain -------------------------------------------------------

class _Boom:
    """A classifier that always fails, standing in for an unreachable vendor."""
    def classify(self, sentence, spans):
        raise RuntimeError("model unavailable")


def _q(text="main"):
    return [SpanQuery(0, text, 0, len(text))]


def test_chain_falls_through_to_the_next_classifier():
    chain = FallbackChain([_Boom(), FakeSpanClassifier({"main": Label.HI})])
    assert chain.classify("main ghar", _q())[0].label is Label.HI
    assert chain.last_used == "FakeSpanClassifier"


def test_chain_reaches_the_offline_heuristic_when_all_models_fail():
    """The guarantee that matters: the device still speaks with no model at all."""
    chain = FallbackChain([_Boom(), _Boom(), HeuristicSpanClassifier()])
    assert chain.classify("main hai", _q("hai"))[0].label is Label.HI
    assert chain.last_used == "HeuristicSpanClassifier"


def test_chain_raises_only_if_everything_fails():
    with pytest.raises(RuntimeError, match="every classifier"):
        FallbackChain([_Boom()]).classify("x", _q())


def test_unresolvable_span_defaults_to_preserving_the_text():
    """The safe default: preserve verbatim rather than risk mangling English."""
    decision = HeuristicSpanClassifier().classify("xyz", _q("wibble"))[0]
    assert decision.label is Label.EN
    assert decision.confidence == "unsure"


def test_escalation_only_asks_about_unsure_spans():
    class _Unsure:
        def classify(self, sentence, spans):
            return [SpanDecision(s.span_id, Label.EN, "unsure") for s in spans]

    better = FakeSpanClassifier({"main": Label.HI})
    chain = FallbackChain([_Unsure(), better], escalate_on_unsure=True)
    assert chain.classify("main ghar ja raha hoon", _q())[0].label is Label.HI
    assert len(better.calls) == 1


def test_no_escalation_by_default_stops_after_an_answer():
    """An unsure answer reaches the next classifier only when enabled."""
    class _Unsure:
        def classify(self, sentence, spans):
            return [SpanDecision(s.span_id, Label.EN, "unsure") for s in spans]

    cloud = FakeSpanClassifier({"main": Label.HI})
    chain = FallbackChain([_Unsure(), cloud])  # escalate_on_unsure=False
    chain.classify("main ghar", _q())
    assert cloud.calls == [], "next classifier ran without escalation enabled"


# --- Caching --------------------------------------------------------------

def test_cache_avoids_a_second_call_for_the_same_context():
    inner = FakeSpanClassifier({"main": Label.HI})
    cached = CachingSpanClassifier(inner)
    for _ in range(3):
        cached.classify("main ghar ja raha hoon", _q())
    assert len(inner.calls) == 1
    assert cached.hit_rate > 0.5


def test_cache_distinguishes_different_contexts():
    """'main office' and 'main gaya' must not share a cache entry."""
    inner = FakeSpanClassifier({}, default=Label.EN)
    cached = CachingSpanClassifier(inner)
    cached.classify("the main office is closed", [SpanQuery(0, "main", 4, 8)])
    cached.classify("main ghar ja raha hoon", [SpanQuery(0, "main", 0, 4)])
    assert len(inner.calls) == 2, "different contexts must miss the cache"


def test_cache_preserves_the_closed_candidate_choice():
    class _PickSecond:
        def classify(self, sentence, spans):
            return [SpanDecision(s.span_id, Label.HI, "certain", 1)
                    for s in spans]

    cached = CachingSpanClassifier(_PickSecond())
    query = [SpanQuery(0, "mai", 5, 8, ("मैं", "में"))]
    assert cached.classify("gale mai", query)[0].candidate_index == 1
    assert cached.classify("gale mai", query)[0].candidate_index == 1


# --- Closed-set sense selection ------------------------------------------

def test_label_only_prompt_and_schema_stay_byte_for_byte_unchanged():
    """The established classifier benchmark must see its exact old contract."""
    spans = [SpanQuery(0, "main", 0, 4)]
    assert system_prompt(spans) == SYSTEM_PROMPT
    assert build_user_message("main ghar", spans) == (
        "Sentence: main ghar\n\nSpans to label:\n0. main"
    )
    properties = response_schema(spans)["properties"]["spans"]["items"][
        "properties"
    ]
    assert "candidate_index" not in properties


def test_sense_selection_schema_allows_only_closed_integer_indices():
    spans = [SpanQuery(0, "mai", 5, 8, ("माई", "मैं", "मई", "में"))]
    schema = response_schema(spans)
    item = schema["properties"]["spans"]["items"]
    assert item["properties"]["candidate_index"]["enum"] == [-1, 0, 1, 2, 3]
    assert "candidate_index" in item["required"]
    assert item["additionalProperties"] is False


def test_parser_validates_candidate_index_against_its_own_span():
    spans = [SpanQuery(0, "mai", 5, 8, ("माई", "मैं", "मई", "में"))]
    valid = {"spans": [{
        "id": 0, "label": "HI", "confidence": "certain",
        "candidate_index": 3,
    }]}
    invalid = {"spans": [{
        "id": 0, "label": "HI", "confidence": "certain",
        "candidate_index": 99, "rewritten_text": "arbitrary output",
    }]}
    assert parse_response(valid, spans)[0] == ("HI", "certain", 3)
    assert parse_response(invalid, spans)[0] == ("HI", "certain", None)


# --- SentencePriorGuard ---------------------------------------------------

def test_guard_overrides_HI_in_a_fully_english_sentence():
    """The measured fix: models over-predict HI on plain English sentences.

    Lifted qwen3:4b collision accuracy from 60.0% to 90.0% on the gold set.
    """
    from echora.classify.port import SentencePriorGuard

    always_hi = FakeSpanClassifier({}, default=Label.HI)
    guard = SentencePriorGuard(always_hi)
    decision = guard.classify(
        "The main office is closed today", [SpanQuery(0, "main", 4, 8)]
    )[0]
    assert decision.label is Label.EN
    assert guard.overrides == 1


def test_guard_does_not_fire_when_the_sentence_has_hindi_evidence():
    """Any Hindi token disables the guard, so real Hinglish is unaffected."""
    from echora.classify.port import SentencePriorGuard

    always_hi = FakeSpanClassifier({}, default=Label.HI)
    guard = SentencePriorGuard(always_hi)
    decision = guard.classify(
        "Main ghar ja raha hoon", [SpanQuery(0, "Main", 0, 4)]
    )[0]
    assert decision.label is Label.HI
    assert guard.overrides == 0


def test_guard_never_overrides_towards_HI():
    """One-sided by design: it may only push to EN, never to Devanagari."""
    from echora.classify.port import SentencePriorGuard

    always_en = FakeSpanClassifier({}, default=Label.EN)
    guard = SentencePriorGuard(always_en)
    decision = guard.classify(
        "Mujhe kal subah jaldi uthna hai", [SpanQuery(0, "kal", 6, 9)]
    )[0]
    assert decision.label is Label.EN
    assert guard.overrides == 0


# --- Proper-noun deferral -------------------------------------------------

def test_midsentence_capitalised_token_defers_to_context(lex):
    """Indian names are often ordinary Hindi words (Aman = अमन "peace").

    Without this the fast path transliterates them with total confidence and
    they never reach the LLM at all.
    """
    tagged = lex.run(segment("Kal Aman aur Ravi aayenge"))
    labels = {s.text: s.label for s in tagged.words()}
    assert labels["Aman"] is Label.AMBIGUOUS
    assert labels["Ravi"] is Label.AMBIGUOUS
    assert labels["Kal"] is Label.HI, "sentence-initial word keeps normal handling"


def test_sentence_initial_name_is_a_known_gap_fixed_by_the_allowlist():
    """Capitalisation carries no signal on the first word, so the allowlist is
    the remedy -- the right mechanism for a single-user device."""
    plain = LexicalClassifier()
    assert plain.classify_token("Neha", sentence_initial=True)[0] is Label.HI

    allowed = LexicalClassifier(protected_terms={"Neha"})
    assert allowed.classify_token("Neha", sentence_initial=True)[0] is Label.NAME


def test_deferral_does_not_fire_on_the_first_word(lex):
    tagged = lex.run(segment("Karunga main ye kaam"))
    assert {s.text: s.label for s in tagged.words()}["Karunga"] is Label.HI
