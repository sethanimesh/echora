"""The SpanClassifier port and its composable implementations.

WHY THIS IS A PORT (and most things here are not)
-------------------------------------------------
Under the project's design rules an interface must name a second real
implementation or it should not exist. This one names four, all of which ship:

* :class:`~echora.classify.ollama_classifier.OllamaSpanClassifier` -- local, default
* :class:`~echora.classify.groq_classifier.GroqSpanClassifier` -- cloud fallback
* :class:`HeuristicSpanClassifier` -- offline, no model at all
* :class:`FakeSpanClassifier` -- deterministic, for tests

The decorators below (:class:`CachingSpanClassifier`, :class:`FallbackChain`)
also implement the port, which is what lets "local first, then cloud, then
offline" be *composition* rather than branching inside every call site.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from communication.backend.hinglish_core.core.model import Label

log = logging.getLogger(__name__)

# The closed label set the LLM may return. Deliberately excludes AMBIGUOUS: the
# LLM's job is to *resolve*, so it is not offered the option of abstaining into
# the same state it was asked to fix. Genuinely undecidable tokens go to OTHER,
# which is safe because OTHER is preserved verbatim.
LLM_LABELS = ("HI", "EN", "NAME", "NUMBER", "CODE_URL", "OTHER")


@dataclass(frozen=True, slots=True)
class SpanQuery:
    """One span needing a decision, with the offsets that identify it."""

    span_id: int
    text: str
    start: int
    end: int
    candidates: tuple[str, ...] = ()
    """Closed Devanagari choices, if the lexicon found more than one reading."""


@dataclass(frozen=True, slots=True)
class SpanDecision:
    """A closed label and optional closed-set reading choice for one span."""

    span_id: int
    label: Label
    confidence: str = "certain"
    """'certain' or 'unsure'. Drives escalation from local to cloud."""
    candidate_index: int | None = None
    """Index into :attr:`SpanQuery.candidates`; never model-authored text."""


@runtime_checkable
class SpanClassifier(Protocol):
    """Classifies ambiguous spans using the surrounding sentence as context.

    THE ANTI-REWRITE GUARANTEE
    --------------------------
    A classifier returns :class:`SpanDecision` objects, which hold a span id, a
    label drawn from a closed enum, and optionally an integer selecting one of
    the lexicon candidates supplied in :class:`SpanQuery` -- no free text of any
    kind. There is structurally nowhere for a rewritten sentence to go. This is
    enforced by the type and by validating the index against the query, not by
    asking the model nicely in a prompt, because prompts are advisory and types
    are not.
    """

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        """Return one decision per span. Must be total: every span gets a label.

        Implementations should not raise on model/network failure -- they should
        let :class:`FallbackChain` handle it by raising a clear exception, which
        the chain converts into a try-the-next-classifier.
        """
        ...


class HeuristicSpanClassifier:
    """Offline resolver. No model, no network, fully deterministic.

    This is the floor the device never falls below: if there is no local model
    and no internet, Echora still produces speech. It is also the reference
    implementation of the fallback policy.

    THE SAFE DEFAULT, AND WHY
    -------------------------
    For a span we cannot resolve, we must choose which error to make. The two
    options are not symmetric:

    * Guess ``HI`` wrongly -> an English word is mangled into Devanagari
      ("meeting" -> "मीटिंग"). The TTS then pronounces a *distorted* word, and
      the listener hears something that is not what the speaker meant. The user
      often cannot tell it happened, because they typed Roman text and never see
      the Devanagari.
    * Guess ``EN`` wrongly -> a Hindi word stays in Roman script ("karunga").
      A Hindi-capable TTS voice reading Roman script mispronounces it, but it
      mispronounces it *as the word it is*, and the distortion is the familiar
      kind a listener routinely repairs from context.

    So unresolved spans default to ``EN`` (preserve verbatim). The rule is:
    **when in doubt, do not transform.** Preservation is recoverable;
    transformation is not. This matters more here than in a normal text
    pipeline, because the person relying on this device may not be able to
    interrupt and correct a mispronunciation easily.
    """

    #: Roman-Hindi function words so distinctive that context is not required.
    #: Deliberately tiny and hand-checked -- this is a safety net, not a lexicon.
    #: Note the absence of "the": it is Hindi थे ("were") but overwhelmingly the
    #: English article, and including it corrupted ordinary English sentences.
    STRONG_HI = frozenset({
        "hai", "hain", "tha", "thi", "hoon", "hun", "raha", "rahi", "rahe",
        "karunga", "karungi", "karenge", "kiya", "karna", "karne", "kar",
        "nahi", "nahin", "nhi", "kya", "kyun", "kyu", "kaise", "kahan", "kab",
        "mera", "meri", "mere", "tera", "teri", "tumhara", "aapka", "unka",
        "bahut", "bohot", "thoda", "zyada", "accha", "acha", "achha",
        "yaar", "bhai", "arre", "matlab", "abhi", "phir", "lekin", "aur",
        "jaunga", "jaungi", "aaunga", "chahiye", "sakta", "sakti", "gaya", "gayi",
    })

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        out: list[SpanDecision] = []
        for span in spans:
            token = span.text.lower()
            if token in self.STRONG_HI:
                out.append(SpanDecision(span.span_id, Label.HI, "certain"))
            else:
                # Preserve verbatim. See the class docstring for why EN, not HI.
                out.append(SpanDecision(span.span_id, Label.EN, "unsure"))
        return out


class FakeSpanClassifier:
    """Scripted classifier for tests. Returns whatever it was constructed with."""

    def __init__(self, decisions: dict[str, Label], default: Label = Label.EN):
        self._decisions = {k.lower(): v for k, v in decisions.items()}
        self._default = default
        self.calls: list[tuple[str, list[SpanQuery]]] = []

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        self.calls.append((sentence, spans))
        return [
            SpanDecision(
                s.span_id, self._decisions.get(s.text.lower(), self._default)
            )
            for s in spans
        ]


class CachingSpanClassifier:
    """Memoises decisions. A large win for this product specifically.

    An assistive speech device serves one person with a highly repetitive core
    vocabulary -- AAC research consistently finds a few hundred words cover the
    large majority of utterances. Caching a *(token, local context)* decision
    therefore pays off far more here than in general text processing.

    The key includes a small window of surrounding tokens rather than the whole
    sentence, because "main" in "main office" and "main gaya" must not share a
    cache entry, while "main gaya tha" and "main gaya hoon" reasonably can.
    Whole-sentence keys would almost never hit; bare-token keys would be wrong.
    """

    def __init__(self, inner: SpanClassifier, window: int = 2, maxsize: int = 50_000):
        self._inner = inner
        self._window = window
        self._maxsize = maxsize
        self._cache: dict[tuple[str, str], SpanDecision] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, sentence: str, span: SpanQuery) -> tuple[str, str]:
        before = sentence[: span.start].split()[-self._window :]
        after = sentence[span.end :].split()[: self._window]
        ctx = " ".join([*before, "\x00", *after]).lower()
        return (span.text.lower(), ctx)

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        pending: list[SpanQuery] = []
        results: dict[int, SpanDecision] = {}

        for span in spans:
            cached = self._cache.get(self._key(sentence, span))
            if cached is not None:
                self.hits += 1
                results[span.span_id] = SpanDecision(
                    span.span_id, cached.label, cached.confidence,
                    cached.candidate_index,
                )
            else:
                self.misses += 1
                pending.append(span)

        if pending:
            for decision in self._inner.classify(sentence, pending):
                results[decision.span_id] = decision
                span = next(s for s in pending if s.span_id == decision.span_id)
                if len(self._cache) < self._maxsize:
                    self._cache[self._key(sentence, span)] = decision

        return [results[s.span_id] for s in spans]


    @property
    def last_used(self) -> str | None:
        """Forward provenance from the wrapped classifier.

        Without this, a decorator hides which vendor actually answered, and the
        CLI's ``--explain`` (and any metric built on it) reports nothing.
        """
        return getattr(self._inner, "last_used", None)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class FallbackChain:
    """Try classifiers in order; use the first that answers.

    This is what implements "local first, Groq as fallback": the chain is
    ``[Ollama, Groq, Heuristic]``. Because the heuristic never fails, the chain
    is total -- the pipeline degrades but never raises, which is the right
    behaviour for a device someone depends on to speak.

    ``escalate_on_unsure`` additionally re-asks the next classifier when the
    current one answers but flags low confidence. It is **off by default**
    because the measured small-model confidence field is not calibrated and can
    otherwise trigger a second request for nearly every span.
    """

    def __init__(
        self,
        classifiers: list[SpanClassifier],
        escalate_on_unsure: bool = False,
    ) -> None:
        if not classifiers:
            raise ValueError("FallbackChain needs at least one classifier")
        self._classifiers = classifiers
        self._escalate = escalate_on_unsure
        self.last_used: str | None = None

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        if not spans:
            return []

        last_error: Exception | None = None
        for index, classifier in enumerate(self._classifiers):
            try:
                decisions = classifier.classify(sentence, spans)
            except Exception as exc:  # noqa: BLE001 - any failure means "try next"
                log.warning(
                    "classifier %s failed (%s); falling back",
                    type(classifier).__name__, exc,
                )
                last_error = exc
                continue

            self.last_used = type(classifier).__name__
            is_last = index == len(self._classifiers) - 1
            if not self._escalate or is_last:
                return decisions

            unsure = [d for d in decisions if d.confidence == "unsure"]
            if not unsure:
                return decisions

            # Re-ask the next classifier about only the uncertain spans, then
            # splice its answers over the confident ones we already have.
            unsure_ids = {d.span_id for d in unsure}
            retry_spans = [s for s in spans if s.span_id in unsure_ids]
            try:
                better = {
                    d.span_id: d
                    for d in self._classifiers[index + 1].classify(
                        sentence, retry_spans
                    )
                }
            except Exception as exc:  # noqa: BLE001
                log.warning("escalation failed (%s); keeping local answers", exc)
                return decisions
            return [better.get(d.span_id, d) for d in decisions]

        raise RuntimeError(
            f"every classifier in the chain failed; last error: {last_error}"
        )


class SentencePriorGuard:
    """Overrides HI predictions in sentences that carry no Hindi evidence at all.

    WHY THIS EXISTS
    ---------------
    Measured against the gold set, every local model tested shares one failure
    mode: on collision tokens it over-predicts ``HI``. For qwen3:4b, 24 of 31
    errors were ``gold=EN, got=HI``, and **18 of those 24 sat in sentences with
    zero Hindi-classified tokens** -- plainly English sentences like "The main
    office is closed today" and "I am going to Delhi tomorrow".

    That is a *predictable* failure, so it is cheaper and more reliable to fix
    deterministically than to fix by changing models. The rule: if the lexical
    stage found no HI token anywhere in the sentence, and at least one confident
    EN token, then an ambiguous span in that sentence is English.

    Measured on the gold set: fires on 20 spans, **20 correct, 0 wrong**.

    THE RULE IS DELIBERATELY ONE-SIDED
    ----------------------------------
    The mirror-image rule -- "strong Hindi evidence and no English, so lean HI"
    -- was tested and **rejected** at 75-84% precision: it wrongly forced proper
    names (Delhi, Zomato, Paytm) and English loanwords into Devanagari. The
    asymmetry matches the pipeline's safety principle: overriding *to* EN
    preserves text and is recoverable, overriding *to* HI transforms it and is
    not. A guard is only worth having when it is essentially always right.
    """

    MIN_EN_TOKENS = 1

    def __init__(self, inner: SpanClassifier, lexical=None) -> None:
        self._inner = inner
        if lexical is None:
            from communication.backend.hinglish_core.classify.lexical import LexicalClassifier
            lexical = LexicalClassifier()
        self._lexical = lexical
        self.overrides = 0


    @property
    def last_used(self) -> str | None:
        """Forward provenance from the wrapped classifier.

        Without this, a decorator hides which vendor actually answered, and the
        CLI's ``--explain`` (and any metric built on it) reports nothing.
        """
        return getattr(self._inner, "last_used", None)

    def _sentence_is_english(self, sentence: str) -> bool:
        from communication.backend.hinglish_core.detect.segment import segment

        tagged = self._lexical.run(segment(sentence))
        hindi = english = 0
        for seg in tagged.segments:
            if not seg.is_word:
                continue
            if seg.label is Label.HI:
                return False  # any Hindi evidence disables the guard
            if seg.label is Label.EN:
                english += 1
        return hindi == 0 and english >= self.MIN_EN_TOKENS

    def classify(self, sentence: str, spans: list[SpanQuery]) -> list[SpanDecision]:
        decisions = self._inner.classify(sentence, spans)
        if not self._sentence_is_english(sentence):
            return decisions

        out: list[SpanDecision] = []
        for decision in decisions:
            if decision.label is Label.HI:
                self.overrides += 1
                out.append(SpanDecision(decision.span_id, Label.EN, "certain"))
            else:
                out.append(decision)
        return out
