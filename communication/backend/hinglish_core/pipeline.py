"""The pipeline facade: raw Hinglish in, TTS-ready text out.

Deliberately a plain class, not a framework object. It orchestrates five stages
that are each independently testable and each usable on their own:

    segment -> lexical fast path -> LLM (ambiguous only) -> transliterate -> render
                                                                              |
                                                              TTS normalise <-+

The Haystack components in :mod:`echora.haystack_components` are thin adapters
over these same functions, so the core never depends on the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from communication.backend.hinglish_core.classify.lexical import LexicalClassifier
from communication.backend.hinglish_core.classify.port import (
    HeuristicSpanClassifier, SpanClassifier, SpanQuery,
)
from communication.backend.hinglish_core.core.model import Label, Segment, Source, Utterance
from communication.backend.hinglish_core.detect.segment import segment
from communication.backend.hinglish_core.translit.transliterator import (
    LexiconTransliterator, Transliterator, apply_transliteration,
)
from communication.backend.hinglish_core.tts.normalize import HinglishTtsNormalizer, TtsNormalizer


@dataclass(slots=True)
class Result:
    """The outcome of one utterance, with enough detail to debug a bad reading."""

    raw: str
    text: str
    """Reconstructed text: Hindi in Devanagari, everything else preserved."""
    speech: str
    """``text`` after TTS normalisation. What you hand to the voice."""
    utterance: Utterance
    llm_spans: int = 0
    """How many spans needed the LLM. The number to watch for cost/latency."""
    classifier_used: str | None = None

    def labels(self) -> list[tuple[str, str]]:
        """``(token, label)`` for every word. Convenience for tests and debugging."""
        return [(s.text, s.label.value) for s in self.utterance.words()]


class Pipeline:
    """Wires the stages together. Construct via :meth:`from_config` normally."""

    def __init__(
        self,
        lexical: LexicalClassifier | None = None,
        classifier: SpanClassifier | None = None,
        transliterator: Transliterator | None = None,
        tts: TtsNormalizer | None = None,
    ) -> None:
        self._lexical = lexical or LexicalClassifier()
        self._classifier = classifier or HeuristicSpanClassifier()
        self._translit = transliterator or LexiconTransliterator()
        self._tts = tts or HinglishTtsNormalizer()

    def run(self, text: str) -> Result:
        utterance = segment(text)
        utterance = self._lexical.run(utterance)

        pending = [s for s in utterance.segments if s.label is Label.AMBIGUOUS]
        used: str | None = None

        if pending:
            queries = [
                SpanQuery(
                    span_id=i, text=s.text, start=s.start, end=s.end,
                    candidates=s.candidates,
                )
                for i, s in enumerate(pending)
            ]
            decisions = {d.span_id: d for d in self._classifier.classify(text, queries)}
            used = getattr(self._classifier, "last_used", None)

            resolved: list[Segment] = []
            index = {id(s): i for i, s in enumerate(pending)}
            for seg in utterance.segments:
                position = index.get(id(seg))
                if position is None:
                    resolved.append(seg)
                    continue
                decision = decisions.get(position)
                if decision is None:
                    # No answer for this span: preserve verbatim. Same rationale
                    # as HeuristicSpanClassifier -- never transform on doubt.
                    resolved.append(seg.decided(Label.EN, Source.FALLBACK))
                else:
                    candidate_index = decision.candidate_index
                    if (
                        decision.label is not Label.HI
                        or candidate_index is None
                        or not 0 <= candidate_index < len(seg.candidates)
                    ):
                        candidate_index = None
                    resolved.append(
                        seg.decided(
                            decision.label, Source.LLM,
                            confidence=decision.confidence,
                            candidate_index=candidate_index,
                        )
                    )
            utterance = utterance.with_segments(resolved)

        utterance = apply_transliteration(utterance, self._translit)
        rendered = utterance.render()

        return Result(
            raw=text,
            text=rendered,
            speech=self._tts.normalize(rendered),
            utterance=utterance,
            llm_spans=len(pending),
            classifier_used=used,
        )
