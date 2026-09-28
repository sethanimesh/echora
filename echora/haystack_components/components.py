"""Haystack components -- thin adapters over the framework-free core.

Haystack remains available at the library orchestration boundary. The unified
communication app uses its own asynchronous message lifecycle.

Every component here is a ~10-line wrapper that delegates to a plain function in
``echora.*``. No logic lives in this file. That matters because:

* The deterministic core stays unit-testable with plain pytest and no framework
  in the loop -- which is why the 100+ tests in ``tests/`` import none of this.
* The offset-preserving data model passes between components as our own frozen
  dataclasses. Haystack's ``Document`` is deliberately NOT used: it flattens to
  ``content`` + ``meta`` and would destroy the character offsets the entire
  design depends on.
* If Haystack is ever dropped or replaced, this file is deleted and nothing else
  changes.

Requires the optional extra::

    pip install 'echora[haystack]'
"""

from __future__ import annotations

from haystack import component

from echora.classify.lexical import LexicalClassifier
from echora.classify.port import SpanClassifier, SpanQuery
from echora.config import Config
from echora.core.model import Label, Source, Utterance
from echora.detect.segment import segment
from echora.translit.transliterator import (
    LexiconTransliterator, Transliterator, apply_transliteration,
)
from echora.tts.normalize import HinglishTtsNormalizer, TtsNormalizer
from echora.tts.fish_audio import (
    FishAudioSynthesizer, SpeechOptions,
)


@component
class Segmenter:
    """Raw text -> a validated total partition."""

    @component.output_types(utterance=Utterance)
    def run(self, text: str):
        return {"utterance": segment(text)}


@component
class LexicalTagger:
    """Applies the deterministic fast path. No network, no model."""

    def __init__(self, protected_terms: set[str] | None = None) -> None:
        self._classifier = LexicalClassifier(protected_terms=protected_terms)

    @component.output_types(utterance=Utterance, ambiguous=int)
    def run(self, utterance: Utterance):
        tagged = self._classifier.run(utterance)
        pending = sum(1 for s in tagged.segments if s.label is Label.AMBIGUOUS)
        return {"utterance": tagged, "ambiguous": pending}


@component
class ContextualResolver:
    """Resolves ambiguous spans via the configured classifier chain.

    Defaults to the chain from :class:`~echora.config.Config`: local Ollama
    first, Groq as fallback, offline heuristic as the floor.
    """

    def __init__(self, classifier: SpanClassifier | None = None) -> None:
        self._classifier = classifier or Config.from_env().build_classifier()

    @component.output_types(utterance=Utterance, llm_spans=int)
    def run(self, utterance: Utterance):
        pending = [s for s in utterance.segments if s.label is Label.AMBIGUOUS]
        if not pending:
            return {"utterance": utterance, "llm_spans": 0}

        queries = [
            SpanQuery(
                span_id=i, text=s.text, start=s.start, end=s.end,
                candidates=s.candidates,
            )
            for i, s in enumerate(pending)
        ]
        decisions = {
            d.span_id: d
            for d in self._classifier.classify(utterance.raw, queries)
        }

        index = {id(s): i for i, s in enumerate(pending)}
        resolved = []
        for seg in utterance.segments:
            position = index.get(id(seg))
            if position is None:
                resolved.append(seg)
                continue
            decision = decisions.get(position)
            if decision is None:
                resolved.append(seg.decided(Label.EN, Source.FALLBACK))
                continue
            candidate_index = decision.candidate_index
            if (
                decision.label is not Label.HI
                or candidate_index is None
                or not 0 <= candidate_index < len(seg.candidates)
            ):
                candidate_index = None
            resolved.append(seg.decided(
                decision.label, Source.LLM,
                confidence=decision.confidence,
                candidate_index=candidate_index,
            ))
        return {
            "utterance": utterance.with_segments(resolved),
            "llm_spans": len(pending),
        }


@component
class DevanagariTransliterator:
    """Transliterates HI spans only."""

    def __init__(self, transliterator: Transliterator | None = None) -> None:
        self._translit = transliterator or LexiconTransliterator()

    @component.output_types(utterance=Utterance, text=str)
    def run(self, utterance: Utterance):
        out = apply_transliteration(utterance, self._translit)
        return {"utterance": out, "text": out.render()}


@component
class TtsNormalization:
    """Pronunciation pass. Runs after reconstruction, never before."""

    def __init__(self, normalizer: TtsNormalizer | None = None) -> None:
        self._normalizer = normalizer or HinglishTtsNormalizer()

    @component.output_types(speech=str)
    def run(self, text: str):
        return {"speech": self._normalizer.normalize(text)}


@component
class FishAudioTTS:
    """Normalised text -> lazy Fish audio/alignment event stream."""

    def __init__(
        self,
        synthesizer: FishAudioSynthesizer | None = None,
        options: SpeechOptions | None = None,
    ) -> None:
        self._synthesizer = synthesizer or Config.from_env().build_synthesizer()
        self._options = options

    @component.output_types(speech_stream=object)
    def run(self, speech: str, options: SpeechOptions | None = None):
        return {
            "speech_stream": self._synthesizer.stream(
                speech, options if options is not None else self._options
            )
        }


def build_pipeline(config: Config | None = None):
    """Assemble the full Haystack pipeline.

    Returns a ``Pipeline`` whose ``run({"segmenter": {"text": ...}})`` yields the
    normalised speech string under ``tts.speech``.
    """
    from haystack import Pipeline as HaystackPipeline

    cfg = config or Config.from_env()
    pipe = HaystackPipeline()
    pipe.add_component("segmenter", Segmenter())
    pipe.add_component("lexical", LexicalTagger(protected_terms=cfg.protected_terms))
    pipe.add_component("resolver", ContextualResolver(cfg.build_classifier()))
    pipe.add_component("translit", DevanagariTransliterator())
    pipe.add_component("tts", TtsNormalization())

    pipe.connect("segmenter.utterance", "lexical.utterance")
    pipe.connect("lexical.utterance", "resolver.utterance")
    pipe.connect("resolver.utterance", "translit.utterance")
    pipe.connect("translit.text", "tts.text")
    return pipe


def build_voice_pipeline(
    config: Config | None = None,
    *,
    synthesizer: FishAudioSynthesizer | None = None,
    options: SpeechOptions | None = None,
):
    """Build the full Haystack path through lazy Fish speech synthesis.

    Network I/O starts only when the returned ``fish.speech_stream`` iterator is
    consumed. ``build_pipeline`` is text-only; classifier requests follow its explicit Config.
    """
    from haystack import Pipeline as HaystackPipeline

    cfg = config or Config.from_env()
    pipe = HaystackPipeline()
    pipe.add_component("segmenter", Segmenter())
    pipe.add_component("lexical", LexicalTagger(protected_terms=cfg.protected_terms))
    pipe.add_component("resolver", ContextualResolver(cfg.build_classifier()))
    pipe.add_component("translit", DevanagariTransliterator())
    pipe.add_component("tts", TtsNormalization())
    pipe.add_component(
        "fish",
        FishAudioTTS(
            synthesizer=synthesizer or cfg.build_synthesizer(), options=options
        ),
    )

    pipe.connect("segmenter.utterance", "lexical.utterance")
    pipe.connect("lexical.utterance", "resolver.utterance")
    pipe.connect("resolver.utterance", "translit.utterance")
    pipe.connect("translit.text", "tts.text")
    pipe.connect("tts.speech", "fish.speech")
    return pipe
