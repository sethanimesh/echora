"""Segmentation: turn a raw string into a total partition of Segments.

This is the only stage that *creates* the partition. Everything downstream
attaches decisions to it. Structural spans are honoured here so a URL becomes a
single indivisible segment rather than being shredded into word-ish pieces.
"""

from __future__ import annotations

import regex as re

from communication.backend.hinglish_core.core.model import Label, Segment, Source, Utterance
from communication.backend.hinglish_core.detect.spans import find_structural_spans

_STRUCTURAL_LABEL = {
    "URL": Label.CODE_URL,
    "EMAIL": Label.CODE_URL,
    "HANDLE": Label.CODE_URL,
    "HASHTAG": Label.CODE_URL,
    "FILEPATH": Label.CODE_URL,
    "CODE_SNAKE": Label.CODE_URL,
    "CODE_CAMEL": Label.CODE_URL,
    "CODE_DOTTED": Label.CODE_URL,
    "CONST": Label.CODE_URL,
    "CURRENCY": Label.NUMBER,
    "PERCENT": Label.NUMBER,
    "TIME": Label.NUMBER,
    "DATE": Label.NUMBER,
    "VERSION": Label.NUMBER,
    "UNIT": Label.NUMBER,
    "NUMBER": Label.NUMBER,
    "EMOJI": Label.OTHER,
}

# A "word" for our purposes: Latin letters or Devanagari, plus internal
# apostrophes and hyphens so "nahi'n" and "IIT-B" stay whole.
_WORD = re.compile(r"[\p{L}\p{M}][\p{L}\p{M}'’-]*")


def segment(raw: str) -> Utterance:
    """Partition ``raw`` into structural spans, words, and everything between.

    Every character lands in exactly one segment. Gaps between words -- spaces,
    punctuation, newlines -- become their own ``is_word=False`` segments rather
    than being dropped, which is what makes exact reconstruction possible.
    """
    structural = find_structural_spans(raw)
    segments: list[Segment] = []
    cursor = 0

    def emit_gap(upto: int) -> None:
        """Emit the non-word run between ``cursor`` and ``upto``, if any."""
        nonlocal cursor
        if upto > cursor:
            segments.append(
                Segment(
                    text=raw[cursor:upto],
                    start=cursor,
                    end=upto,
                    label=Label.OTHER,
                    source=Source.NONWORD,
                    is_word=False,
                )
            )
            cursor = upto

    for s_start, s_end, kind in structural:
        # Words living before this structural span.
        for m in _WORD.finditer(raw, cursor, s_start):
            emit_gap(m.start())
            segments.append(
                Segment(text=m.group(), start=m.start(), end=m.end())
            )
            cursor = m.end()
        emit_gap(s_start)

        segments.append(
            Segment(
                text=raw[s_start:s_end],
                start=s_start,
                end=s_end,
                label=_STRUCTURAL_LABEL.get(kind, Label.OTHER),
                source=Source.STRUCTURAL,
                is_word=False,  # frozen: never classified, never transliterated
            )
        )
        cursor = s_end

    # Trailing region after the last structural span.
    for m in _WORD.finditer(raw, cursor):
        emit_gap(m.start())
        segments.append(Segment(text=m.group(), start=m.start(), end=m.end()))
        cursor = m.end()
    emit_gap(len(raw))

    return Utterance.from_segments(raw, segments)
