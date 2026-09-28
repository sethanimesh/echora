"""Core data model.

The load-bearing idea: an utterance is an **immutable total partition** of the
original string. Every character of the original belongs to exactly one segment,
segments are contiguous, ordered, and never move. Stages never mutate text and
never rebuild the string by joining tokens -- they attach *decisions* to
segments, and reconstruction is a single fold over the partition.

That makes two classic failure modes structurally impossible rather than merely
tested against:

* **Lost characters.** Whitespace, punctuation and emoji live *in* the partition
  as their own segments, so there is no path where "join the tokens" silently
  drops them.
* **Offset drift.** A Devanagari replacement is usually a different length than
  its Roman source. Because offsets always refer to the *original* string and
  are never rewritten, a length-changing replacement cannot shift the offsets of
  later segments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterator


class Label(str, Enum):
    """What a segment turned out to be.

    The closed set is shared by the lexical fast path and the LLM stage, so a
    label never has to be translated between stages.
    """

    HI = "HI"
    """Hindi written in Roman script. The only label that gets transliterated."""

    EN = "EN"
    """English. Preserved exactly."""

    NAME = "NAME"
    """Proper noun. Preserved exactly -- names must never be transliterated."""

    NUMBER = "NUMBER"
    """Numeric content. Preserved here; the TTS stage may expand it later."""

    CODE_URL = "CODE_URL"
    """URL, email, path, identifier. Preserved exactly."""

    OTHER = "OTHER"
    """Punctuation, whitespace, emoji, or genuinely undecidable. Preserved."""

    AMBIGUOUS = "AMBIGUOUS"
    """Not yet decided. Never appears in a finished utterance -- the pipeline
    resolves every AMBIGUOUS segment via the LLM stage or the fallback policy
    before reconstruction."""


class Source(str, Enum):
    """Which stage decided a segment's label.

    Kept because it is the difference between a debuggable pipeline and an
    opaque one: when output is wrong, the first question is always *who decided
    this*. It also drives metrics (what fraction of spans reached the LLM) and
    lets the decision cache record provenance.
    """

    STRUCTURAL = "STRUCTURAL"
    LEXICON = "LEXICON"
    ALLOWLIST = "ALLOWLIST"
    LLM = "LLM"
    FALLBACK = "FALLBACK"
    CACHE = "CACHE"
    NONWORD = "NONWORD"


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous slice of the original utterance, plus what we decided about it.

    ``text`` is always exactly ``original[start:end]``. This is an invariant the
    whole design leans on, and :meth:`Utterance.validate` asserts it.

    ``output`` is the replacement text, or ``None`` meaning "emit ``text``
    unchanged". Only the transliteration and TTS-normalisation stages ever set
    it. Keeping the replacement *beside* the original rather than overwriting it
    means every stage can still see what the user actually typed, and a bad
    transliteration is inspectable rather than lost.
    """

    text: str
    start: int
    end: int
    label: Label = Label.AMBIGUOUS
    source: Source | None = None
    output: str | None = None
    is_word: bool = True
    """False for whitespace/punctuation/emoji runs, which are never classified."""
    candidates: tuple[str, ...] = ()
    """Ranked Devanagari candidates from the lexicon, best first. Retained so a
    downstream stage (or a human correcting the device) can pick a different
    reading without re-running lookup."""
    candidate_index: int | None = None
    """Context-selected index into ``candidates``; never model-authored text."""
    confidence: str | None = None
    """Set by the LLM stage: 'certain' or 'unsure'. Drives cloud escalation."""

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"inverted span: {self.start}..{self.end}")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"text length {len(self.text)} != span width {self.end - self.start}"
            )

    @property
    def rendered(self) -> str:
        """What this segment contributes to the final string."""
        return self.text if self.output is None else self.output

    def decided(self, label: Label, source: Source, **kw) -> Segment:
        """Return a copy with a decision attached. Never mutates."""
        return replace(self, label=label, source=source, **kw)


@dataclass(frozen=True, slots=True)
class Utterance:
    """An original string and its total partition into segments.

    Construct with :meth:`from_segments` so the partition is checked. The
    ``raw`` string is never modified for the lifetime of the object -- it is the
    ground truth that reconstruction is measured against.
    """

    raw: str
    segments: tuple[Segment, ...]

    @classmethod
    def from_segments(cls, raw: str, segments: list[Segment]) -> Utterance:
        u = cls(raw=raw, segments=tuple(segments))
        u.validate()
        return u

    def validate(self) -> None:
        """Assert the total-partition invariant.

        Cheap (O(n) over segments) and called on construction, because a
        partition that has silently stopped covering the string produces
        corrupted speech rather than an exception, and corrupted speech from an
        assistive device is the failure this whole design exists to prevent.
        """
        cursor = 0
        for seg in self.segments:
            if seg.start != cursor:
                raise ValueError(
                    f"partition gap/overlap at {cursor}: segment starts at {seg.start}"
                )
            if seg.text != self.raw[seg.start : seg.end]:
                raise ValueError(
                    f"segment text {seg.text!r} != raw slice "
                    f"{self.raw[seg.start:seg.end]!r} at {seg.start}..{seg.end}"
                )
            cursor = seg.end
        if cursor != len(self.raw):
            raise ValueError(
                f"partition covers {cursor} chars but raw is {len(self.raw)}"
            )

    def words(self) -> Iterator[Segment]:
        """Segments eligible for classification (skips whitespace/punctuation)."""
        return (s for s in self.segments if s.is_word)

    def with_segments(self, segments: list[Segment]) -> Utterance:
        """Return a new utterance with a replacement partition, re-validated."""
        return Utterance.from_segments(self.raw, segments)

    def render(self) -> str:
        """Fold the partition into the output string.

        This is the *only* place output text is produced. There is deliberately
        no other path -- no token joining, no whitespace re-insertion, no
        regeneration -- so preservation of everything we did not deliberately
        change is a property of the data structure, not of a stage remembering
        to be careful.
        """
        return "".join(seg.rendered for seg in self.segments)

    def unchanged(self) -> bool:
        """True if no stage has replaced any segment."""
        return all(s.output is None for s in self.segments)
