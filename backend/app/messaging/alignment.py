"""Line the competing beams up position by position, and audit words against them.

The recognizer returns several transcriptions of one utterance. Where they
disagree, that is almost always one sound heard several ways rather than
different words the speaker said. Aligning them separates the positions every
beam agreed on from the positions still in contention, which is what lets the
rest of the pipeline reason about evidence instead of about sentences.

This lives apart from the Groq chain because the personal layer needs the same
alignment -- to decide which of a speaker's known words the recognizer actually
produced and which anchored details are safe to apply -- and reaching across
packages for a private name is how modules quietly grow into each other.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..schemas import Hypothesis


EMPTY_SLOT = "—"


def _tokens(text: str) -> list[str]:
    normalized = text.lower().replace("won't", "not").replace("can't", "cannot")
    return re.findall(r"[a-z0-9]+", normalized)


def _slot_alignment(hypotheses: list[Hypothesis]) -> list[dict[str, object]]:
    """Line the beams up against the leading beam, position by position."""
    reference = _tokens(hypotheses[0].literal_text)
    if not reference:
        return []
    total_weight = sum(item.search_weight for item in hypotheses) or 1.0
    columns: list[dict[str, float]] = [{} for _ in reference]
    for item in hypotheses:
        tokens = _tokens(item.literal_text)
        aligned: list[str] = [""] * len(reference)
        for tag, i1, i2, j1, j2 in SequenceMatcher(a=reference, b=tokens, autojunk=False).get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    aligned[i1 + offset] = tokens[j1 + offset]
            elif tag == "replace":
                span = tokens[j1:j2]
                for position in range(i1, i2):
                    offset = position - i1
                    aligned[position] = span[offset] if offset < len(span) else ""
        for position, token in enumerate(aligned):
            word = token or EMPTY_SLOT
            columns[position][word] = columns[position].get(word, 0.0) + item.search_weight / total_weight
    return [
        {
            "position": position + 1,
            "stable": len(column) == 1,
            "options": [
                {"word": word, "share": round(share, 3)}
                for word, share in sorted(column.items(), key=lambda pair: -pair[1])
            ],
        }
        for position, column in enumerate(columns)
    ]


def settled_words(slots: list[dict[str, object]], threshold: float) -> set[str]:
    """Words the beams overwhelmingly agreed on, by share of search weight.

    Requiring a slot to be perfectly stable sounds safer and is not: beam search
    reliably splits one word across spellings, so a real utterance of "coffee"
    arrives as coffee 0.88 / coffey 0.06 / coffeey 0.06 and almost no position is
    ever stable. Share is the honest measure, and it is the one the rest of the
    pipeline already uses for key terms -- a genuine ambiguity between two real
    words splits the weight near evenly and stays unsettled, while a spelling
    wobble does not.
    """
    words: set[str] = set()
    for slot in slots:
        for option in slot["options"]:  # type: ignore[union-attr]
            word = str(option["word"])
            if word != EMPTY_SLOT and float(option["share"]) >= threshold:
                words.add(word)
    return words


def _alignment_template(slots: list[dict[str, object]]) -> str:
    parts = []
    for slot in slots:
        options = slot["options"]
        if slot["stable"]:
            parts.append(str(options[0]["word"]))
        else:
            parts.append("[" + "|".join(str(option["word"]) for option in options) + "]")
    return " ".join(parts)


def _slot_options(slots: list[dict[str, object]]) -> list[list[str]]:
    return [
        [str(option["word"]) for option in slot["options"] if option["word"] != EMPTY_SLOT]
        for slot in slots
    ]


def _word_alternatives(slots: list[dict[str, object]], reading: list[str]) -> dict[str, list[str]]:
    """Other words heard at each position the chosen reading used."""
    alternatives: dict[str, list[str]] = {}
    for slot, chosen in zip(slots, reading, strict=False):
        others = [
            str(option["word"])
            for option in slot["options"]
            if option["word"] not in (chosen, EMPTY_SLOT)
        ]
        if others:
            alternatives[chosen] = others
    return alternatives


def _grounded_reading(reading: str, slots: list[dict[str, object]]) -> list[str] | None:
    """The one property enforced in code: every word was actually heard.

    Checked positionally when the lengths agree, otherwise against the union of
    slot options. Returns None when nothing survives.
    """
    words = _tokens(reading)
    options = _slot_options(slots)
    if not words or not options:
        return None
    if len(words) == len(options):
        if all(word in allowed for word, allowed in zip(words, options, strict=True)):
            return words
        return None
    # Never silently remove an unsupported word while keeping the message the
    # model wrote for it. Only ordered omissions of genuinely uncertain slots
    # are allowed; a stable word cannot disappear from a generated reading.
    positions = {0}
    for slot, allowed in zip(slots, options, strict=True):
        next_positions = set()
        for position in positions:
            if position < len(words) and words[position] in allowed:
                next_positions.add(position + 1)
            if not slot["stable"]:
                next_positions.add(position)
        positions = next_positions
    return words if len(words) in positions else None
