"""Roman-Hindi -> Devanagari transliteration, applied ONLY to HI spans.

WHY LEXICON LOOKUP RATHER THAN A NEURAL MODEL
---------------------------------------------
The obvious choice would be AI4Bharat's IndicXlit. Investigation ruled it out as
a *runtime* dependency: it requires Python 3.10 plus monkeypatches to import at
all (fairseq's mutable-default dataclass breaks on 3.11+), pulls ~1 GB, and
takes ~35 ms/word against ~30 ns for a dict lookup. Shipping that inside a
device someone depends on to speak is a fragility and maintenance burden that
buys nothing here -- on the sample tested, the lexicon was also *more* accurate,
because Dakshina encodes what humans actually wrote rather than what a model
generalises.

The right use of a neural model is at BUILD time, as an asset generator: run it
offline over a frequency list, keep candidates that agree with a second oracle,
and bake the results into the shipped JSON. That converts a fragile runtime
dependency into a static file. See ``tools/build_lexicon.py``.

The port exists because that build-time generator, plus the identity
transliterator used to isolate pipeline tests, are both real second
implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from communication.backend.hinglish_core.core.model import Label, Segment, Source, Utterance


@runtime_checkable
class Transliterator(Protocol):
    """Converts a Roman-Hindi token to Devanagari."""

    def transliterate(self, token: str, candidates: tuple[str, ...] = ()) -> str | None:
        """Return the Devanagari form, or ``None`` to abstain.

        Abstaining is a first-class outcome: returning ``None`` leaves the token
        in Roman script, which is the recoverable failure. Guessing badly is not.
        """
        ...


class IdentityTransliterator:
    """Returns nothing, ever. Used to test the pipeline without conversion."""

    def transliterate(self, token: str, candidates: tuple[str, ...] = ()) -> str | None:
        return None


class LexiconTransliterator:
    """Dictionary lookup with a normalisation retry, then abstention.

    Three tiers, in order of decreasing confidence:

    1. Candidates already attached by the lexical stage (no second lookup).
    2. Direct lexicon lookup.
    3. A normalised-key retry that folds the common Hinglish spelling variants
       (aa->a, ee->i, oo->u, doubled consonants, trailing h). This is what
       catches "bohot"/"bahut" and "accha"/"acha" without a neural model.

    If all three miss, it abstains rather than inventing a spelling.
    """

    def __init__(self, lexicon: dict[str, list[str]] | None = None) -> None:
        if lexicon is None:
            from communication.backend.hinglish_core.classify.lexical import load_lexicon
            lexicon = load_lexicon()
        self._lexicon = lexicon
        self._normalised = self._build_normalised_index(lexicon)

    @staticmethod
    def _normalise(token: str) -> str:
        """Fold spelling variation that does not change the Hindi word."""
        t = token.lower()
        for a, b in (("aa", "a"), ("ee", "i"), ("ii", "i"),
                     ("oo", "u"), ("uu", "u")):
            t = t.replace(a, b)
        # Collapse doubled consonants: "accha" -> "acha".
        out: list[str] = []
        for ch in t:
            if out and ch == out[-1] and ch not in "aeiou":
                continue
            out.append(ch)
        t = "".join(out)
        if t.endswith("h") and len(t) > 2:
            t = t[:-1]
        return t

    def _build_normalised_index(
        self, lexicon: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for roman, devs in lexicon.items():
            index.setdefault(self._normalise(roman), devs)
        return index

    def transliterate(self, token: str, candidates: tuple[str, ...] = ()) -> str | None:
        if candidates:
            return candidates[0]
        direct = self._lexicon.get(token.lower())
        if direct:
            return direct[0]
        fuzzy = self._normalised.get(self._normalise(token))
        if fuzzy:
            return fuzzy[0]
        return None


def apply_transliteration(
    utterance: Utterance, transliterator: Transliterator
) -> Utterance:
    """Set ``output`` on every HI segment. Everything else is untouched.

    Casing: Devanagari has no case, so a capitalised Roman source simply maps to
    the Devanagari form. No case restoration is attempted or needed.
    """
    out: list[Segment] = []
    for seg in utterance.segments:
        if seg.label is not Label.HI or not seg.is_word:
            out.append(seg)
            continue
        # Already Devanagari (user typed it directly): leave it alone.
        if any(0x0900 <= ord(c) <= 0x097F for c in seg.text):
            out.append(seg)
            continue
        candidates = seg.candidates
        if (
            seg.candidate_index is not None
            and 0 <= seg.candidate_index < len(candidates)
        ):
            selected = candidates[seg.candidate_index]
            candidates = (selected, *(
                candidate for index, candidate in enumerate(candidates)
                if index != seg.candidate_index
            ))
        devanagari = transliterator.transliterate(seg.text, candidates)
        out.append(seg if devanagari is None else
                   Segment(**{**{f: getattr(seg, f) for f in seg.__slots__},
                              "output": devanagari}))
    return utterance.with_segments(out)
