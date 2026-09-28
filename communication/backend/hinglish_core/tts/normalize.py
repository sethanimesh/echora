"""TTS normalisation -- runs AFTER reconstruction, kept separate on purpose.

This stage is about *pronunciation*, not language. It expands acronyms, numbers,
currency and units into speakable words. It is deliberately not fused with
transliteration because the two answer different questions ("what language is
this?" vs "how should a voice say this?"), change for different reasons, and are
useful independently -- a future English-only deployment would want this stage
and not the other.

It operates on the reconstructed string via the same structural patterns, so it
never needs the segment model. That keeps it swappable per TTS vendor: a voice
that already speaks "API" correctly can skip acronym expansion entirely.
"""

from __future__ import annotations

import regex as re
from typing import Protocol, runtime_checkable

# Acronyms a Hindi/English TTS voice typically mangles. Spelled out letter by
# letter with separators the voice will pause on.
ACRONYM_SPEECH = {
    "API": "A P I", "AWS": "A W S", "LLM": "L L M", "URL": "U R L",
    "CPU": "C P U", "GPU": "G P U", "RAM": "R A M", "SQL": "S Q L",
    "HTTP": "H T T P", "JSON": "J S O N", "CSV": "C S V", "PDF": "P D F",
    "USB": "U S B", "WiFi": "wifi", "OTP": "O T P", "ATM": "A T M",
    "IIT": "I I T", "UPI": "U P I", "SMS": "S M S", "ID": "I D",
}

_UNIT_SPEECH = {
    "kg": "kilogram", "km": "kilometre", "cm": "centimetre", "mm": "millimetre",
    "gb": "gigabyte", "mb": "megabyte", "kb": "kilobyte", "tb": "terabyte",
    "hr": "hour", "hrs": "hours", "min": "minute", "mins": "minutes",
    "sec": "second", "secs": "seconds", "ml": "millilitre",
}

_CURRENCY_WORD = {"₹": "rupees", "$": "dollars", "€": "euros", "£": "pounds"}


@runtime_checkable
class TtsNormalizer(Protocol):
    """Rewrites text for pronunciation. Swappable per TTS vendor."""

    def normalize(self, text: str) -> str: ...


class NullTtsNormalizer:
    """Passes text through. For voices that handle this themselves."""

    def normalize(self, text: str) -> str:
        return text


class HinglishTtsNormalizer:
    """Expands acronyms, currency, and units for an Indic-capable TTS voice.

    Numbers are left as digits: every mainstream TTS engine already reads digits
    correctly and in the target language, whereas expanding them here would
    force us to pick a language for the number words and would frequently pick
    wrong in code-mixed speech.
    """

    def __init__(self, acronyms: dict[str, str] | None = None) -> None:
        self._acronyms = acronyms if acronyms is not None else dict(ACRONYM_SPEECH)

    def normalize(self, text: str) -> str:
        out = text

        # Currency: "₹500" -> "500 rupees" (symbol follows the amount in speech).
        def _currency(m: re.Match) -> str:
            return f"{m.group('amt')} {_CURRENCY_WORD.get(m.group('sym'), '')}".strip()

        out = re.sub(
            r"(?P<sym>[₹$€£])\s?(?P<amt>\d[\d,]*(?:\.\d{1,2})?)", _currency, out
        )

        # Units: "5kg" -> "5 kilogram".
        def _unit(m: re.Match) -> str:
            word = _UNIT_SPEECH.get(m.group("u").lower())
            return f"{m.group('n')} {word}" if word else m.group(0)

        out = re.sub(
            r"\b(?P<n>\d+(?:\.\d+)?)\s?(?P<u>[A-Za-z]{1,4})\b", _unit, out
        )

        # Acronyms: whole-token match only, so "API" is expanded but "Apis" is not.
        def _acronym(m: re.Match) -> str:
            return self._acronyms.get(m.group(0), m.group(0))

        if self._acronyms:
            pattern = r"\b(?:" + "|".join(
                re.escape(a) for a in sorted(self._acronyms, key=len, reverse=True)
            ) + r")\b"
            out = re.sub(pattern, _acronym, out)

        return out
