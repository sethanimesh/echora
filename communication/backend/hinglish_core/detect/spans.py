"""Deterministic structural span detection -- runs *before* language ID.

Anything matched here is frozen verbatim: never classified, never transliterated,
never sent to the LLM. Getting a URL or a price mangled into Devanagari is a
severe failure, and these shapes are decidable without any language knowledge,
so they are settled first and cheaply.

Uses ``regex`` rather than ``re`` because emoji detection needs
``\\p{Extended_Pictographic}`` and ``\\p{Regional_Indicator}``. It arrives as a
``wordfreq`` dependency, so it costs nothing extra.
"""

from __future__ import annotations

import regex as re

DEV = r"ऀ-ॿ"  # Devanagari block

# Ordered by precedence: earlier patterns win ties against later ones when two
# matches are the same length. Self-delimiting structures (URL, email) come
# first because their internals routinely look like other patterns -- a URL
# containing "v1.2" must stay one URL, not fragment into a version span.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("URL", re.compile(
        r'(?:https?|ftp)://[^\s<>"\']+'
        r'|\bwww\.[^\s<>"\']+'
        r'|\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+'
        r'(?:com|org|net|edu|gov|io|ai|co|in|dev|app|me|xyz)\b(?:/[^\s<>"\']*)?')),

    ("EMAIL", re.compile(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?'
        r'(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b')),

    ("HANDLE", re.compile(r'(?<![\w@])@[A-Za-z0-9_]{2,30}\b')),

    ("HASHTAG", re.compile(r'(?<![\w#])#[\w' + DEV + r']{1,60}\b')),

    ("FILEPATH", re.compile(
        r'(?:~|\.{1,2})?/(?:[\w.\-@]+/)*[\w.\-@]+(?:\.[A-Za-z0-9]{1,8})?'
        r'|\b[A-Za-z]:\\(?:[\w.\- ]+\\)*[\w.\-]+'
        r'|\b[\w\-]+\.(?:py|js|ts|json|yaml|yml|toml|md|txt|csv|pdf|png|jpg'
        r'|sh|cfg|ini|log|wav|mp3)\b')),

    # Money and measures before bare NUMBER, so "500 rupees" is one span.
    ("CURRENCY", re.compile(
        r'(?:[₹$€£¥]\s?\d[\d,]*(?:\.\d{1,2})?)'
        r'|(?:\b(?:Rs|INR|USD|EUR|GBP)\.?\s?\d[\d,]*(?:\.\d{1,2})?)'
        r'|(?:\b\d[\d,]*(?:\.\d{1,2})?\s?'
        r'(?:rs|rupees?|INR|USD|k|cr|crore|lakhs?|lacs?)\b)',
        re.IGNORECASE)),

    ("PERCENT", re.compile(r'\b\d+(?:\.\d+)?\s?(?:%|percent|pc)\b|\d+(?:\.\d+)?%')),

    ("TIME", re.compile(
        r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:[ap]\.?m\.?)?'
        r'|\b\d{1,2}\s?[ap]\.?m\.?(?![a-z])'
        r'|\b\d{1,2}(?::\d{2})?\s?बजे',  # "बजे"
        re.IGNORECASE)),

    ("DATE", re.compile(
        r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b'
        r'|\b\d{4}-\d{2}-\d{2}\b'
        r'|\b\d{1,2}(?:st|nd|rd|th)\s+'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b'
        r'|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
        r'\s+\d{1,2}(?:st|nd|rd|th)?\b',
        re.IGNORECASE)),

    ("VERSION", re.compile(r'\bv?\d+\.\d+(?:\.\d+)*(?:-[\w.]+)?\b')),

    ("UNIT", re.compile(
        r'\b\d+(?:\.\d+)?\s?'
        r'(?:kg|g|mg|km|m|cm|mm|kb|mb|gb|tb|hz|khz|mhz|ghz|ml|l|hrs?|hours?'
        r'|mins?|minutes?|secs?|seconds?|days?|weeks?|months?|years?)\b',
        re.IGNORECASE)),

    ("NUMBER", re.compile(r'\b\d[\d,]*(?:\.\d+)?(?:st|nd|rd|th)?\b')),

    # Identifier shapes. These are decidable from form alone and are common in
    # the technical speech this device's user actually produces.
    ("CONST", re.compile(r'\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b')),
    ("CODE_SNAKE", re.compile(r'\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b')),
    ("CODE_CAMEL", re.compile(r'\b[a-z]+(?:[A-Z][a-z0-9]+)+\b')),
    ("CODE_DOTTED", re.compile(r'\b[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)+\b')),

    # Emoji: hand-rolled rather than \p{Emoji}, which also matches ASCII digits
    # and '#'. Handles ZWJ sequences, flags, skin-tone modifiers and keycaps.
    ("EMOJI", re.compile(
        r'(?:\p{Extended_Pictographic}(?:️)?'
        r'(?:\p{Emoji_Modifier})?'
        r'(?:‍\p{Extended_Pictographic}(?:️)?)*'
        r'|\p{Regional_Indicator}{2}'
        r'|[0-9#*]️?⃣)+')),
]

# ACRONYM is deliberately NOT in PATTERNS. It is not structurally decidable:
# "KYA HUA BHAI" has exactly the same shape as "API AWS LLM". Treating all-caps
# as structural would freeze shouty Hinglish into Roman script forever. It is
# emitted as a soft hint instead, which the Hindi lexicon can veto downstream.
ACRONYM = re.compile(r"\b[A-Z]{2,6}(?:-[A-Z0-9]{1,4})?\b")


def find_structural_spans(text: str) -> list[tuple[int, int, str]]:
    """Return non-overlapping ``(start, end, kind)`` spans, left to right.

    Collects every candidate match from every pattern, then resolves overlaps by
    **longest match wins**, with pattern precedence as the tie-break. Longest-wins
    is what keeps ``https://x.com/a/v1.2`` a single URL instead of letting the
    VERSION pattern carve a hole in the middle of it.
    """
    candidates: list[tuple[int, int, int, str]] = []
    for idx, (kind, pat) in enumerate(PATTERNS):
        for m in pat.finditer(text):
            if m.end() > m.start():
                candidates.append((m.start(), m.end(), idx, kind))

    # Longest first; then by pattern precedence; then leftmost, for determinism.
    candidates.sort(key=lambda c: (-(c[1] - c[0]), c[2], c[0]))

    kept: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, _idx, kind in candidates:
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        kept.append((start, end, kind))
        occupied.append((start, end))

    kept.sort(key=lambda s: s[0])
    return kept


def acronym_hint(token: str) -> bool:
    """True if the token *looks* like an acronym. A hint, never a decision."""
    return bool(ACRONYM.fullmatch(token))
