"""Bounded checks for concrete wording changes; not a semantic proof.

These guards supplement preserved evidence and alternatives. They deliberately
do not claim to validate arbitrary translations, pronouns, or every possible
fact a generative model can invent.
"""
from __future__ import annotations

import re


FAMILIES = (
    ("arm", "arms"), ("leg", "legs"), ("hand", "hands"),
    ("foot", "feet"), ("knee", "knees"), ("head",), ("chest",),
    ("left",), ("right",), ("pain", "painful", "hurt", "hurts", "hurting", "ache", "aching"),
    ("water",), ("tea", "chai"), ("coffee",), ("juice",), ("soup",),
    ("sugar", "cheeni"), ("milk", "doodh"), ("salt", "namak"),
    ("aspirin",), ("ibuprofen",), ("paracetamol", "acetaminophen"),
    ("insulin",), ("medicine", "medication", "medicines", "medications"),
    ("dose", "dosage"), ("tablet", "tablets"), ("pill", "pills"),
    ("hot",), ("cold",), ("black",), ("green",),
    ("window", "windows"), ("door", "doors"),
    ("open", "opened", "opening"), ("close", "closed", "closing", "shut"),
)
NUMBERS = {word: str(i) for i, word in enumerate(
    ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))}


def words(text: str) -> set[str]:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"\b(?:don't|doesn't|didn't|can't|cannot|won't|isn't|wasn't|shouldn't)\b", "not", text)
    return set(re.findall(r"[a-z0-9]+", text))


def concepts(tokens: set[str]) -> set[str]:
    return {family[0] for family in FAMILIES if tokens.intersection(family)}


def quantities(tokens: set[str]) -> set[str]:
    return {NUMBERS.get(token, token) for token in tokens if token.isdigit() or token in NUMBERS}


def preserves_bounded_facts(source: str, message: str, *, reference: str = "", approved_details: list[str] | None = None) -> bool:
    source_words, message_words = words(source), words(message)
    licensed = words(" ".join([source, reference, *(approved_details or [])]))
    # Approved personal detail can add a brand/temperature, while current words
    # still have to survive. An old reference is only an optional source of a
    # subject; it never excuses deleting negation in the current utterance.
    source_concepts, message_concepts = concepts(source_words), concepts(message_words)
    if not source_concepts <= message_concepts or not message_concepts <= concepts(licensed):
        return False
    negation = {"not", "no", "never", "without", "neither"}
    if bool(source_words & negation) != bool(message_words & negation):
        # A reference can carry existing negation only when the current words do
        # not explicitly replace it with a positive with/quantity instruction.
        if not reference or source_words & negation or source_words & {"with", "make", "change"}:
            return False
        if not words(reference) & negation:
            return False
    if not quantities(source_words) <= quantities(message_words):
        return False
    if not quantities(message_words) <= quantities(licensed):
        return False
    return True
