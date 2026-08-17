"""What Echora is allowed to know about one speaker, and what that licenses.

A profile holds two very different kinds of knowledge, and keeping them apart is
the whole safety argument. The lexicon names words this speaker's life contains,
which only ever helps choose between words the recognizer already produced. A
specialization rule is stronger: it puts wording into the message that no beam
carried, so every rule is checked here at load time and again at use time.

Rules are stored as three parts rather than two. Substituting the bare anchor
would turn "Please pass me the Dove soap." back into "Please pass me soap.", so
the plain wording is declared alongside the specialized one and the anchor is
only the word that ties both to the audio.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..messaging.alignment import _tokens
from ..schemas import CommunicationContext, PersonaKind


class LexiconEntry(BaseModel):
    """A word this speaker's life contains, offered only when the beams produced it."""

    id: str
    word: str
    aliases: list[str] = Field(default_factory=list)
    display: str
    kind: PersonaKind
    note: str = ""


class SpecializationRule(BaseModel):
    """The speaker's own version of a thing: `soap` -> `the Dove soap`."""

    id: str
    anchor: str
    plain: str
    surface: str
    kind: PersonaKind
    note: str = ""


class PersonaProfile(BaseModel):
    id: str
    label: str
    blurb: str = ""
    icon: str = "circle"
    baseline: bool = True
    context_default: CommunicationContext = "general"
    speaker_note: str = ""
    lexicon: list[LexiconEntry] = Field(default_factory=list)
    specializations: list[SpecializationRule] = Field(default_factory=list)
    created_at: str = ""

    def heard_forms(self) -> dict[str, LexiconEntry]:
        """Every spelling the recognizer might produce, mapped to its entry."""
        forms: dict[str, LexiconEntry] = {}
        for entry in self.lexicon:
            for form in [entry.word, *entry.aliases]:
                tokens = _tokens(form)
                if len(tokens) == 1:
                    forms.setdefault(tokens[0], entry)
        return forms

    def rule_for(self, anchor: str) -> SpecializationRule | None:
        tokens = _tokens(anchor)
        if len(tokens) != 1:
            return None
        for rule in self.specializations:
            if _tokens(rule.anchor) == tokens:
                return rule
        return None


def _is_subsequence(inner: list[str], outer: list[str]) -> bool:
    iterator = iter(outer)
    return all(word in iterator for word in inner)


def valid_rule(rule: SpecializationRule) -> bool:
    """A rule may only add words around a word the speaker actually said.

    The anchor must be one token, it must appear in both wordings, and the plain
    wording must be a subsequence of the specialized one. That last condition is
    what makes reverting a pure deletion: anything the rule adds can be taken
    back out without leaving the sentence rearranged.
    """
    anchor = _tokens(rule.anchor)
    plain = _tokens(rule.plain)
    surface = _tokens(rule.surface)
    if len(anchor) != 1 or not plain or not surface:
        return False
    if anchor[0] not in plain or anchor[0] not in surface:
        return False
    if len(surface) <= len(plain):
        return False
    return _is_subsequence(plain, surface)


def parse_profile(raw: dict) -> PersonaProfile | None:
    """Read one profile, dropping what it cannot verify instead of refusing to load.

    A single malformed rule in a hand-edited file must not take the speaker's
    whole profile -- and with it their disambiguation prior -- out of service.
    """
    try:
        profile = PersonaProfile.model_validate(raw)
    except Exception:
        return None
    kept = [rule for rule in profile.specializations if valid_rule(rule)]
    seen: set[str] = set()
    unique: list[SpecializationRule] = []
    for rule in kept:
        anchor = _tokens(rule.anchor)[0]
        if anchor not in seen:
            seen.add(anchor)
            unique.append(rule)
    return profile.model_copy(update={"specializations": unique})
