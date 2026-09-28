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
from ..schemas import (
    AudienceKind,
    CommunicationContext,
    CommunicationRegister,
    Listener,
    PersonaKind,
)


# An entry with no settings listed applies everywhere, which is what every entry
# did before scoping existed. Listing settings narrows it: "Madras filter coffee"
# is the right words at home and the wrong ones said across a counter.
def _in_scope(settings: list[CommunicationContext], context: CommunicationContext) -> bool:
    return not settings or context in settings


class LexiconEntry(BaseModel):
    """A word this speaker's life contains, offered only when the beams produced it."""

    id: str
    word: str
    aliases: list[str] = Field(default_factory=list)
    display: str
    kind: PersonaKind
    note: str = ""
    settings: list[CommunicationContext] = Field(default_factory=list)

    def applies_in(self, context: CommunicationContext) -> bool:
        return _in_scope(self.settings, context)


class SpecializationRule(BaseModel):
    """The speaker's own version of a thing: `soap` -> `the Dove soap`."""

    id: str
    anchor: str
    plain: str
    surface: str
    kind: PersonaKind
    note: str = ""
    settings: list[CommunicationContext] = Field(default_factory=list)

    def applies_in(self, context: CommunicationContext) -> bool:
        return _in_scope(self.settings, context)


class AudienceProfile(BaseModel):
    """One explicitly selectable person, role, or generic audience."""

    id: str
    label: str
    kind: AudienceKind = "person"
    relationship: str = ""
    icon: str = ""
    listener: Listener
    style: CommunicationRegister = Field(default_factory=CommunicationRegister)
    # Tone can vary by the four contexts without sending a named place to the
    # server. Exact named places only decide whether this icon is offered in the
    # browser; their identifiers never leave it with the audio request.
    style_by_setting: dict[CommunicationContext, CommunicationRegister] = Field(
        default_factory=dict
    )
    visible_in_settings: list[CommunicationContext] = Field(default_factory=list)
    visible_in_places: list[str] = Field(default_factory=list)
    # A selected audience can suppress details it already knows. Empty means no
    # such suppression, preserving the pre-audience specialization behaviour.
    known_detail_ids: list[str] = Field(default_factory=list)

    def register_for(self, context: CommunicationContext) -> CommunicationRegister:
        return self.style_by_setting.get(context, self.style)


class PersonaProfile(BaseModel):
    id: str
    label: str
    blurb: str = ""
    icon: str = "circle"
    baseline: bool = True
    context_default: CommunicationContext = "general"
    # What this speaker's settings usually mean for who is listening, where they
    # differ from the default. A speaker who never goes out alone declares
    # {"outdoors": "familiar"}; an absent setting takes the default.
    listener_by_setting: dict[CommunicationContext, Listener] = Field(default_factory=dict)
    style: CommunicationRegister = Field(default_factory=CommunicationRegister)
    style_by_setting: dict[CommunicationContext, CommunicationRegister] = Field(
        default_factory=dict
    )
    speaker_note: str = ""
    lexicon: list[LexiconEntry] = Field(default_factory=list)
    specializations: list[SpecializationRule] = Field(default_factory=list)
    audiences: list[AudienceProfile] = Field(default_factory=list)
    created_at: str = ""

    def audience(self, audience_id: str) -> AudienceProfile | None:
        return next((item for item in self.audiences if item.id == audience_id), None)

    def register_for(self, context: CommunicationContext) -> CommunicationRegister:
        return self.style_by_setting.get(context, self.style)

    def heard_forms(self, context: CommunicationContext | None = None) -> dict[str, LexiconEntry]:
        """Every spelling the recognizer might produce, mapped to its entry.

        Scoped to one setting when given a setting; otherwise the whole lexicon,
        which is what the audit vocabulary wants.
        """
        forms: dict[str, LexiconEntry] = {}
        for entry in self.lexicon:
            if context is not None and not entry.applies_in(context):
                continue
            for form in [entry.word, *entry.aliases]:
                tokens = _tokens(form)
                if len(tokens) == 1:
                    forms.setdefault(tokens[0], entry)
        return forms

    def rule_for(
        self, anchor: str, context: CommunicationContext | None = None
    ) -> SpecializationRule | None:
        tokens = _tokens(anchor)
        if len(tokens) != 1:
            return None
        for rule in self.specializations:
            if _tokens(rule.anchor) != tokens:
                continue
            if context is not None and not rule.applies_in(context):
                return None
            return rule
        return None


class UserProfileInput(BaseModel):
    """The complete speaker-reviewed document accepted by the writable endpoint."""

    label: str = "You"
    blurb: str = "Your own profile."
    context_default: CommunicationContext = "general"
    listener_by_setting: dict[CommunicationContext, Listener] = Field(default_factory=dict)
    style: CommunicationRegister = Field(default_factory=CommunicationRegister)
    style_by_setting: dict[CommunicationContext, CommunicationRegister] = Field(
        default_factory=dict
    )
    lexicon: list[LexiconEntry] = Field(default_factory=list)
    specializations: list[SpecializationRule] = Field(default_factory=list)
    audiences: list[AudienceProfile] = Field(default_factory=list)


class UserProfileSaveResponse(BaseModel):
    profile: PersonaProfile
    refused: list[str] = Field(default_factory=list)


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
    audience_ids: set[str] = set()
    audiences: list[AudienceProfile] = []
    for audience in profile.audiences:
        if not audience.id.strip() or audience.id in audience_ids:
            continue
        audience_ids.add(audience.id)
        audiences.append(audience)
    return profile.model_copy(update={"specializations": unique, "audiences": audiences})
