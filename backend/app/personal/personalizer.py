"""The one entry point the request path touches, and the one that never raises.

This follows the policy `messaging/speech.py` set: personal context is an
enhancement and must never be a dependency. Whatever goes wrong in here -- an
unreadable profile or malformed rule -- the answer is None,
and the message chain runs exactly as it did before this layer existed. A
speaker waiting to be heard is not helped by an error.
"""

from __future__ import annotations

import asyncio

from ..config import Settings
from ..messaging.alignment import _slot_alignment, _tokens
from ..schemas import (
    CommunicationContext,
    CommunicationRegister,
    Hypothesis,
    Listener,
    PersonaSummary,
    PersonalBrief,
    ResolvedAudience,
    resolve_listener,
)
from .lexicon import lexicon_hints, specialization_offers
from .profile import (
    LexiconEntry,
    PersonaProfile,
    SpecializationRule,
    UserProfileInput,
    parse_profile,
    valid_rule,
)
from .store import USER_PROFILE_ID, BaselineStore, LiveStore, summarize


class Personalizer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.baseline = BaselineStore(settings.persona_root)
        self.live = LiveStore(settings.personal_root)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ profiles

    def known_ids(self) -> set[str]:
        return set(self.baseline.profiles()) | {USER_PROFILE_ID}

    def personas(self) -> list[PersonaSummary]:
        summaries: list[PersonaSummary] = []
        for profile_id, profile in sorted(self.baseline.profiles().items()):
            live = self.live.profile(profile_id)
            summaries.append(summarize(live or profile, baseline=live is None))
        user = self.live.profile(USER_PROFILE_ID)
        summaries.append(summarize(user or self._empty_user(), baseline=user is None))
        return summaries

    def _empty_user(self) -> PersonaProfile:
        return PersonaProfile(
            id=USER_PROFILE_ID,
            label="You",
            blurb="Your own profile. It starts empty and fills as you use Echora.",
            icon="you",
            baseline=False,
        )

    def _resolve(self, profile_id: str) -> PersonaProfile | None:
        """Return the live profile, copying a baseline across the first time it is used."""
        if not profile_id:
            return None
        live = self.live.profile(profile_id)
        if live is not None:
            return live
        if profile_id == USER_PROFILE_ID:
            return self._empty_user()
        shipped = self.baseline.profile(profile_id)
        if shipped is None:
            return None
        self.live.seed(shipped)
        return self.live.profile(profile_id) or shipped

    def listener_for(self, profile_id: str, context: CommunicationContext) -> Listener | None:
        """What this profile says about who listens in this setting, or None to use the default.

        Silent like everything else here: a profile that will not load simply has
        no opinion, and the setting's own default stands.
        """
        try:
            profile = self._resolve(profile_id)
        except Exception:
            return None
        if profile is None:
            return None
        return profile.listener_by_setting.get(context)

    def resolve_audience(
        self,
        profile_id: str,
        context: CommunicationContext,
        audience_id: str = "",
        declared: Listener | None = None,
    ) -> ResolvedAudience:
        """Resolve one visible, frozen audience without ever guessing a person."""
        try:
            profile = self._resolve(profile_id) if profile_id else None
        except Exception:
            profile = None
        audience = profile.audience(audience_id) if profile is not None and audience_id else None
        by_profile = profile.listener_by_setting.get(context) if profile is not None else None
        listener = resolve_listener(
            context,
            by_audience=audience.listener if audience is not None else None,
            declared=declared,
            by_profile=by_profile,
        )
        if audience is not None:
            register = audience.register_for(context)
            style_source = (
                "audience_setting" if context in audience.style_by_setting else "audience"
            )
            return ResolvedAudience(
                audience_id=audience.id,
                audience_label=audience.label,
                audience_kind=audience.kind,
                listener=listener,
                style=register,
                listener_source="audience",
                style_source=style_source,
            )
        if declared is not None:
            listener_source = "place"
        elif by_profile is not None:
            listener_source = "profile"
        else:
            listener_source = "setting"
        if profile is not None:
            register = profile.register_for(context)
            style_source = (
                "profile_setting" if context in profile.style_by_setting else "profile"
            )
        else:
            register = CommunicationRegister()
            style_source = "default"
        return ResolvedAudience(
            audience_label=(
                "Someone who knows you" if listener == "familiar" else "Someone new"
            ),
            listener=listener,
            style=register,
            listener_source=listener_source,
            style_source=style_source,
        )

    def _audit_vocabulary(self, profile: PersonaProfile) -> list[str]:
        """Every content word this profile could contribute, for the post-hoc check.

        Wider than the hints deliberately: the word worth catching is one the
        recognizer never produced, which is exactly the word never offered.
        """
        words: set[str] = set()
        for entry in profile.lexicon:
            for form in [entry.word, entry.display, *entry.aliases]:
                words |= set(_tokens(form))
        for rule in profile.specializations:
            words |= set(_tokens(rule.surface)) - set(_tokens(rule.plain))
        return sorted(words)

    # ------------------------------------------------------------------ the request path

    async def brief(
        self,
        hypotheses: list[Hypothesis],
        context: CommunicationContext,
        profile_id: str,
        listener: Listener = "familiar",
        audience_id: str = "",
    ) -> PersonalBrief | None:
        try:
            async with self._lock:
                return await asyncio.to_thread(
                    self._brief, hypotheses, context, profile_id, listener, audience_id
                )
        except Exception:
            # Deliberately silent. Personal context is worth having and never
            # worth failing a message over.
            return None

    def _brief(
        self,
        hypotheses: list[Hypothesis],
        context: CommunicationContext,
        profile_id: str,
        listener: Listener = "familiar",
        audience_id: str = "",
    ) -> PersonalBrief | None:
        profile = self._resolve(profile_id)
        if profile is None:
            return None
        slots = _slot_alignment(hypotheses)
        if not slots:
            return None
        audience = profile.audience(audience_id) if audience_id else None
        offers = (
            specialization_offers(
                profile, slots, self.settings.personal_anchor_share, context
            )
            if self.settings.personal_specializations
            else []
        )
        if audience is not None and audience.known_detail_ids:
            known = set(audience.known_detail_ids)
            offers = [offer for offer in offers if offer.source not in known]
        brief = PersonalBrief(
            profile_id=profile.id,
            profile_label=profile.label,
            speaker_note=profile.speaker_note,
            lexicon=lexicon_hints(profile, slots, self.settings.personal_max_hints, context),
            specializations=offers,
            audit_vocabulary=self._audit_vocabulary(profile),
        )
        return None if brief.is_empty() else brief

    # ------------------------------------------------------------------ onboarding

    def describe(self, profile_id: str) -> PersonaProfile | None:
        try:
            return self._resolve(profile_id)
        except Exception:
            return None

    def save_user_profile(
        self,
        people: list[str],
        places: list[str],
        things: list[str],
        details: list[tuple[str, str]],
    ) -> tuple[PersonaProfile, list[str]]:
        """Write the onboarding answers into the speaker's own profile.

        A detail the speaker types is checked by exactly the same rule a shipped
        one is, and a pair that could not be reverted cleanly is refused and
        named rather than quietly stored.
        """
        existing = self.live.profile(USER_PROFILE_ID) or self._empty_user()
        lexicon: list[LexiconEntry] = []
        for kind, values in (("person", people), ("place", places), ("object", things)):
            for value in values:
                display = value.strip()
                words = _tokens(display)
                if not words:
                    continue
                lexicon.append(
                    LexiconEntry(
                        id=f"user/{kind}/{words[-1]}",
                        word=words[-1],
                        display=display,
                        kind=kind,  # type: ignore[arg-type]
                    )
                )
        rules: list[SpecializationRule] = []
        refused: list[str] = []
        for word, wording in details:
            plain = word.strip()
            surface = wording.strip()
            anchor = _tokens(plain)
            if not anchor or not surface:
                continue
            rule = SpecializationRule(
                id=f"user/detail/{anchor[-1]}",
                anchor=anchor[-1],
                plain=plain,
                surface=surface,
                kind="brand",
            )
            if valid_rule(rule):
                rules.append(rule)
            else:
                refused.append(f"{plain} -> {surface}")
        profile = existing.model_copy(
            update={"lexicon": lexicon, "specializations": rules, "baseline": False}
        )
        self.live.save_profile(profile)
        return profile, refused

    def save_user_document(
        self, request: UserProfileInput
    ) -> tuple[PersonaProfile, list[str]]:
        """Replace the reviewed document atomically, retaining only valid details."""
        existing = self.live.profile(USER_PROFILE_ID) or self._empty_user()
        rules: list[SpecializationRule] = []
        refused: list[str] = []
        for rule in request.specializations:
            if valid_rule(rule):
                rules.append(rule)
            else:
                refused.append(f"{rule.plain} -> {rule.surface}")
        detail_ids = {rule.id for rule in rules}
        audiences = [
            audience.model_copy(
                update={
                    "known_detail_ids": [
                        item for item in audience.known_detail_ids if item in detail_ids
                    ]
                }
            )
            for audience in request.audiences
        ]
        candidate = existing.model_copy(
            update={
                "label": request.label.strip() or "You",
                "blurb": request.blurb.strip(),
                "context_default": request.context_default,
                "listener_by_setting": dict(request.listener_by_setting),
                "style": request.style,
                "style_by_setting": dict(request.style_by_setting),
                "lexicon": list(request.lexicon),
                "specializations": rules,
                "audiences": audiences,
                "baseline": False,
            }
        )
        profile = parse_profile(candidate.model_dump())
        if profile is None:
            raise ValueError("the profile document could not be validated")
        self.live.save_profile(profile)
        return profile, refused
