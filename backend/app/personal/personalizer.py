"""The one entry point the request path touches, and the one that never raises.

This follows the policy `messaging/speech.py` set: personal context is an
enhancement and must never be a dependency. Whatever goes wrong in here -- a
missing bundle, an unreadable profile, an encoder fault -- the answer is None,
and the message chain runs exactly as it did before this layer existed. A
speaker waiting to be heard is not helped by an error.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from ..config import Settings
from ..messaging.alignment import _slot_alignment, _tokens
from ..schemas import (
    AcceptedMessage,
    CommunicationContext,
    Hypothesis,
    Listener,
    PersonaSummary,
    PersonalBrief,
)
from .consolidation import merge_into, prune
from .embedder import Embedder
from .lexicon import lexicon_hints, specialization_offers
from .profile import LexiconEntry, PersonaProfile, SpecializationRule, valid_rule
from .retrieval import ExampleIndex, index_text, query_terms
from .store import USER_PROFILE_ID, BaselineStore, LiveStore, summarize


class Personalizer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Eager, like the recognizer: a bundle that will not load should fail now,
        # where lifespan records it, not on the first utterance.
        self.embedder = Embedder(
            settings.embedder_root, settings.embedder_device, settings.embedder_max_tokens
        )
        self.baseline = BaselineStore(settings.persona_root)
        self.live = LiveStore(settings.personal_root)
        self._indexes: dict[str, ExampleIndex] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ profiles

    def known_ids(self) -> set[str]:
        return set(self.baseline.profiles()) | {USER_PROFILE_ID}

    def personas(self) -> list[PersonaSummary]:
        summaries: list[PersonaSummary] = []
        for profile_id, profile in sorted(self.baseline.profiles().items()):
            live = self.live.profile(profile_id)
            summaries.append(
                summarize(
                    live or profile,
                    len(self.live.history(profile_id) if live else self.baseline.history(profile_id)),
                    baseline=live is None,
                )
            )
        user = self.live.profile(USER_PROFILE_ID)
        summaries.append(
            summarize(
                user or self._empty_user(),
                len(self.live.history(USER_PROFILE_ID)),
                baseline=user is None,
            )
        )
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
        self.live.seed(shipped, self.baseline.history(profile_id))
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

    def _index(self, profile_id: str) -> ExampleIndex:
        cached = self._indexes.get(profile_id)
        if cached is not None:
            return cached
        index = ExampleIndex(self.embedder.dimension)
        for record in self.live.history(profile_id):
            index.add(record, self.embedder.encode(index_text(record)))
        self._indexes[profile_id] = index
        return index

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
    ) -> PersonalBrief | None:
        try:
            async with self._lock:
                return await asyncio.to_thread(
                    self._brief, hypotheses, context, profile_id, listener
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
    ) -> PersonalBrief | None:
        profile = self._resolve(profile_id)
        if profile is None:
            return None
        slots = _slot_alignment(hypotheses)
        if not slots:
            return None
        started = time.perf_counter()
        text, weights = query_terms(slots)
        examples = []
        if text.strip():
            vector = self.embedder.encode(text, weights)
            examples = self._index(profile.id).search(
                vector, context, listener, datetime.now(timezone.utc), self.settings
            )
        brief = PersonalBrief(
            profile_id=profile.id,
            profile_label=profile.label,
            speaker_note=profile.speaker_note,
            lexicon=lexicon_hints(profile, slots, self.settings.personal_max_hints, context),
            specializations=(
                specialization_offers(
                    profile, slots, self.settings.personal_anchor_share, context
                )
                if self.settings.personal_specializations
                else []
            ),
            examples=examples,
            audit_vocabulary=self._audit_vocabulary(profile),
            retrieval_seconds=round(time.perf_counter() - started, 3),
        )
        return None if brief.is_empty() else brief

    async def remember(
        self,
        profile_id: str,
        context: CommunicationContext,
        heard: str,
        message: str,
        listener: Listener = "familiar",
    ) -> tuple[bool, bool, str]:
        """Store an accepted message. Returns (stored, merged, reason)."""
        try:
            async with self._lock:
                return await asyncio.to_thread(self._remember, profile_id, context, heard, message)
        except Exception as error:
            return False, False, f"not stored ({type(error).__name__})"

    def _remember(
        self,
        profile_id: str,
        context: CommunicationContext,
        heard: str,
        message: str,
        listener: Listener = "familiar",
    ) -> tuple[bool, bool, str]:
        profile = self._resolve(profile_id)
        if profile is None or not message.strip():
            return False, False, "no such profile"
        now = datetime.now(timezone.utc)
        record = AcceptedMessage(
            id=f"{profile.id}-{uuid.uuid4().hex[:12]}",
            profile_id=profile.id,
            heard=heard.strip(),
            message=message.strip(),
            context=context,
            listener=listener,
            hour=now.hour,
            accepted_at=now.isoformat().replace("+00:00", "Z"),
            source="user",
        )
        index = self._index(profile.id)
        vector = self.embedder.encode(index_text(record))
        merged, _ = merge_into(index, record, vector, self.settings.personal_merge_threshold)
        prune(index, now, context, listener, self.settings)
        self.live.save_history(profile.id, index.records)
        return True, merged, "merged into an existing message" if merged else "stored"

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
        self._indexes.pop(USER_PROFILE_ID, None)
        return profile, refused
