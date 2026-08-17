"""Match a speaker's known words against what the recognizer actually produced.

Both functions here are filters, never generators. A profile word the beams did
not contain is never mentioned to the model, which is what makes the
disambiguation prior grounding-safe: the model is being helped to choose among
the recognizer's own words, not tempted toward a word that would then be thrown
out by the grounding check and take a working message down with it.
"""

from __future__ import annotations

from ..messaging.alignment import settled_words
from ..schemas import LexiconHint, SpecializationOffer
from .profile import PersonaProfile


def _slot_words(slot: dict[str, object]) -> list[str]:
    return [str(option["word"]) for option in slot["options"]]  # type: ignore[index]


def lexicon_hints(
    profile: PersonaProfile,
    slots: list[dict[str, object]],
    limit: int,
) -> list[LexiconHint]:
    """Known words this utterance contains, contested positions first.

    A contested position is where the prior does real work -- telling the model
    that `marge` is a person it should expect beats confirming a word every beam
    already agreed on.
    """
    forms = profile.heard_forms()
    if not forms:
        return []
    contested: list[LexiconHint] = []
    stable: list[LexiconHint] = []
    seen: set[str] = set()
    for slot in slots:
        for word in _slot_words(slot):
            entry = forms.get(word)
            if entry is None or entry.id in seen:
                continue
            seen.add(entry.id)
            hint = LexiconHint(
                word=word,
                display=entry.display,
                kind=entry.kind,
                note=entry.note[:60],
            )
            (stable if slot["stable"] else contested).append(hint)
    return (contested + stable)[:limit]


def specialization_offers(
    profile: PersonaProfile,
    slots: list[dict[str, object]],
    anchor_share: float,
) -> list[SpecializationOffer]:
    """Details whose anchor the beams overwhelmingly agreed on.

    A single surviving message speaks itself the moment it arrives, so a detail
    riding on a genuinely contested sound would turn a possible mishearing into a
    confident and very specific wrong statement. An anchor that holds most of its
    slot is not that case -- its rivals are spellings, not other words.
    """
    if not profile.specializations:
        return []
    offers: list[SpecializationOffer] = []
    seen: set[str] = set()
    for word in sorted(settled_words(slots, anchor_share)):
        rule = profile.rule_for(word)
        if rule is None or rule.id in seen:
            continue
        seen.add(rule.id)
        offers.append(
            SpecializationOffer(
                anchor=word,
                plain=rule.plain,
                surface=rule.surface,
                source=rule.id,
                kind=rule.kind,
            )
        )
    return offers
