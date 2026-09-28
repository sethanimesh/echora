"""Match a speaker's known words against what the recognizer actually produced.

Both functions here are filters, never generators. A profile word the beams did
not contain is never mentioned to the model, which is what makes the
disambiguation prior grounding-safe: the model is being helped to choose among
the recognizer's own words, not tempted toward a word that would then be thrown
out by the grounding check and take a working message down with it.
"""

from __future__ import annotations

from ..schemas import CommunicationContext, LexiconHint, SpecializationOffer
from .profile import PersonaProfile


_ANCHOR_EQUIVALENTS: dict[str, frozenset[str]] = {
    # Conservative spelling families observed in this recognizer's beams. They
    # are aggregated because they are the same communicative word, unlike
    # genuinely different alternatives such as soup/suit or coffee/toffee.
    "tea": frozenset({"tea", "tee", "teaa"}),
    "coffee": frozenset({"coffee", "coffey", "coffe"}),
    "cream": frozenset({"cream", "creem", "crem"}),
}


def _slot_words(slot: dict[str, object]) -> list[str]:
    return [str(option["word"]) for option in slot["options"]]  # type: ignore[index]


def lexicon_hints(
    profile: PersonaProfile,
    slots: list[dict[str, object]],
    limit: int,
    context: CommunicationContext = "general",
) -> list[LexiconHint]:
    """Known words this utterance contains, contested positions first.

    A contested position is where the prior does real work -- telling the model
    that `marge` is a person it should expect beats confirming a word every beam
    already agreed on.

    Every spelling that matched gets its own hint, including several that lead to
    the same person. Reporting one hint per entry looks tidier and throws away
    the answer: with beams `donna | dawn | dorn | danna`, naming only `donna` and
    `dawn` leaves `dorn` an unexplained third word, and the speaker is asked to
    choose between "Dawn." and "Dorn." -- two spellings of the same carer. The
    profile already knows they are one person; this is what says so.
    """
    forms = profile.heard_forms(context)
    if not forms:
        return []
    contested: list[LexiconHint] = []
    stable: list[LexiconHint] = []
    seen: set[str] = set()
    for slot in slots:
        for word in _slot_words(slot):
            entry = forms.get(word)
            if entry is None or word in seen:
                continue
            seen.add(word)
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
    context: CommunicationContext = "general",
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
    for rule in profile.specializations:
        if not rule.applies_in(context):
            continue
        canonical = rule.anchor.strip().lower()
        family = _ANCHOR_EQUIVALENTS.get(canonical, frozenset({canonical}))
        for slot in slots:
            matched = [
                str(option["word"])
                for option in slot["options"]  # type: ignore[index]
                if str(option["word"]) in family  # type: ignore[index]
            ]
            share = sum(
                float(option["share"])
                for option in slot["options"]  # type: ignore[index]
                if str(option["word"]) in family  # type: ignore[index]
            )
            if matched and share >= anchor_share:
                offers.append(
                    SpecializationOffer(
                        anchor=canonical,
                        matches=matched,
                        plain=rule.plain,
                        surface=rule.surface,
                        source=rule.id,
                        kind=rule.kind,
                    )
                )
                break
    return offers
