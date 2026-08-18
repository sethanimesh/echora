"""Fold an accepted message into the store instead of piling it on top.

A speaker who needs water asks for water. Appending every acceptance would leave
forty near-identical rows, and since retrieval returns only four neighbours those
rows would crowd out the rare utterance that actually needed retrieving -- the
store would grow while getting less useful.

So an acceptance either merges into the entry it already resembles, taking that
entry's count up and its timestamp forward, or it is genuinely new and gets a
row. Eviction uses the same score retrieval ranks by, so what the store keeps and
what retrieval reaches for can never drift apart.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..config import Settings
from ..schemas import AcceptedMessage, CommunicationContext, Listener
from .retrieval import ExampleIndex, score


def merge_into(
    index: ExampleIndex,
    record: AcceptedMessage,
    vector: np.ndarray,
    threshold: float,
) -> tuple[bool, AcceptedMessage]:
    """Merge with the nearest entry above the threshold, else insert. True when merged."""
    if len(index):
        cosines = index.similarities(vector)
        # Only entries accepted to the same kind of listener are candidates. A
        # merge takes the longer wording and stamps the newcomer's setting over
        # the old one, so without this guard "Where is the washroom?" could
        # absorb "I want to use the washroom.", keep the home wording because it
        # is longer, and relabel it -- and the speaker would be left with neither
        # phrasing intact. Gating on the listener rather than the setting is
        # deliberate: home and care are both familiar and should still fold
        # together, so the store splits at worst in two, not in four.
        cosines = np.where(
            np.array([entry.listener == record.listener for entry in index.records]),
            cosines,
            -1.0,
        )
        position = int(np.argmax(cosines))
        if float(cosines[position]) >= threshold:
            existing = index.records[position]
            # Keep the fuller wording: a later, more complete phrasing of the same
            # need is the one worth showing the model next time.
            message = existing.message if len(existing.message) >= len(record.message) else record.message
            merged = existing.model_copy(
                update={
                    "message": message,
                    "heard": record.heard or existing.heard,
                    "context": record.context,
                    "hour": record.hour,
                    "accepted_at": record.accepted_at,
                    "uses": existing.uses + 1,
                }
            )
            # The stored vector drifts toward how this speaker actually says it,
            # as a running mean rather than a jump to the newest phrasing.
            blended = index.vectors[position] * (merged.uses - 1) + vector
            norm = float(np.linalg.norm(blended)) or 1.0
            index.replace(position, merged, blended / norm)
            return True, merged
    index.add(record, vector)
    return False, record


def prune(
    index: ExampleIndex,
    now: datetime,
    context: CommunicationContext,
    listener: Listener,
    settings: Settings,
) -> int:
    """Drop the weakest entries once the store is over its cap. Returns how many went."""
    excess = len(index) - settings.personal_store_cap
    if excess <= 0:
        return 0
    ranked = sorted(
        range(len(index)),
        key=lambda position: score(
            index.records[position], 1.0, now, context, listener, settings
        ),
    )
    index.drop(set(ranked[:excess]))
    return excess
