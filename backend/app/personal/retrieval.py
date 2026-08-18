"""Find the handful of past messages that resemble what was just said.

The query is the part worth reading twice. What arrives is not a sentence but a
beam set, and this recognizer puts the exact literal somewhere in its top five
only about a third of the time -- so keying retrieval on the leading beam would
inherit precisely the error the rest of the pipeline exists to correct.

Instead every position contributes all of its variants at once, each weighted by
the share of search weight it holds. A position every beam agreed on arrives at
full strength; a sound heard five ways contributes the mass of one word split
across five spellings rather than five words of noise. The result approximates
the embedding of the true utterance under the beam distribution, in one pass.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import log

import numpy as np

from ..config import Settings
from ..messaging.alignment import EMPTY_SLOT
from ..schemas import AcceptedMessage, CommunicationContext, Listener, PersonalExample


def query_terms(slots: list[dict[str, object]]) -> tuple[str, dict[str, float]]:
    """Build one weighted query string from the whole alignment."""
    words: list[str] = []
    weights: dict[str, float] = {}
    for slot in slots:
        for option in slot["options"]:  # type: ignore[union-attr]
            word = str(option["word"])
            if word == EMPTY_SLOT:
                continue
            words.append(word)
            share = float(option["share"])
            weights[word] = max(weights.get(word, 0.0), share)
    return " ".join(words), weights


def _bucket(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def _parse(moment: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(moment.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _recency(record: AcceptedMessage, now: datetime, half_life_days: float) -> float:
    when = _parse(record.accepted_at)
    if when is None:
        return 1.0
    age = max((now - when).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age / half_life_days)


def _frequency(record: AcceptedMessage) -> float:
    """Log-scaled, so a phrase said constantly is favoured without crowding out the rest."""
    return 1.0 + log(1.0 + max(record.uses - 1, 0))


def score(
    record: AcceptedMessage,
    cosine: float,
    now: datetime,
    context: CommunicationContext,
    listener: Listener,
    settings: Settings,
) -> float:
    """Similarity, then the decays. Context, listener and time are boosts, never filters.

    Filtering by setting would leave a small store with nothing to return, and
    `general` genuinely tells us nothing either way, so it stays neutral.

    The listener is penalized harder than the setting, because the two mismatches
    cost different things. A care example read at home is mildly off -- the words
    are right and the shape is right. A home example read among strangers is
    actively misleading, because its shape is the thing being copied: "I need the
    toilet." is what a carer is told and exactly not what a stranger is asked. So
    a different listener costs more than a different setting, and the two
    multiply when both differ.
    """
    base = max(cosine, 0.0)
    if base <= 0.0:
        return 0.0
    same_context = record.context == context or "general" in (record.context, context)
    same_listener = record.listener == listener
    same_time = _bucket(record.hour) == _bucket(now.hour)
    return (
        base
        * _recency(record, now, settings.personal_half_life_days)
        * _frequency(record)
        * (1.0 if same_context else settings.personal_context_penalty)
        * (1.0 if same_listener else settings.personal_listener_penalty)
        * (1.0 if same_time else settings.personal_time_penalty)
    )


def index_text(record: AcceptedMessage) -> str:
    return f"{record.heard} | {record.message}"


class ExampleIndex:
    """Past messages plus their vectors. Brute force, because N is in the hundreds."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.records: list[AcceptedMessage] = []
        self.vectors = np.zeros((0, dimension), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: AcceptedMessage, vector: np.ndarray) -> None:
        self.records.append(record)
        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1).astype(np.float32)])

    def replace(self, position: int, record: AcceptedMessage, vector: np.ndarray) -> None:
        self.records[position] = record
        self.vectors[position] = vector.astype(np.float32)

    def drop(self, positions: set[int]) -> None:
        if not positions:
            return
        keep = [index for index in range(len(self.records)) if index not in positions]
        self.records = [self.records[index] for index in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dimension), dtype=np.float32)

    def similarities(self, vector: np.ndarray) -> np.ndarray:
        if not len(self.records):
            return np.zeros((0,), dtype=np.float32)
        return self.vectors @ vector.astype(np.float32)

    def search(
        self,
        vector: np.ndarray,
        context: CommunicationContext,
        listener: Listener,
        now: datetime,
        settings: Settings,
    ) -> list[PersonalExample]:
        """The nearest few, or nothing at all while the store is still too small.

        A bad example is worse than no example: it shows the model a fluent
        sentence on roughly the right topic, which is exactly what it might copy
        when the audio is poor. Below the floor, the section is simply absent.
        """
        if len(self.records) < max(settings.personal_examples, 1):
            return []
        cosines = self.similarities(vector)
        scored = [
            (score(record, float(cosine), now, context, listener, settings), record)
            for record, cosine in zip(self.records, cosines, strict=True)
        ]
        scored = [item for item in scored if item[0] >= settings.personal_min_similarity]
        scored.sort(key=lambda item: -item[0])
        return [
            PersonalExample(
                heard=record.heard,
                message=record.message,
                context=record.context,
                listener=record.listener,
                score=round(value, 4),
            )
            for value, record in scored[: settings.personal_examples]
        ]
