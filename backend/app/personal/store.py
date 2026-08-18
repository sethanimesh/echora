"""Two stores: the baselines that ship with the repo, and what a speaker accumulates.

`data/personas/` holds the six demonstration speakers. It is tracked in git, it is
the reproducible starting point, and nothing in this package can write to it --
`BaselineStore` has no write method at all, so a mistaken call is an
AttributeError while the code is being written rather than a silent edit to a
committed fixture.

Everything that grows lives under `data/personal/`, which is gitignored. A
persona is copied there the first time it is used, so the demo genuinely
remembers across restarts while the baseline it started from stays pristine and
resetting is one directory delete.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ..schemas import AcceptedMessage, PersonaSummary
from .profile import PersonaProfile, parse_profile


USER_PROFILE_ID = "user"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            rows.append(parsed)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Replace the file atomically, so a crash mid-write cannot truncate a history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _messages(rows: list[dict]) -> list[AcceptedMessage]:
    records: list[AcceptedMessage] = []
    for row in rows:
        try:
            records.append(AcceptedMessage.model_validate(row))
        except Exception:
            continue
    return records


class BaselineStore:
    """The shipped personas. Read-only by construction: there is no writer here."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._profiles: dict[str, PersonaProfile] = {}
        self._history: dict[str, list[AcceptedMessage]] = {}
        for row in _read_jsonl(root / "personas.jsonl"):
            profile = parse_profile(row)
            if profile is None:
                continue
            self._profiles[profile.id] = profile
            self._history[profile.id] = _messages(_read_jsonl(root / "history" / f"{profile.id}.jsonl"))

    def profiles(self) -> dict[str, PersonaProfile]:
        return dict(self._profiles)

    def profile(self, profile_id: str) -> PersonaProfile | None:
        return self._profiles.get(profile_id)

    def history(self, profile_id: str) -> list[AcceptedMessage]:
        return list(self._history.get(profile_id, []))


class LiveStore:
    """Everything a speaker accumulates. The only writer in this package."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _profile_path(self, profile_id: str) -> Path:
        return self._under(self.root / profile_id / "profile.json")

    def _history_path(self, profile_id: str) -> Path:
        return self._under(self.root / profile_id / "history.jsonl")

    def _under(self, path: Path) -> Path:
        """Refuse any path that escapes the live root.

        The profile id reaches this from a form field, so `../` in it would
        otherwise be a write anywhere on disk -- including over a baseline.
        """
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ValueError(f"refusing to touch {path}, which is outside the live store")
        return resolved

    def exists(self, profile_id: str) -> bool:
        return self._profile_path(profile_id).is_file()

    def profile(self, profile_id: str) -> PersonaProfile | None:
        path = self._profile_path(profile_id)
        if not path.is_file():
            return None
        try:
            return parse_profile(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def history(self, profile_id: str) -> list[AcceptedMessage]:
        return _messages(_read_jsonl(self._history_path(profile_id)))

    def save_profile(self, profile: PersonaProfile) -> None:
        path = self._profile_path(profile.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def save_history(self, profile_id: str, records: list[AcceptedMessage]) -> None:
        _write_jsonl(self._history_path(profile_id), [record.model_dump() for record in records])

    def seed(self, profile: PersonaProfile, history: list[AcceptedMessage]) -> None:
        """Copy a baseline into the live store the first time it is used."""
        if self.exists(profile.id):
            return
        self.save_profile(profile.model_copy(update={"baseline": False}))
        self.save_history(profile.id, history)


def summarize(profile: PersonaProfile, history_size: int, baseline: bool) -> PersonaSummary:
    return PersonaSummary(
        id=profile.id,
        label=profile.label,
        blurb=profile.blurb,
        icon=profile.icon,
        context_default=profile.context_default,
        listener_by_setting=dict(profile.listener_by_setting),
        lexicon_size=len(profile.lexicon),
        specialization_size=len(profile.specializations),
        history_size=history_size,
        baseline=baseline,
    )
