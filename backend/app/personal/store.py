"""Two stores: shipped profiles and speaker-controlled live profiles.

`data/personas/` holds the six demonstration speakers. It is tracked in git, it is
the reproducible starting point, and nothing in this package can write to it --
`BaselineStore` has no write method at all, so a mistaken call is an
AttributeError while the code is being written rather than a silent edit to a
committed fixture.

An explicitly selected or edited profile lives under `data/personal/`, which is
gitignored. Messages are never recorded here: personal context is declarative,
not inferred from whatever the assistant happened to say before.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schemas import PersonaSummary
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


class BaselineStore:
    """The shipped personas. Read-only by construction: there is no writer here."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._profiles: dict[str, PersonaProfile] = {}
        for row in _read_jsonl(root / "personas.jsonl"):
            profile = parse_profile(row)
            if profile is None:
                continue
            self._profiles[profile.id] = profile

    def profiles(self) -> dict[str, PersonaProfile]:
        return dict(self._profiles)

    def profile(self, profile_id: str) -> PersonaProfile | None:
        return self._profiles.get(profile_id)

class LiveStore:
    """Speaker-controlled profile data. The only writer in this package."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _profile_path(self, profile_id: str) -> Path:
        return self._under(self.root / profile_id / "profile.json")

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

    def save_profile(self, profile: PersonaProfile) -> None:
        path = self._profile_path(profile.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def seed(self, profile: PersonaProfile) -> None:
        """Copy a baseline into the live store the first time it is used."""
        if self.exists(profile.id):
            return
        self.save_profile(profile.model_copy(update={"baseline": False}))


def summarize(profile: PersonaProfile, baseline: bool) -> PersonaSummary:
    return PersonaSummary(
        id=profile.id,
        label=profile.label,
        blurb=profile.blurb,
        icon=profile.icon,
        context_default=profile.context_default,
        listener_by_setting=dict(profile.listener_by_setting),
        lexicon_size=len(profile.lexicon),
        specialization_size=len(profile.specializations),
        audience_size=len(profile.audiences),
        baseline=baseline,
    )
