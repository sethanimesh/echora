"""Named places, and the switch that lets the speaker's location choose one.

A place is a label with an optional location. It resolves to one of the four
communication contexts and behaves exactly as that context already does -- the
same guidance sentence in the Groq prompt, the same retrieval pool, the same
stamp on an accepted message. Nothing here widens `CommunicationContext`, so a
custom place can never hand the message chain a setting it does not know.

Three places ship and cannot be removed, so the interface always has something
to offer and a speaker cannot delete their way into an empty screen. Everything
else is theirs to add and remove.

The store is tolerant in the same way the persona loader is: a missing,
unreadable, or half-corrupt file yields a usable document rather than an
exception. Losing a location tag is a small annoyance; failing to open is not.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

from .schemas import (
    DEFAULT_RADIUS_M,
    MAX_RADIUS_M,
    MIN_RADIUS_M,
    CommunicationContext,
    Place,
    PlaceSettings,
    default_listener,
)


# The context each built-in borrows is also its id, so the three shipped places
# read the same in the store as they do in a history record.
BUILTIN_PLACES: tuple[tuple[str, str, CommunicationContext], ...] = (
    ("home", "Home", "home"),
    ("care", "Care", "care"),
    ("outdoors", "Outdoors", "outdoors"),
)
BUILTIN_IDS = tuple(place_id for place_id, _, _ in BUILTIN_PLACES)

# "General" is not a place. It is the absence of one -- what the interface falls
# back to when nothing is detected -- so it is never stored or tagged.
FALLBACK_CONTEXT: CommunicationContext = "general"

MAX_PLACES = 24
MAX_LABEL = 40
EARTH_RADIUS_M = 6_371_008.8


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_places() -> list[Place]:
    """The three shipped places, declaring nothing about who is there.

    `listener=None` is not the same as `listener=default_listener(context)`, and
    the difference is the whole middle rung of the precedence chain. A place that
    declares a listener outranks the speaker's profile; a place that declares
    nothing lets the profile speak, and only then falls to the setting's own
    default. Stamping the default here would make every shipped place look like a
    declaration, and a profile could never be heard.
    """
    return [
        Place(id=place_id, label=label, context=context, listener=None, builtin=True)
        for place_id, label, context in BUILTIN_PLACES
    ]


def default_settings() -> PlaceSettings:
    """Detection off, three untagged built-ins. What a first run looks like."""
    return PlaceSettings(auto_detect=False, places=default_places())


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    return slug[:MAX_LABEL] or "place"


def meters_between(
    first_lat: float, first_lon: float, second_lat: float, second_lon: float
) -> float:
    """Great-circle distance. Haversine is far more precision than a radius needs."""
    lat1, lon1, lat2, lon2 = map(radians, (first_lat, first_lon, second_lat, second_lon))
    inner = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(min(1.0, sqrt(inner)))


def normalize(incoming: PlaceSettings) -> PlaceSettings:
    """The single validation path for anything about to be stored.

    Every rule that protects the built-ins lives here rather than in the route,
    so a place read back from disk gets the same treatment as one that arrived
    over the wire. Built-ins keep their id, label and context whatever the
    request says; custom places get a unique slug; a place with only half a
    coordinate pair is treated as untagged.
    """
    seen: dict[str, Place] = {}
    for place in incoming.places:
        builtin = _builtin_for(place.id)
        if builtin is not None:
            # A built-in may be tagged and retagged, never renamed or rebound.
            place = place.model_copy(
                update={
                    "id": builtin[0],
                    "label": builtin[1],
                    "context": builtin[2],
                    "builtin": True,
                }
            )
        else:
            label = " ".join(place.label.split())[:MAX_LABEL]
            if not label:
                continue
            place = place.model_copy(
                update={
                    "id": _unique(slugify(label), seen),
                    "label": label,
                    "builtin": False,
                }
            )
        # The listener is deliberately NOT in either update above. A built-in
        # keeps its id, label and context whatever the request says, but who is
        # there is the speaker's to declare even for the shipped Outdoors -- a
        # speaker who only ever goes out with their daughter needs exactly that.
        #
        # None is preserved rather than filled in. It means "this place says
        # nothing", which is what lets the profile be consulted before the
        # setting's default. A built-in that merely repeats its own setting's
        # default has said nothing either, so it is read back as no declaration:
        # that keeps behaviour identical for a speaker with no profile, and stops
        # a document written before abstention existed from permanently masking
        # one. To make a built-in differ from its setting, set the other value.
        if place.builtin and place.listener == default_listener(place.context):
            place = place.model_copy(update={"listener": None})
        place = _with_location(place)
        if place.id in seen and not place.builtin:
            continue
        seen[place.id] = place

    places = [seen[place_id] for place_id in seen]
    # Anything the request dropped comes back. There is no request that can
    # leave the interface with fewer than the three shipped places.
    missing = [place for place in default_places() if place.id not in seen]
    ordered = _in_builtin_order(places + missing)
    # `updated_at` is carried through untouched: it records when the speaker last
    # changed something, so only `PlaceStore.save` may stamp it. Re-stamping here
    # would make merely opening the settings panel look like an edit.
    return PlaceSettings(
        auto_detect=incoming.auto_detect,
        places=ordered[:MAX_PLACES],
        updated_at=incoming.updated_at,
    )


def _builtin_for(place_id: str) -> tuple[str, str, CommunicationContext] | None:
    for entry in BUILTIN_PLACES:
        if entry[0] == place_id:
            return entry
    return None


def _unique(slug: str, taken: dict[str, Place]) -> str:
    """Keep two places called the same thing distinguishable rather than merged."""
    if slug not in taken and slug not in BUILTIN_IDS:
        return slug
    for suffix in range(2, 100):
        candidate = f"{slug}-{suffix}"
        if candidate not in taken and candidate not in BUILTIN_IDS:
            return candidate
    return f"{slug}-{os.urandom(3).hex()}"


def _with_location(place: Place) -> Place:
    """Clamp the radius, and treat half a coordinate pair as no location at all."""
    radius = min(max(place.radius_m, MIN_RADIUS_M), MAX_RADIUS_M)
    if place.latitude is None or place.longitude is None:
        return place.model_copy(
            update={"latitude": None, "longitude": None, "tagged_at": None, "radius_m": radius}
        )
    return place.model_copy(update={"radius_m": radius, "tagged_at": place.tagged_at or _now()})


def _in_builtin_order(places: list[Place]) -> list[Place]:
    """Built-ins first, in the shipped order; custom places after, as added."""
    builtins = [place for place_id in BUILTIN_IDS for place in places if place.id == place_id]
    return builtins + [place for place in places if place.id not in BUILTIN_IDS]


def nearest(
    settings: PlaceSettings, latitude: float, longitude: float, accuracy_m: float = 0.0
) -> tuple[Place, float] | None:
    """The closest tagged place the speaker is plausibly inside, or None.

    The accuracy allowance matters: a browser fix indoors is routinely tens of
    metres out, which would otherwise make a correctly tagged home fail to match
    its own front room. Kept on the server as the reference implementation --
    the interface resolves client-side so coordinates never leave the browser.
    """
    tolerance = min(max(accuracy_m, 0.0), 100.0)
    best: tuple[Place, float] | None = None
    for place in settings.places:
        if place.latitude is None or place.longitude is None:
            continue
        distance = meters_between(latitude, longitude, place.latitude, place.longitude)
        if distance > place.radius_m + tolerance:
            continue
        if best is None or distance < best[1]:
            best = (place, distance)
    return best


class PlaceStore:
    """Reads and writes the one places document. Never raises on a bad file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> PlaceSettings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default_settings()
        except Exception:
            return default_settings()
        return self._parse(raw)

    def save(self, settings: PlaceSettings) -> PlaceSettings:
        stored = normalize(settings).model_copy(update={"updated_at": _now()})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Replace atomically, so a crash mid-write cannot leave a document that
        # loses every location tag on the next read.
        handle, temporary = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(stored.model_dump_json(indent=2))
            os.replace(temporary, self.path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return stored

    def _parse(self, raw: object) -> PlaceSettings:
        """Drop individual unreadable places rather than failing the document."""
        if not isinstance(raw, dict):
            return default_settings()
        places: list[Place] = []
        for row in raw.get("places") or []:
            if not isinstance(row, dict):
                continue
            try:
                places.append(Place.model_validate(row))
            except Exception:
                continue
        updated_at = raw.get("updated_at")
        return normalize(
            PlaceSettings(
                auto_detect=bool(raw.get("auto_detect", False)),
                places=places,
                updated_at=updated_at if isinstance(updated_at, str) else None,
            )
        )
