"""Places: the built-ins cannot be lost, and a bad file cannot lock the speaker out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.places import (
    BUILTIN_IDS,
    PlaceStore,
    default_settings,
    meters_between,
    nearest,
    normalize,
    slugify,
)
from app.schemas import MAX_RADIUS_M, MIN_RADIUS_M, Place, PlaceSettings


def _settings(*places: Place, auto_detect: bool = False) -> PlaceSettings:
    return PlaceSettings(auto_detect=auto_detect, places=list(places))


def _by_id(settings: PlaceSettings, place_id: str) -> Place:
    return next(place for place in settings.places if place.id == place_id)


# --------------------------------------------------------------- the built-ins


def test_a_first_run_ships_three_untagged_places_with_detection_off() -> None:
    settings = default_settings()
    assert [place.id for place in settings.places] == list(BUILTIN_IDS)
    assert settings.auto_detect is False
    assert all(place.builtin and not place.tagged for place in settings.places)
    # Every built-in borrows the context of the same name, so a place and a
    # request and profile read identically.
    assert all(place.context == place.id for place in settings.places)


def test_builtins_come_back_when_a_request_leaves_them_out() -> None:
    stored = normalize(_settings(auto_detect=True))
    assert [place.id for place in stored.places] == list(BUILTIN_IDS)
    assert stored.auto_detect is True


def test_a_builtin_cannot_be_renamed_or_rebound_to_another_context() -> None:
    stored = normalize(_settings(Place(id="home", label="Somewhere", context="care", builtin=False)))
    home = _by_id(stored, "home")
    assert (home.label, home.context, home.builtin) == ("Home", "home", True)


def test_a_builtin_can_still_be_tagged_and_cleared() -> None:
    tagged = normalize(
        _settings(Place(id="home", label="Home", context="home", latitude=12.9716, longitude=80.2594))
    )
    assert _by_id(tagged, "home").tagged
    assert _by_id(tagged, "home").tagged_at

    cleared = normalize(_settings(*[place.model_copy(update={"latitude": None}) for place in tagged.places]))
    assert not _by_id(cleared, "home").tagged
    assert _by_id(cleared, "home").tagged_at is None


# ------------------------------------------------------------- custom places


def test_a_custom_place_is_slugged_and_keeps_its_own_wording() -> None:
    stored = normalize(_settings(Place(id="ignored", label="  Shopping   Centre  ", context="outdoors")))
    custom = _by_id(stored, "shopping-centre")
    assert (custom.label, custom.context, custom.builtin) == ("Shopping Centre", "outdoors", False)
    # Built-ins stay first so the chip row does not reorder as places are added.
    assert [place.id for place in stored.places][:3] == list(BUILTIN_IDS)


def test_two_places_with_the_same_name_stay_distinguishable() -> None:
    stored = normalize(
        _settings(
            Place(id="a", label="Shopping centre", context="outdoors"),
            Place(id="b", label="shopping centre", context="outdoors"),
        )
    )
    assert [place.id for place in stored.places if not place.builtin] == [
        "shopping-centre",
        "shopping-centre-2",
    ]


def test_a_custom_place_cannot_squat_on_a_builtin_id() -> None:
    stored = normalize(_settings(Place(id="whatever", label="Home", context="care")))
    assert _by_id(stored, "home").builtin is True
    assert _by_id(stored, "home-2").context == "care"


def test_a_place_with_no_label_is_dropped() -> None:
    stored = normalize(_settings(Place(id="x", label="   ", context="home")))
    assert [place.id for place in stored.places] == list(BUILTIN_IDS)


def test_a_custom_place_can_be_removed_but_a_builtin_survives_the_same_request() -> None:
    with_custom = normalize(_settings(Place(id="a", label="Shopping centre", context="outdoors")))
    kept = [place for place in with_custom.places if place.builtin]
    assert [place.id for place in normalize(_settings(*kept)).places] == list(BUILTIN_IDS)


def test_the_place_list_is_capped() -> None:
    many = [Place(id=f"p{index}", label=f"Place {index}", context="home") for index in range(60)]
    assert len(normalize(_settings(*many)).places) == 24


@pytest.mark.parametrize(
    ("label", "expected"),
    [("Shopping Centre", "shopping-centre"), ("Mum's!!", "mum-s"), ("...", "place"), ("Café 42", "caf-42")],
)
def test_slugify(label: str, expected: str) -> None:
    assert slugify(label) == expected


# ------------------------------------------------------------------ locations


def test_the_radius_is_clamped_rather_than_refused() -> None:
    """A silly radius is corrected. Refusing it would drop the place and its tag."""
    wide = normalize(_settings(Place(id="a", label="Wide", context="home", radius_m=999_999)))
    assert _by_id(wide, "wide").radius_m == MAX_RADIUS_M
    narrow = normalize(_settings(Place(id="a", label="Narrow", context="home", radius_m=0)))
    assert _by_id(narrow, "narrow").radius_m == MIN_RADIUS_M


def test_half_a_coordinate_pair_is_treated_as_untagged() -> None:
    stored = normalize(_settings(Place(id="a", label="Half", context="home", latitude=12.97)))
    half = _by_id(stored, "half")
    assert (half.latitude, half.longitude, half.tagged_at) == (None, None, None)


def test_an_impossible_coordinate_is_refused_by_the_model() -> None:
    with pytest.raises(Exception):
        Place(id="a", label="Nowhere", context="home", latitude=91.0, longitude=0.0)


def test_meters_between_matches_a_known_distance() -> None:
    # One tenth of a degree of latitude is a shade over 11 km, anywhere on earth.
    assert round(meters_between(12.9716, 80.2594, 13.0716, 80.2594)) == pytest.approx(11_119, abs=5)
    assert meters_between(12.9716, 80.2594, 12.9716, 80.2594) == 0.0


def test_nearest_picks_the_closest_place_the_speaker_is_inside() -> None:
    settings = normalize(
        _settings(
            Place(id="a", label="Home", context="home", latitude=12.9716, longitude=80.2594, radius_m=150),
            Place(id="b", label="Clinic", context="care", latitude=12.9720, longitude=80.2594, radius_m=150),
        )
    )
    found = nearest(settings, 12.97185, 80.2594)
    assert found is not None and found[0].id == "clinic"


def test_nearest_returns_nothing_when_out_of_range_or_untagged() -> None:
    settings = normalize(
        _settings(Place(id="a", label="Home", context="home", latitude=12.9716, longitude=80.2594, radius_m=150))
    )
    assert nearest(settings, 12.9800, 80.2594) is None
    assert nearest(default_settings(), 12.9716, 80.2594) is None


def test_a_poor_fix_still_matches_the_place_it_is_standing_in() -> None:
    """Browser geolocation indoors is tens of metres out; a tagged home must survive that."""
    settings = normalize(
        _settings(Place(id="a", label="Home", context="home", latitude=12.9716, longitude=80.2594, radius_m=150))
    )
    assert nearest(settings, 12.9733, 80.2594) is None
    assert nearest(settings, 12.9733, 80.2594, accuracy_m=100) is not None
    # The allowance is capped, so a hopeless fix cannot match half the city.
    assert nearest(settings, 12.9900, 80.2594, accuracy_m=100_000) is None


# ---------------------------------------------------------------- the store


def test_a_missing_file_loads_the_defaults_without_writing_anything(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert PlaceStore(path).load() == default_settings()
    assert not path.exists()


def test_a_round_trip_keeps_the_switch_and_the_tag(tmp_path: Path) -> None:
    store = PlaceStore(tmp_path / "nested" / "settings.json")
    stored = store.save(
        _settings(
            Place(id="home", label="Home", context="home", latitude=12.9716, longitude=80.2594),
            Place(id="x", label="Shopping centre", context="outdoors"),
            auto_detect=True,
        )
    )
    assert stored.updated_at
    reloaded = store.load()
    assert reloaded.auto_detect is True
    assert _by_id(reloaded, "home").tagged
    assert _by_id(reloaded, "shopping-centre").context == "outdoors"


def test_the_write_is_atomic_and_leaves_no_temporary_behind(tmp_path: Path) -> None:
    store = PlaceStore(tmp_path / "settings.json")
    store.save(default_settings())
    assert [path.name for path in tmp_path.iterdir()] == ["settings.json"]


@pytest.mark.parametrize("body", ["", "not json at all", "[]", "null", '{"places": "nonsense"}'])
def test_an_unreadable_file_still_yields_a_usable_document(tmp_path: Path, body: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(body, encoding="utf-8")
    loaded = PlaceStore(path).load()
    assert [place.id for place in loaded.places] == list(BUILTIN_IDS)
    assert loaded.auto_detect is False


def test_one_corrupt_place_does_not_take_the_rest_with_it(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "auto_detect": True,
                "places": [
                    "not an object",
                    {"id": "broken", "label": "Broken", "context": "spaceship"},
                    {"id": "x", "label": "Shopping centre", "context": "outdoors"},
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = PlaceStore(path).load()
    assert loaded.auto_detect is True
    assert "shopping-centre" in {place.id for place in loaded.places}
    assert "broken" not in {place.id for place in loaded.places}


def test_reading_the_document_does_not_look_like_an_edit(tmp_path: Path) -> None:
    """`updated_at` records when the speaker last changed something, not the last read."""
    store = PlaceStore(tmp_path / "settings.json")
    stored = store.save(default_settings())
    assert store.load().updated_at == stored.updated_at == store.load().updated_at
    assert store.save(default_settings()).updated_at != stored.updated_at


# ------------------------------------------------- abstaining from the listener


def test_a_shipped_place_declares_no_listener_so_the_profile_can_be_heard() -> None:
    """`None` and "the setting's own default" are not the same claim.

    A place that declares a listener outranks the speaker's profile. If the
    shipped places arrived already stamped with their setting's default, every
    one of them would look like a declaration and the middle rung of the
    precedence chain -- what the profile says this setting usually means -- could
    never be reached.
    """
    for place in default_settings().places:
        assert place.listener is None


def test_a_declared_listener_survives_the_round_trip_on_a_builtin() -> None:
    """The one thing a speaker may change about a built-in still sticks."""
    stored = normalize(_settings(Place(id="outdoors", label="Outdoors", context="outdoors", listener="familiar")))
    assert _by_id(stored, "outdoors").listener == "familiar"


def test_a_builtin_repeating_its_own_default_is_read_back_as_no_declaration() -> None:
    """Saying what the setting already says is not a declaration.

    This is also the migration: a document written before a place could abstain
    stamped every built-in with its default, which would otherwise mask a
    profile forever. Behaviour is unchanged for a speaker with no profile,
    because the chain ends on that same default.
    """
    stored = normalize(
        _settings(
            Place(id="outdoors", label="Outdoors", context="outdoors", listener="unfamiliar"),
            Place(id="home", label="Home", context="home", listener="familiar"),
        )
    )
    assert _by_id(stored, "outdoors").listener is None
    assert _by_id(stored, "home").listener is None


def test_a_custom_place_keeps_a_listener_that_matches_its_default() -> None:
    """Only the built-ins are migrated: a custom place means what it says."""
    stored = normalize(_settings(Place(id="", label="Chemist", context="outdoors", listener="unfamiliar")))
    assert _by_id(stored, "chemist").listener == "unfamiliar"
