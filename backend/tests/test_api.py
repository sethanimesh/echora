from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app import main
from app.audio import encode_wav
from app.messaging.groq_chain import MessageChainResult
from app.personal.profile import PersonaProfile
from app.places import BUILTIN_IDS, PlaceStore
from app.schemas import (
    AudioQuality,
    CommunicationRegister,
    Hypothesis,
    MessageCandidate,
    PersonalBrief,
    PersonaSummary,
    RankerDecision,
    RawAsrResult,
    ResolvedAudience,
    SpeechAudio,
)


class FakeBackend:
    device = "test/float32"

    async def transcribe(self, audio, beams):
        return RawAsrResult(
            backend="local",
            device=self.device,
            hypotheses=[Hypothesis(id="h1", literal_text="I water", sequence_score=-0.2, search_weight=1)],
            audio_quality=AudioQuality(seconds=0.5, peak_dbfs=-3, rms_dbfs=-9, clipped_samples=0, low_level_warning=False),
            decode_seconds=0.1,
        )


class FailingBackend:
    device = "remote-gpu"

    async def transcribe(self, audio, beams):
        raise TimeoutError("cloud worker is still warming")


class FakeChain:
    def __init__(self, settings):
        pass

    stances: list[tuple[str, str]] = []
    styles: list[CommunicationRegister | None] = []

    async def run(
        self, hypotheses, context="general", brief=None, listener="familiar", register=None
    ):
        FakeChain.stances.append((context, listener))
        FakeChain.styles.append(register)
        return MessageChainResult(
            ranker=RankerDecision(
                decision="selected",
                selected_message_id="m1",
                display_hypothesis_ids=["h1"],
                display_message_ids=["m1"],
                reason="Only candidate",
                source="groq",
            ),
            messages=[
                MessageCandidate(
                    message_id="m1",
                    hypothesis_id="h1",
                    source_hypothesis_ids=["h1"],
                    source_literals=["I water"],
                    literal_text="I water",
                    interpreted_intent="i water",
                    corrected_text="I would like some water.",
                    repair_status="corrected",
                    repair_note="Heard as \u201ci water\u201d.",
                )
            ],
            ranking_seconds=0,
            grammar_seconds=0,
            warnings=[],
        )


def _candidate(message_id: str, text: str) -> MessageCandidate:
    return MessageCandidate(
        message_id=message_id,
        hypothesis_id="h1",
        source_hypothesis_ids=["h1"],
        source_literals=["I water"],
        literal_text="I water",
        interpreted_intent="i water",
        corrected_text=text,
        repair_status="corrected",
        repair_note="Heard as \u201ci water\u201d.",
    )


class AmbiguousChain:
    """Two readings survived, so the speaker still has to choose between them."""

    def __init__(self, settings):
        pass

    async def run(
        self, hypotheses, context="general", brief=None, listener="familiar", register=None
    ):
        messages = [_candidate("m1", "I would like some water."), _candidate("m2", "I would like to wait.")]
        return MessageChainResult(
            ranker=RankerDecision(
                decision="ambiguous",
                display_hypothesis_ids=["h1"],
                display_message_ids=["m1", "m2"],
                reason="These could be different messages; please choose.",
                source="groq",
            ),
            messages=messages,
        )


class FakeSpeech:
    spoken: list[str] = []

    def __init__(self, settings):
        pass

    async def synthesize(self, text):
        FakeSpeech.spoken.append(text)
        return SpeechAudio(audio_base64="UklGRg==", voice="hannah", model="test-tts")


class SilentSpeech:
    """Groq TTS could not answer. The message must still be returned."""

    def __init__(self, settings):
        pass

    async def synthesize(self, text):
        return None


class FakePersonal:
    """Stands in for the personal layer so no test loads the real encoder weights."""

    briefs: list[tuple[str, str]] = []
    stances: list[tuple[str, str]] = []
    declares: dict[str, str] = {}
    current: PersonaProfile = PersonaProfile(id="user", label="You", baseline=False)

    def __init__(self, settings):
        pass

    def known_ids(self):
        return {"krishnan", "user"}

    def personas(self):
        return [PersonaSummary(id="krishnan", label="Krishnan")]

    def describe(self, profile_id):
        return self.current if profile_id == "user" else PersonaProfile(id=profile_id, label="Krishnan")

    def save_user_document(self, request):
        FakePersonal.current = PersonaProfile(
            id="user",
            label=request.label,
            blurb=request.blurb,
            baseline=False,
            context_default=request.context_default,
            listener_by_setting=request.listener_by_setting,
            style=request.style,
            style_by_setting=request.style_by_setting,
            lexicon=request.lexicon,
            specializations=request.specializations,
            audiences=request.audiences,
        )
        return FakePersonal.current, []

    def listener_for(self, profile_id, context):
        return FakePersonal.declares.get(context)

    def resolve_audience(self, profile_id, context, audience_id="", declared=None):
        if profile_id in self.known_ids() and audience_id == "priya":
            return ResolvedAudience(
                audience_id="priya",
                audience_label="Priya",
                audience_kind="person",
                listener="familiar",
                style=CommunicationRegister(
                    brevity="natural", courtesy="please", formality="informal"
                ),
                listener_source="audience",
                style_source="audience",
            )
        listener = declared or FakePersonal.declares.get(context) or (
            "unfamiliar" if context == "outdoors" else "familiar"
        )
        return ResolvedAudience(
            audience_label="Usual here",
            listener=listener,
            style=CommunicationRegister(),
            listener_source=(
                "place"
                if declared
                else "profile"
                if FakePersonal.declares.get(context)
                else "setting"
            ),
            style_source="profile",
        )

    async def brief(
        self, hypotheses, context, profile_id, listener="familiar", audience_id=""
    ):
        FakePersonal.briefs.append((profile_id, context))
        FakePersonal.stances.append((context, listener))
        if profile_id not in self.known_ids():
            return None
        return PersonalBrief(profile_id=profile_id, profile_label="Krishnan", speaker_note="note")

def client(monkeypatch, chain=FakeChain, speech=FakeSpeech, personal=FakePersonal, places_path=None):
    FakeSpeech.spoken = []
    FakeChain.stances = []
    FakeChain.styles = []
    FakePersonal.briefs = []
    FakePersonal.stances = []
    FakePersonal.declares = {}
    FakePersonal.current = PersonaProfile(id="user", label="You", baseline=False)
    monkeypatch.setattr(main, "create_backend", lambda settings: FakeBackend())
    monkeypatch.setattr(main, "GroqMessageChain", chain)
    monkeypatch.setattr(main, "GroqSpeech", speech)
    monkeypatch.setattr(main, "Personalizer", personal)
    if places_path is not None:
        # Anything that writes places goes to a temporary file. The real document
        # lives in data/personal/, which a test must never touch.
        monkeypatch.setattr(main, "PlaceStore", lambda _path: PlaceStore(places_path))
    return TestClient(main.app)


def _post(
    api,
    context: str | None = None,
    persona: str | None = None,
    listener: str | None = None,
    audience: str | None = None,
):
    audio = np.sin(np.linspace(0, 100, 8_000)).astype(np.float32) * 0.1
    form = {}
    if context:
        form["context"] = context
    if persona:
        form["persona"] = persona
    if listener:
        form["listener"] = listener
    if audience:
        form["audience"] = audience
    return api.post(
        "/api/v1/transcriptions",
        files={"audio": ("voice.wav", encode_wav(audio), "audio/wav")},
        data=form or None,
    )


def test_health_and_transcription_contract(monkeypatch) -> None:
    with client(monkeypatch) as api:
        assert api.get("/api/v1/health").status_code == 200
        audio = np.sin(np.linspace(0, 100, 8_000)).astype(np.float32) * 0.1
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(audio), "audio/wav")},
            data={"context": "home"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["hypotheses"][0]["literal_text"] == "I water"
        assert body["messages"][0]["literal_text"] == "I water"
        assert body["messages"][0]["corrected_text"] == "I would like some water."
        assert body["context"] == "home"
        assert body["recommended_message_id"] == "m1"
        assert body["beam_weights_are_calibrated_confidence"] is False
        assert body["user_confirmation_required"] is False


def test_invalid_context_is_rejected(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
            data={"context": "spaceship"},
        )
        assert response.status_code == 422


def test_invalid_and_oversized_audio(monkeypatch) -> None:
    with client(monkeypatch) as api:
        invalid = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.xyz", b"bad", "application/octet-stream")},
        )
        assert invalid.status_code == 422
        main.app.state.settings = SimpleNamespace(
            **{**main.app.state.settings.__dict__, "max_upload_bytes": 3}
        )
        oversized = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", b"1234", "audio/wav")},
        )
        assert oversized.status_code == 413


def test_silence_and_duration_are_rejected(monkeypatch) -> None:
    with client(monkeypatch) as api:
        silence = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("silent.wav", encode_wav(np.zeros(8_000, dtype=np.float32)), "audio/wav")},
        )
        assert silence.status_code == 422
        main.app.state.settings = SimpleNamespace(
            **{**main.app.state.settings.__dict__, "max_audio_seconds": 0.2}
        )
        long_recording = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("long.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
        )
        assert long_recording.status_code == 422


def test_model_and_cloud_worker_unavailable_are_explicit(monkeypatch) -> None:
    monkeypatch.setattr(main, "create_backend", lambda settings: (_ for _ in ()).throw(RuntimeError("missing model")))
    monkeypatch.setattr(main, "GroqMessageChain", FakeChain)
    with TestClient(main.app) as api:
        assert api.get("/api/v1/health").json()["model_ready"] is False
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
        )
        assert response.status_code == 503
        assert "missing model" in response.json()["detail"]

    monkeypatch.setattr(main, "create_backend", lambda settings: FailingBackend())
    with TestClient(main.app) as api:
        response = api.post(
            "/api/v1/transcriptions",
            files={"audio": ("voice.wav", encode_wav(np.ones(8_000, dtype=np.float32) * 0.1), "audio/wav")},
        )
        assert response.status_code == 503
        assert "TimeoutError" in response.json()["detail"]


def test_one_surviving_message_is_synthesized_with_the_result(monkeypatch) -> None:
    """An unambiguous message ships its own audio, so the browser can speak it at once."""
    with client(monkeypatch) as api:
        body = _post(api).json()
        assert body["ranker"]["decision"] == "selected"
        assert body["speech"]["audio_base64"] == "UklGRg=="
        assert body["speech"]["media_type"] == "audio/wav"
        # The realized message is spoken, never the raw literal.
        assert FakeSpeech.spoken == ["I would like some water."]


def test_an_ambiguous_result_is_never_spoken_for_the_speaker(monkeypatch) -> None:
    with client(monkeypatch, chain=AmbiguousChain) as api:
        body = _post(api).json()
        assert body["needs_user_choice"] is True
        assert body["speech"] is None
        assert FakeSpeech.spoken == []


def test_failed_synthesis_still_returns_the_message(monkeypatch) -> None:
    with client(monkeypatch, speech=SilentSpeech) as api:
        response = _post(api)
        assert response.status_code == 200
        body = response.json()
        assert body["speech"] is None
        assert body["messages"][0]["corrected_text"] == "I would like some water."


def test_autoplay_can_be_switched_off(monkeypatch) -> None:
    with client(monkeypatch) as api:
        main.app.state.settings = SimpleNamespace(
            **{**main.app.state.settings.__dict__, "speech_autoplay": False}
        )
        body = _post(api).json()
        assert body["speech"] is None
        assert FakeSpeech.spoken == []


def test_the_speech_endpoint_serves_audio_and_reports_unavailability(monkeypatch) -> None:
    with client(monkeypatch) as api:
        ok = api.post("/api/v1/speech", json={"text": "My leg hurts."})
        assert ok.status_code == 200
        assert ok.json()["audio_base64"] == "UklGRg=="
        assert FakeSpeech.spoken == ["My leg hurts."]

    # A 503 is the signal for the interface to fall back to the browser voice.
    with client(monkeypatch, speech=SilentSpeech) as api:
        assert api.post("/api/v1/speech", json={"text": "My leg hurts."}).status_code == 503


# ---------------------------------------------------------------- personal context


def test_a_profile_reaches_the_message_chain(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = _post(api, context="home", persona="krishnan")
    assert response.status_code == 200
    assert response.json()["persona"] == "krishnan"
    assert FakePersonal.briefs == [("krishnan", "home")]


def test_an_explicit_audience_freezes_listener_and_style(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = _post(
            api,
            context="outdoors",
            persona="krishnan",
            listener="unfamiliar",
            audience="priya",
        )
    assert response.status_code == 200
    body = response.json()
    assert body["listener"] == "familiar"
    assert body["audience"]["audience_id"] == "priya"
    assert body["audience"]["audience_label"] == "Priya"
    assert body["audience"]["listener_source"] == "audience"
    assert body["audience"]["style"]["courtesy"] == "please"
    assert FakeChain.stances[-1] == ("outdoors", "familiar")
    assert FakeChain.styles[-1].formality == "informal"


def test_idle_preview_and_transcription_use_the_same_audience_resolver(monkeypatch) -> None:
    request = {
        "context": "outdoors",
        "persona": "krishnan",
        "audience": "priya",
        "declared_listener": "unfamiliar",
    }
    with client(monkeypatch) as api:
        preview = api.post("/api/v1/audience/resolve", json=request)
        result = _post(
            api,
            context="outdoors",
            persona="krishnan",
            listener="unfamiliar",
            audience="priya",
        )
    assert preview.status_code == 200
    assert result.status_code == 200
    assert preview.json() == result.json()["audience"]


def test_a_request_without_a_profile_never_asks_the_personal_layer(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = _post(api)
    assert response.status_code == 200
    assert response.json()["persona"] is None
    assert FakePersonal.briefs == []


def test_complete_profile_document_round_trips_without_losing_audiences(monkeypatch) -> None:
    document = {
        "label": "Animesh",
        "blurb": "My reviewed profile.",
        "context_default": "home",
        "listener_by_setting": {"outdoors": "unfamiliar"},
        "style": {"brevity": "natural", "courtesy": "plain", "formality": "neutral"},
        "style_by_setting": {
            "care": {"brevity": "complete", "courtesy": "please", "formality": "formal"}
        },
        "lexicon": [
            {
                "id": "user/person/priya",
                "word": "priya",
                "aliases": [],
                "display": "Priya",
                "kind": "person",
            }
        ],
        "specializations": [],
        "audiences": [
            {
                "id": "priya",
                "label": "Priya",
                "relationship": "Daughter",
                "listener": "familiar",
                "style": {"brevity": "natural", "courtesy": "please", "formality": "informal"},
                "visible_in_settings": ["home"],
            }
        ],
    }
    with client(monkeypatch) as api:
        saved = api.put("/api/v1/profile", json=document)
        read = api.get("/api/v1/profile")
    assert saved.status_code == 200
    assert saved.json()["profile"]["audiences"][0]["label"] == "Priya"
    assert read.status_code == 200
    assert read.json()["label"] == "Animesh"
    assert read.json()["audiences"][0]["style"]["formality"] == "informal"


def test_an_unknown_profile_is_a_warning_and_still_speaks(monkeypatch) -> None:
    """A stale value in the interface must never stand between a speaker and their message."""
    with client(monkeypatch) as api:
        response = _post(api, persona="nobody")
    body = response.json()
    assert response.status_code == 200
    assert body["messages"][0]["corrected_text"] == "I would like some water."
    assert any("nobody" in warning for warning in body["warnings"])


def test_the_personal_layer_failing_to_load_leaves_transcription_working(monkeypatch) -> None:
    class Broken:
        def __init__(self, settings):
            raise RuntimeError("profile store unavailable")

    with client(monkeypatch, personal=Broken) as api:
        assert api.get("/api/v1/health").json()["personal_ready"] is False
        assert api.get("/api/v1/personas").json() == []
        response = _post(api, persona="krishnan")
    assert response.status_code == 200
    assert response.json()["messages"][0]["corrected_text"] == "I would like some water."


def test_persistent_accepted_history_endpoint_does_not_exist(monkeypatch) -> None:
    with client(monkeypatch) as api:
        response = api.post(
            "/api/v1/accepted",
            headers={"X-Echora-Client": "1"},
            json={"persona": "krishnan", "heard": "tea", "message": "Tea, please."},
        )
    assert response.status_code == 404


# ------------------------------------------------------------------- places


def test_places_start_at_the_shipped_defaults(monkeypatch, tmp_path) -> None:
    with client(monkeypatch, places_path=tmp_path / "settings.json") as api:
        body = api.get("/api/v1/places").json()
    assert [place["id"] for place in body["places"]] == list(BUILTIN_IDS)
    assert body["auto_detect"] is False


def test_places_round_trip_through_the_api(monkeypatch, tmp_path) -> None:
    path = tmp_path / "settings.json"
    with client(monkeypatch, places_path=path) as api:
        saved = api.post(
            "/api/v1/places",
            json={
                "auto_detect": True,
                "places": [
                    {"id": "home", "label": "Home", "context": "home", "builtin": True,
                     "latitude": 12.9716, "longitude": 80.2594, "radius_m": 150},
                    {"id": "new", "label": "Shopping centre", "context": "outdoors"},
                ],
            },
        )
        assert saved.status_code == 200
        # The response is what was stored, not an echo of the request.
        assert saved.json() == api.get("/api/v1/places").json()

    body = saved.json()
    assert body["auto_detect"] is True
    custom = next(place for place in body["places"] if place["id"] == "shopping-centre")
    assert (custom["context"], custom["builtin"]) == ("outdoors", False)
    assert next(place for place in body["places"] if place["id"] == "home")["tagged_at"]
    assert path.is_file()


def test_the_api_will_not_let_a_request_delete_or_rebind_a_builtin(monkeypatch, tmp_path) -> None:
    with client(monkeypatch, places_path=tmp_path / "settings.json") as api:
        emptied = api.post("/api/v1/places", json={"auto_detect": False, "places": []}).json()
        assert [place["id"] for place in emptied["places"]] == list(BUILTIN_IDS)

        rebound = api.post(
            "/api/v1/places",
            json={"places": [{"id": "home", "label": "Ward 4", "context": "care"}]},
        ).json()
        home = next(place for place in rebound["places"] if place["id"] == "home")
        assert (home["label"], home["context"]) == ("Home", "home")


def test_an_impossible_coordinate_is_rejected(monkeypatch, tmp_path) -> None:
    with client(monkeypatch, places_path=tmp_path / "settings.json") as api:
        response = api.post(
            "/api/v1/places",
            json={"places": [{"id": "x", "label": "Nowhere", "context": "home",
                              "latitude": 900.0, "longitude": 0.0}]},
        )
    assert response.status_code == 422


def test_a_silly_radius_is_clamped_rather_than_rejected(monkeypatch, tmp_path) -> None:
    with client(monkeypatch, places_path=tmp_path / "settings.json") as api:
        body = api.post(
            "/api/v1/places",
            json={"places": [{"id": "x", "label": "Wide", "context": "home",
                              "latitude": 12.9, "longitude": 80.2, "radius_m": 999999}]},
        ).json()
    assert next(place for place in body["places"] if place["id"] == "wide")["radius_m"] == 2000


def test_reading_places_is_never_an_error(monkeypatch, tmp_path) -> None:
    """A document that cannot be read must not stop the interface from opening."""
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    with client(monkeypatch, places_path=path) as api:
        response = api.get("/api/v1/places")
    assert response.status_code == 200
    assert [place["id"] for place in response.json()["places"]] == list(BUILTIN_IDS)


def test_a_place_never_reaches_the_message_chain_as_a_setting(monkeypatch, tmp_path) -> None:
    """A custom place borrows a built-in context; the chain only ever sees the four."""
    with client(monkeypatch, places_path=tmp_path / "settings.json") as api:
        api.post("/api/v1/places", json={"places": [{"id": "x", "label": "Shopping centre",
                                                     "context": "outdoors"}]})
        rejected = _post(api, context="shopping-centre")
        assert rejected.status_code == 422
        accepted = _post(api, context="outdoors")
        assert accepted.status_code == 200
        assert accepted.json()["context"] == "outdoors"


def test_the_setting_alone_decides_who_is_listening(monkeypatch) -> None:
    """No place, no profile: outdoors reaches the chain as strangers, home does not."""
    with client(monkeypatch) as api:
        assert _post(api, context="home").json()["listener"] == "familiar"
        assert _post(api, context="outdoors").json()["listener"] == "unfamiliar"
        assert ("outdoors", "unfamiliar") in FakeChain.stances


def test_a_place_outranks_the_profile_and_the_profile_outranks_the_default(monkeypatch) -> None:
    """Precedence mirrors the places layer: what the speaker tapped wins."""
    with client(monkeypatch) as api:
        # The profile says it is usually family out there.
        FakePersonal.declares = {"outdoors": "familiar"}
        assert _post(api, context="outdoors", persona="krishnan").json()["listener"] == "familiar"
        # A place the speaker tapped says otherwise, and outranks it.
        response = _post(api, context="outdoors", persona="krishnan", listener="unfamiliar")
        assert response.json()["listener"] == "unfamiliar"
        # With neither, the setting's own default stands.
        FakePersonal.declares = {}
        assert _post(api, context="outdoors", persona="krishnan").json()["listener"] == "unfamiliar"


def test_an_unknown_listener_is_refused_but_an_absent_one_is_not(monkeypatch) -> None:
    """The stance is part of the prompt, so a value the server does not know is a 422.

    An empty field is a different thing: it means the caller has nothing to say
    about who is listening, which is the ordinary case for a plain setting.
    """
    with client(monkeypatch) as api:
        assert _post(api, context="home", listener="a-stranger").status_code == 422
        assert _post(api, context="home").status_code == 200


def test_the_stance_reaches_the_personal_layer(monkeypatch) -> None:
    with client(monkeypatch) as api:
        _post(api, context="outdoors", persona="krishnan")
        assert ("outdoors", "unfamiliar") in FakePersonal.stances
