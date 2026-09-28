"""Fish Audio transport, public types, CLI helpers, and live smoke coverage."""

from __future__ import annotations

import base64
import builtins
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from echora.cli import (
    _highlight_stream, _process_text, _resolve_audio_format, _run_enrollment,
    _set_dotenv_value, build_parser,
)
from echora.config import Config
from echora.pipeline import Pipeline
from echora.tts.fish_audio import (
    DEFAULT_FISH_VOICE_ID,
    FISH_MODEL,
    FishAudioSynthesizer,
    SpeechError,
    SpeechEvent,
    SpeechOptions,
    SpeechStyle,
    VoiceEnrollment,
)


def _event(
    audio: bytes,
    *,
    seq: int = 0,
    offset: float = 0.0,
    segments: list[dict] | None = None,
) -> str:
    return "data: " + json.dumps({
        "audio_base64": base64.b64encode(audio).decode(),
        "content": "hello world",
        "alignment": None if segments is None else {
            "audio_duration": 1.0,
            "segments": segments,
        },
        "chunk_seq": seq,
        "chunk_audio_offset_sec": offset,
    })


class _Response:
    def __init__(self, lines=(), *, status=200, error=None):
        self._lines = list(lines)
        self.status_code = status
        self._error = error or {"message": "failed"}
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def json(self):
        return self._error

    def close(self):
        self.closed = True


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_stream_orders_audio_and_offsets_alignment():
    response = _Response([
        _event(b"a", segments=None),
        _event(b"b", segments=[{"text": "hello", "start": 0, "end": .4}]),
        _event(
            b"c", seq=1, offset=1.5,
            segments=[{"text": "world", "start": .1, "end": .6}],
        ),
    ])
    session = _Session(response)
    client = FishAudioSynthesizer(api_key="key", session=session)

    result = client.synthesize("hello world")

    assert result.audio == b"abc"
    assert [segment.text for segment in result.alignment] == ["hello", "world"]
    assert result.alignment[1].start == pytest.approx(1.6)
    assert response.closed
    _, request = session.calls[0]
    assert request["headers"]["model"] == FISH_MODEL == "s2.1-pro-free"
    assert request["json"]["normalize"] is False
    assert request["json"]["condition_on_previous_chunks"] is True
    assert request["json"]["reference_id"] == DEFAULT_FISH_VOICE_ID
    assert request["json"]["text"] == (
        "[calm, soothing, natural conversational delivery] hello world"
    )


def test_latest_snapshot_replaces_earlier_snapshot():
    response = _Response([
        _event(b"a", segments=[{"text": "old", "start": 0, "end": .2}]),
        _event(b"b", segments=[
            {"text": "new", "start": 0, "end": .2},
            {"text": "words", "start": .2, "end": .6},
        ]),
    ])
    result = FishAudioSynthesizer(
        api_key="key", session=_Session(response)
    ).synthesize("hello")
    assert [segment.text for segment in result.alignment] == ["new", "words"]


def test_style_is_composed_and_voice_is_forwarded():
    response = _Response([_event(b"audio")])
    session = _Session(response)
    client = FishAudioSynthesizer(
        api_key="key", voice_id="configured", session=session
    )
    list(client.stream("नमस्ते friend", SpeechOptions(style=SpeechStyle.WARM)))
    payload = session.calls[0][1]["json"]
    assert payload["text"] == "[warm, friendly tone] नमस्ते friend"
    assert payload["reference_id"] == "configured"


@pytest.mark.parametrize("style", list(SpeechStyle))
def test_every_style_is_valid(style):
    text = SpeechOptions(style=style).compose("Hello")
    if style is SpeechStyle.NEUTRAL:
        assert text == "Hello"
    else:
        assert text.startswith("[")


def test_simple_style_and_bilingual_female_voice_are_defaults(monkeypatch):
    monkeypatch.delenv("ECHORA_FISH_VOICE_ID", raising=False)
    options = SpeechOptions()
    assert options.style is SpeechStyle.SIMPLE
    assert options.compose("Hello") == (
        "[calm, soothing, natural conversational delivery] Hello"
    )
    assert Config.from_env(dotenv=None).fish_voice_id == DEFAULT_FISH_VOICE_ID


def test_directions_and_message_brackets_are_validated():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SpeechOptions(style=SpeechStyle.WARM, direction="gentle")
    with pytest.raises(ValueError, match="square brackets"):
        SpeechOptions(direction="[angry]")
    with pytest.raises(ValueError, match="square brackets"):
        SpeechOptions().compose("Please say [urgent]")
    with pytest.raises(ValueError, match="speed"):
        SpeechOptions(speed=2.1)
    with pytest.raises(ValueError, match="volume"):
        SpeechOptions(volume=-21)


def test_protocol_and_provider_errors_are_typed():
    malformed = FishAudioSynthesizer(
        api_key="key", session=_Session(_Response(["data: not-json"]))
    )
    with pytest.raises(SpeechError) as caught:
        list(malformed.stream("hello"))
    assert caught.value.kind == "protocol"

    limited = FishAudioSynthesizer(
        api_key="key",
        session=_Session(_Response(status=429, error={"message": "slow down"})),
    )
    with pytest.raises(SpeechError) as caught:
        list(limited.stream("hello"))
    assert caught.value.kind == "rate_limit"
    assert caught.value.status_code == 429

    empty = FishAudioSynthesizer(api_key="key", session=_Session(_Response([])))
    with pytest.raises(SpeechError) as caught:
        empty.synthesize("hello")
    assert caught.value.kind == "protocol"


def test_voice_enrollment_is_private_and_returns_id(tmp_path):
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"wav")
    calls = []

    class _Voices:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="voice-123", title="My voice", state="created")

    fake = SimpleNamespace(voices=_Voices())
    enrolled = FishAudioSynthesizer(
        api_key="key", voice_client=fake
    ).enroll_voice([sample], ["Exact transcript."], title="My voice")
    assert enrolled.id == "voice-123"
    assert calls[0]["visibility"] == "private"
    assert calls[0]["train_mode"] == "fast"
    assert calls[0]["enhance_audio_quality"] is True


def test_cli_audio_and_timestamp_outputs_are_atomic(tmp_path, monkeypatch, capsys):
    audio_path = tmp_path / "speech.opus"
    timestamps_path = tmp_path / "speech.json"
    args = build_parser().parse_args([
        "-c", "offline", "--json", "--audio-out", str(audio_path),
        "--timestamps-out", str(timestamps_path), "--style", "warm",
        "Main kal jaunga",
    ])
    event = SpeechEvent(
        audio=b"opus",
        content="hello",
        chunk_seq=0,
        alignment=(
            SimpleNamespace(text="hello", start=0.0, end=0.5, chunk_seq=0),
        ),
    )

    class _Synth:
        def stream(self, speech, options):
            assert speech != "Main kal jaunga", "normalization must happen first"
            assert options.style is SpeechStyle.WARM
            yield event

    config = Config(classifier="offline")
    monkeypatch.setattr(Config, "build_synthesizer", lambda self: _Synth())
    status = _process_text(Pipeline.from_config(config), "Main kal jaunga", args, config)
    assert status == 0
    assert audio_path.read_bytes() == b"opus"
    assert json.loads(timestamps_path.read_text())["segments"][0]["text"] == "hello"
    output = json.loads(capsys.readouterr().out)
    assert output["audio"]["format"] == "opus"


def test_cli_hard_failure_prints_speech_and_creates_no_file(
    tmp_path, monkeypatch, capsys
):
    audio_path = tmp_path / "never.mp3"
    args = build_parser().parse_args([
        "-c", "offline", "--audio-out", str(audio_path), "Main kal jaunga"
    ])

    class _Synth:
        def stream(self, speech, options):
            raise SpeechError("rate limited", kind="rate_limit")
            yield  # pragma: no cover - make this a generator

    config = Config(classifier="offline")
    monkeypatch.setattr(Config, "build_synthesizer", lambda self: _Synth())
    status = _process_text(Pipeline.from_config(config), "Main kal jaunga", args, config)
    captured = capsys.readouterr()
    assert status == 1
    assert captured.out.strip()
    assert "rate limited" in captured.err
    assert not audio_path.exists()


def test_cli_format_inference_and_conflicts(tmp_path):
    inferred = build_parser().parse_args([
        "--audio-out", str(tmp_path / "speech.wav"), "hello"
    ])
    assert _resolve_audio_format(inferred) == "wav"
    conflict = build_parser().parse_args([
        "--audio-out", str(tmp_path / "speech.wav"), "--format", "mp3", "hello"
    ])
    with pytest.raises(ValueError, match="conflicts"):
        _resolve_audio_format(conflict)


def test_approximate_highlight_prints_only_new_snapshot_words(capsys):
    first = SpeechEvent(
        b"a", "hello", 0,
        (SimpleNamespace(text="hello"),),
    )
    second = SpeechEvent(
        b"b", "hello world", 0,
        (SimpleNamespace(text="hello"), SimpleNamespace(text="world")),
    )
    collector = {"audio": [], "snapshots": {}}
    assert b"".join(_highlight_stream(
        [first, second], enabled=True, collector=collector
    )) == b"ab"
    assert capsys.readouterr().err.split() == ["hello", "world"]


def test_enrollment_cli_can_set_default_without_replacing_env(
    tmp_path, monkeypatch, capsys
):
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"audio")
    (tmp_path / ".env").write_text("KEEP=this\nECHORA_FISH_VOICE_ID=old\n")
    args = build_parser().parse_args([
        "--enroll-voice", str(sample), "--voice-transcript", "Hello.",
        "--voice-title", "Mine", "--confirm-consent", "--set-default",
    ])

    class _Synth:
        def enroll_voice(self, paths, transcripts, *, title):
            return VoiceEnrollment("new-id", title, "created")

    config = Config(classifier="offline")
    monkeypatch.setattr(Config, "build_synthesizer", lambda self: _Synth())
    monkeypatch.chdir(tmp_path)
    assert _run_enrollment(args, config) == 0
    dotenv = (tmp_path / ".env").read_text()
    assert "KEEP=this" in dotenv
    assert dotenv.count("ECHORA_FISH_VOICE_ID=") == 1
    assert "ECHORA_FISH_VOICE_ID=new-id" in dotenv
    assert capsys.readouterr().out.strip() == "new-id"


def test_enrollment_requires_interactive_confirmation(tmp_path, monkeypatch):
    sample = tmp_path / "voice.wav"
    sample.write_bytes(b"audio")
    args = build_parser().parse_args([
        "--enroll-voice", str(sample), "--voice-transcript", "Hello.",
        "--voice-title", "Mine",
    ])
    called = []

    class _Synth:
        def enroll_voice(self, paths, transcripts, *, title):
            called.append(True)
            return VoiceEnrollment("id", title, "created")

    class _TTY:
        def isatty(self):
            return True

    monkeypatch.setattr(Config, "build_synthesizer", lambda self: _Synth())
    monkeypatch.setattr("sys.stdin", _TTY())
    monkeypatch.setattr(builtins, "input", lambda prompt: "yes")
    assert _run_enrollment(args, Config(classifier="offline")) == 0
    assert called == [True]


def test_dotenv_adds_missing_key(tmp_path):
    path = tmp_path / ".env"
    _set_dotenv_value(path, "ECHORA_FISH_VOICE_ID", "abc")
    assert path.read_text() == "ECHORA_FISH_VOICE_ID=abc\n"


@pytest.mark.skipif(
    not (os.environ.get("FISH_API_KEY") and os.environ.get("ECHORA_RUN_FISH_LIVE") == "1"),
    reason="set FISH_API_KEY and ECHORA_RUN_FISH_LIVE=1 for the free API smoke test",
)
def test_live_free_tts_smoke():
    result = FishAudioSynthesizer().synthesize(
        "नमस्ते, this is an Echora smoke test.",
        SpeechOptions(format="opus", latency="balanced"),
    )
    assert result.audio
    assert result.alignment
