"""The optional Haystack voice boundary stays lazy and injectable."""

import pytest
pytest.importorskip("haystack", reason="Install the optional Haystack extra to exercise framework adapters")

from echora.config import Config
from echora.haystack_components.components import build_voice_pipeline
from echora.tts.fish_audio import SpeechEvent


def test_voice_pipeline_returns_lazy_speech_stream():
    calls = []

    class _Synth:
        def stream(self, speech, options=None):
            calls.append(speech)
            yield SpeechEvent(b"audio", speech, 0, ())

    pipeline = build_voice_pipeline(
        Config(classifier="offline"), synthesizer=_Synth()
    )
    output = pipeline.run({"segmenter": {"text": "Main kal jaunga"}})
    stream = output["fish"]["speech_stream"]
    assert calls == [], "network stream must stay lazy until consumed"
    assert b"".join(event.audio for event in stream) == b"audio"
    assert calls and "कल" in calls[0]
