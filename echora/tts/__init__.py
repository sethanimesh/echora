"""Pronunciation normalization and optional speech synthesis backends."""

from echora.tts.fish_audio import (
    AlignmentSegment,
    DEFAULT_FISH_VOICE_ID,
    FishAudioSynthesizer,
    SpeechError,
    SpeechEvent,
    SpeechOptions,
    SpeechStyle,
    SynthesisResult,
    VoiceEnrollment,
)

__all__ = [
    "AlignmentSegment",
    "DEFAULT_FISH_VOICE_ID",
    "FishAudioSynthesizer",
    "SpeechError",
    "SpeechEvent",
    "SpeechOptions",
    "SpeechStyle",
    "SynthesisResult",
    "VoiceEnrollment",
]
