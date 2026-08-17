"""Shared API and inference schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CommunicationContext = Literal["general", "home", "care", "outdoors"]
PersonaKind = Literal["person", "place", "object", "routine", "brand", "food"]


class Hypothesis(BaseModel):
    id: str
    literal_text: str
    sequence_score: float
    search_weight: float = Field(ge=0.0, le=1.0)


class AudioQuality(BaseModel):
    seconds: float
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipped_samples: int
    low_level_warning: bool


class RawAsrResult(BaseModel):
    backend: Literal["local", "pod", "runpod"]
    model: str = "echora-qwen3-asr-command-v3"
    device: str
    hypotheses: list[Hypothesis]
    audio_quality: AudioQuality
    decode_seconds: float


class Specialization(BaseModel):
    """One profile-declared detail applied to a word the recognizer actually produced.

    Nothing here is model-authored. `anchor` is a word that survived the grounding
    check, and `surface` is copied verbatim from the profile, so a specialization
    can be undone mechanically rather than on the model's promise.
    """

    anchor: str
    plain: str
    surface: str
    source: str
    kind: PersonaKind
    profile_id: str


class PersonalizationTrace(BaseModel):
    """What the personal layer offered and what survived, so refusals are observable."""

    profile_id: str
    profile_label: str
    lexicon_hints: list[str] = Field(default_factory=list)
    examples_used: int = 0
    specializations_offered: int = 0
    specializations_applied: int = 0
    specializations_refused: int = 0
    retrieval_seconds: float = 0.0


class LexiconHint(BaseModel):
    """A known word the recognizer produced in this utterance, offered as a prior."""

    word: str
    display: str
    kind: PersonaKind
    note: str = ""


class SpecializationOffer(BaseModel):
    anchor: str
    plain: str
    surface: str
    source: str
    kind: PersonaKind


class PersonalExample(BaseModel):
    heard: str
    message: str
    context: CommunicationContext
    score: float


class PersonalBrief(BaseModel):
    """Everything the personal layer contributes to one request, assembled up front."""

    profile_id: str
    profile_label: str
    speaker_note: str = ""
    lexicon: list[LexiconHint] = Field(default_factory=list)
    specializations: list[SpecializationOffer] = Field(default_factory=list)
    examples: list[PersonalExample] = Field(default_factory=list)
    # Every content word this profile could supply, used only to audit the
    # finished message. Deliberately wider than `lexicon`, which lists just the
    # words this utterance actually contained: the word that needs catching is
    # the one the recognizer never produced, so it is never a hint.
    audit_vocabulary: list[str] = Field(default_factory=list)
    retrieval_seconds: float = 0.0

    def is_empty(self) -> bool:
        return not (self.lexicon or self.specializations or self.examples)

    def offer(self, anchor: str) -> SpecializationOffer | None:
        lowered = anchor.strip().lower()
        for candidate in self.specializations:
            if candidate.anchor.lower() == lowered:
                return candidate
        return None


class PersonaSummary(BaseModel):
    id: str
    label: str
    blurb: str = ""
    icon: str = "circle"
    context_default: CommunicationContext = "general"
    lexicon_size: int = 0
    specialization_size: int = 0
    history_size: int = 0
    baseline: bool = True


class AcceptedMessage(BaseModel):
    """One message the speaker settled on, stored so later utterances can learn from it."""

    id: str
    profile_id: str
    heard: str
    message: str
    context: CommunicationContext = "general"
    hour: int = Field(default=12, ge=0, le=23)
    accepted_at: str
    uses: int = 1
    source: Literal["baseline", "user"] = "user"


class AcceptedMessageRequest(BaseModel):
    persona: str
    context: CommunicationContext = "general"
    heard: str
    message: str


class ProfileDetailInput(BaseModel):
    """One "I have a particular version of this" pair from the onboarding form."""

    word: str
    wording: str


class ProfileRequest(BaseModel):
    """The short onboarding form. Everything else is learned from accepted messages."""

    people: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    things: list[str] = Field(default_factory=list)
    details: list[ProfileDetailInput] = Field(default_factory=list)


class ProfileResponse(BaseModel):
    saved: bool
    lexicon_size: int = 0
    specialization_size: int = 0
    refused: list[str] = Field(default_factory=list)
    reason: str = ""


class AcceptedMessageResponse(BaseModel):
    stored: bool
    merged: bool = False
    reason: str


class RankerDecision(BaseModel):
    decision: Literal["selected", "ambiguous"]
    selected_message_id: str | None = None
    display_hypothesis_ids: list[str]
    display_message_ids: list[str] = Field(default_factory=list)
    reason: str
    source: Literal["groq", "unavailable"]
    assistant_model: str | None = None
    personalization: PersonalizationTrace | None = None


class MessageCandidate(BaseModel):
    message_id: str
    hypothesis_id: str
    source_hypothesis_ids: list[str]
    source_literals: list[str]
    literal_text: str
    interpreted_intent: str
    corrected_text: str
    repair_status: Literal["unchanged", "corrected", "unavailable"]
    repair_note: str
    # Chosen word -> other variants the recognizer produced at that slot, so the
    # interface can mark an uncertain word inline instead of showing rival sentences.
    word_alternatives: dict[str, list[str]] = Field(default_factory=dict)
    # Profile-derived details present in corrected_text. Empty for every message
    # that used no personal knowledge, which is the whole un-personalized path.
    specializations: list[Specialization] = Field(default_factory=list)
    # corrected_text with every applied detail substituted back to its plain
    # wording, derived in code. None when there is nothing to revert, so the
    # interface asks a null check rather than comparing two strings.
    plain_text: str | None = None


class SpeechAudio(BaseModel):
    """Synthesized speech for one message, inlined so it plays without a round trip."""

    audio_base64: str
    media_type: str = "audio/wav"
    voice: str
    model: str


class SpeechRequest(BaseModel):
    text: str


class Timing(BaseModel):
    audio_decode_seconds: float
    asr_seconds: float
    ranking_seconds: float
    grammar_seconds: float
    speech_seconds: float = 0.0
    total_seconds: float


class TranscriptionResponse(BaseModel):
    request_id: str
    backend: str
    model: str
    device: str
    context: CommunicationContext
    persona: str | None = None
    hypotheses: list[Hypothesis]
    audio_quality: AudioQuality
    ranker: RankerDecision
    messages: list[MessageCandidate]
    recommended_message_id: str | None
    needs_user_choice: bool
    # Present only when one message survived the chain: that message is spoken on
    # arrival, so the audio ships with the result rather than costing a round trip.
    speech: SpeechAudio | None = None
    beam_weights_are_calibrated_confidence: bool = False
    # An unambiguous message is spoken as soon as it lands, and choosing among
    # ambiguous options speaks the chosen one. Neither waits for a confirm step.
    user_confirmation_required: bool = False
    timing: Timing
    warnings: list[str]


class RemoteTranscriptionRequest(BaseModel):
    audio_base64: str
    filename: str = "recording.wav"
    beams: int = Field(default=5, ge=2, le=8)


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded", "starting"]
    asr_backend: str
    model_ready: bool
    groq_configured: bool
    personal_ready: bool = False
    detail: str


class ModelInfo(BaseModel):
    name: str
    foundation: str
    adapter_sha256: str
    device: str
    beams: int
    literal_prompt: str
    supported_contexts: list[CommunicationContext]
    beam_weights_are_calibrated_confidence: bool = False
    semantic_repair_is_part_of_asr: bool = False
    limitations: list[str]
