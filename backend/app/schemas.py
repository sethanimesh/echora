"""Shared API and inference schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CommunicationContext = Literal["general", "home", "care", "outdoors"]
PersonaKind = Literal["person", "place", "object", "routine", "brand", "food"]

# Who is listening. The setting says where the speaker is; this says whether the
# person they are speaking to knows them, and that is what decides the act. A
# familiar listener can fetch and do, so a need is stated to them. An unfamiliar
# one can only answer, point or serve, so the same need is asked of them:
# "washroom" is "I want to use the washroom." at home and "Where is the
# washroom?" on a platform. Two values, orthogonal to the four settings -- a
# place still borrows one of those four and never becomes a fifth.
Listener = Literal["familiar", "unfamiliar"]
AudienceKind = Literal["person", "role", "generic"]
RegisterBrevity = Literal["short", "natural", "complete"]
RegisterCourtesy = Literal["plain", "please"]
RegisterFormality = Literal["informal", "neutral", "formal"]

# The listener a setting implies when nothing overrides it. Only `outdoors`
# assumes strangers; everywhere else someone who knows the speaker is present,
# which is what every setting assumed before this layer existed.
LISTENER_DEFAULTS: dict[CommunicationContext, Listener] = {
    "general": "familiar",
    "home": "familiar",
    "care": "familiar",
    "outdoors": "unfamiliar",
}


def default_listener(context: CommunicationContext) -> Listener:
    return LISTENER_DEFAULTS.get(context, "familiar")


def resolve_listener(
    context: CommunicationContext,
    by_audience: Listener | None = None,
    declared: Listener | None = None,
    by_profile: Listener | None = None,
) -> Listener:
    """Who is listening, strongest claim first.

    `by_audience` is a person or audience the speaker explicitly tapped for this
    session. It is the strongest claim because it identifies who is actually in
    front of them. `declared` is what the selected place says when no audience
    was chosen. `by_profile` is what their profile says this setting usually
    means. None is required, and the setting default ends the chain.

    It lives here, as one expression, because every caller that recomputes it
    gets a chance to disagree -- and a caller that fills in the default in place
    of an absent `declared` silently deletes the middle rung.
    """
    return by_audience or declared or by_profile or default_listener(context)


class CommunicationRegister(BaseModel):
    """Bounded wording choices; never a licence to invent social content."""

    brevity: RegisterBrevity = "natural"
    courtesy: RegisterCourtesy = "plain"
    formality: RegisterFormality = "neutral"


DEFAULT_REGISTER = CommunicationRegister()


class ResolvedAudience(BaseModel):
    """The frozen audience/register snapshot used for one utterance."""

    audience_id: str | None = None
    audience_label: str
    audience_kind: AudienceKind = "generic"
    listener: Listener
    style: CommunicationRegister = Field(default_factory=CommunicationRegister)
    listener_source: Literal["audience", "place", "profile", "setting"]
    style_source: Literal[
        "audience_setting", "audience", "profile_setting", "profile", "default"
    ]


class AudienceResolutionRequest(BaseModel):
    context: CommunicationContext = "general"
    persona: str = ""
    audience: str = ""
    # Null is the selected place abstaining, exactly as in transcription.
    declared_listener: Listener | None = None

# A tagged place is matched by distance, so the radius is the whole match rule.
# 150 m covers a house or a ward without swallowing the next street.
DEFAULT_RADIUS_M = 150
MIN_RADIUS_M = 25
MAX_RADIUS_M = 2000


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
    """What explicit profile context was eligible and deterministically applied."""

    profile_id: str
    profile_label: str
    lexicon_hints: list[str] = Field(default_factory=list)
    specializations_offered: int = 0
    specializations_applied: int = 0


class LexiconHint(BaseModel):
    """A known word the recognizer produced in this utterance, offered as a prior."""

    word: str
    display: str
    kind: PersonaKind
    note: str = ""


class SpecializationOffer(BaseModel):
    anchor: str
    # Recognizer spellings that contributed to the same declared anchor. These
    # are explicit conservative equivalences, not free-form fuzzy matching.
    matches: list[str] = Field(default_factory=list)
    plain: str
    surface: str
    source: str
    kind: PersonaKind


class PersonalBrief(BaseModel):
    """Everything the personal layer contributes to one request, assembled up front."""

    profile_id: str
    profile_label: str
    speaker_note: str = ""
    lexicon: list[LexiconHint] = Field(default_factory=list)
    specializations: list[SpecializationOffer] = Field(default_factory=list)
    # Every content word this profile could supply, used only to audit the
    # finished message. Deliberately wider than `lexicon`, which lists just the
    # words this utterance actually contained: the word that needs catching is
    # the one the recognizer never produced, so it is never a hint.
    audit_vocabulary: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.speaker_note or self.lexicon or self.specializations)

    def offer(self, anchor: str) -> SpecializationOffer | None:
        lowered = anchor.strip().lower()
        for candidate in self.specializations:
            if candidate.anchor.lower() == lowered or lowered in candidate.matches:
                return candidate
        return None


class PersonaSummary(BaseModel):
    id: str
    label: str
    blurb: str = ""
    icon: str = "circle"
    context_default: CommunicationContext = "general"
    listener_by_setting: dict[CommunicationContext, Listener] = Field(default_factory=dict)
    lexicon_size: int = 0
    specialization_size: int = 0
    audience_size: int = 0
    baseline: bool = True


class Place(BaseModel):
    """A named place that resolves to one of the four communication contexts.

    A place is a label and an optional location. It never introduces a new
    context value: `context` is the built-in whose prior and specialization scope the
    place borrows, so a custom place behaves exactly as that built-in already
    does. `listener` is the one thing a place says on its own account, and it is
    orthogonal to the four: two places can both borrow `outdoors` and still meet
    different people. `latitude`/`longitude` stay None until the speaker tags the
    place while standing in it.
    """

    id: str
    label: str
    context: CommunicationContext
    # Whether the people here know the speaker. Defaults from the borrowed
    # context and is then the speaker's to set, including on the built-ins: a
    # speaker who only ever goes out with their daughter needs the shipped
    # Outdoors place to keep the familiar register.
    listener: Listener | None = None
    builtin: bool = False
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    # Deliberately unbounded here and clamped in `places.normalize` instead: a
    # hand-edited radius should be corrected, not rejected, because rejecting it
    # drops the whole place and loses a location tag with it.
    radius_m: int = DEFAULT_RADIUS_M
    tagged_at: str | None = None

    @property
    def tagged(self) -> bool:
        return self.latitude is not None and self.longitude is not None


class PlaceSettings(BaseModel):
    """The whole places document: the detection switch and every known place."""

    auto_detect: bool = False
    places: list[Place] = Field(default_factory=list)
    updated_at: str | None = None


class PlaceSettingsRequest(BaseModel):
    """A full replacement of the document. The server normalizes before storing."""

    auto_detect: bool = False
    places: list[Place] = Field(default_factory=list)


class ProfileDetailInput(BaseModel):
    """One "I have a particular version of this" pair from the onboarding form."""

    word: str
    wording: str


class ProfileRequest(BaseModel):
    """The short onboarding form for explicit speaker-controlled context."""

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
    # Retrieved sources are references considered by the one composition pass;
    # additions list only exact, reversible profile wording actually applied.
    retrieved_sources: list[dict] = Field(default_factory=list)
    contextual_additions: list[dict] = Field(default_factory=list)


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
    listener: Listener = "familiar"
    audience: ResolvedAudience
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
    supported_listeners: list[Listener]
    beam_weights_are_calibrated_confidence: bool = False
    semantic_repair_is_part_of_asr: bool = False
    limitations: list[str]
