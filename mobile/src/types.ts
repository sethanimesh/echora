export type CommunicationContext = "general" | "home" | "care" | "outdoors";
export type Listener = "familiar" | "unfamiliar";
export type AudienceKind = "person" | "role" | "generic";

export type CommunicationStyle = {
  brevity: "short" | "natural" | "complete";
  courtesy: "plain" | "please";
  formality: "informal" | "neutral" | "formal";
};

export type AudienceProfile = {
  id: string;
  label: string;
  kind: AudienceKind;
  relationship: string;
  icon: string;
  listener: Listener;
  style: CommunicationStyle;
  style_by_setting: Partial<Record<CommunicationContext, CommunicationStyle>>;
  visible_in_settings: CommunicationContext[];
  visible_in_places: string[];
  known_detail_ids: string[];
};

export type LexiconEntry = {
  id: string;
  word: string;
  aliases: string[];
  display: string;
  kind: "person" | "place" | "object" | "routine" | "brand" | "food";
  note: string;
  settings: CommunicationContext[];
};

export type SpecializationRule = {
  id: string;
  anchor: string;
  plain: string;
  surface: string;
  kind: LexiconEntry["kind"];
  note: string;
  settings: CommunicationContext[];
};

export type PersonaProfile = {
  revision?: number;
  id: string;
  label: string;
  blurb: string;
  icon: string;
  baseline: boolean;
  context_default: CommunicationContext;
  listener_by_setting: Partial<Record<CommunicationContext, Listener>>;
  style: CommunicationStyle;
  style_by_setting: Partial<Record<CommunicationContext, CommunicationStyle>>;
  speaker_note: string;
  lexicon: LexiconEntry[];
  specializations: SpecializationRule[];
  audiences: AudienceProfile[];
  created_at: string;
};

export type ResolvedAudience = {
  audience_id: string | null;
  audience_label: string;
  audience_kind: AudienceKind;
  listener: Listener;
  style: CommunicationStyle;
  listener_source: "audience" | "place" | "profile" | "setting";
  style_source: "audience_setting" | "audience" | "profile_setting" | "profile" | "default";
};

export type Hypothesis = {
  id: string;
  literal_text: string;
  sequence_score: number;
  search_weight: number;
};

export type Specialization = {
  anchor: string;
  plain: string;
  surface: string;
  source: string;
  kind: string;
  profile_id: string;
};

export type PersonalizationTrace = {
  profile_id: string;
  profile_label: string;
  lexicon_hints: string[];
  specializations_offered: number;
  specializations_applied: number;
};

export type PersonaSummary = {
  revision?: number;
  id: string;
  label: string;
  blurb: string;
  icon: string;
  context_default: CommunicationContext;
  listener_by_setting: Partial<Record<CommunicationContext, Listener>>;
  lexicon_size: number;
  specialization_size: number;
  audience_size: number;
  baseline: boolean;
};

export type Place = {
  id: string;
  label: string;
  context: CommunicationContext;
  listener: Listener | null;
  builtin: boolean;
  latitude: number | null;
  longitude: number | null;
  radius_m: number;
  tagged_at: string | null;
};

export type PlaceSettings = {
  auto_detect: boolean;
  places: Place[];
  updated_at: string | null;
};

export type MessageCandidate = {
  message_id: string;
  hypothesis_id: string;
  source_hypothesis_ids: string[];
  source_literals: string[];
  literal_text: string;
  interpreted_intent: string;
  corrected_text: string;
  repair_status: "unchanged" | "corrected" | "unavailable";
  repair_note: string;
  word_alternatives: Record<string, string[]>;
  specializations: Specialization[];
  plain_text: string | null;
  retrieved_sources?: ContextSource[];
  contextual_additions?: ContextAddition[];
};

export type ContextSource = { id?: string; source_id?: string; entry_id?: string; kind?: string; text?: string; message?: string; label?: string };
export type ContextAddition = { anchor?: string; text?: string; wording?: string; surface?: string; source_id?: string };
export type HypothesisRanking = {
  status: string;
  route?: string;
  decision: "selected" | "ambiguous";
  selected_hypothesis_id: string | null;
  ordered_hypothesis_ids?: string[];
  reason: string;
  scores?: unknown;
  artifacts?: unknown;
};
export type SavedMemory = {
  id: string;
  message: string;
  language: string;
  created_at: number;
  scope: { scenario?: string; core_context?: string; recipient?: string; audience_id?: string; listener?: string };
  profile_revision: number;
};
export type MemoryList = { profile_id: string; memory_revision: number; memories: SavedMemory[] };
export type RememberResult = { memory_id: string; memory_revision: number; message_revision: number; created: boolean; memory: SavedMemory };

export type SpeechAudio = {
  audio_base64: string;
  media_type: string;
  voice: string;
  model: string;
};

export type Transcription = {
  auto_speak?: boolean;
  ranking?: HypothesisRanking | null;
  decision_source?: string;
  job_revision: number;
  question?: string;
  conversation_reference?: string | null;
  request_id: string;
  backend: string;
  model: string;
  device: string;
  context: CommunicationContext;
  listener: Listener;
  audience: ResolvedAudience;
  persona: string | null;
  hypotheses: Hypothesis[];
  audio_quality: {
    seconds: number;
    peak_dbfs: number | null;
    rms_dbfs: number | null;
    clipped_samples: number;
    low_level_warning: boolean;
  };
  ranker: {
    decision: "selected" | "ambiguous";
    selected_message_id: string | null;
    display_hypothesis_ids: string[];
    display_message_ids: string[];
    reason: string;
    source: "groq" | "unavailable";
    assistant_model: string | null;
    personalization: PersonalizationTrace | null;
  };
  messages: MessageCandidate[];
  recommended_message_id: string | null;
  needs_user_choice: boolean;
  speech: SpeechAudio | null;
  timing: {
    audio_decode_seconds: number;
    asr_seconds: number;
    ranking_seconds: number;
    grammar_seconds: number;
    speech_seconds: number;
    total_seconds: number;
  };
  warnings: string[];
};

export type Health = {
  status: "ready" | "degraded" | "starting";
  asr_backend: string;
  model_ready: boolean;
  groq_configured: boolean;
  personal_ready: boolean;
  detail: string;
};

export type ProfileResponse = {
  saved: boolean;
  lexicon_size: number;
  specialization_size: number;
  refused: string[];
  reason: string;
};

export type ProfileSaveResponse = {
  profile: PersonaProfile;
  refused: string[];
};
