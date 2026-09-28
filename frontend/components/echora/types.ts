import type { ContextSelection, Profile } from './contextTypes';
import type { VoiceSuggestionResult } from './VoiceSuggestion';
import type { FaceSuggestionResult } from './FaceSuggestion';
import type { Delivery } from './delivery';
export type RecognitionBackend = 'adapted' | 'whisper';
export type RecognitionOption = {
  id: RecognitionBackend;
  label: string;
  ready: boolean;
  model: string;
  message?: string;
};
export type Candidate = {
  available?: boolean;
  id: string;
  text: string;
  reading: string;
  source_hypothesis_ids: string[];
  source_literals: string[];
  plain_text?: string | null;
  specializations?: { anchor: string; surface: string }[];
  retrieved_sources?: ContextSource[];
  contextual_additions?: ContextAddition[];
};
export type ContextSource = {
  id?: string;
  source_id?: string;
  entry_id?: string;
  kind?: string;
  text?: string;
  message?: string;
  label?: string;
};
export type ContextAddition = {
  anchor?: string;
  text?: string;
  wording?: string;
  surface?: string;
  source_id?: string;
  plain?: string;
};
export type HypothesisRanking = {
  status: string;
  route?: string;
  decision: 'selected' | 'ambiguous';
  selected_hypothesis_id: string | null;
  ordered_hypothesis_ids?: string[];
  reason: string;
  scores?: unknown;
  artifacts?: unknown;
};
export type Job = {
  auto_speak_revision?: number | null;
  ranking?: HypothesisRanking | null;
  selected_candidate_id?: string | null;
  decision_source?: string;
  evidence?: {
    kind: string;
    backend: string;
    model: string;
    hypotheses: {
      id: string;
      literal_text: string;
      sequence_score?: number;
      search_weight?: number;
    }[];
  } | null;
  candidates?: Candidate[];
  id: string;
  revision: number;
  created_at: number;
  status:
    | 'review'
    | 'transcribing'
    | 'drafting'
    | 'preparing'
    | 'synthesizing'
    | 'suggesting'
    | 'checking_face'
    | 'analyzing_delivery'
    | 'confirmed'
    | 'cancelled';
  modality: string;
  original: string;
  source_text: string;
  conversation?: { text: string; job_id: string } | null;
  independent?: boolean;
  output_script?: 'auto' | 'latin' | 'devanagari';
  context?: {
    selection: ContextSelection;
    profile: Profile | null;
    core_context?: string;
    resolved_audience?: {
      audience_label: string;
      listener: string;
      style: { brevity: string; courtesy: string; formality: string };
      listener_source: string;
      style_source: string;
    };
  } | null;
  context_trace?: {
    scenario: string;
    recipient: string;
    listener: string;
    situation: string;
    profile_label: string;
    moment_title?: string;
    applied_details: { anchor: string; wording: string }[];
  } | null;
  text: string;
  question: string;
  options: string[];
  language: string;
  output_language: string;
  error: { code: string; message: string; stage?: string } | null;
  metadata: Record<
    string,
    { model: string; elapsed_ms: number; audio_seconds?: number }
  >;
  delivery_suggestion?: VoiceSuggestionResult;
  face_suggestion?: FaceSuggestionResult;
  analysis_id?: string;
  synthesis?: {
    state: 'generating' | 'ready' | 'error' | 'stopped';
    confirmation_id: string;
    message?: string;
  };
  prepared_speech?: {
    revision: number;
    display_text: string;
    pronunciation_text: string;
    speech_text: string;
    changes: { original: string; pronunciation: string }[];
    metadata: { model: string; elapsed_ms: number; ambiguous_spans: number };
  } | null;
  confirmed: {
    id: string;
    text: string;
    speech_text?: string;
    version: number;
    delivery?: Delivery;
  } | null;
};
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const form = body instanceof FormData;
  const r = await fetch('/api' + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers:
      body === undefined
        ? {}
        : {
            'X-Echora-Client': '1',
            ...(form ? {} : { 'Content-Type': 'application/json' }),
          },
    ...(body === undefined ? {} : { body: form ? body : JSON.stringify(body) }),
  });
  let data;
  try {
    data = await r.json();
  } catch {
    throw new Error(
      'The communication service is unavailable. Please restart it.',
    );
  }
  const detail = (data as { detail?: unknown })?.detail;
  if (!r.ok)
    throw new Error(
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail) && typeof detail[0]?.msg === 'string'
          ? detail[0].msg.replace(/^Value error, /, '')
          : 'The request could not be completed. Please review your input.',
    );
  return data as T;
}
