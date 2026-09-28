import { fetch } from "expo/fetch";
import { File } from "expo-file-system";
import { NativeModules } from "react-native";
import type { CommunicationContext, Health, Listener, PersonaProfile, PersonaSummary, PlaceSettings, ProfileSaveResponse, ResolvedAudience, SpeechAudio, Transcription, HypothesisRanking, ContextSource, ContextAddition, MemoryList, RememberResult } from "./types";

const configuredUrl = process.env.EXPO_PUBLIC_ECHORA_API_URL?.trim();
function developmentHost(): string {
  try {
    const scriptUrl = NativeModules.SourceCode?.scriptURL as string | undefined;
    if (scriptUrl) return new URL(scriptUrl).hostname;
  } catch { /* An installed app can use the explicit URL or simulator fallback. */ }
  return "127.0.0.1";
}
export const API_URL = (configuredUrl || `http://${developmentHost()}:8000`).replace(/\/$/, "");
const NATIVE = `${API_URL}/api/v1/communication`;
let token = "";
let bootstrapPending: Promise<void> | null = null;
let requestGeneration = 0;
let activeJob: Job | null = null;
let editQueue: Promise<void> = Promise.resolve();
const profileRevisions = new Map<string, number>();

type Candidate = { id: string; text: string; reading?: string; source_hypothesis_ids?: string[]; source_literals?: string[]; plain_text?: string | null; specializations?: Transcription["messages"][number]["specializations"]; word_alternatives?: Record<string, string[]>; available?: boolean; retrieved_sources?: ContextSource[]; contextual_additions?: ContextAddition[] };
type Job = { id: string; revision: number; status: string; original: string; source_text: string; text: string; question: string; options: string[]; error?: {message: string} | null; candidates: Candidate[]; selected_candidate_id?: string; auto_speak_revision?: number | null; ranking?: HypothesisRanking | null; decision_source?: string; evidence: { backend: string; model?: string; hypotheses: Transcription["hypotheses"]; audio_quality?: Transcription["audio_quality"] }; context?: { selection: {profile_id?: string}; core_context: CommunicationContext; resolved_audience: ResolvedAudience } | null; warnings?: string[]; conversation?: {text: string} | null; confirmed?: {id: string; speech_text: string} | null };

async function bodyOrError<T>(response: Response): Promise<T> {
  const body = await response.json() as T & { detail?: string };
  if (!response.ok) throw new Error(body.detail || `Echora returned ${response.status}`);
  return body;
}
async function session(): Promise<void> {
  if (token) return;
  if (!bootstrapPending) bootstrapPending = (async () => {
    const body = await bodyOrError<{session_token: string}>(await fetch(`${NATIVE}/session`));
    token = body.session_token;
  })().finally(() => { bootstrapPending = null; });
  await bootstrapPending;
}
async function request<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  await session();
  const multipart = body instanceof FormData;
  const response = await fetch(`${NATIVE}${path}`, {
    method, headers: {"X-Echora-Session": token, "X-Echora-Client": "1", ...(body && !multipart ? {"content-type": "application/json"} : {})},
    body: body === undefined ? undefined : multipart ? body as FormData : JSON.stringify(body),
  });
  if (response.status === 401) token = "";
  return bodyOrError<T>(response);
}
function requireCurrent(job: Job) {
  if (!activeJob || activeJob.id !== job.id || job.revision < activeJob.revision) throw new Error("This message is no longer active.");
}
function update(job: Job) { requireCurrent(job); activeJob = job; return job; }
function display(job: Job): Transcription {
  const hypotheses = job.evidence.hypotheses || [];
  const fallback: Candidate[] = job.options?.length ? job.options.map((text, index) => ({id: `option-${index}`, text})) : [{id: "raw", text: job.text || job.original, available: false}];
  const candidates = job.options?.length ? fallback : job.candidates?.length ? job.candidates : fallback;
  const canSpeak = job.auto_speak_revision === job.revision && !job.error && !job.question && ["review", "confirmed"].includes(job.status);
  const userEdited = job.decision_source === "user" && !job.candidates?.length && !job.options?.length;
  const selectedId = candidates.find(candidate => candidate.id === job.selected_candidate_id)?.id || (canSpeak || userEdited ? candidates[0]?.id : null);
  const machineUnresolved = job.ranking?.decision === "ambiguous" && job.ranking.route !== "legacy";
  const reviewed = Boolean(selectedId && job.status !== "cancelled" && !job.error && !job.question && (!machineUnresolved || job.decision_source === "user"));
  const audience = job.context?.resolved_audience || {audience_id: null, audience_label: "Someone who knows you", audience_kind: "generic", listener: "familiar", style: {brevity: "natural", courtesy: "plain", formality: "neutral"}, listener_source: "setting", style_source: "default"} as ResolvedAudience;
  const messages = candidates.map(candidate => ({message_id: candidate.id, hypothesis_id: candidate.source_hypothesis_ids?.[0] || "", source_hypothesis_ids: candidate.source_hypothesis_ids || [], source_literals: candidate.source_literals || [], literal_text: candidate.reading || job.original, interpreted_intent: candidate.reading || candidate.text, corrected_text: candidate.text, repair_status: candidate.available === false ? "unavailable" as const : "corrected" as const, repair_note: "Literal evidence is preserved separately.", word_alternatives: candidate.word_alternatives || {}, specializations: candidate.specializations || [], plain_text: candidate.plain_text || null, retrieved_sources: candidate.retrieved_sources || [], contextual_additions: candidate.contextual_additions || []}));
  return {auto_speak: Boolean(canSpeak && reviewed), ranking: job.ranking, decision_source: job.decision_source, request_id: job.id, job_revision: job.revision, backend: job.evidence.backend, model: job.evidence.model || "", device: "server", context: job.context?.core_context || "general", listener: audience.listener, audience, persona: job.context?.selection.profile_id || null, hypotheses,
    audio_quality: job.evidence.audio_quality || {seconds: 0, peak_dbfs: null, rms_dbfs: null, clipped_samples: 0, low_level_warning: false},
    ranker: {decision: reviewed ? "selected" : "ambiguous", selected_message_id: reviewed ? selectedId : null, display_hypothesis_ids: hypotheses.map(h => h.id), display_message_ids: messages.map(m => m.message_id), reason: job.question || job.error?.message || job.ranking?.reason || "", source: job.error ? "unavailable" : "groq", assistant_model: null, personalization: null},
    messages, recommended_message_id: reviewed ? selectedId : null, needs_user_choice: !reviewed, speech: null,
    timing: {audio_decode_seconds: 0, asr_seconds: 0, ranking_seconds: 0, grammar_seconds: 0, speech_seconds: 0, total_seconds: 0}, warnings: [...(job.warnings || []), ...(job.error ? [job.error.message] : [])], question: job.question, conversation_reference: job.conversation?.text || null};
}

export async function readHealth(): Promise<Health> { return bodyOrError<Health>(await fetch(`${API_URL}/api/v1/health`)); }
export async function readPersonas(): Promise<PersonaSummary[]> {
  const profiles = await request<PersonaSummary[]>("/profiles");
  profiles.forEach(profile => profileRevisions.set(profile.id, profile.revision || 0));
  return profiles;
}
export async function readProfile(persona = "user"): Promise<PersonaProfile> {
  const profile = await request<PersonaProfile>(`/profiles/${encodeURIComponent(persona)}`);
  profileRevisions.set(profile.id, profile.revision || 0);
  return profile;
}
export async function resolveAudience(input: {context: CommunicationContext; persona: string; audience: string; declared_listener: Listener | null; place_id?: string}): Promise<ResolvedAudience> {
  return request<ResolvedAudience>("/audience/resolve", "POST", input);
}
export async function readPlaces(): Promise<PlaceSettings> { return bodyOrError<PlaceSettings>(await fetch(`${API_URL}/api/v1/places`)); }
export async function savePlaces(settings: PlaceSettings): Promise<PlaceSettings> {
  return bodyOrError<PlaceSettings>(await fetch(`${API_URL}/api/v1/places`, {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({auto_detect: settings.auto_detect, places: settings.places})}));
}
export async function cancelActiveMessage(): Promise<void> {
  requestGeneration += 1;
  const previous = activeJob;
  activeJob = null;
  // A completed approved message remains the next utterance's bounded reference.
  // In-flight work is cancelled and late poll/speech results become ineligible.
  if (previous && !previous.confirmed && previous.status !== "cancelled") {
    await request(`/messages/${previous.id}/cancel`, "POST").catch(() => undefined);
  }
}
export async function transcribeAudio(uri: string, filename: string, context: CommunicationContext, persona: string, listener: Listener | null, audience: string, place = ""): Promise<Transcription> {
  const pendingGeneration = requestGeneration;
  await editQueue;
  if (pendingGeneration !== requestGeneration) throw new Error("Recording cancelled.");
  const cancellation = cancelActiveMessage();
  const generation = requestGeneration;
  await cancellation;
  if (generation !== requestGeneration) throw new Error("Recording cancelled.");
  if (persona && !profileRevisions.has(persona)) await readProfile(persona);
  if (generation !== requestGeneration) throw new Error("Recording cancelled.");
  const selectedProfile = persona === "user" && !profileRevisions.get(persona) ? "" : persona;
  const form = new FormData();
  form.append("file", new File(uri), filename);
  form.append("recognition_backend", "adapted");
  form.append("selection", JSON.stringify({profile_id: selectedProfile, profile_revision: profileRevisions.get(selectedProfile) || 0, scenario: context === "outdoors" ? "outside" : context, core_context: context, declared_listener: listener, audience_id: audience, place_id: place}));
  const job = await request<Job>("/audio", "POST", form);
  if (generation !== requestGeneration) { await request(`/messages/${job.id}/cancel`, "POST").catch(() => undefined); throw new Error("Recording cancelled."); }
  activeJob = job;
  const deadline = Date.now() + 300_000;
  while (["transcribing", "drafting", "preparing"].includes(activeJob.status)) {
    await new Promise(resolve => setTimeout(resolve, 250));
    if (generation !== requestGeneration) throw new Error("Recording cancelled.");
    if (Date.now() > deadline) { await cancelActiveMessage(); throw new Error("Recognition took too long. Please try again."); }
    const state = await request<{job: Job | null}>("/state");
    if (!state.job || state.job.id !== job.id || generation !== requestGeneration) throw new Error("This recording is no longer active.");
    activeJob = state.job;
  }
  if (activeJob.status === "cancelled") throw new Error("Recording cancelled.");
  if (!activeJob.text && activeJob.error) throw new Error(activeJob.error.message);
  return display(activeJob);
}
export async function chooseMessage(candidateId: string): Promise<Transcription> {
  const generation = requestGeneration;
  await editQueue;
  if (!activeJob || generation !== requestGeneration) throw new Error("This message is no longer active.");
  const current = activeJob;
  if (candidateId.startsWith("option-")) {
    const text = current.options[Number(candidateId.slice(7))];
    update(await request<Job>(`/messages/${current.id}/edit`, "POST", {revision: current.revision, text, selection: true}));
  } else if (candidateId !== "raw") {
    update(await request<Job>(`/messages/${current.id}/choose`, "POST", {revision: current.revision, candidate_id: candidateId}));
  }
  return display(activeJob);
}
export function editMessage(text: string): Promise<void> {
  const jobId = activeJob?.id;
  const operation = editQueue.then(async () => {
    if (!activeJob || activeJob.id !== jobId || !text.trim() || activeJob.text === text.trim()) return;
    update(await request<Job>(`/messages/${activeJob.id}/edit`, "POST", {revision: activeJob.revision, text}));
  });
  editQueue = operation.catch(() => undefined);
  return operation;
}
export async function synthesizeSpeech(text: string): Promise<{speech: SpeechAudio | null; speech_text: string}> {
  const generation = requestGeneration;
  await editQueue;
  if (!activeJob || generation !== requestGeneration) throw new Error("This message is no longer active.");
  let current = activeJob;
  if (current.text.trim() !== text.trim() || current.question || current.error) {
    current = update(await request<Job>(`/messages/${current.id}/edit`, "POST", {revision: current.revision, text}));
  }
  current = update(await request<Job>(`/messages/${current.id}/confirm`, "POST", {revision: current.revision}));
  if (!current.confirmed) throw new Error("The message is not ready to speak.");
  const spoken = await request<{job: Job; speech: SpeechAudio | null; speech_text: string}>(`/messages/${current.id}/speech`, "POST", {revision: current.revision, confirmation_id: current.confirmed.id});
  update(spoken.job);
  return spoken;
}
export async function replaceProfile(input: Omit<PersonaProfile, "id" | "icon" | "baseline" | "speaker_note" | "created_at">, persona = "user", revision = 0): Promise<ProfileSaveResponse> {
  const {revision: _revision, ...profile} = input;
  const response = await request<ProfileSaveResponse>(`/profiles/${encodeURIComponent(persona)}`, "PUT", {revision, profile});
  profileRevisions.set(response.profile.id, response.profile.revision || 0);
  return response;
}


export const readMemories = (profileId: string) =>
  request<MemoryList>(`/profiles/${encodeURIComponent(profileId)}/memories`);

export async function rememberActiveMessage(text: string): Promise<RememberResult> {
  const generation = requestGeneration;
  await editQueue;
  if (!activeJob || generation !== requestGeneration) throw new Error("This message is no longer active.");
  if (activeJob.text.trim() !== text.trim()) await editMessage(text);
  if (!activeJob || generation !== requestGeneration) throw new Error("This message is no longer active.");
  const current = activeJob;
  const profileId = current.context?.selection.profile_id;
  if (!profileId) throw new Error("Choose your own profile before remembering this message.");
  const memories = await readMemories(profileId);
  requireCurrent(current);
  if (generation !== requestGeneration) throw new Error("This message is no longer active.");
  return request<RememberResult>(`/messages/${current.id}/remember`, "POST", {
    revision: current.revision, memory_revision: memories.memory_revision,
  });
}

type MemoryChange = { profile_id: string; memory_revision: number; deleted_ids: string[] };
export const deleteMemory = (profileId: string, memoryId: string, revision: number) =>
  request<MemoryChange>(`/profiles/${encodeURIComponent(profileId)}/memories/${encodeURIComponent(memoryId)}/delete`, "POST", {memory_revision: revision});
export const clearMemories = (profileId: string, revision: number) =>
  request<MemoryChange>(`/profiles/${encodeURIComponent(profileId)}/memories/clear`, "POST", {memory_revision: revision});
export async function deleteProfile(profileId: string, profileRevision: number, memoryRevision: number) {
  const response = await request<{deleted: boolean; profile_id: string}>(`/profiles/${encodeURIComponent(profileId)}/delete`, "POST", {
    profile_revision: profileRevision, memory_revision: memoryRevision,
  });
  if (response.deleted) profileRevisions.delete(profileId);
  return response;
}
export async function refreshActiveMessage(): Promise<Transcription | null> {
  const previous = activeJob;
  const state = await request<{job: Job | null}>("/state");
  if (!previous || activeJob?.id !== previous.id) return null;
  if (!state.job || state.job.id !== previous.id || state.job.status === "cancelled") {
    activeJob = null;
    requestGeneration += 1;
    return null;
  }
  return display(update(state.job));
}

/** Native has no EventSource: observe revocations while a result is displayed. */
export function watchMessageInvalidation(onInvalidated: (next: Transcription | null) => void, onUnavailable: () => void) {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const watchedId = activeJob?.id;
  const check = async () => {
    try {
      if (!watchedId || activeJob?.id !== watchedId) return;
      const state = await request<{job: Job | null}>("/state");
      if (stopped || activeJob?.id !== watchedId) return;
      const current = activeJob;
      const next = state.job;
      if (next?.id === current.id && next.revision < current.revision) return;
      const revoked = !next || next.id !== current.id || next.status === "cancelled" ||
        (next.revision > current.revision && next.ranking?.status === "stale_context") ||
        (next.context?.selection.profile_id !== current.context?.selection.profile_id) ||
        Boolean(current.confirmed && current.confirmed.id !== next.confirmed?.id);
      if (revoked) {
        requestGeneration += 1;
        activeJob = next?.id === current.id && next.status !== "cancelled" ? next : null;
        onInvalidated(activeJob ? display(activeJob) : null);
      }
    } catch {
      if (!stopped) onUnavailable();
    } finally {
      if (!stopped && activeJob?.id === watchedId) timer = setTimeout(check, 500);
    }
  };
  void check();
  return () => { stopped = true; if (timer !== undefined) clearTimeout(timer); };
}
