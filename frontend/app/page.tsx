"use client";

import { ChangeEvent, ReactNode, Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";


// three.js is by far the heaviest thing on this page, and the orb it draws is
// decoration. Loading it separately keeps it out of the first paint, and means a
// speaker on reduced motion -- who never mounts the orb -- never downloads it.
const Orb = lazy(() => import("@/components/ui/orb").then((module) => ({ default: module.Orb })));

/**
 * The orb, sized in flow so the stage can lay out around it rather than having
 * text guess where it landed. It is mounted per stage rather than once for the
 * session: that costs a WebGL context on each entry, which is cheap next to the
 * arithmetic of positioning an overlay against centred, variable-height content.
 *
 * `live` feeds it the real microphone level; without it the orb animates itself
 * from its state, which is how the speaking side stays clear of the audio element.
 */
function StageOrb({
  state,
  ready,
  live,
  volumeRef,
}: {
  state: "listening" | "talking";
  ready: boolean;
  live?: boolean;
  volumeRef: React.RefObject<number>;
}) {
  // The orb's body comes from its "out" channel. In manual mode that defaults to
  // zero because nothing supplies it, and the sphere collapses into hard lobes --
  // held steady here so the microphone only ever modulates a shape that is already
  // whole. Auto mode supplies its own, so this is ignored there.
  const bodyRef = useRef(0.45);
  if (!ready) return null;
  return (
    <div className="orb-slot" aria-hidden="true">
      <Suspense fallback={null}>
        <Orb
          agentState={state}
          volumeMode={live ? "manual" : "auto"}
          inputVolumeRef={volumeRef}
          outputVolumeRef={bodyRef}
          colors={ORB_COLORS}
        />
      </Suspense>
    </div>
  );
}
import { useHydrated, useReducedMotion } from "@/lib/browser";

type Hypothesis = {
  id: string;
  literal_text: string;
  sequence_score: number;
  search_weight: number;
};

type Specialization = {
  anchor: string;
  plain: string;
  surface: string;
  source: string;
  kind: string;
  profile_id: string;
};

// What the personal layer offered and what survived it. Shown in the evidence
// sheet because a detail that does not appear has several very different
// reasons -- scoped out of this setting, refused after the fact, or never
// offered because the recognizer was not sure enough of the word it rides on --
// and none of them are visible in the message itself.
type PersonalizationTrace = {
  profile_id: string;
  profile_label: string;
  lexicon_hints: string[];
  examples_used: number;
  specializations_offered: number;
  specializations_applied: number;
  specializations_refused: number;
  retrieval_seconds: number;
};

type PersonaSummary = {
  id: string;
  label: string;
  blurb: string;
  icon: string;
  context_default: CommunicationContext;
  listener_by_setting: Partial<Record<CommunicationContext, Listener>>;
  lexicon_size: number;
  specialization_size: number;
  history_size: number;
  baseline: boolean;
};

type MessageCandidate = {
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
};

type SpeechAudio = {
  audio_base64: string;
  media_type: string;
  voice: string;
  model: string;
};

type CommunicationContext = "general" | "home" | "care" | "outdoors";

// Who is listening. The setting says where the speaker is; this says whether the
// person they are speaking to knows them, which is what decides whether a need
// is stated or asked. Orthogonal to the four settings, never a fifth one.
type Listener = "familiar" | "unfamiliar";

const LISTENER_DEFAULTS: Record<CommunicationContext, Listener> = {
  general: "familiar",
  home: "familiar",
  care: "familiar",
  outdoors: "unfamiliar",
};

function defaultListener(context: CommunicationContext): Listener {
  return LISTENER_DEFAULTS[context] ?? "familiar";
}

// What a setting falls back to when neither the place nor the profile has said
// anything. Only ever used to describe that fallback in words -- the value the
// request is actually made with is resolved on the server.
function listenerLabel(context: CommunicationContext): string {
  return defaultListener(context) === "familiar"
    ? "people who know me"
    : "people who do not know me";
}

// A place is a label with an optional location. It never introduces a new
// setting: `context` is the built-in whose prior it borrows, so a custom place
// behaves exactly as that built-in already does.
type Place = {
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

type PlaceSettings = {
  auto_detect: boolean;
  places: Place[];
  updated_at: string | null;
};

type Transcription = {
  request_id: string;
  backend: string;
  model: string;
  device: string;
  context: CommunicationContext;
  // What the server actually resolved, after the place → profile → default
  // chain. Not the same as what the browser sent, which is why an accepted
  // message is stamped from here rather than from local state.
  listener: Listener;
  persona: string | null;
  hypotheses: Hypothesis[];
  ranker: {
    decision: "selected" | "ambiguous";
    selected_message_id: string | null;
    reason: string;
    source: string;
    personalization: PersonalizationTrace | null;
  };
  messages: MessageCandidate[];
  recommended_message_id: string | null;
  needs_user_choice: boolean;
  speech: SpeechAudio | null;
  timing: { total_seconds: number; asr_seconds: number };
  warnings: string[];
};

type Health = {
  status: "ready" | "degraded" | "starting";
  asr_backend: string;
  model_ready: boolean;
  groq_configured: boolean;
  personal_ready: boolean;
  detail: string;
};

type Phase = "idle" | "recording" | "working" | "ready" | "error";
type Stage = "idle" | "recording" | "working" | "choosing" | "composing" | "error";
type Overlay = "evidence" | "about" | "persona" | "settings" | null;

const contexts: { value: CommunicationContext; label: string; hint: string }[] = [
  { value: "general", label: "General", hint: "No setting assumptions" },
  { value: "home", label: "At home", hint: "Everyday needs and household help" },
  { value: "care", label: "Care", hint: "Symptoms, comfort, and assistance" },
  { value: "outdoors", label: "Outdoors", hint: "Travel, safety, and nearby places" },
];

const promises = [
  { number: "01", title: "Your words stay visible", body: "The literal transcript is never replaced by a cleaned-up message." },
  { number: "02", title: "Uncertainty stays honest", body: "When the evidence is unclear, Echora asks you instead of pretending." },
  { number: "03", title: "You are in control", body: "A message you have chosen is spoken straight away, and stays editable so you can say it again." },
];

const API = process.env.NEXT_PUBLIC_ECHORA_API_URL || "http://127.0.0.1:8000";
const BAR_COUNT = 25;
// A 44-byte silent WAV, played inside the tap that starts a recording. Browsers
// only grant autoplay from a user gesture, and the message is spoken after an
// await, long after that gesture ended -- so an element is unlocked here and
// reused for every later clip.
const SILENCE = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";
const MAX_SECONDS = 45;
const bars = Array.from({ length: BAR_COUNT }, (_, index) => index);
// The orb takes its colours as a prop rather than from CSS, so the two ends of
// the palette are repeated here: --green and --coral.
const ORB_COLORS: [string, string] = ["#1e5a4a", "#87aa75"];

function formatBackend(value?: string) {
  if (value === "local") return "Local Mac";
  if (value === "pod") return "RunPod Pod";
  if (value === "runpod") return "RunPod Serverless";
  return "Connecting";
}

function sameHealth(a: Health | null, b: Health) {
  return Boolean(
    a &&
      a.status === b.status &&
      a.asr_backend === b.asr_backend &&
      a.model_ready === b.model_ready &&
      a.groq_configured === b.groq_configured &&
      a.detail === b.detail,
  );
}

// Mirrors `places.meters_between` on the server. Resolution happens here so the
// speaker's coordinates never leave the browser -- only the borrowed setting is
// sent, through the `context` field that already existed.
function metersBetween(fromLat: number, fromLon: number, toLat: number, toLon: number) {
  const rad = (value: number) => (value * Math.PI) / 180;
  const inner =
    Math.sin(rad(toLat - fromLat) / 2) ** 2 +
    Math.cos(rad(fromLat)) * Math.cos(rad(toLat)) * Math.sin(rad(toLon - fromLon) / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(inner)));
}

// The accuracy allowance matters: a browser fix indoors is routinely tens of
// metres out, which would otherwise stop a correctly tagged home from matching
// its own front room. Capped, so a hopeless fix cannot match half the city.
function nearestPlace(places: Place[], latitude: number, longitude: number, accuracy = 0) {
  const tolerance = Math.min(Math.max(accuracy, 0), 100);
  let best: { place: Place; meters: number } | null = null;
  for (const item of places) {
    if (item.latitude === null || item.longitude === null) continue;
    const meters = metersBetween(latitude, longitude, item.latitude, item.longitude);
    if (meters > item.radius_m + tolerance) continue;
    if (!best || meters < best.meters) best = { place: item, meters };
  }
  return best;
}

function isTagged(place: Place) {
  return place.latitude !== null && place.longitude !== null;
}

function behaviourLabel(context: CommunicationContext) {
  return contexts.find((item) => item.value === context)?.label || context;
}

// The idle screen cannot scroll, so the chip row is capped and the rest stays in
// the settings sheet. The selected place is always one of the ones shown.
const IDLE_CHIPS = 5;

function idleChips(places: Place[], selected: string) {
  if (places.length <= IDLE_CHIPS) return { shown: places, hidden: 0 };
  const front = places.filter((item) => item.id === selected);
  const kept = [...front, ...places.filter((item) => item.id !== selected)].slice(0, IDLE_CHIPS);
  // Filtered back into stored order, so the row does not rearrange itself as
  // the selection moves around.
  return { shown: places.filter((item) => kept.includes(item)), hidden: places.length - kept.length };
}

function groupedWeight(result: Transcription, candidate: MessageCandidate) {
  return candidate.source_hypothesis_ids.reduce(
    (total, id) => total + (result.hypotheses.find((item) => item.id === id)?.search_weight || 0),
    0,
  );
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<Transcription | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [speaking, setSpeaking] = useState(false);
  // The wording that was last actually spoken. Comparing it to the textarea is
  // what tells the speaker their edit has not been said yet.
  const [spokenText, setSpokenText] = useState("");
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [context, setContext] = useState<CommunicationContext>("general");
  // What the active place *declares*, and nothing more. `null` means it declares
  // nothing, which is not the same as declaring the setting's default: the
  // server reads an absent listener as permission to ask the profile what this
  // setting usually means for this speaker before falling back to the default.
  // The whole chain lives on the server, so this never tries to compute it.
  const [listener, setListener] = useState<Listener | null>(null);
  // "" is General: no place, no assumptions. Places are stored on this machine;
  // the setting they resolve to is still session-only and never saved.
  const [place, setPlace] = useState("");
  const [placeSettings, setPlaceSettings] = useState<PlaceSettings | null>(null);
  const [detected, setDetected] = useState<{ id: string; label: string; meters: number } | null>(null);
  const [overlay, setOverlay] = useState<Overlay>(null);
  const [personas, setPersonas] = useState<PersonaSummary[]>([]);
  // "" means nobody in particular, which is exactly how Echora behaved before
  // personal context existed: no prior, no history, no details.
  const [persona, setPersona] = useState("");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playerRef = useRef<HTMLAudioElement | null>(null);
  const clipUrlRef = useRef<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const waveRef = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const mountedRef = useRef(false);
  const overlayRef = useRef<Overlay>(null);
  // Once the speaker has chosen a place by hand, detection stops overriding it
  // for the rest of the session. A tap always outranks a guess.
  const manualPlaceRef = useRef(false);
  // Read by the orb inside its own render loop. A ref, not state: the level
  // changes 30 times a second and must never re-render the screen.
  const inputVolumeRef = useRef(0);

  // The stage is derived, not stored. The backend only fills recommended_message_id
  // when a single message survived the chain, so a confident result opens straight
  // on "composing" and an ambiguous one opens on "choosing".
  const stage: Stage = phase === "ready" ? (selectedId ? "composing" : "choosing") : phase;

  // The orb is WebGL, so it must never render on the server, and it is skipped
  // outright under reduced motion -- that setting exists to stop exactly this kind
  // of continuous movement, and skipping it also avoids loading three.js at all.
  // The orb marks the two states where sound is actually moving: in, while the
  // speaker talks, and out, while the message is spoken. Idle, working, choosing
  // and error are waiting or reading, and it would only compete there.
  const reducedMotion = useReducedMotion();
  const orbReady = useHydrated() && !reducedMotion;

  useEffect(() => {
    const check = async () => {
      try {
        const response = await fetch(`${API}/api/v1/health`);
        if (!response.ok) throw new Error("Health check failed");
        const next = (await response.json()) as Health;
        setHealth((previous) => (sameHealth(previous, next) ? previous : next));
        setHealthError(false);
      } catch {
        setHealthError(true);
      }
    };
    check();
    const interval = window.setInterval(check, 15000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    fetch(`${API}/api/v1/personas`)
      .then((response) => (response.ok ? response.json() : []))
      .then((list: PersonaSummary[]) => setPersonas(list))
      .catch(() => setPersonas([]));
  }, []);

  useEffect(() => {
    fetch(`${API}/api/v1/places`)
      .then((response) => (response.ok ? response.json() : null))
      .then((body: PlaceSettings | null) => setPlaceSettings(body))
      .catch(() => setPlaceSettings(null));
  }, []);

  // Detection runs only while idle, so a reading can never land mid-utterance,
  // and runs again each time the speaker comes back to the idle screen. Every
  // failure -- permission refused, no fix, nothing tagged -- falls back to the
  // chips below, which is exactly how this screen behaved before places existed.
  useEffect(() => {
    if (stage !== "idle") return;
    if (!placeSettings?.auto_detect || manualPlaceRef.current) return;
    if (!placeSettings.places.some(isTagged)) return;
    if (typeof navigator === "undefined" || !navigator.geolocation) return;
    let cancelled = false;
    navigator.geolocation.getCurrentPosition(
      (position) => {
        if (cancelled || manualPlaceRef.current) return;
        const found = nearestPlace(
          placeSettings.places,
          position.coords.latitude,
          position.coords.longitude,
          position.coords.accuracy,
        );
        if (!found) {
          setDetected(null);
          return;
        }
        setDetected({ id: found.place.id, label: found.place.label, meters: Math.round(found.meters) });
        setPlace(found.place.id);
        setContext(found.place.context);
        setListener(found.place.listener);
      },
      () => {
        if (!cancelled) setDetected(null);
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 },
    );
    return () => {
      cancelled = true;
    };
  }, [placeSettings, stage]);

  useEffect(
    () => () => {
      stopMedia();
      stopSpeaking();
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
    },
    [],
  );

  useEffect(() => {
    if (phase !== "working") return;
    const interval = window.setInterval(() => setProgress((value) => Math.min(value + 1, 3)), 1800);
    return () => window.clearInterval(interval);
  }, [phase]);

  useEffect(() => {
    if (phase === "recording" && elapsed >= MAX_SECONDS) stopRecording();
  }, [phase, elapsed]);

  useEffect(() => {
    overlayRef.current = overlay;
  }, [overlay]);

  // Move focus to whatever the speaker should touch next, but never on first paint and
  // never on overlay close — <dialog> already returns focus to the button that opened it.
  useEffect(() => {
    if (!mountedRef.current) {
      mountedRef.current = true;
      return;
    }
    if (overlayRef.current) return;
    stageRef.current?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
  }, [stage]);

  const active = useMemo(
    () => personas.find((item) => item.id === persona) || null,
    [personas, persona],
  );

  const selected = useMemo(
    () => result?.messages.find((candidate) => candidate.message_id === selectedId) || null,
    [result, selectedId],
  );

  // True once the speaker has changed the wording away from whatever was spoken,
  // so the button can offer to say the new version rather than repeat the old one.
  const edited = message.trim() !== spokenText;

  const progressLabel = [
    "Listening closely to the recording",
    "Finding literal speech candidates",
    "Comparing the available evidence",
    "Preparing clear wording",
  ][progress];

  function stopMedia() {
    if (timerRef.current) clearInterval(timerRef.current);
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    audioContextRef.current?.close().catch(() => undefined);
    timerRef.current = null;
    animationRef.current = null;
    inputVolumeRef.current = 0;
    streamRef.current = null;
    audioContextRef.current = null;
  }

  // Bar levels are written straight to the DOM as a scaleY factor, and the orb's
  // one input level goes to a ref beside them. Routing either through state would
  // re-render the whole screen 30 times a second, and animating `height` would
  // force layout every frame; `transform` stays on the compositor.
  function animateWaveform(stream: MediaStream) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const audio = new AudioContext();
    const analyser = audio.createAnalyser();
    analyser.fftSize = 128;
    analyser.smoothingTimeConstant = 0.75;
    audio.createMediaStreamSource(stream).connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const level = new Float32Array(BAR_COUNT).fill(0.2);
    audioContextRef.current = audio;
    let last = 0;
    const draw = (now: number) => {
      animationRef.current = requestAnimationFrame(draw);
      if (now - last < 33) return; // ~30fps is plenty and halves the work
      last = now;
      analyser.getByteFrequencyData(data);
      const node = waveRef.current;
      if (!node) return;
      let sum = 0;
      for (let index = 0; index < BAR_COUNT; index += 1) {
        const target = Math.min(1, Math.max(0.14, (data[index * 2] / 255) * 1.35));
        level[index] += (target - level[index]) * 0.35; // easing lives here, not in CSS
        (node.children[index] as HTMLElement | undefined)?.style.setProperty("--l", level[index].toFixed(3));
        sum += level[index];
      }
      // One scalar for the orb, from the levels this loop already computed -- no
      // second AudioContext. Damped hard: at full scale the orb swells into a
      // shape behind the stage text and the reading suffers, and the bars already
      // carry the live signal accurately, so this only has to breathe.
      inputVolumeRef.current = Math.min(0.42, (sum / BAR_COUNT) * 0.42);
    };
    animationRef.current = requestAnimationFrame(draw);
  }

  // Unlocked while a tap is still the active gesture. The composed message is
  // spoken after an await, by which point the browser no longer counts the tap as
  // permission, so the same element has to be primed here and reused later.
  function primePlayer() {
    if (playerRef.current) return playerRef.current;
    const player = new Audio(SILENCE);
    player.play().catch(() => undefined);
    playerRef.current = player;
    return player;
  }

  function stopSpeaking() {
    playerRef.current?.pause();
    window.speechSynthesis?.cancel();
  }

  function playClip(clip: SpeechAudio) {
    const player = primePlayer();
    if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
    const bytes = Uint8Array.from(atob(clip.audio_base64), (character) => character.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: clip.media_type }));
    clipUrlRef.current = url;
    player.src = url;
    return new Promise<void>((resolve, reject) => {
      player.onended = () => resolve();
      player.onerror = () => reject(new Error("The clip could not be played"));
      player.play().catch(reject);
    });
  }

  function browserSpeak(text: string) {
    return new Promise<void>((resolve) => {
      if (!window.speechSynthesis) return resolve();
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.92;
      utterance.onend = () => resolve();
      utterance.onerror = () => resolve();
      window.speechSynthesis.speak(utterance);
    });
  }

  // Speech never fails loudly. `clip` is audio the backend already made; `null`
  // means it tried and could not, so there is no point asking again; `undefined`
  // means nobody has asked yet. Whatever goes wrong, the browser's own voice
  // still says the sentence -- silence would leave the speaker unheard.
  async function speak(text: string, clip?: SpeechAudio | null) {
    const spoken = text.trim();
    if (!spoken) return;
    stopSpeaking();
    setSpeaking(true);
    setSpokenText(spoken);
    try {
      if (clip) {
        await playClip(clip);
        return;
      }
      if (clip === null) {
        await browserSpeak(spoken);
        return;
      }
      const response = await fetch(`${API}/api/v1/speech`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: spoken }),
      });
      if (!response.ok) throw new Error("Speech is unavailable");
      await playClip((await response.json()) as SpeechAudio);
    } catch {
      await browserSpeak(spoken);
    } finally {
      setSpeaking(false);
    }
  }

  async function startRecording() {
    setError("");
    setResult(null);
    setSelectedId(null);
    setSpokenText("");
    setCopied(false);
    primePlayer();
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError("This browser cannot record audio. Use the audio-file button instead.");
      setPhase("error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
      });
      const preferred = ["audio/webm;codecs=opus", "audio/mp4", "audio/webm"].find((type) =>
        MediaRecorder.isTypeSupported(type),
      );
      const recorder = new MediaRecorder(stream, preferred ? { mimeType: preferred } : undefined);
      recorderRef.current = recorder;
      streamRef.current = stream;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => event.data.size && chunksRef.current.push(event.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        stopMedia();
        await submitAudio(blob, recorder.mimeType.includes("mp4") ? "recording.m4a" : "recording.webm");
      };
      recorder.start(250);
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((value) => value + 1), 1000);
      animateWaveform(stream);
      setPhase("recording");
    } catch (caught) {
      const denied = caught instanceof DOMException && caught.name === "NotAllowedError";
      setError(denied ? "Microphone access was denied. You can still upload an audio file." : "The microphone could not start.");
      setPhase("error");
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  function cancelRecording() {
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
    }
    stopMedia();
    setElapsed(0);
    setPhase("idle");
  }

  async function submitAudio(blob: Blob, filename: string) {
    setPhase("working");
    setProgress(0);
    setError("");
    setSpokenText("");
    const form = new FormData();
    form.append("audio", blob, filename);
    form.append("context", context);
    form.append("persona", persona);
    // Empty means "nothing declared here", not "familiar". The server treats it
    // as permission to ask the profile before falling back to the setting.
    form.append("listener", listener ?? "");
    try {
      const response = await fetch(`${API}/api/v1/transcriptions`, { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "The recording could not be transcribed");
      const next = body as Transcription;
      setResult(next);
      const initial = next.recommended_message_id || (next.messages.length === 1 ? next.messages[0].message_id : null);
      setSelectedId(initial);
      const initialMessage = next.messages.find((candidate) => candidate.message_id === initial);
      setMessage(initialMessage?.corrected_text || "");
      setPhase("ready");
      // Nothing is left to disambiguate, so the message says itself. This tracks
      // the ranker rather than the option count: when the chain is unavailable it
      // shows one *raw* beam, and reading unreviewed ASR aloud is exactly what the
      // literal evidence is meant to protect the speaker from.
      if (initialMessage && next.ranker.decision === "selected") {
        void speak(initialMessage.corrected_text, next.speech);
        remember(initialMessage, initialMessage.corrected_text, next);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong while transcribing");
      setPhase("error");
    }
  }

  function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    primePlayer();
    if (file) submitAudio(file, file.name);
    event.target.value = "";
  }

  // Choosing is the decision. The tap is also a live gesture, so the chosen
  // wording can be spoken straight back without waiting for a second confirmation.
  // A message the speaker settled on is what the personal layer learns from.
  // It is fire-and-forget: failing to remember something is never worth
  // interrupting someone mid-conversation for.
  function remember(candidate: MessageCandidate, text: string, from: Transcription | null = result) {
    if (!persona || !from) return;
    // An unavailable candidate is a raw beam with a label on it, not a message
    // the assistant formed. Storing one would put unreviewed ASR into the pool
    // this speaker's later messages are written from -- teaching the profile
    // that they say "nead coffee" because that is what the recognizer typed.
    if (candidate.repair_status === "unavailable") return;
    void fetch(`${API}/api/v1/accepted`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        persona,
        // Stamped from what the server resolved, not from local state. The two
        // differ exactly when the profile decided the listener, and a record
        // filed under the wrong stance is worse than no record: it is retrieved
        // as an example of a shape this speaker never used.
        context: from.context,
        listener: from.listener,
        heard: candidate.interpreted_intent,
        message: text,
      }),
    }).catch(() => undefined);
  }

  // Take a profile detail back out. The plain wording is derived by the backend,
  // so this is a swap to a string that was computed, not one the model promised.
  function revert(candidate: MessageCandidate) {
    if (!candidate.plain_text) return;
    setMessage(candidate.plain_text);
    void speak(candidate.plain_text);
  }

  function choose(candidate: MessageCandidate) {
    setSelectedId(candidate.message_id);
    setMessage(candidate.corrected_text);
    setCopied(false);
    void speak(candidate.corrected_text);
    remember(candidate, candidate.corrected_text);
  }

  function compareOptions() {
    stopSpeaking();
    setSelectedId(null);
    setSpokenText("");
    setCopied(false);
  }

  async function copyMessage() {
    if (!message.trim()) return;
    await navigator.clipboard.writeText(message.trim());
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  function speakMessage() {
    void speak(message);
  }

  // A tap is a decision, so it sticks for the session and clears the detected
  // note rather than sitting underneath a line that contradicts it.
  function pickPlace(next: string) {
    manualPlaceRef.current = true;
    setDetected(null);
    setPlace(next);
    const chosen = (placeSettings?.places || []).find((item) => item.id === next);
    const nextContext: CommunicationContext = chosen ? chosen.context : "general";
    setContext(nextContext);
    // Only what the place declares. A place that declares nothing, and plain
    // General, both send no listener at all -- which is what lets the server
    // reach the profile before falling back to the setting's own default.
    // Filling in the default here would silently outrank the profile.
    setListener(chosen?.listener ?? null);
  }

  function reset() {
    stopSpeaking();
    setResult(null);
    setSelectedId(null);
    setMessage("");
    setSpeaking(false);
    setSpokenText("");
    setError("");
    setElapsed(0);
    setOverlay(null);
    setPhase("idle");
  }

  const modelReady = Boolean(health?.model_ready) && !healthError;
  const busy = phase === "recording" || phase === "working";
  const liveText =
    stage === "recording" ? "Listening"
    : stage === "working" ? "Working on your message"
    : stage === "choosing" ? `${result?.messages.length ?? 0} options ready. Which one sounds right?`
    : stage === "composing" ? (speaking ? "Speaking your message" : edited ? "Message edited and ready to speak" : "Message spoken")
    : stage === "error" ? "Something went wrong"
    : "Ready to record";

  return (
    <main className="shell" data-stage={stage}>
      <header className="bar">
        <span className="brand">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>Echora</span>
        </span>
        <div className={`model-status ${modelReady ? "online" : "offline"}`} role="status">
          <span className="status-dot" />
          <span>{healthError ? "Backend offline" : `${formatBackend(health?.asr_backend)} · ${modelReady ? "Ready" : "Starting"}`}</span>
        </div>
        <span className="bar-spacer" />
        {result && (
          <button className="bar-action" onClick={() => setOverlay("evidence")}>
            Evidence
          </button>
        )}
        {result && (
          <button className="bar-action strong" onClick={reset}>
            Start over
          </button>
        )}
        {personas.length > 0 && (
          <button className="bar-action persona-action" onClick={() => setOverlay("persona")}>
            <PersonaIcon name={active?.icon || "none"} />
            <span>{active ? active.label.split(",")[0] : "No profile"}</span>
          </button>
        )}
        <button className="bar-action" onClick={() => setOverlay("settings")}>
          Places
        </button>
        <button className="bar-action" onClick={() => setOverlay("about")}>
          About
        </button>
      </header>

      <div className="stage" data-stage={stage} ref={stageRef}>
        {stage === "idle" && (
          <IdleStage
            modelReady={modelReady}
            detail={health?.detail}
            place={place}
            places={placeSettings?.places || []}
            detected={detected}
            onPickPlace={pickPlace}
            onSettings={() => setOverlay("settings")}
            busy={busy}
            onRecord={startRecording}
            onPickFile={() => fileRef.current?.click()}
          />
        )}
        {stage === "recording" && (
          <RecordingStage
            elapsed={elapsed}
            waveRef={waveRef}
            orbReady={orbReady}
            volumeRef={inputVolumeRef}
            onStop={stopRecording}
            onCancel={cancelRecording}
          />
        )}
        {stage === "working" && <WorkingStage label={progressLabel} progress={progress} />}
        {stage === "choosing" && result && <ChooseStage result={result} onChoose={choose} />}
        {stage === "composing" && result && selected && (
          <ComposeStage
            result={result}
            selectedId={selectedId}
            message={message}
            speaking={speaking}
            edited={edited}
            copied={copied}
            onChoose={choose}
            onCompare={compareOptions}
            onEdit={setMessage}
            onCopy={copyMessage}
            onSpeak={speakMessage}
            onEvidence={() => setOverlay("evidence")}
            onRevert={revert}
            orbReady={orbReady}
            volumeRef={inputVolumeRef}
          />
        )}
        {stage === "error" && <ErrorStage message={error} onRetry={reset} />}
      </div>

      <p className="visually-hidden" aria-live="polite">{liveText}</p>

      <footer className="hint">
        <span>Your chosen message is spoken as soon as it is ready.</span>
        <span>No conversation is saved after refresh</span>
      </footer>

      <input
        ref={fileRef}
        className="visually-hidden"
        type="file"
        accept="audio/*,.wav,.webm,.ogg,.m4a,.mp3"
        onChange={upload}
      />

      <Sheet open={overlay === "evidence"} onClose={() => setOverlay(null)} title="Literal ASR evidence">
        {result && <EvidenceBody result={result} />}
      </Sheet>

      <Sheet open={overlay === "persona"} onClose={() => setOverlay(null)} title="Who is speaking?">
        <PersonaBody
          personas={personas}
          persona={persona}
          onPick={(next) => {
            setPersona(next);
            const picked = personas.find((item) => item.id === next);
            // A profile default is the weakest of the three signals: a tap and a
            // detected place both outrank it.
            if (!picked || !next || manualPlaceRef.current || detected) return;
            const match = (placeSettings?.places || []).find((item) => item.context === picked.context_default);
            setPlace(match ? match.id : "");
            setContext(picked.context_default);
            // Still only what the place declares. What the profile says about
            // this setting is the server's to apply, and applying it here as
            // well would pin it to the profile's *default* setting -- so a
            // speaker who declares a listener for outdoors while living at home
            // would never have that declaration read at all.
            setListener(match?.listener ?? null);
          }}
        />
      </Sheet>

      <Sheet open={overlay === "settings"} onClose={() => setOverlay(null)} title="Places">
        <SettingsBody
          settings={placeSettings}
          onSaved={(next) => {
            setPlaceSettings(next);
            // A place that no longer exists cannot stay selected.
            if (place && !next.places.some((item) => item.id === place)) {
              setPlace("");
              setContext("general");
            }
            if (!next.auto_detect) setDetected(null);
          }}
        />
      </Sheet>

      <Sheet open={overlay === "about"} onClose={() => setOverlay(null)} title="About Echora">
        <ul className="promise-list">
          {promises.map((item) => (
            <li key={item.number}>
              <span className="promise-number">{item.number}</span>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </li>
          ))}
        </ul>
        <p className="sheet-note">
          Local-first assistive communication. No conversation is saved after refresh. The places you name stay on
          this machine.
        </p>
      </Sheet>
    </main>
  );
}

function IdleStage({
  modelReady,
  detail,
  place,
  places,
  detected,
  onPickPlace,
  onSettings,
  busy,
  onRecord,
  onPickFile,
}: {
  modelReady: boolean;
  detail?: string;
  place: string;
  places: Place[];
  detected: { id: string; label: string; meters: number } | null;
  onPickPlace: (id: string) => void;
  onSettings: () => void;
  busy: boolean;
  onRecord: () => void;
  onPickFile: () => void;
}) {
  const { shown, hidden } = idleChips(places, place);
  return (
    <section className="stage-idle">
      {/* The bloom is a wrapper, not a layer on the button: anything that gives the
          button a stacking context traps a negative z-index child above its own
          background, and the halo ends up painted across the glyph. */}
      <div className="mic-wrap">
        <button
          className="mic-button"
          onClick={onRecord}
          disabled={!modelReady}
          aria-label="Start recording"
          data-autofocus
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <rect x="9" y="2" width="6" height="11" rx="3" />
            <path d="M5 11a7 7 0 0 0 14 0" />
            <path d="M12 18v3" />
          </svg>
        </button>
      </div>
      <strong className="stage-title">{modelReady ? "Tap to speak" : detail || "Waiting for the speech model"}</strong>
      <span className="stage-hint">Take your time. Pauses are welcome.</span>

      <fieldset className="context-chips" disabled={busy}>
        <legend>Where are you speaking?</legend>
        <div>
          <label className={place === "" ? "active" : ""} title={contexts[0].hint}>
            <input
              type="radio"
              name="context"
              value=""
              checked={place === ""}
              onChange={() => onPickPlace("")}
            />
            <span>{contexts[0].label}</span>
          </label>
          {shown.map((option) => (
            <label
              key={option.id}
              className={place === option.id ? "active" : ""}
              title={`Behaves as ${behaviourLabel(option.context)}`}
            >
              <input
                type="radio"
                name="context"
                value={option.id}
                checked={place === option.id}
                onChange={() => onPickPlace(option.id)}
              />
              <span>{option.label}</span>
            </label>
          ))}
          {hidden > 0 && (
            <button type="button" className="chip-more" onClick={onSettings}>
              {hidden} more…
            </button>
          )}
        </div>
        {detected && (
          <p className="detected-note">
            Detected <b>{detected.label}</b> · {detected.meters} m away. Tap another if that is wrong.
          </p>
        )}
      </fieldset>

      <button className="text-button" onClick={onPickFile}>Choose an audio file</button>
    </section>
  );
}

function RecordingStage({
  elapsed,
  waveRef,
  orbReady,
  volumeRef,
  onStop,
  onCancel,
}: {
  elapsed: number;
  waveRef: React.RefObject<HTMLDivElement | null>;
  orbReady: boolean;
  volumeRef: React.RefObject<number>;
  onStop: () => void;
  onCancel: () => void;
}) {
  return (
    <section className="stage-recording">
      {/* Both, deliberately. The orb is presence; the bars are evidence that a
          quiet voice is actually reaching the microphone, which is information a
          speaker with difficult speech needs and a glowing shape cannot give. */}
      <StageOrb state="listening" ready={orbReady} live volumeRef={volumeRef} />
      <div className="recording-label"><span className="recording-dot" />Listening · {elapsed}s</div>
      <div className="waveform" ref={waveRef} aria-label="Live microphone level">
        {bars.map((index) => <i key={index} />)}
      </div>
      <button className="primary-button" onClick={onStop} data-autofocus>
        <span className="stop-glyph" aria-hidden="true" />Finish speaking
      </button>
      <button className="text-button" onClick={onCancel}>Cancel recording</button>
    </section>
  );
}

function WorkingStage({ label, progress }: { label: string; progress: number }) {
  return (
    <section className="stage-working" role="status">
      <div className="listening-orbit" aria-hidden="true"><span /><span /><span /></div>
      <strong className="stage-title">{label}</strong>
      <p className="stage-hint">Your literal transcript stays separate from wording suggestions.</p>
      <div className="progress-track"><i style={{ width: `${30 + progress * 22}%` }} /></div>
    </section>
  );
}

function ChooseStage({ result, onChoose }: { result: Transcription; onChoose: (candidate: MessageCandidate) => void }) {
  return (
    <section className="stage-choosing">
      <h2 className="stage-heading">Which one sounds right?</h2>
      <p className="stage-hint">{result.ranker.reason}</p>
      {result.warnings
        .filter((warning) => warning !== result.ranker.reason)
        .map((warning) => <p className="warning" key={warning}>{warning}</p>)}

      <div className="candidate-grid" data-count={result.messages.length}>
        {result.messages.map((candidate, index) => (
          <button
            className="candidate-card"
            key={candidate.message_id}
            onClick={() => onChoose(candidate)}
            aria-pressed={false}
            data-autofocus={index === 0 ? true : undefined}
          >
            <span className="candidate-topline">
              <b>Option {index + 1}</b>
              <i>{Math.round(groupedWeight(result, candidate) * 100)}% grouped search weight</i>
            </span>
            <span className="candidate-message">{candidate.corrected_text}</span>
            <span className="literal-label">All transcriptions</span>
            <span className="literal-text">{candidate.source_literals.map((text) => `“${text}”`).join(" · ")}</span>
            <span className="select-indicator">Choose this</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function ComposeStage({
  result,
  selectedId,
  message,
  speaking,
  edited,
  copied,
  onChoose,
  onCompare,
  onEdit,
  onCopy,
  onSpeak,
  onEvidence,
  onRevert,
  orbReady,
  volumeRef,
}: {
  result: Transcription;
  selectedId: string | null;
  message: string;
  speaking: boolean;
  edited: boolean;
  copied: boolean;
  onChoose: (candidate: MessageCandidate) => void;
  onCompare: () => void;
  onEdit: (value: string) => void;
  onCopy: () => void;
  onSpeak: () => void;
  onEvidence: () => void;
  onRevert: (candidate: MessageCandidate) => void;
  orbReady: boolean;
  volumeRef: React.RefObject<number>;
}) {
  const multiple = result.messages.length > 1;
  const current = result.messages.find((candidate) => candidate.message_id === selectedId) || null;
  const details = current?.specializations || [];
  return (
    <section className="stage-composing" data-speaking={speaking}>
      {/* Only while the message is actually being spoken. It leaves as soon as the
          voice stops, so the orb always means sound is moving right now. */}
      {speaking && <StageOrb state="talking" ready={orbReady} volumeRef={volumeRef} />}
      {multiple && (
        <div className="pills" role="group" aria-label="Message options">
          {result.messages.map((candidate) => (
            <button
              key={candidate.message_id}
              className="pill"
              title={candidate.corrected_text}
              aria-pressed={candidate.message_id === selectedId}
              onClick={() => onChoose(candidate)}
            >
              {candidate.corrected_text}
            </button>
          ))}
        </div>
      )}

      <h2 className="stage-heading">{result.needs_user_choice ? "Your message" : "Here’s the clearest match"}</h2>

      <label className="visually-hidden" htmlFor="message">Message to communicate</label>
      <textarea
        id="message"
        className="compose-message"
        value={message}
        onChange={(event) => onEdit(event.target.value)}
        rows={2}
      />

      {details.length > 0 && current?.plain_text && (
        <p className="detail-note">
          <span className="detail-mark" aria-hidden="true">◆</span>
          <span>
            Using your wording for <b>{details.map((item) => item.anchor).join(", ")}</b>
          </span>
          <button className="text-button" onClick={() => onRevert(current)}>
            Say it plainly instead
          </button>
        </p>
      )}

      <p className="evidence-caption">
        Based on literal evidence · <button className="text-button" onClick={onEvidence}>see all transcriptions</button>
      </p>

      <div className="speak-actions">
        <div className="speak-label">
          <span aria-hidden="true">{speaking ? "♪" : edited ? "✎" : "✓"}</span>
          <div>
            <b>{speaking ? "Speaking" : edited ? "Edited" : "Spoken aloud"}</b>
            <small>{edited ? "Say it again with your wording" : "Edit the wording and say it again"}</small>
          </div>
        </div>
        <button onClick={onCopy} disabled={!message.trim()}>{copied ? "Copied" : "Copy"}</button>
        <button
          className="speak-button"
          onClick={onSpeak}
          disabled={!message.trim() || speaking}
          data-autofocus
        >
          {speaking ? "Speaking…" : edited ? "Speak this" : "Speak again"}
        </button>
      </div>

      {multiple && <button className="text-button" onClick={onCompare}>Compare all options</button>}
    </section>
  );
}

function ErrorStage({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <section className="stage-error">
      <div className="error-banner" role="alert">
        <strong>We could not finish that.</strong>
        <span>{message}</span>
      </div>
      <button className="primary-button" onClick={onRetry} data-autofocus>Try again</button>
    </section>
  );
}

function EvidenceBody({ result }: { result: Transcription }) {
  const trace = result.ranker.personalization;
  return (
    <>
      <div className="evidence-list">
        {result.hypotheses.map((item) => (
          <div key={item.id}>
            <span>{item.literal_text}</span>
            <code>{(item.search_weight * 100).toFixed(1)}% search weight · score {item.sequence_score.toFixed(3)}</code>
          </div>
        ))}
      </div>
      <p className="sheet-note">Search weights compare only these beam hypotheses. They are not calibrated confidence.</p>

      {/* The stance the message was actually written under. It is worth showing
          because it is not always the one the screen implies: a place may
          declare nothing and let the speaker's profile decide. */}
      <h3 className="sheet-subhead">What this message was written for</h3>
      <div className="evidence-option">
        <span className="intent-label">Setting</span>
        <span className="intent-text">{behaviourLabel(result.context)}</span>
        <span className="intent-label">Listener</span>
        <span className="intent-text">
          {result.listener === "familiar"
            ? "someone who knows the speaker — a need is stated to them"
            : "someone who does not know the speaker — a need is asked of them, and kept short"}
        </span>
      </div>

      {/* A detail that did not appear looks identical to one that was never
          declared. These three counts are the difference: offered=0 means the
          recognizer was not sure enough of the word it rides on, or the detail
          is scoped out of this setting; refused>0 means it was written and then
          taken back out because nothing licensed it. */}
      {trace && (
        <>
          <h3 className="sheet-subhead">What {trace.profile_label.split(",")[0]}’s profile contributed</h3>
          <div className="evidence-option">
            <span className="intent-label">Known words the recognizer produced</span>
            <span className="intent-text">{trace.lexicon_hints.join(" · ") || "none in this utterance"}</span>
            <span className="intent-label">Personal details</span>
            <span className="intent-text">
              {trace.specializations_offered === 0
                ? "none offered — either scoped out of this setting, or the word one rides on was not settled enough in the beams"
                : `${trace.specializations_offered} offered · ${trace.specializations_applied} used · ${trace.specializations_refused} refused`}
            </span>
            <span className="intent-label">Past accepted messages used as examples</span>
            <span className="intent-text">{trace.examples_used}</span>
          </div>
        </>
      )}

      <h3 className="sheet-subhead">How each option was formed</h3>
      {result.messages.map((candidate, index) => (
        <div className="evidence-option" key={candidate.message_id}>
          <b>Option {index + 1} · {candidate.corrected_text}</b>
          <span className="intent-label">Interpreted meaning</span>
          <span className="intent-text">{candidate.interpreted_intent}</span>
          {Object.entries(candidate.word_alternatives || {}).map(([word, options]) => (
            <span className="word-alternative" key={word}>
              <b>{word}</b> — also heard as <i>{options.join(" · ")}</i>
            </span>
          ))}
          <span className="repair-note">{candidate.repair_note}</span>
        </div>
      ))}
    </>
  );
}

function Sheet({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDialogElement | null>(null);

  // Native <dialog> gives focus trapping, Escape to close and an inert background.
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (open && !node.open) node.showModal();
    if (!open && node.open) node.close();
  }, [open]);

  return (
    <dialog className="sheet" ref={ref} onClose={onClose}>
      <div className="sheet-head">
        <h2>{title}</h2>
        <button className="bar-action" onClick={onClose}>Close</button>
      </div>
      {open && children}
    </dialog>
  );
}

// Abstract marks, not portraits. A gallery of little figures standing in for
// disabled or ethnically-coded people is a worse idea the longer you look at it,
// and a silhouette people can tell apart at a glance is what this needs to do.
// Every mark inherits `currentColor`, so the selected chip tints its icon free.
const ICONS: Record<string, ReactNode> = {
  arc: <><path d="M4 18a8 8 0 0 1 16 0" /><path d="M9 18a3 3 0 0 1 6 0" /></>,
  wave: <path d="M3 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0" />,
  chevron: <><path d="M6 8l6 5 6-5" /><path d="M6 14l6 5 6-5" /></>,
  orbit: <><circle cx="12" cy="12" r="3.5" /><ellipse cx="12" cy="12" rx="9" ry="4.5" /></>,
  stack: <><path d="M5 7h14" /><path d="M7 12h10" /><path d="M9 17h6" /></>,
  bloom: <><circle cx="12" cy="12" r="2.5" /><path d="M12 4v3M12 17v3M4 12h3M17 12h3" /></>,
  ridge: <path d="M3 18l5-8 4 5 3-4 6 7z" />,
  tide: <><path d="M3 10c2-3 4-3 6 0s4 3 6 0 4-3 6 0" /><path d="M3 16c2-3 4-3 6 0s4 3 6 0 4-3 6 0" /></>,
  you: <><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="2.5" /></>,
  none: <circle cx="12" cy="12" r="8" strokeDasharray="3 3" />,
};

function PersonaIcon({ name }: { name: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name] || ICONS.none}
    </svg>
  );
}

function PersonaBody({
  personas,
  persona,
  onPick,
}: {
  personas: PersonaSummary[];
  persona: string;
  onPick: (id: string) => void;
}) {
  return (
    <>
      <p className="sheet-note">
        A profile lets Echora use what it knows about a speaker to choose between the words it heard, and to
        say their particular version of a thing. It never invents a word the recogniser did not produce.
      </p>
      <ul className="persona-list">
        <li>
          <button className="persona-card" aria-pressed={persona === ""} onClick={() => onPick("")}>
            <span className="persona-mark"><PersonaIcon name="none" /></span>
            <span className="persona-text">
              <b>No profile</b>
              <small>Nothing personal is used. This is how Echora behaves for a stranger.</small>
            </span>
          </button>
        </li>
        {personas.map((item) => (
          <li key={item.id}>
            <button className="persona-card" aria-pressed={persona === item.id} onClick={() => onPick(item.id)}>
              <span className="persona-mark"><PersonaIcon name={item.icon} /></span>
              <span className="persona-text">
                <b>{item.label}</b>
                <small>{item.blurb}</small>
                <em>
                  {item.history_size} remembered · {item.lexicon_size} known words · {item.specialization_size} details
                </em>
              </span>
            </button>
          </li>
        ))}
      </ul>
      {persona === "user" && <Onboarding />}
    </>
  );
}

// Places, and the switch that lets location choose between them. Every change
// sends the whole document and takes the server's answer back, so what is shown
// is what was actually stored rather than what was asked for -- the built-ins
// come back whatever the request said, and a silly radius comes back corrected.
function SettingsBody({
  settings,
  onSaved,
}: {
  settings: PlaceSettings | null;
  onSaved: (next: PlaceSettings) => void;
}) {
  const [status, setStatus] = useState("");
  const [label, setLabel] = useState("");
  const [behaviour, setBehaviour] = useState<CommunicationContext>("home");
  // Null by default: a new place says nothing about who is there until the
  // speaker says so, and the setting plus their profile take it from there.
  const [company, setCompany] = useState<Listener | null>(null);
  const [busy, setBusy] = useState(false);

  if (!settings) {
    return <p className="sheet-note">Places are unavailable, so the setting stays yours to choose by hand.</p>;
  }
  // Bound to a const so the handlers below keep the narrowed type.
  const doc = settings;

  async function persist(auto_detect: boolean, places: Place[], note: string) {
    setBusy(true);
    setStatus("Saving…");
    try {
      const response = await fetch(`${API}/api/v1/places`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ auto_detect, places }),
      });
      if (!response.ok) throw new Error("save failed");
      onSaved((await response.json()) as PlaceSettings);
      setStatus(note);
    } catch {
      setStatus("That could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  // Tagging asks the browser where it is once, right now, while the speaker is
  // standing in the place they are naming. No address lookup, no map, nothing
  // leaves this machine.
  function tag(target: Place) {
    if (typeof navigator === "undefined" || !navigator.geolocation) {
      setStatus("This browser cannot report a location.");
      return;
    }
    setStatus(`Finding where you are…`);
    navigator.geolocation.getCurrentPosition(
      (position) => {
        void persist(
          doc.auto_detect,
          doc.places.map((item) =>
            item.id === target.id
              ? {
                  ...item,
                  latitude: position.coords.latitude,
                  longitude: position.coords.longitude,
                  tagged_at: null,
                }
              : item,
          ),
          `${target.label} is now tagged here.`,
        );
      },
      (error) => {
        setStatus(
          error.code === error.PERMISSION_DENIED
            ? "Location permission was refused, so places stay yours to choose by hand."
            : "Your location could not be found just now.",
        );
      },
      { enableHighAccuracy: true, timeout: 10000 },
    );
  }

  function clearTag(target: Place) {
    void persist(
      doc.auto_detect,
      doc.places.map((item) =>
        item.id === target.id ? { ...item, latitude: null, longitude: null, tagged_at: null } : item,
      ),
      `The location for ${target.label} was removed.`,
    );
  }

  // Three states, not two. "Leave it to me" is the absence of a declaration, and
  // it is the only one of the three that lets a profile be heard -- so it has to
  // be a value the speaker can choose and see, not an invisible default.
  function setPlaceListener(target: Place, next: Listener | null) {
    void persist(
      doc.auto_detect,
      doc.places.map((item) => (item.id === target.id ? { ...item, listener: next } : item)),
      next === "familiar"
        ? `Messages for ${target.label} will be said to someone who knows you.`
        : next === "unfamiliar"
          ? `Messages for ${target.label} will be asked of someone who does not know you.`
          : `${target.label} will follow whoever is speaking, then ${listenerLabel(target.context)}.`,
    );
  }

  function remove(target: Place) {
    void persist(
      doc.auto_detect,
      doc.places.filter((item) => item.id !== target.id),
      `${target.label} was removed.`,
    );
  }

  function add() {
    const trimmed = label.trim();
    if (!trimmed) return;
    setLabel("");
    void persist(
      doc.auto_detect,
      [
        ...doc.places,
        {
          id: "",
          label: trimmed,
          context: behaviour,
          listener: company,
          builtin: false,
          latitude: null,
          longitude: null,
          radius_m: 150,
          tagged_at: null,
        },
      ],
      `${trimmed} was added. Tag it while you are there.`,
    );
  }

  const tagged = doc.places.filter(isTagged).length;

  return (
    <div className="settings">
      <p className="sheet-note">
        A place is a name and, if you tag it, a location. It borrows the behaviour of one of the four settings — it
        never changes how a message is worked out. What it does say on its own account is whether the people there
        know you: somewhere they do, “washroom” becomes “I need the washroom.”; somewhere they do not, it becomes
        “Where is the washroom?”. Leave that to whoever is speaking and their own profile decides it instead, which
        is what a speaker who never goes out alone wants. The places you name stay on this machine.
      </p>

      <div className="settings-toggle">
        <input
          id="auto-detect"
          type="checkbox"
          checked={doc.auto_detect}
          disabled={busy}
          onChange={(event) =>
            persist(
              event.target.checked,
              doc.places,
              event.target.checked
                ? tagged > 0
                  ? "Echora will choose the setting from where you are, and say so."
                  : "Turned on. Tag a place while you are there and it will start working."
                : "Turned off. The setting stays yours to choose.",
            )
          }
        />
        <label htmlFor="auto-detect">
          <b>Choose the setting from where I am</b>
          <small>
            {tagged > 0
              ? `${tagged} of ${doc.places.length} places are tagged. Echora always shows which one it picked.`
              : "Nothing is tagged yet, so this will do nothing until you tag a place."}
          </small>
        </label>
      </div>

      <ul className="settings-places">
        {doc.places.map((item) => (
          <li key={item.id}>
            <span className="settings-place">
              <b>{item.label}</b>
              <small>
                Behaves as {behaviourLabel(item.context)}
                {item.listener === "familiar"
                  ? " · people here know me"
                  : item.listener === "unfamiliar"
                    ? " · people here do not know me"
                    : ` · whoever is speaking decides, then ${listenerLabel(item.context)}`}
                {isTagged(item) ? ` · tagged, within ${item.radius_m} m` : " · not tagged"}
              </small>
            </span>
            <span className="settings-actions">
              {/* Deliberately offered on the built-ins too: a speaker who only
                  ever goes out with their daughter needs the shipped Outdoors
                  to keep the familiar register. */}
              <select
                aria-label={`Who is usually at ${item.label}`}
                disabled={busy}
                value={item.listener ?? ""}
                onChange={(event) =>
                  setPlaceListener(item, (event.target.value || null) as Listener | null)
                }
              >
                <option value="">Leave it to whoever is speaking</option>
                <option value="familiar">People here know me</option>
                <option value="unfamiliar">People here do not know me</option>
              </select>
              <button className="text-button" disabled={busy} onClick={() => tag(item)}>
                {isTagged(item) ? "Retag here" : "Tag here"}
              </button>
              {isTagged(item) && (
                <button className="text-button" disabled={busy} onClick={() => clearTag(item)}>
                  Clear
                </button>
              )}
              {!item.builtin && (
                <button className="text-button" disabled={busy} onClick={() => remove(item)}>
                  Remove
                </button>
              )}
            </span>
          </li>
        ))}
      </ul>

      <div className="settings-add">
        <label htmlFor="place-label">Add a place</label>
        <div className="onboarding-pair">
          <input
            id="place-label"
            placeholder="Shopping centre"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
          />
          <span aria-hidden="true">→</span>
          <select
            aria-label="Which setting it behaves as"
            value={behaviour}
            onChange={(event) => setBehaviour(event.target.value as CommunicationContext)}
          >
            {contexts.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <div className="onboarding-pair">
          <select
            aria-label="Who is usually there"
            value={company ?? ""}
            onChange={(event) => setCompany((event.target.value || null) as Listener | null)}
          >
            <option value="">Leave it to whoever is speaking</option>
            <option value="familiar">People here know me</option>
            <option value="unfamiliar">People here do not know me</option>
          </select>
        </div>
        <button className="speak-button" disabled={busy || !label.trim()} onClick={add}>
          Add place
        </button>
      </div>

      {status && <p className="sheet-note" role="status">{status}</p>}
    </div>
  );
}

// The only form in the app. Everything else Echora learns, it learns from
// messages the speaker actually accepted.
function Onboarding() {
  const [people, setPeople] = useState("");
  const [places, setPlaces] = useState("");
  const [things, setThings] = useState("");
  const [word, setWord] = useState("");
  const [wording, setWording] = useState("");
  const [status, setStatus] = useState("");

  const lines = (value: string) => value.split(/[\n,]/).map((item) => item.trim()).filter(Boolean);

  async function save() {
    setStatus("Saving…");
    try {
      const response = await fetch(`${API}/api/v1/profile`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          people: lines(people),
          places: lines(places),
          things: lines(things),
          details: word.trim() && wording.trim() ? [{ word: word.trim(), wording: wording.trim() }] : [],
        }),
      });
      if (!response.ok) throw new Error("save failed");
      const body = await response.json();
      setStatus(
        body.refused?.length
          ? `Saved. “${body.refused[0]}” was not kept: the fuller wording has to contain the ordinary one.`
          : `Saved ${body.lexicon_size} words and ${body.specialization_size} details.`,
      );
    } catch {
      setStatus("That could not be saved.");
    }
  }

  return (
    <div className="onboarding">
      <h3>Tell Echora about yourself</h3>
      <p>One per line. You can leave any of these empty and fill them in later.</p>
      <label htmlFor="ob-people">People you talk to</label>
      <textarea id="ob-people" rows={2} value={people} onChange={(event) => setPeople(event.target.value)} />
      <label htmlFor="ob-places">Rooms and places you are in</label>
      <textarea id="ob-places" rows={2} value={places} onChange={(event) => setPlaces(event.target.value)} />
      <label htmlFor="ob-things">Things you ask for often</label>
      <textarea id="ob-things" rows={2} value={things} onChange={(event) => setThings(event.target.value)} />
      <label htmlFor="ob-word">Something you have a particular version of</label>
      <div className="onboarding-pair">
        <input id="ob-word" placeholder="the soap" value={word} onChange={(event) => setWord(event.target.value)} />
        <span aria-hidden="true">→</span>
        <input
          aria-label="How you would say it"
          placeholder="the Dove soap"
          value={wording}
          onChange={(event) => setWording(event.target.value)}
        />
      </div>
      <button className="speak-button" onClick={save}>Save profile</button>
      {status && <p className="sheet-note" role="status">{status}</p>}
    </div>
  );
}
