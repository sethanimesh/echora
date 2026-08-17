"use client";

import { ChangeEvent, Dispatch, SetStateAction, useEffect, useMemo, useRef, useState } from "react";

type Hypothesis = {
  id: string;
  literal_text: string;
  sequence_score: number;
  search_weight: number;
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
};

type SpeechAudio = {
  audio_base64: string;
  media_type: string;
  voice: string;
  model: string;
};

type CommunicationContext = "general" | "home" | "care" | "outdoors";

type Transcription = {
  request_id: string;
  backend: string;
  model: string;
  device: string;
  context: CommunicationContext;
  hypotheses: Hypothesis[];
  ranker: {
    decision: "selected" | "ambiguous";
    selected_message_id: string | null;
    reason: string;
    source: string;
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
  detail: string;
};

type Phase = "idle" | "recording" | "working" | "ready" | "error";
type Stage = "idle" | "recording" | "working" | "choosing" | "composing" | "error";
type Overlay = "evidence" | "about" | null;

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
  const [overlay, setOverlay] = useState<Overlay>(null);

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

  // The stage is derived, not stored. The backend only fills recommended_message_id
  // when a single message survived the chain, so a confident result opens straight
  // on "composing" and an ambiguous one opens on "choosing".
  const stage: Stage = phase === "ready" ? (selectedId ? "composing" : "choosing") : phase;

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
    streamRef.current = null;
    audioContextRef.current = null;
  }

  // Bar levels are written straight to the DOM as a scaleY factor. Routing them
  // through state would re-render the whole screen 60 times a second, and animating
  // `height` would force layout every frame; `transform` stays on the compositor.
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
      for (let index = 0; index < BAR_COUNT; index += 1) {
        const target = Math.min(1, Math.max(0.14, (data[index * 2] / 255) * 1.35));
        level[index] += (target - level[index]) * 0.35; // easing lives here, not in CSS
        (node.children[index] as HTMLElement | undefined)?.style.setProperty("--l", level[index].toFixed(3));
      }
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
  function choose(candidate: MessageCandidate) {
    setSelectedId(candidate.message_id);
    setMessage(candidate.corrected_text);
    setCopied(false);
    void speak(candidate.corrected_text);
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
        <button className="bar-action" onClick={() => setOverlay("about")}>
          About
        </button>
      </header>

      <div className="stage" ref={stageRef}>
        {stage === "idle" && (
          <IdleStage
            modelReady={modelReady}
            detail={health?.detail}
            context={context}
            setContext={setContext}
            busy={busy}
            onRecord={startRecording}
            onPickFile={() => fileRef.current?.click()}
          />
        )}
        {stage === "recording" && (
          <RecordingStage elapsed={elapsed} waveRef={waveRef} onStop={stopRecording} onCancel={cancelRecording} />
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
          />
        )}
        {stage === "error" && <ErrorStage message={error} onRetry={reset} />}
      </div>

      <p className="visually-hidden" aria-live="polite">{liveText}</p>

      <footer className="hint">
        <span>Your chosen message is spoken as soon as it is ready.</span>
        <span>No history is saved after refresh</span>
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
        <p className="sheet-note">Local-first assistive communication. No history is saved after refresh.</p>
      </Sheet>
    </main>
  );
}

function IdleStage({
  modelReady,
  detail,
  context,
  setContext,
  busy,
  onRecord,
  onPickFile,
}: {
  modelReady: boolean;
  detail?: string;
  context: CommunicationContext;
  setContext: Dispatch<SetStateAction<CommunicationContext>>;
  busy: boolean;
  onRecord: () => void;
  onPickFile: () => void;
}) {
  return (
    <section className="stage-idle">
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
      <strong className="stage-title">{modelReady ? "Tap to speak" : detail || "Waiting for the speech model"}</strong>
      <span className="stage-hint">Take your time. Pauses are welcome.</span>

      <fieldset className="context-chips" disabled={busy}>
        <legend>Where are you speaking?</legend>
        <div>
          {contexts.map((option) => (
            <label key={option.value} className={context === option.value ? "active" : ""} title={option.hint}>
              <input
                type="radio"
                name="context"
                value={option.value}
                checked={context === option.value}
                onChange={() => setContext(option.value)}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <button className="text-button" onClick={onPickFile}>Choose an audio file</button>
    </section>
  );
}

function RecordingStage({
  elapsed,
  waveRef,
  onStop,
  onCancel,
}: {
  elapsed: number;
  waveRef: React.RefObject<HTMLDivElement | null>;
  onStop: () => void;
  onCancel: () => void;
}) {
  return (
    <section className="stage-recording">
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
}) {
  const multiple = result.messages.length > 1;
  return (
    <section className="stage-composing">
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
