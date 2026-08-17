"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";

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
  timing: { total_seconds: number; asr_seconds: number };
  warnings: string[];
};

type CommunicationContext = "general" | "home" | "care" | "outdoors";

const contexts: { value: CommunicationContext; label: string; hint: string }[] = [
  { value: "general", label: "General", hint: "No setting assumptions" },
  { value: "home", label: "At home", hint: "Everyday needs and household help" },
  { value: "care", label: "Hospital or care", hint: "Symptoms, comfort, and assistance" },
  { value: "outdoors", label: "Outdoors", hint: "Travel, safety, and nearby places" },
];

type Health = {
  status: "ready" | "degraded" | "starting";
  asr_backend: string;
  model_ready: boolean;
  groq_configured: boolean;
  detail: string;
};

const API = process.env.NEXT_PUBLIC_ECHORA_API_URL || "http://127.0.0.1:8000";
const calmBars = Array.from({ length: 25 }, (_, index) => 18 + ((index * 13) % 22));

function formatBackend(value?: string) {
  if (value === "local") return "Local Mac";
  if (value === "pod") return "RunPod Pod";
  if (value === "runpod") return "RunPod Serverless";
  return "Connecting";
}

export default function Home() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [phase, setPhase] = useState<"idle" | "recording" | "working" | "ready" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [bars, setBars] = useState(calmBars);
  const [result, setResult] = useState<Transcription | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [context, setContext] = useState<CommunicationContext>("general");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const animationRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const response = await fetch(`${API}/api/v1/health`);
        if (!response.ok) throw new Error("Health check failed");
        setHealth(await response.json());
        setHealthError(false);
      } catch {
        setHealthError(true);
      }
    };
    check();
    const interval = window.setInterval(check, 15000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => () => stopMedia(), []);

  useEffect(() => {
    if (phase !== "working") return;
    setProgress(0);
    const interval = window.setInterval(() => setProgress((value) => Math.min(value + 1, 3)), 1800);
    return () => window.clearInterval(interval);
  }, [phase]);

  const selected = useMemo(
    () => result?.messages.find((candidate) => candidate.message_id === selectedId) || null,
    [result, selectedId],
  );

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
    setBars(calmBars);
  }

  function animateWaveform(stream: MediaStream) {
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 128;
    context.createMediaStreamSource(stream).connect(analyser);
    const data = new Uint8Array(analyser.frequencyBinCount);
    audioContextRef.current = context;
    const draw = () => {
      analyser.getByteFrequencyData(data);
      setBars(calmBars.map((_, index) => Math.max(8, Math.min(58, data[index * 2] * 0.23))));
      animationRef.current = requestAnimationFrame(draw);
    };
    draw();
  }

  async function startRecording() {
    setError("");
    setResult(null);
    setConfirmed(false);
    setCopied(false);
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
      timerRef.current = setInterval(() => {
        setElapsed((value) => {
          if (value >= 44) recorder.stop();
          return value + 1;
        });
      }, 1000);
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
    setError("");
    setConfirmed(false);
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
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Something went wrong while transcribing");
      setPhase("error");
    }
  }

  function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) submitAudio(file, file.name);
    event.target.value = "";
  }

  function choose(candidate: MessageCandidate) {
    setSelectedId(candidate.message_id);
    setMessage(candidate.corrected_text);
    setConfirmed(false);
    setCopied(false);
  }

  function confirmMessage() {
    if (!selected || !message.trim()) return;
    setConfirmed(true);
  }

  async function copyMessage() {
    if (!confirmed) return;
    await navigator.clipboard.writeText(message.trim());
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  function speakMessage() {
    if (!confirmed || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(message.trim());
    utterance.rate = 0.92;
    window.speechSynthesis.speak(utterance);
  }

  function reset() {
    setResult(null);
    setSelectedId(null);
    setMessage("");
    setConfirmed(false);
    setError("");
    setElapsed(0);
    setPhase("idle");
  }

  const modelReady = Boolean(health?.model_ready) && !healthError;

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Echora home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>Echora</span>
        </a>
        <div className={`model-status ${modelReady ? "online" : "offline"}`} role="status">
          <span className="status-dot" />
          <span>{healthError ? "Backend offline" : `${formatBackend(health?.asr_backend)} · ${modelReady ? "Ready" : "Starting"}`}</span>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span>Literal speech, then your choice</span></div>
        <h1>Say it your way.</h1>
        <p className="lede">Take your time. Pauses are welcome. Echora listens for the words you spoke and lets you decide what to communicate.</p>

        <fieldset className="context-picker" disabled={phase === "recording" || phase === "working"}>
          <legend>Where are you speaking?</legend>
          <div>
            {contexts.map((option) => (
              <label key={option.value} className={context === option.value ? "active" : ""}>
                <input
                  type="radio"
                  name="context"
                  value={option.value}
                  checked={context === option.value}
                  onChange={() => setContext(option.value)}
                />
                <span><b>{option.label}</b><small>{option.hint}</small></span>
              </label>
            ))}
          </div>
          <p>The setting helps interpret likely meaning, but never changes the literal transcript.</p>
        </fieldset>

        <div className={`voice-card ${phase}`}>
          {phase === "recording" ? (
            <>
              <div className="recording-label"><span className="recording-dot" />Listening · {elapsed}s</div>
              <div className="waveform" aria-label="Live microphone level">
                {bars.map((height, index) => <i key={index} style={{ height }} />)}
              </div>
              <button className="stop-button" onClick={stopRecording}><span />Finish speaking</button>
              <button className="text-button" onClick={cancelRecording}>Cancel recording</button>
            </>
          ) : phase === "working" ? (
            <div className="working-state" role="status" aria-live="polite">
              <div className="listening-orbit"><span /><span /><span /></div>
              <strong>{progressLabel}</strong>
              <p>This can take a few seconds. Your literal transcript stays separate from wording suggestions.</p>
              <div className="progress-track"><i style={{ width: `${30 + progress * 22}%` }} /></div>
            </div>
          ) : (
            <>
              <button className="mic-button" onClick={startRecording} disabled={!modelReady} aria-label="Start recording">
                <span className="mic-shape" aria-hidden="true"><i /></span>
              </button>
              <strong className="voice-title">{modelReady ? "Tap to speak" : health?.detail || "Waiting for the speech model"}</strong>
              <span className="voice-hint">Speak naturally for up to 45 seconds</span>
              <div className="divider"><span>or</span></div>
              <button className="upload-button" onClick={() => fileRef.current?.click()}>Choose an audio file</button>
              <input ref={fileRef} className="visually-hidden" type="file" accept="audio/*,.wav,.webm,.ogg,.m4a,.mp3" onChange={upload} />
            </>
          )}
        </div>
        {error && <div className="error-banner" role="alert"><strong>We could not finish that.</strong><span>{error}</span><button onClick={reset}>Try again</button></div>}
      </section>

      {result && phase === "ready" && (
        <section className="results" aria-live="polite">
          <div className="section-heading">
            <div>
              <span className="kicker">Your message</span>
              <h2>{result.needs_user_choice ? "Which one sounds right?" : "Here’s your message"}</h2>
            </div>
            <button className="new-recording" onClick={reset}>Start over</button>
          </div>

          <p className="decision-note">{result.ranker.reason}</p>
          <p className="context-note">Setting used: <b>{contexts.find((item) => item.value === result.context)?.label}</b></p>
          {result.warnings.map((warning) => <p className="warning" key={warning}>{warning}</p>)}

          <div className={`candidate-grid ${result.messages.length === 1 ? "single" : ""}`}>
            {result.messages.map((candidate, index) => {
              const evidenceWeight = candidate.source_hypothesis_ids.reduce(
                (total, id) => total + (result.hypotheses.find((item) => item.id === id)?.search_weight || 0),
                0,
              );
              const active = candidate.message_id === selectedId;
              return (
                <button className={`candidate-card ${active ? "selected" : ""}`} key={candidate.message_id} onClick={() => choose(candidate)} aria-pressed={active}>
                  <span className="candidate-topline"><b>Option {index + 1}</b><i>{Math.round(evidenceWeight * 100)}% grouped search weight</i></span>
                  <span className="candidate-message">{candidate.corrected_text}</span>
                  {Object.keys(candidate.word_alternatives || {}).length > 0 && (
                    <span className="word-alternatives">
                      {Object.entries(candidate.word_alternatives).map(([word, options]) => (
                        <span className="word-alternative" key={word}>
                          <b>{word}</b> — also heard as
                          <i>{options.join(" · ")}</i>
                        </span>
                      ))}
                    </span>
                  )}
                  <span className="intent-label">Heard as</span>
                  <span className="intent-text">{candidate.interpreted_intent}</span>
                  <span className="literal-label">All transcriptions</span>
                  <span className="literal-text">{candidate.source_literals.map((text) => `“${text}”`).join(" · ")}</span>
                  <span className="repair-note">{candidate.repair_note}</span>
                  <span className="select-indicator">{active ? "Selected" : "Choose this"}</span>
                </button>
              );
            })}
          </div>

          {selected && (
            <div className="confirm-panel">
              <label htmlFor="message"><span>Message to communicate</span><small>You can edit this before confirming.</small></label>
              <textarea id="message" value={message} onChange={(event) => { setMessage(event.target.value); setConfirmed(false); }} rows={2} />
              {!confirmed ? (
                <button className="confirm-button" onClick={confirmMessage} disabled={!message.trim()}>Confirm this message</button>
              ) : (
                <div className="confirmed-actions">
                  <div className="confirmed-label"><span>✓</span><div><b>Confirmed by you</b><small>Ready to communicate</small></div></div>
                  <button onClick={copyMessage}>{copied ? "Copied" : "Copy"}</button>
                  <button className="speak-button" onClick={speakMessage}>Speak aloud</button>
                </div>
              )}
            </div>
          )}

          <details className="evidence">
            <summary>View all literal ASR evidence</summary>
            <div className="evidence-list">
              {result.hypotheses.map((item) => (
                <div key={item.id}><span>{item.literal_text}</span><code>{(item.search_weight * 100).toFixed(1)}% search weight · score {item.sequence_score.toFixed(3)}</code></div>
              ))}
            </div>
            <p>Search weights compare only these beam hypotheses. They are not calibrated confidence.</p>
          </details>
        </section>
      )}

      <section className="promise">
        <div><span className="promise-number">01</span><h3>Your words stay visible</h3><p>The literal transcript is never replaced by a cleaned-up message.</p></div>
        <div><span className="promise-number">02</span><h3>Uncertainty stays honest</h3><p>When the evidence is unclear, Echora asks you instead of pretending.</p></div>
        <div><span className="promise-number">03</span><h3>You are in control</h3><p>Nothing is copied or spoken until you explicitly confirm it.</p></div>
      </section>

      <footer><span>Echora · Local-first assistive communication</span><span>No history is saved after refresh</span></footer>
    </main>
  );
}
