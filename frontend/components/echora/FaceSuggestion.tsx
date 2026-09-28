'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { FaceCamera } from './faceCamera';
import { tones, type Tone } from './delivery';
export type FaceSuggestionResult = {
  id: string;
  revision: number;
  state: 'analyzing' | 'ready' | 'error' | 'stopped';
  visibility?: 'clear_face' | 'no_face' | 'multiple_faces' | 'obscured';
  cue?: 'smile' | 'broad_smile' | 'positive_expression' | 'neutral' | 'unclear';
  tone?: Tone | null;
  message?: string;
  model?: string;
  elapsed_ms?: number;
  comparison?: {
    local: FaceModelResult;
    gemini: FaceModelResult;
    agreement: 'same_style' | 'different_style' | 'not_comparable';
    selected: 'gemini';
    policy: string;
  };
};
export type FaceModelResult = {
  state: 'ready' | 'error';
  model?: string;
  visibility?: string;
  cue?: string;
  tone?: Tone | null;
  elapsed_ms?: number;
  message?: string;
  frames?: {
    visibility: string;
    cue: string;
    label?: string;
    score?: number;
    margin?: number;
  }[];
};
const cueDescriptions = {
  positive_expression:
    'The local classifier found a consistent positive-expression pattern.',
  smile: 'A smile was visible in at least two snapshots.',
  broad_smile: 'A broad smile was visible in at least two snapshots.',
  neutral: 'No distinctive expression was visible in at least two snapshots.',
  unclear: 'The snapshots did not show a consistent usable cue.',
};
export function faceDescription(suggestion: FaceSuggestionResult) {
  if (suggestion.visibility === 'no_face')
    return 'The model could not find a clear face in every snapshot.';
  if (suggestion.visibility === 'multiple_faces')
    return 'More than one face was visible. Try again with only you in view.';
  if (suggestion.visibility === 'obscured')
    return 'The view was unclear. Try adjusting the lighting or camera position.';
  return cueDescriptions[suggestion.cue ?? 'unclear'];
}
export default function FaceSuggestion({
  configured,
  disabled,
  cameraBlocked,
  suggestion,
  voiceTone,
  onCheck,
  onStop,
  onAccept,
}: {
  configured: boolean;
  disabled: boolean;
  cameraBlocked: boolean;
  suggestion?: FaceSuggestionResult | null;
  voiceTone?: Tone | null;
  onCheck: (frames: Blob[]) => Promise<void>;
  onStop: () => void;
  onAccept: () => void;
}) {
  const [camera] = useState(() => new FaceCamera());
  const video = useRef<HTMLVideoElement>(null);
  const alive = useRef(true);
  const operation = useRef(0);
  const [open, setOpen] = useState(false);
  const [starting, setStarting] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [dismissed, setDismissed] = useState('');
  const analyzing = suggestion?.state === 'analyzing';
  const stopCamera = useCallback(() => {
    operation.current += 1;
    camera.stop();
    setOpen(false);
    setStarting(false);
    setReady(false);
    setCapturing(false);
  }, [camera]);
  useEffect(() => {
    alive.current = true;
    const hide = () => {
      if (document.hidden) stopCamera();
    };
    document.addEventListener('visibilitychange', hide);
    return () => {
      alive.current = false;
      camera.stop();
      document.removeEventListener('visibilitychange', hide);
    };
  }, [camera, stopCamera]);
  useEffect(() => {
    if (disabled || cameraBlocked) {
      camera.stop();
      operation.current += 1;
      queueMicrotask(() => {
        if (!alive.current) return;
        setOpen(false);
        setStarting(false);
        setReady(false);
        setCapturing(false);
      });
    }
  }, [disabled, cameraBlocked, camera]);
  async function start() {
    if (!video.current || disabled || cameraBlocked || starting) return;
    const token = ++operation.current;
    setOpen(true);
    setStarting(true);
    setError('');
    try {
      const active = await camera.start(video.current);
      if (alive.current && token === operation.current && active)
        setReady(true);
    } catch {
      if (alive.current && token === operation.current) {
        stopCamera();
        setError(
          'The camera could not open. Allow camera access in your browser, or keep using your voice and manual tone.',
        );
      }
    } finally {
      if (alive.current && token === operation.current) setStarting(false);
    }
  }
  async function check() {
    if (!ready || disabled || capturing) return;
    const token = ++operation.current;
    setCapturing(true);
    setError('');
    try {
      const frames = await camera.capture();
      if (!alive.current || token !== operation.current) return;
      stopCamera();
      await onCheck(frames);
    } catch (error) {
      if (alive.current && token === operation.current) {
        stopCamera();
        setError(
          error instanceof Error
            ? error.message
            : 'Camera capture failed. Nothing was sent.',
        );
      }
    } finally {
      if (alive.current && token === operation.current) setCapturing(false);
    }
  }
  return (
    <div className="voice-suggestion face-suggestion">
      <h4>
        Include facial cues <span>Optional camera check</span>
      </h4>
      <p id="face-cue-note">
        Preview stays on this device. “Check facial cues” sends three snapshots
        to Google Gemini Flash and closes the camera. It can suggest a tone from
        visible cues; it cannot tell how you feel. Speaking pace and your words
        stay unchanged.
      </p>
      <video
        ref={video}
        muted
        playsInline
        autoPlay
        hidden={!open}
        className="face-camera-preview"
        aria-label="Your camera preview"
      />
      <div className="delivery-options">
        {!open && !analyzing && (
          <button
            type="button"
            disabled={disabled || cameraBlocked || !configured}
            aria-describedby="face-cue-note"
            onClick={() => void start()}
          >
            Enable camera preview
          </button>
        )}
        {open && (
          <>
            <button
              type="button"
              disabled={!ready || starting || capturing || disabled}
              onClick={() => void check()}
            >
              {starting
                ? 'Opening camera…'
                : capturing
                  ? 'Capturing three snapshots…'
                  : 'Check facial cues'}
            </button>
            <button type="button" onClick={stopCamera}>
              Turn camera off
            </button>
          </>
        )}
        {analyzing && (
          <button type="button" onClick={onStop}>
            Stop camera analysis
          </button>
        )}
      </div>
      {cameraBlocked && (
        <p>
          Stop webcam head or gaze control in Accessibility before opening this
          camera preview.
        </p>
      )}
      {!configured && (
        <p>
          Gemini is not configured for camera cues. You can still choose your
          tone.
        </p>
      )}
      <div aria-live="polite" aria-atomic="true">
        {error && <p>{error}</p>}
        {analyzing && <p>Checking the snapshots… The camera is off.</p>}
        {suggestion?.state === 'error' && <p>{suggestion.message}</p>}
        {suggestion?.state === 'stopped' && (
          <p>Camera analysis stopped. Your delivery is unchanged.</p>
        )}
        {suggestion?.state === 'ready' && dismissed !== suggestion.id && (
          <>
            <p>
              {faceDescription(suggestion)}{' '}
              {suggestion.tone
                ? `Camera suggestion: ${tones[suggestion.tone]}.`
                : 'No tone change suggested.'}
            </p>
            {voiceTone && suggestion.tone && (
              <p>
                {voiceTone === suggestion.tone
                  ? `Voice and camera both suggest ${tones[voiceTone]}.`
                  : `Voice suggests ${tones[voiceTone]}; camera suggests ${tones[suggestion.tone]}. Choose whichever fits your intended delivery.`}
              </p>
            )}
            <div className="delivery-options">
              {suggestion.tone && (
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => {
                    onAccept();
                    setDismissed(suggestion.id);
                  }}
                >
                  Use camera tone
                </button>
              )}
              <button
                type="button"
                disabled={disabled}
                onClick={() => setDismissed(suggestion.id)}
              >
                Dismiss camera suggestion
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
