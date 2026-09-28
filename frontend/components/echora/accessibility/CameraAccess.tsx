/* eslint-disable react/react-compiler -- Imperative camera/pointer loop uses refs to share fresh samples without restarting hardware on React renders. */
'use client';
import { useEffect, useRef, useState } from 'react';
import {
  faceFeature,
  fitCalibration,
  mapPoint,
  type Calibration,
  type Point,
} from './cameraMath';
import type { PointerSample } from './AccessInput';
const training: Point[] = [
  { x: 0.5, y: 0.5 },
  { x: 0.15, y: 0.15 },
  { x: 0.5, y: 0.15 },
  { x: 0.85, y: 0.15 },
  { x: 0.85, y: 0.5 },
  { x: 0.85, y: 0.85 },
  { x: 0.5, y: 0.85 },
  { x: 0.15, y: 0.85 },
  { x: 0.15, y: 0.5 },
];
const validation: Point[] = [
  { x: 0.3, y: 0.3 },
  { x: 0.7, y: 0.7 },
  { x: 0.5, y: 0.5 },
];
export default function CameraAccess({
  mode,
  pointer,
  stop,
}: {
  mode: 'head' | 'gaze';
  pointer: React.RefObject<PointerSample>;
  stop: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const enabledRef = useRef(false);
  const [enabled, setEnabled] = useState(false);
  const [run, setRun] = useState(0),
    [status, setStatus] = useState('Starting camera…'),
    [stage, setStage] = useState<
      'loading' | 'calibrating' | 'validating' | 'ready' | 'failed'
    >('loading');
  const [target, setTarget] = useState<Point>(training[0]),
    [progress, setProgress] = useState(0);
  const callbacks = useRef({ stop });
  callbacks.current = { stop };
  useEffect(() => {
    let disposed = false,
      stream: MediaStream | null = null,
      worker: Worker | null = null,
      interval = 0,
      startupTimeout = 0,
      inFlight = false,
      lastFrame = -1;
    let step = 0,
      started = 0,
      model: Calibration | null = null,
      smoothed: Point | null = null;
    let samples: Point[] = [];
    const allSamples: { feature: Point; target: Point }[] = [],
      errors: number[] = [];
    let phase: 'calibrating' | 'validating' | 'ready' = 'calibrating';
    pointer.current = null;
    enabledRef.current = false;
    setEnabled(false);
    setStage('loading');
    setStatus('Starting camera…');
    setProgress(0);
    setTarget(training[0]);
    const release = () => {
      stream?.getTracks().forEach((t) => t.stop());
      worker?.terminate();
      worker = null;
      window.clearTimeout(startupTimeout);
      window.clearInterval(interval);
      pointer.current = null;
    };
    const fail = (message: string) => {
      if (disposed) return;
      release();
      setStage('failed');
      setStatus(message);
    };
    const reset = () => {
      started = 0;
      samples = [];
      pointer.current = null;
      setProgress(0);
    };
    const receive = (points: Point[], time: number) => {
      if (disposed) return;
      const now = performance.now();
      const feature =
        now - time < 400 && !document.hidden && document.hasFocus()
          ? faceFeature(points, mode)
          : null;
      if (!feature) {
        reset();
        smoothed = null;
        setStatus('Selection paused. Keep one face visible with eyes open.');
        return;
      }
      if (phase === 'ready' && model) {
        const mapped = mapPoint(model, feature);
        if (!Number.isFinite(mapped.x) || !Number.isFinite(mapped.y)) {
          reset();
          return;
        }
        smoothed = smoothed
          ? {
              x: smoothed.x * 0.72 + mapped.x * 0.28,
              y: smoothed.y * 0.72 + mapped.y * 0.28,
            }
          : mapped;
        pointer.current = {
          x: Math.max(12, Math.min(innerWidth - 12, smoothed.x * innerWidth)),
          y: Math.max(12, Math.min(innerHeight - 12, smoothed.y * innerHeight)),
          time: now,
          select: enabledRef.current,
        };
        setStatus(
          !enabledRef.current
            ? 'Practice first: check that the pointer follows you, then enable selections.'
            : mode === 'head'
              ? 'Move your head gently. Hold over a control to select.'
              : 'Look at a control and hold to select. Keep your head steady.',
        );
        return;
      }
      setStatus(
        phase === 'validating'
          ? 'Checking accuracy. Follow the dot.'
          : mode === 'head'
            ? 'Move your head gently toward the dot, then hold.'
            : 'Keep your head still. Look directly at the dot.',
      );
      if (!started) started = now;
      const elapsed = now - started;
      setProgress(Math.min(1, elapsed / 3000));
      if (elapsed > 1200) samples.push(feature);
      if (elapsed < 3000 || samples.length < 10) return;
      const sortedX = samples.map((p) => p.x).sort((a, b) => a - b),
        sortedY = samples.map((p) => p.y).sort((a, b) => a - b);
      const median = {
        x: sortedX[Math.floor(samples.length / 2)],
        y: sortedY[Math.floor(samples.length / 2)],
      };
      if (phase === 'calibrating') {
        allSamples.push({ feature: median, target: training[step] });
        step++;
        if (step === training.length) {
          model = fitCalibration(allSamples);
          if (!model) {
            fail(
              'Calibration was not consistent enough. Adjust your lighting and position, then retry, or try head control.',
            );
            return;
          }
          phase = 'validating';
          step = 0;
          setStage('validating');
        }
      } else if (model) {
        const prediction = mapPoint(model, median);
        errors.push(
          Math.hypot(
            prediction.x - validation[step].x,
            prediction.y - validation[step].y,
          ),
        );
        step++;
        if (step === validation.length) {
          if (
            Math.max(...errors) > 0.18 ||
            errors.reduce((a, b) => a + b, 0) / errors.length > 0.12
          ) {
            fail(
              'The pointer is not accurate enough yet. Selection is off. Retry calibration or choose another input method.',
            );
            return;
          }
          phase = 'ready';
          setStage('ready');
          reset();
          return;
        }
      }
      setTarget(phase === 'calibrating' ? training[step] : validation[step]);
      reset();
    };
    const start = async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia)
          throw new Error('Camera access is unavailable in this browser.');
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: 'user',
          },
          audio: false,
        });
        if (disposed) {
          release();
          return;
        }
        for (const track of stream.getVideoTracks())
          track.onended = () => fail('The camera stopped. Retry to reconnect.');
        const el = video.current;
        if (!el) {
          release();
          return;
        }
        el.srcObject = stream;
        await el.play();
        if (disposed) {
          release();
          return;
        }
        setStatus('Loading local camera controls…');
        worker = new Worker(new URL('./face.worker.ts', import.meta.url), {
          type: 'module',
        });
        startupTimeout = window.setTimeout(
          () =>
            fail(
              'Camera controls took too long to load. Retry or choose pointer dwell.',
            ),
          30000,
        );
        worker.onerror = (event) =>
          fail(
            `Camera worker could not start: ${event.message || 'The browser could not load the camera worker.'}`,
          );
        worker.onmessage = (event: MessageEvent) => {
          const data = event.data;
          if (data.type === 'error') {
            const detail =
              typeof data.message === 'string'
                ? data.message.slice(0, 400)
                : 'No error details were returned.';
            fail(
              `Camera ${data.stage === 'frame' ? 'frame processing' : 'startup'} failed: ${detail}`,
            );
            return;
          }
          if (data.type === 'ready') {
            window.clearTimeout(startupTimeout);
            setStage('calibrating');
            interval = window.setInterval(async () => {
              if (
                disposed ||
                inFlight ||
                document.hidden ||
                el.readyState < 2 ||
                lastFrame === el.currentTime
              )
                return;
              inFlight = true;
              lastFrame = el.currentTime;
              try {
                const bitmap = await createImageBitmap(el);
                if (disposed || !worker) {
                  bitmap.close();
                  return;
                }
                worker?.postMessage(
                  { type: 'frame', bitmap, time: performance.now() },
                  [bitmap],
                );
              } catch {
                inFlight = false;
                fail(
                  'The camera frame could not be read. Retry camera controls.',
                );
              }
            }, 100);
          } else if (data.type === 'result') {
            inFlight = false;
            receive(data.points, data.time);
          }
        };
        worker.postMessage({
          type: 'init',
          base: new URL('/accessibility', location.origin).href,
        });
      } catch (error) {
        if (!disposed)
          fail(
            error instanceof DOMException && error.name === 'NotAllowedError'
              ? 'Camera permission was declined. Allow camera access and retry, or use another input method.'
              : error instanceof Error
                ? error.message
                : 'Camera could not start.',
          );
      }
    };
    void start();
    const hide = () => {
      if (document.hidden) callbacks.current.stop();
    };
    const resize = () => {
      pointer.current = null;
      setRun((v) => v + 1);
    };
    document.addEventListener('visibilitychange', hide);
    window.addEventListener('resize', resize);
    return () => {
      disposed = true;
      release();
      document.removeEventListener('visibilitychange', hide);
      window.removeEventListener('resize', resize);
    };
  }, [mode, run, pointer]);
  return (
    <section
      className={`camera-access ${stage === 'ready' ? 'camera-ready' : ''}`}
      aria-label="Camera input trial"
    >
      <div
        className="camera-info"
        style={
          stage !== 'ready'
            ? {
                left: target.x > 0.5 ? 16 : 'auto',
                right: target.x > 0.5 ? 'auto' : 16,
                top: target.y > 0.5 ? 16 : 'auto',
                bottom: target.y > 0.5 ? 'auto' : 90,
              }
            : undefined
        }
      >
        <strong>
          {mode === 'head' ? 'Head pointer' : 'Eye gaze'} · experimental
        </strong>
        <video
          ref={video}
          autoPlay
          playsInline
          muted
          aria-label="Local camera preview"
        />
        <output aria-live="polite">{status}</output>
        <p>
          Camera frames stay on this device. Calibration is forgotten when
          stopped.
        </p>
        <div>
          <button onClick={() => setRun((v) => v + 1)}>Recalibrate</button>
          <button onClick={stop}>Stop camera</button>
        </div>
        {stage === 'ready' && !enabled && (
          <button
            data-access-enable
            onClick={() => {
              enabledRef.current = true;
              setEnabled(true);
            }}
          >
            Enable selections
          </button>
        )}
        {stage === 'ready' && (
          <div>
            <button
              onClick={() =>
                window.scrollBy({
                  top: -innerHeight * 0.55,
                  behavior: 'instant',
                })
              }
            >
              Scroll up
            </button>
            <button
              onClick={() =>
                window.scrollBy({
                  top: innerHeight * 0.55,
                  behavior: 'instant',
                })
              }
            >
              Scroll down
            </button>
          </div>
        )}
      </div>
      {(stage === 'calibrating' || stage === 'validating') && (
        <div
          className="calibration-dot"
          aria-hidden="true"
          style={{
            left: `${target.x * 100}%`,
            top: `${target.y * 100}%`,
            background: `conic-gradient(#315e8b ${progress * 360}deg,#e7edf4 0)`,
          }}
        >
          <span />
        </div>
      )}
    </section>
  );
}
