'use client';
import { RecordingFaces } from './recordingFaces';
import { useEffect, useRef, useState } from 'react';
export function useRecorder() {
  const [recording, setRecording] = useState(false),
    [starting, setStarting] = useState(false),
    [seconds, setSeconds] = useState(0);
  const [frames, setFrames] = useState<Blob[]>([]);
  const [cameraNote, setCameraNote] = useState('');
  const faceSampler = useRef(new RecordingFaces());
  const capturedFrames = useRef<Blob[]>([]);
  const cameraPreview = useRef<HTMLVideoElement | null>(null);
  const faceRequested = useRef(false);
  const [audio, setAudio] = useState<Blob | null>(null),
    [url, setUrl] = useState(''),
    [error, setError] = useState('');
  const analyser = useRef<AnalyserNode | null>(null),
    recorder = useRef<MediaRecorder | null>(null),
    stream = useRef<MediaStream | null>(null),
    context = useRef<AudioContext | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null),
    mounted = useRef(true),
    generation = useRef(0);
  function release() {
    faceSampler.current.stop();
    if (cameraPreview.current) cameraPreview.current.srcObject = null;
    cameraPreview.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    analyser.current = null;
    if (context.current) {
      void context.current.close();
      context.current = null;
    }
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  }
  function stop() {
    if (recorder.current?.state !== 'recording') generation.current++;
    capturedFrames.current = faceSampler.current.stop();
    if (recorder.current?.state === 'recording') recorder.current.stop();
    release();
    setRecording(false);
    setStarting(false);
  }
  function disposeRecording() {
    mounted.current = false;
    generation.current++;
    const device = recorder.current;
    if (device) device.onstop = null;
    if (device?.state === 'recording') device.stop();
    release();
  }
  useEffect(() => {
    mounted.current = true;
    const hide = () => {
      if (document.hidden && stream.current?.getVideoTracks().length) stop();
    };
    document.addEventListener('visibilitychange', hide);
    return () => {
      document.removeEventListener('visibilitychange', hide);
      disposeRecording();
    };
  }, []);
  useEffect(() => {
    const next = audio ? URL.createObjectURL(audio) : '';
    const frame = requestAnimationFrame(() => setUrl(next));
    return () => {
      cancelAnimationFrame(frame);
      if (next) URL.revokeObjectURL(next);
    };
  }, [audio]);
  async function start(options?: {
    includeFace: boolean;
    video: HTMLVideoElement | null;
  }) {
    const attempt = ++generation.current;
    setStarting(true);
    setError('');
    setAudio(null);
    setFrames([]);
    setCameraNote('');
    capturedFrames.current = [];
    faceRequested.current = !!options?.includeFace;
    setSeconds(0);
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder)
        throw new Error(
          'Recording is unavailable in this browser. You can upload an audio file or type instead.',
        );
      const input = await navigator.mediaDevices.getUserMedia({
        video: options?.includeFace
          ? {
              facingMode: 'user',
              width: { ideal: 640 },
              height: { ideal: 480 },
            }
          : false,
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      });
      if (!mounted.current || attempt !== generation.current) {
        input.getTracks().forEach((t) => t.stop());
        return;
      }
      stream.current = input;
      if (options?.includeFace && options.video) {
        cameraPreview.current = options.video;
        options.video.srcObject = input;
        await options.video.play();
      }
      if (!mounted.current || attempt !== generation.current) {
        input.getTracks().forEach((t) => t.stop());
        return;
      }
      try {
        const ctx = new AudioContext();
        context.current = ctx;
        await ctx.resume();
        const meter = ctx.createAnalyser();
        meter.fftSize = 1024;
        ctx.createMediaStreamSource(input).connect(meter);
        analyser.current = meter;
      } catch {
        /* Recording can work without an audio visualizer. */
      }
      if (!mounted.current || attempt !== generation.current) {
        input.getTracks().forEach((track) => track.stop());
        return;
      }
      const mime = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm'].find(
        (v) => MediaRecorder.isTypeSupported(v),
      );
      const device = new MediaRecorder(
        new MediaStream(input.getAudioTracks()),
        mime ? { mimeType: mime } : undefined,
      );
      recorder.current = device;
      const chunks: Blob[] = [];
      device.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      device.onstop = () => {
        if (mounted.current && attempt === generation.current) {
          setAudio(new Blob(chunks, { type: device.mimeType }));
          const snapshots = capturedFrames.current.length
            ? capturedFrames.current
            : faceSampler.current.stop();
          setFrames(snapshots);
          if (faceRequested.current)
            setCameraNote(
              snapshots.length === 3
                ? 'Three snapshots from this recording will be submitted with your audio.'
                : 'No complete snapshot set was captured. This recording will use voice cues only.',
            );
          setRecording(false);
        }
        if (attempt === generation.current) release();
        else input.getTracks().forEach((track) => track.stop());
      };
      device.onerror = () => {
        if (!mounted.current || attempt !== generation.current) return;
        setError('Recording stopped unexpectedly. Please try again.');
        stop();
      };
      device.start(250);
      if (options?.includeFace && options.video)
        faceSampler.current.start(options.video);
      setRecording(true);
      const started = Date.now();
      timer.current = setInterval(() => {
        const elapsed = Math.floor((Date.now() - started) / 1000);
        setSeconds(elapsed);
        if (elapsed >= 60) stop();
      }, 250);
    } catch (e) {
      if (!mounted.current || attempt !== generation.current) return;
      release();
      setError(
        e instanceof DOMException && e.name === 'NotAllowedError'
          ? options?.includeFace
            ? 'Microphone or camera access was declined. Allow both, or turn off facial cues and try voice only.'
            : 'Microphone access was declined. Allow it in your browser, upload a recording, or type your message.'
          : e instanceof Error
            ? e.message
            : 'Could not start recording.',
      );
    } finally {
      if (mounted.current && attempt === generation.current) setStarting(false);
    }
  }
  return {
    recording,
    frames,
    cameraNote,
    starting,
    seconds,
    audio,
    url,
    error,
    analyser,
    start,
    stop,
    discardFaces: () => {
      capturedFrames.current = [];
      setFrames([]);
      setCameraNote(
        'Facial snapshots removed. Only your audio will be submitted.',
      );
    },
    clear: () => {
      setAudio(null);
      setFrames([]);
      setCameraNote('');
    },
  };
}
