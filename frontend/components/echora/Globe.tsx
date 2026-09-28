'use client';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VoiceMotion } from './voiceMotion';
import { createOrganicSphere } from './organic-sphere/renderSphere';

export type GlobeState =
  | 'idle'
  | 'listening'
  | 'processing'
  | 'speaking'
  | 'error';

export default function Globe({
  state,
  analyser,
  paused,
  playbackAnalyser,
}: {
  state: GlobeState;
  analyser: React.RefObject<AnalyserNode | null>;
  paused: boolean;
  playbackAnalyser?: React.RefObject<AnalyserNode | null>;
}) {
  const host = useRef<HTMLDivElement>(null);
  const current = useRef({ state, paused });
  useEffect(() => {
    current.current = { state, paused };
  }, [state, paused]);
  const [fallback, setFallback] = useState(false);
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        alpha: true,
        antialias: true,
        powerPreference: 'low-power',
      });
    } catch {
      queueMicrotask(() => setFallback(true));
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.7));
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.NoToneMapping;
    element.appendChild(renderer.domElement);
    const sphere = createOrganicSphere(renderer);
    renderer.debug.onShaderError = () => setFallback(true);
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
    let samples = new Float32Array(1024);
    const motion = new VoiceMotion();
    let lastAnalyser: AnalyserNode | null = null;
    let frame = 0,
      last = 0,
      disposed = false,
      inView = true;
    const resize = () => {
      const w = element.clientWidth,
        h = element.clientHeight;
      if (w && h) {
        sphere.resize(w, h);
        sphere.render();
      }
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    const visible = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
    });
    visible.observe(element);
    const lost = (event: Event) => {
      event.preventDefault();
      setFallback(true);
    };
    renderer.domElement.addEventListener('webglcontextlost', lost);
    const tick = (now: number) => {
      if (disposed) return;
      frame = requestAnimationFrame(tick);
      if (document.hidden || !inView || now - last < 32) return;
      const dt = Math.min((now - last) / 1000, 0.07);
      last = now;
      const still = reduce.matches || current.current.paused;
      const meter =
        current.current.state === 'listening'
          ? analyser.current
          : current.current.state === 'speaking'
            ? (playbackAnalyser?.current ?? null)
            : null;
      if (meter !== lastAnalyser) {
        if (meter) motion.reset();
        lastAnalyser = meter;
      }
      if (meter) {
        if (samples.length !== meter.fftSize)
          samples = new Float32Array(meter.fftSize);
        meter.getFloatTimeDomainData(samples);
      }
      const energy = motion.step(meter ? samples : null, dt);
      if (
        sphere.update(dt, energy, current.current.state === 'processing', still)
      ) {
        sphere.render();
      }
    };
    resize();
    frame = requestAnimationFrame(tick);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      visible.disconnect();
      renderer.domElement.removeEventListener('webglcontextlost', lost);
      sphere.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [analyser, playbackAnalyser]);
  return (
    <div className="globe-wrap" aria-hidden="true">
      <div
        ref={host}
        className={`globe-canvas ${fallback ? 'is-hidden' : ''}`}
      />
      {fallback && <div className="static-globe" />}
    </div>
  );
}
