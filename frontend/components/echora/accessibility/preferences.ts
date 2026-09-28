export type AccessPreferences = {
  textSize: 'standard' | 'large' | 'largest';
  contrast: boolean;
  largeTargets: boolean;
  reduceMotion: boolean;
  simpleView: boolean;
  speechRate: number;
  keyboard: boolean;
  dwellMs: number;
  scanMs: number;
  switchKey: 'Space' | 'Enter' | 'F8';
};
export const defaults: AccessPreferences = {
  textSize: 'standard',
  contrast: false,
  largeTargets: false,
  reduceMotion: false,
  simpleView: false,
  speechRate: 0.9,
  keyboard: false,
  dwellMs: 1400,
  scanMs: 1800,
  switchKey: 'Space',
};
export function normalizePreferences(raw: unknown): AccessPreferences {
  const r =
    raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {};
  const number = (key: keyof AccessPreferences, min: number, max: number) =>
    typeof r[key] === 'number' && Number.isFinite(r[key])
      ? Math.min(max, Math.max(min, r[key] as number))
      : (defaults[key] as number);
  return {
    textSize:
      r.textSize === 'large' || r.textSize === 'largest'
        ? r.textSize
        : 'standard',
    contrast: r.contrast === true,
    largeTargets: r.largeTargets === true,
    reduceMotion: r.reduceMotion === true,
    simpleView: r.simpleView === true,
    speechRate: number('speechRate', 0.5, 1.5),
    keyboard: r.keyboard === true,
    dwellMs: number('dwellMs', 800, 4000),
    scanMs: number('scanMs', 800, 5000),
    switchKey:
      r.switchKey === 'Enter' || r.switchKey === 'F8' ? r.switchKey : 'Space',
  };
}

// A target fires once per visit. Leaving it (or losing tracking) re-arms dwell.
export class DwellGate<T> {
  target: T | null = null;
  since = 0;
  fired = false;
  update(target: T | null, now: number, duration: number) {
    if (target !== this.target) {
      this.target = target;
      this.since = now;
      this.fired = false;
    }
    if (!target || this.fired) return { progress: 0, activate: false };
    const progress = Math.min(1, Math.max(0, (now - this.since) / duration));
    if (progress === 1) this.fired = true;
    return { progress, activate: progress === 1 };
  }
}
