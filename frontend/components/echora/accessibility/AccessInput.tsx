/* eslint-disable react/react-compiler -- Imperative camera/pointer loop uses refs to share fresh samples without restarting hardware on React renders. */
'use client';
import { useEffect, useRef, useState } from 'react';
import { DwellGate, type AccessPreferences } from './preferences';
import type { InputMode } from './AccessPanel';
export type PointerSample = {
  x: number;
  y: number;
  time: number;
  select?: boolean;
} | null;
const selector =
  'button, a[href], input:not([type="hidden"]):not([type="file"]), textarea, select, summary';
function usable(el: HTMLElement) {
  return (
    el.isConnected &&
    !el.matches(':disabled, [aria-disabled="true"]') &&
    !el.closest('[inert], [hidden]') &&
    el.getClientRects().length > 0 &&
    getComputedStyle(el).visibility !== 'hidden'
  );
}
function scope() {
  return (
    Array.from(document.querySelectorAll<HTMLElement>('[role="dialog"]'))
      .filter(usable)
      .at(-1) || document
  );
}
function targets() {
  return Array.from(scope().querySelectorAll<HTMLElement>(selector)).filter(
    usable,
  );
}
function targetAt(x: number, y: number) {
  const el = document.elementFromPoint(x, y)?.closest<HTMLElement>(selector);
  return el && scope().contains(el) && usable(el) ? el : null;
}
export default function AccessInput({
  mode,
  preferences,
  cameraPointer,
  stop,
  editField,
}: {
  mode: InputMode;
  preferences: AccessPreferences;
  cameraPointer: React.RefObject<PointerSample>;
  stop: () => void;
  editField: (field: HTMLInputElement | HTMLTextAreaElement) => void;
}) {
  const [label, setLabel] = useState('');
  const [point, setPoint] = useState<{
    x: number;
    y: number;
    progress: number;
  } | null>(null);
  const callbacks = useRef({ stop, editField });
  callbacks.current = { stop, editField };
  useEffect(() => {
    if (mode === 'off') {
      setPoint(null);
      setLabel('');
      return;
    }
    const dwell = new DwellGate<HTMLElement>();
    let hovered: PointerSample = null,
      highlighted: HTMLElement | null = null;
    let scanIndex = -1,
      scanAt = performance.now(),
      frame = 0,
      lastKey = 0;
    const clear = () => {
      highlighted?.removeAttribute('data-access-target');
      highlighted = null;
    };
    const mark = (target: HTMLElement | null) => {
      if (highlighted === target) return;
      clear();
      highlighted = target;
      target?.setAttribute('data-access-target', 'true');
      if (mode === 'scan') {
        target?.scrollIntoView({ block: 'nearest', behavior: 'instant' });
        setLabel(
          target?.getAttribute('aria-label') ||
            target?.textContent?.trim().slice(0, 90) ||
            'Input field',
        );
      }
    };
    const activate = (target: HTMLElement) => {
      if (!usable(target)) return;
      if (
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLInputElement &&
          [
            'text',
            'search',
            'email',
            'tel',
            'url',
            'number',
            'password',
          ].includes(target.type))
      ) {
        callbacks.current.editField(target);
      } else if (target instanceof HTMLSelectElement) {
        window.dispatchEvent(
          new CustomEvent('echora-access-select', { detail: target }),
        );
      } else target.click();
      scanAt = performance.now();
    };
    const key = (e: KeyboardEvent) => {
      if (e.code === 'Escape') {
        callbacks.current.stop();
        return;
      }
      if (
        mode !== 'scan' ||
        e.code !== preferences.switchKey ||
        e.altKey ||
        e.ctrlKey ||
        e.metaKey
      )
        return;
      e.preventDefault();
      e.stopPropagation();
      if (e.repeat || performance.now() - lastKey < 350) return;
      lastKey = performance.now();
      if (highlighted) activate(highlighted);
    };
    const move = (e: PointerEvent) => {
      if (e.pointerType !== 'touch')
        hovered = { x: e.clientX, y: e.clientY, time: performance.now() };
    };
    const leave = () => {
      hovered = null;
      dwell.update(null, 0, 1);
    };
    const blur = () => {
      leave();
    };
    const visibility = () => {
      if (document.hidden) {
        blur();
        callbacks.current.stop();
      }
    };
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      if (!document.hasFocus()) {
        dwell.update(null, now, 1);
        setPoint(null);
        return;
      }
      if (mode === 'scan') {
        const items = targets();
        if (
          !highlighted ||
          !items.includes(highlighted) ||
          now - scanAt >= preferences.scanMs
        ) {
          scanIndex = highlighted
            ? items.indexOf(highlighted) + 1
            : scanIndex + 1;
          mark(items.length ? items[scanIndex % items.length] : null);
          scanAt = now;
        }
      } else {
        const sample = mode === 'dwell' ? hovered : cameraPointer.current;
        const valid = sample && (mode === 'dwell' || now - sample.time < 400);
        const candidate = valid ? targetAt(sample.x, sample.y) : null;
        const target =
          sample?.select === false &&
          !candidate?.hasAttribute('data-access-enable')
            ? null
            : candidate;
        mark(target);
        const result = dwell.update(target, now, preferences.dwellMs);
        setPoint(
          valid
            ? { x: sample.x, y: sample.y, progress: result.progress }
            : null,
        );
        if (result.activate && target) activate(target);
      }
    };
    document.addEventListener('keydown', key, true);
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerleave', leave);
    document.addEventListener('visibilitychange', visibility);
    window.addEventListener('blur', blur);
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      clear();
      document.removeEventListener('keydown', key, true);
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerleave', leave);
      document.removeEventListener('visibilitychange', visibility);
      window.removeEventListener('blur', blur);
    };
  }, [
    mode,
    preferences.dwellMs,
    preferences.scanMs,
    preferences.switchKey,
    cameraPointer,
  ]);
  return (
    <>
      {mode !== 'off' && (
        <div className="access-active" aria-label="Active input assistance">
          <output aria-live="polite" aria-atomic="true">
            {mode === 'scan'
              ? `Scanning · ${label}`
              : mode === 'dwell'
                ? 'Pointer dwell on'
                : 'Camera assistance on'}
          </output>
          <button onClick={stop}>Pause assistance</button>
        </div>
      )}
      {point && (
        <div
          aria-hidden="true"
          className="access-pointer"
          style={{
            left: point.x,
            top: point.y,
            background: `conic-gradient(#315e8b ${point.progress * 360}deg, transparent 0)`,
          }}
        />
      )}
    </>
  );
}
