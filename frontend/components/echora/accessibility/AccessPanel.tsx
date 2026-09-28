'use client';
import { Accessibility, X } from 'lucide-react';
import { speechPaces } from '../delivery';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import type { AccessPreferences } from './preferences';
export type InputMode =
  | 'off'
  | 'dwell'
  | 'scan'
  | 'head'
  | 'gaze'
  | 'neural-gaze';
export default function AccessPanel({
  open,
  onClose,
  preferences: p,
  update,
  mode,
  setMode,
  storageError,
  reset,
}: {
  open: boolean;
  onClose: () => void;
  preferences: AccessPreferences;
  update: (patch: Partial<AccessPreferences>) => void;
  mode: InputMode;
  setMode: (mode: InputMode) => void;
  storageError: string;
  reset: () => void;
}) {
  function toggle(
    key:
      | 'contrast'
      | 'largeTargets'
      | 'reduceMotion'
      | 'simpleView'
      | 'keyboard',
    label: string,
    note: string,
  ) {
    return (
      <button
        className="access-toggle"
        aria-pressed={p[key]}
        onClick={() => update({ [key]: !p[key] })}
      >
        <span>
          <strong>{label}</strong>
          <small>{note}</small>
        </span>
        <span className="toggle-state">{p[key] ? 'On' : 'Off'}</span>
      </button>
    );
  }
  function choices<T extends string | number>(
    label: string,
    value: T,
    items: { value: T; label: string }[],
    change: (value: T) => void,
  ) {
    return (
      <fieldset>
        <legend>{label}</legend>
        <div className="access-choices">
          {items.map((item) => (
            <button
              key={item.value}
              aria-pressed={value === item.value}
              onClick={() => change(item.value)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </fieldset>
    );
  }
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <DialogContent className="access-panel" showCloseButton={false}>
        <div className="access-heading">
          <DialogTitle id="access-title">
            <Accessibility size={22} /> Accessibility
          </DialogTitle>
          <button onClick={onClose} aria-label="Close accessibility">
            <X size={22} />
          </button>
        </div>
        <DialogDescription>
          Make Echora comfortable for you. Display and timing choices stay on
          this device.
        </DialogDescription>
        {storageError && <output>{storageError}</output>}
        <h3>Reading & display</h3>
        {choices(
          'Text size',
          p.textSize,
          [
            { value: 'standard', label: 'Standard' },
            { value: 'large', label: 'Large' },
            { value: 'largest', label: 'Largest' },
          ],
          (textSize) => update({ textSize }),
        )}
        {toggle(
          'contrast',
          'Stronger contrast',
          'Clearer text and borders, with a light background.',
        )}
        {toggle(
          'largeTargets',
          'Larger controls',
          'More room to point, tap, or dwell.',
        )}
        {toggle(
          'reduceMotion',
          'Reduce motion',
          'Keep the globe still and remove interface animation.',
        )}
        {toggle(
          'simpleView',
          'Simpler view',
          'Hide the decorative globe and technical test tools.',
        )}
        <h3>Input & movement</h3>
        <p>
          Choose one assistance method. These start only when selected here;
          camera access never resumes automatically.
        </p>
        {choices(
          'Input assistance',
          mode,
          [
            { value: 'off', label: 'Standard' },
            { value: 'dwell', label: 'Pointer dwell' },
            { value: 'scan', label: 'Switch scanning' },
            { value: 'head', label: 'Head pointer · trial' },
            { value: 'neural-gaze', label: 'Eye gaze · GazeFollower' },
            { value: 'gaze', label: 'WebGazer · previous trial' },
          ],
          (next) => {
            setMode(next);
            if (next !== 'off') update({ largeTargets: true });
          },
        )}
        <p>
          Pointer dwell also works with an eye tracker or head mouse that moves
          your system pointer. Hold over a control to select it; move away
          before selecting it again.
        </p>
        {choices(
          'Time before dwell selection',
          p.dwellMs,
          [
            { value: 1000, label: '1 second' },
            { value: 1400, label: '1.4 seconds' },
            { value: 2200, label: '2.2 seconds' },
            { value: 3200, label: '3.2 seconds' },
          ],
          (dwellMs) => update({ dwellMs }),
        )}
        {choices(
          'Time per scan target',
          p.scanMs,
          [
            { value: 1200, label: '1.2 seconds' },
            { value: 1800, label: '1.8 seconds' },
            { value: 2800, label: '2.8 seconds' },
            { value: 4000, label: '4 seconds' },
          ],
          (scanMs) => update({ scanMs }),
        )}
        {choices(
          'Switch key',
          p.switchKey,
          [
            { value: 'Space', label: 'Space' },
            { value: 'Enter', label: 'Enter' },
            { value: 'F8', label: 'F8' },
          ],
          (switchKey) => update({ switchKey }),
        )}
        <p>
          For a switch that sends a keyboard key: scanning highlights each
          control, then your switch selects it. Escape stops assistance. The
          Pause assistance button is always available.
        </p>
        {toggle(
          'keyboard',
          'On-screen keyboard & phrases',
          'Type letters or select everyday messages without a physical keyboard.',
        )}
        <p>
          On a Mac, you can also use System Settings → Accessibility → Pointer
          Control → Head Pointer, then choose Pointer dwell here. The WebGazer
          trial uses a separate board with four large choices.
        </p>
        <h3>Listening</h3>
        {choices('Speaking pace', p.speechRate, speechPaces, (speechRate) =>
          update({ speechRate }),
        )}
        <p>
          All messages remain visible. Screen readers can use the labelled
          controls, message field, and spoken status updates. Nothing is spoken
          until you confirm.
        </p>
        <div className="access-footer">
          <button
            onClick={() => {
              setMode('off');
              reset();
            }}
          >
            Reset accessibility
          </button>
          <button onClick={onClose}>Done</button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
