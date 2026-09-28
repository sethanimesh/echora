'use client';
import { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
const phrases = [
  ['💧', 'Water', 'Please bring me water.'],
  ['🍽', 'Food', 'I would like something to eat.'],
  ['🚻', 'Bathroom', 'Please help me get to the bathroom.'],
  ['🤝', 'Help', 'I need help, please.'],
  ['💛', 'Company', 'Please sit with me.'],
  ['🛑', 'Stop', 'Please stop.'],
  ['✓', 'Yes', 'Yes.'],
  ['✕', 'No', 'No.'],
  ['⏳', 'Wait', 'Please give me a moment.'],
];
export function setFieldText(
  field: HTMLInputElement | HTMLTextAreaElement,
  value: string,
) {
  if (field.disabled || !field.isConnected) return;
  const proto =
    field instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value')?.set?.call(
    field,
    value.slice(0, field.maxLength > 0 ? field.maxLength : 2000),
  );
  field.dispatchEvent(new Event('input', { bubbles: true }));
}
export default function AccessKeyboard({
  field,
  onClose,
}: {
  field: HTMLInputElement | HTMLTextAreaElement | null;
  onClose: () => void;
}) {
  const [value, setValue] = useState(field?.value || ''),
    [numbers, setNumbers] = useState(false),
    [shift, setShift] = useState(false);
  const change = (next: string) => {
    if (!field || field.disabled) return;
    const limited = next.slice(0, field.maxLength > 0 ? field.maxLength : 2000);
    setValue(limited);
    setFieldText(field, limited);
  };
  return (
    <Dialog
      open={!!field}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="access-keyboard" showCloseButton={false}>
        <DialogTitle>On-screen keyboard</DialogTitle>
        <DialogDescription>
          Letters add to the end. Your words still need review before speaking.
        </DialogDescription>
        <output className="keyboard-preview" aria-live="polite">
          {value || 'Choose a phrase or add letters.'}
        </output>
        <div className="symbol-board">
          {phrases.map(([icon, label, text]) => (
            <button
              key={label}
              onClick={() => change(value ? value + ' ' + text : text)}
            >
              <span aria-hidden="true">{icon}</span>
              {label}
            </button>
          ))}
        </div>
        <div className="letter-board">
          {(numbers ? '1234567890.,?!-@' : 'abcdefghijklmnopqrstuvwxyz')
            .split('')
            .map((letter) => (
              <button
                key={letter}
                onClick={() =>
                  change(value + (shift ? letter.toUpperCase() : letter))
                }
              >
                {shift ? letter.toUpperCase() : letter}
              </button>
            ))}
        </div>
        <div className="keyboard-actions">
          <button aria-pressed={shift} onClick={() => setShift(!shift)}>
            Capital letters
          </button>
          <button onClick={() => setNumbers(!numbers)}>
            {numbers ? 'ABC' : '123 & punctuation'}
          </button>
          <button onClick={() => change(value + ' ')}>Space</button>
          <button
            onClick={() => change(Array.from(value).slice(0, -1).join(''))}
          >
            Backspace
          </button>
          <button onClick={onClose}>Done typing</button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
export function AccessSelect() {
  const [field, setField] = useState<HTMLSelectElement | null>(null);
  useEffect(() => {
    const open = (e: Event) =>
      setField((e as CustomEvent<HTMLSelectElement>).detail);
    window.addEventListener('echora-access-select', open);
    return () => window.removeEventListener('echora-access-select', open);
  }, []);
  return (
    <Dialog
      open={!!field}
      onOpenChange={(open) => {
        if (!open) setField(null);
      }}
    >
      <DialogContent className="access-select" showCloseButton={false}>
        <DialogTitle>Choose an option</DialogTitle>
        <DialogDescription>
          {field?.labels?.[0]?.textContent?.split('\n')[0] ||
            'Select a value for this control.'}
        </DialogDescription>
        <div className="access-options">
          {field &&
            Array.from(field.options).map((option, i) => (
              <button
                key={i}
                disabled={option.disabled}
                aria-pressed={field.value === option.value}
                onClick={() => {
                  if (field.isConnected && !field.disabled) {
                    field.value = option.value;
                    field.dispatchEvent(new Event('change', { bubbles: true }));
                  }
                  setField(null);
                }}
              >
                {option.text}
              </button>
            ))}
        </div>
        <button onClick={() => setField(null)}>Cancel</button>
      </DialogContent>
    </Dialog>
  );
}
