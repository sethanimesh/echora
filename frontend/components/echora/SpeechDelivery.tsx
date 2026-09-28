'use client';
import type { ReactNode } from 'react';
import { tones, speechPaces, type Tone } from './delivery';

export default function SpeechDelivery({
  tone,
  rate,
  disabled,
  onTone,
  onRate,
  voice,
  fishReady,
  onVoice,
  children,
}: {
  children?: ReactNode;
  tone: Tone;
  rate: number;
  disabled: boolean;
  onTone: (tone: Tone) => void;
  onRate: (rate: number) => void;
  voice: 'device' | 'fish';
  fishReady: boolean;
  onVoice: (voice: 'device' | 'fish') => void;
}) {
  return (
    <section className="delivery-panel" aria-label="Voice and expression">
      <div className="delivery-heading">
        <h3>How should it sound?</h3>
        <span>Your words, your delivery</span>
      </div>
      <fieldset disabled={disabled}>
        <legend>Voice</legend>
        <div className="delivery-options">
          <button
            type="button"
            aria-pressed={voice === 'device'}
            onClick={() => onVoice('device')}
          >
            Device voice
          </button>
          <button
            type="button"
            aria-pressed={voice === 'fish'}
            disabled={!fishReady}
            onClick={() => onVoice('fish')}
          >
            Fish Audio{!fishReady ? ' · setup needed' : ''}
          </button>
        </div>
      </fieldset>
      <fieldset disabled={disabled} aria-describedby="delivery-note">
        <legend>Tone</legend>
        <div className="delivery-options">
          {(Object.keys(tones) as Tone[]).map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={tone === value}
              onClick={() => onTone(value)}
            >
              {tones[value]}
            </button>
          ))}
        </div>
      </fieldset>
      <fieldset disabled={disabled}>
        <legend>Speaking pace</legend>
        <div className="delivery-options">
          {speechPaces.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              aria-pressed={rate === value}
              onClick={() => onRate(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </fieldset>
      <p id="delivery-note">
        {voice === 'fish'
          ? 'Confirm & speak sends your approved words, tone and pace to Fish Audio. No recording or camera frames are sent. Generated speech may vary; check its words and pronunciation during this trial.'
          : 'Device voice uses your pace. Tone applies when you choose Fish Audio.'}
        {!fishReady &&
          ' Fish Audio needs an API key and voice ID on this computer before it can be tested.'}
      </p>
      {children}
    </section>
  );
}
