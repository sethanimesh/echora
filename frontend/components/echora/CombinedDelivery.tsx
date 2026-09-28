'use client';
import { useState } from 'react';
import { tones, speechPaces } from './delivery';
import { combinedDelivery } from './deliveryMerge';
import FaceComparison from './FaceComparison';
import type { VoiceSuggestionResult } from './VoiceSuggestion';
import type { FaceSuggestionResult } from './FaceSuggestion';
export default function CombinedDelivery({
  voice,
  face,
  currentRate,
  disabled,
  onAccept,
}: {
  voice?: VoiceSuggestionResult | null;
  face?: FaceSuggestionResult | null;
  currentRate: number;
  disabled: boolean;
  onAccept: () => void;
}) {
  const [dismissed, setDismissed] = useState(false);
  const result = combinedDelivery(voice, face, currentRate);
  const pace = speechPaces.find((p) => p.value === result.rate);
  const completed = !!voice || !!face;
  return (
    <div className="voice-suggestion">
      <h4>
        Your delivery <span>From this recording</span>
      </h4>
      <p>
        Echora combines your voice and any clear facial cues into one tone and
        pace.
      </p>
      <div aria-live="polite" aria-atomic="true">
        {!completed && (
          <p>Record a message to receive a delivery suggestion here.</p>
        )}
        {result.pending && <p>{result.explanation}</p>}
        {face?.state === 'analyzing' && (
          <p>The camera is off while we check your snapshots.</p>
        )}
        {voice?.state === 'error' && <p>Voice check: {voice.message}</p>}
        {face?.state === 'error' && <p>Camera check: {face.message}</p>}
        {face?.comparison?.local.state === 'error' && (
          <p>Local model comparison: {face.comparison.local.message}</p>
        )}
        {face?.comparison?.gemini.state === 'error' && (
          <p>
            Facial delivery check: {face.comparison.gemini.message} The
            experimental local result has not been substituted.
          </p>
        )}
        {voice?.state === 'stopped' && <p>Voice check stopped.</p>}
        {face?.state === 'stopped' && <p>Camera check stopped.</p>}
        {!dismissed && completed && !result.pending && (
          <>
            {result.available && (
              <p>
                Suggested delivery:{' '}
                <strong>
                  {result.tone ? tones[result.tone] : 'Keep current tone'} ·{' '}
                  {pace
                    ? `${pace.label} (${pace.value}×)`
                    : 'Keep current pace'}
                </strong>
                .
              </p>
            )}
            <p>{result.explanation}</p>
            <div className="delivery-options">
              {result.available && (
                <button
                  type="button"
                  disabled={disabled}
                  onClick={() => {
                    onAccept();
                    setDismissed(true);
                  }}
                >
                  Use suggested delivery
                </button>
              )}
              <button
                type="button"
                disabled={disabled}
                onClick={() => setDismissed(true)}
              >
                Keep my settings
              </button>
            </div>
          </>
        )}
        {dismissed && <p>Delivery reviewed. Confirm &amp; speak when ready.</p>}
      </div>
      {face?.comparison && !result.pending && (
        <FaceComparison comparison={face.comparison} />
      )}
    </div>
  );
}
