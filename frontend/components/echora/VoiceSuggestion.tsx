'use client';
import { tones, speechPaces, type Tone } from './delivery';

export type VoiceSuggestionResult = {
  id: string;
  revision: number;
  state: 'analyzing' | 'ready' | 'error' | 'stopped';
  audio_status?: 'single_speaker' | 'no_speech' | 'overlapping' | 'unclear';
  tone?: Tone | null;
  rate?: number | null;
  cue?: 'soft' | 'bright' | 'steady' | 'emphatic' | 'unclear';
  message?: string;
};
const descriptions = {
  soft: 'The recording may have a soft delivery.',
  bright: 'The recording may have a lively delivery.',
  steady: 'The recording may have an even delivery.',
  emphatic: 'The recording may have deliberate emphasis.',
  unclear: 'There was not a clear enough cue to suggest a tone.',
};

export default function VoiceSuggestion({
  configured,
  available,
  disabled,
  suggestion,
  dismissed,
  onSuggest,
  onStop,
  onAccept,
  onDismiss,
}: {
  configured: boolean;
  available: boolean;
  disabled: boolean;
  suggestion?: VoiceSuggestionResult | null;
  dismissed: boolean;
  onSuggest: () => void;
  onStop: () => void;
  onAccept: () => void;
  onDismiss: () => void;
}) {
  const analyzing = suggestion?.state === 'analyzing';
  const pace = speechPaces.find((pace) => pace.value === suggestion?.rate);
  const hasSuggestion = !!suggestion?.tone || !!pace;
  return (
    <div className="voice-suggestion">
      <h4>
        Your delivery <span>Review suggestion</span>
      </h4>
      <p id="voice-cue-consent">
        Tone and pace are suggested when you transcribe. “Use suggested
        delivery” updates the controls; unclear settings keep your current
        choices. Suggestions are approximate, not a reliable assessment of how
        you feel. Retrying sends the recording to Gemini Flash again.
      </p>
      <div className="delivery-options">
        {analyzing ? (
          <button type="button" onClick={onStop}>
            Stop voice analysis
          </button>
        ) : (
          <button
            type="button"
            disabled={disabled || !available || !configured}
            aria-describedby="voice-cue-consent"
            onClick={onSuggest}
          >
            {suggestion
              ? 'Retry delivery suggestion'
              : 'Suggest delivery from my voice'}
          </button>
        )}
      </div>
      {!configured && (
        <p>
          Voice suggestions are not configured. You can select a tone above.
        </p>
      )}
      {!available && (
        <p>
          Record and transcribe a message in this visit to try a suggestion.
        </p>
      )}
      <div aria-live="polite" aria-atomic="true">
        {analyzing && (
          <p>
            Listening for delivery cues… Your words, tone and pace stay
            unchanged.
          </p>
        )}
        {suggestion?.state === 'error' && <p>{suggestion.message}</p>}
        {suggestion?.state === 'stopped' && (
          <p>Analysis stopped. Your words, tone and pace are unchanged.</p>
        )}
        {suggestion?.state === 'ready' && !dismissed && (
          <>
            <p>
              {suggestion.audio_status === 'no_speech'
                ? 'The model did not identify usable speech in this recording.'
                : suggestion.audio_status === 'overlapping'
                  ? 'The model could not separate the overlapping voices.'
                  : descriptions[suggestion.cue ?? 'unclear']}{' '}
              Tone:{' '}
              {suggestion.tone
                ? tones[suggestion.tone]
                : 'keep current (unclear)'}
              . Pace:{' '}
              {pace
                ? `${pace.label} (${pace.value}×)`
                : 'keep current (unclear)'}
              .
            </p>
            <div className="delivery-options">
              {hasSuggestion && (
                <button type="button" disabled={disabled} onClick={onAccept}>
                  Use suggested delivery
                </button>
              )}
              <button type="button" disabled={disabled} onClick={onDismiss}>
                Dismiss suggestion
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
