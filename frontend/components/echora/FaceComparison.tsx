'use client';
import { useState } from 'react';
import type { FaceSuggestionResult, FaceModelResult } from './FaceSuggestion';
import { faceReport } from './faceReport';
const labels: Record<string, string> = {
  smile: 'Smile',
  broad_smile: 'Broad smile',
  positive_expression: 'Positive expression',
  neutral: 'Neutral expression',
  unclear: 'No clear style',
};
function description(result: FaceModelResult) {
  if (result.state === 'error') return result.message ?? 'Check unavailable';
  if (result.visibility === 'no_face') return 'No face detected';
  if (result.visibility === 'multiple_faces') return 'Multiple faces; no style';
  if (result.visibility === 'obscured') return 'Unclear view; no style';
  return labels[result.cue ?? 'unclear'] ?? 'No clear style';
}
export default function FaceComparison({
  comparison,
}: {
  comparison: NonNullable<FaceSuggestionResult['comparison']>;
}) {
  const [trial, setTrial] = useState('unlabelled');
  const [feedback, setFeedback] = useState('not_listened');
  function save() {
    const report = faceReport(comparison, trial, feedback);
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }),
    );
    const link = document.createElement('a');
    link.href = url;
    link.download = `echora-face-comparison-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return (
    <details className="face-comparison">
      <summary>Facial model comparison</summary>
      <p>
        Both models checked the same snapshots. Gemini remains the facial input
        to your delivery while we evaluate the local model.
      </p>
      <table>
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Result</th>
            <th scope="col">Time</th>
          </tr>
        </thead>
        <tbody>
          {(
            [
              ['On this computer · EmotiEffLib', comparison.local],
              ['Gemini Flash', comparison.gemini],
            ] as const
          ).map(([name, result]) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td>{description(result)}</td>
              <td>
                {result.elapsed_ms == null
                  ? '—'
                  : `${(result.elapsed_ms / 1000).toFixed(2)} s`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        {comparison.agreement === 'same_style'
          ? 'Both suggest the same broad style.'
          : comparison.agreement === 'different_style'
            ? 'The suggested styles differ.'
            : 'There are not two clear styles to compare.'}{' '}
        Agreement alone does not establish accuracy.
      </p>
      <label>
        Test performed{' '}
        <select value={trial} onChange={(e) => setTrial(e.target.value)}>
          <option value="unlabelled">Not labelled</option>
          <option value="ordinary_speech">Ordinary speech</option>
          <option value="neutral">Neutral expression</option>
          <option value="smile">Smile</option>
          <option value="broad_smile">Broad smile</option>
          <option value="no_face">Out of camera view</option>
          <option value="poor_lighting">Poor lighting</option>
          <option value="head_movement">Head movement</option>
        </select>
      </label>
      <label>
        How did the speech sound?{' '}
        <select value={feedback} onChange={(e) => setFeedback(e.target.value)}>
          <option value="not_listened">Not listened yet</option>
          <option value="fits">Fits what I intended</option>
          <option value="too_fast">Too fast</option>
          <option value="too_slow">Too slow</option>
          <option value="tone_mismatch">Tone did not fit</option>
        </select>
      </label>
      <button type="button" onClick={save}>
        Save comparison report
      </button>
      <p>
        The report contains results and your test notes, without pictures, audio
        or message text.
      </p>
    </details>
  );
}
