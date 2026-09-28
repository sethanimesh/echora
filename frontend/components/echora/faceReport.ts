import type { FaceSuggestionResult, FaceModelResult } from './FaceSuggestion';
// Whitelist results so future image/audio/message fields cannot enter an export.
function summary(result: FaceModelResult) {
  return {
    state: result.state,
    model: result.model,
    visibility: result.visibility,
    cue: result.cue,
    tone: result.tone,
    elapsed_ms: result.elapsed_ms,
    message: result.message,
    frames: result.frames?.map((frame) => ({
      visibility: frame.visibility,
      cue: frame.cue,
      label: frame.label,
      score: frame.score,
      margin: frame.margin,
    })),
  };
}
export function faceReport(
  comparison: NonNullable<FaceSuggestionResult['comparison']>,
  trial: string,
  feedback: string,
) {
  return {
    schema: 'echora-face-comparison-1',
    created_at: new Date().toISOString(),
    trial,
    listening_feedback: feedback,
    comparison: {
      local: summary(comparison.local),
      gemini: summary(comparison.gemini),
      agreement: comparison.agreement,
      selected: comparison.selected,
      policy: comparison.policy,
    },
    note: 'Model outputs and timings only. No images, audio, transcript or identity. Scores are uncalibrated. Gemini is the delivery baseline.',
  };
}
