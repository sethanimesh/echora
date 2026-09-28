import type { Tone } from './delivery';
type VoiceCue =
  | {
      state: string;
      tone?: Tone | null;
      rate?: number | null;
      audio_status?: string;
    }
  | null
  | undefined;
type FaceCue =
  | { state: string; visibility?: string; cue?: string; tone?: Tone | null }
  | null
  | undefined;

const rates = [0.65, 0.9, 1, 1.2] as const;
const validTone = (tone: Tone | null | undefined) =>
  tone && ['neutral', 'warm', 'cheerful', 'firm'].includes(tone) ? tone : null;

// Playback styling policy, not measured speaking speed or inferred emotional truth.
// The face provider requires a clear face in all three frames and agreement in
// at least two. Neutral/unclear expressions never imply slower speech.
export function combinedDelivery(
  voice: VoiceCue,
  face: FaceCue,
  currentRate = 1,
) {
  const pending = voice?.state === 'analyzing' || face?.state === 'analyzing';
  const usableVoice =
    voice?.state === 'ready' &&
    (!voice.audio_status || voice.audio_status === 'single_speaker');
  const voiceTone = usableVoice ? validTone(voice.tone) : null;
  const voiceRate =
    usableVoice && rates.some((rate) => rate === voice.rate)
      ? voice.rate!
      : null;
  const cue =
    face?.state === 'ready' && face.visibility === 'clear_face'
      ? face.cue
      : null;
  const expressive =
    cue === 'smile' || cue === 'broad_smile' || cue === 'positive_expression';
  let tone = voiceTone;
  let rate = voiceRate;
  if (expressive) {
    // Preserve bright prosody with a smile; soften emphasized prosody plus a
    // smile into warm delivery rather than making the user arbitrate sources.
    tone =
      voiceTone === 'firm'
        ? 'warm'
        : voiceTone === 'cheerful' || cue === 'broad_smile'
          ? 'cheerful'
          : 'warm';
    const baseline =
      voiceRate ?? (Number.isFinite(currentRate) ? currentRate : 1);
    // Pace follows the combined style: a smile must not slow a lively voice
    // after we have chosen cheerful delivery from both cues.
    const target = tone === 'cheerful' ? 1.2 : 0.9;
    // At most one adjacent supported pace toward the facial style. Use the
    // current setting when voice pace is unavailable; never invent measured pace.
    rate =
      baseline < target
        ? (rates.find((value) => value > baseline && value <= target) ?? target)
        : baseline > target
          ? ([...rates]
              .reverse()
              .find((value) => value < baseline && value >= target) ?? target)
          : target;
  } else if (!tone && cue === 'neutral') {
    tone = 'neutral';
  }
  const baseline =
    voiceRate ?? (Number.isFinite(currentRate) ? currentRate : 1);
  const toneReason =
    voiceTone === 'firm'
      ? 'A clear smile softens the emphasized voice into a warm tone.'
      : voiceTone === 'cheerful'
        ? 'A smile and lively vocal intonation support a cheerful tone.'
        : cue === 'broad_smile'
          ? 'A broad smile adds a cheerful tone to the delivery.'
          : 'A consistent smile adds warmth to the delivery.';
  const paceReason =
    rate === baseline
      ? 'The starting pace already fits this style, so it is kept.'
      : rate != null && rate > baseline
        ? 'Pace moves one step livelier to match.'
        : 'Pace moves one step gentler to match.';
  const explanation = pending
    ? 'Combining the cues from your recording…'
    : expressive
      ? `${toneReason} ${voiceRate == null ? 'Your current pace is the starting point. ' : ''}${paceReason}`
      : voiceTone != null || voiceRate != null
        ? 'The voice cues set the delivery; no clear expressive facial cue changes it.'
        : tone != null
          ? 'A neutral facial cue suggests a neutral tone; your current pace is kept.'
          : 'No clear delivery suggestion. Your current settings are kept.';
  return {
    tone: pending ? null : tone,
    rate: pending ? null : rate,
    pending,
    explanation,
    available: !pending && (tone != null || rate != null),
  };
}
