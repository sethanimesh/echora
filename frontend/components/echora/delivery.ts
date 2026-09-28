export const tones = {
  neutral: 'Neutral',
  warm: 'Warm',
  cheerful: 'Cheerful',
  firm: 'Firm',
} as const;
export type Tone = keyof typeof tones;
export type Delivery = { tone: Tone; rate: number; source: 'user' };
export const speechPaces = [
  { value: 0.65, label: 'Slower' },
  { value: 0.9, label: 'Gentle' },
  { value: 1, label: 'Standard' },
  { value: 1.2, label: 'Faster' },
];
export function deliveryMatches(
  delivery: Delivery | undefined,
  tone: Tone,
  rate: number,
) {
  return (
    delivery?.tone === tone &&
    delivery.rate === rate &&
    delivery.source === 'user'
  );
}

// Unknown cues preserve the user's existing choice; a suggestion never starts speech.
export function deliveryFromSuggestion(
  suggestion:
    | { state: string; tone?: Tone | null; rate?: number | null }
    | undefined,
  tone: Tone,
  rate: number,
): { tone: Tone; rate: number } | null {
  if (!suggestion || suggestion.state !== 'ready') return null;
  const nextTone = suggestion.tone;
  const nextRate = suggestion.rate;
  if (nextTone != null && !(nextTone in tones)) return null;
  if (nextRate != null && !speechPaces.some((pace) => pace.value === nextRate))
    return null;
  if (nextTone == null && nextRate == null) return null;
  return { tone: nextTone ?? tone, rate: nextRate ?? rate };
}
