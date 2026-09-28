import type { ContextSelection, CoreContext } from './contextTypes';
export type Place = {
  id: string;
  label: string;
  context: CoreContext;
  listener: 'familiar' | 'unfamiliar' | null;
  builtin: boolean;
  latitude: number | null;
  longitude: number | null;
  radius_m: number;
  tagged_at: string | null;
};
export type PlaceSettings = {
  auto_detect: boolean;
  places: Place[];
  updated_at?: string;
};
export function placeSelection(
  selection: ContextSelection,
  place: Place,
): ContextSelection {
  return {
    ...selection,
    place_id: place.id,
    core_context: place.context,
    scenario: place.context === 'outdoors' ? 'outside' : place.context,
    declared_listener: place.listener,
    moment_id: '',
  };
}
export function matchPlace(
  places: Place[],
  latitude: number,
  longitude: number,
  accuracy: number,
): Place | null {
  if (!Number.isFinite(accuracy) || accuracy > 250) return null;
  const radians = (value: number) => (value * Math.PI) / 180;
  const candidates = places.filter((place) => {
    if (place.latitude === null || place.longitude === null) return false;
    const a =
      Math.sin(radians(place.latitude - latitude) / 2) ** 2 +
      Math.cos(radians(latitude)) *
        Math.cos(radians(place.latitude)) *
        Math.sin(radians(place.longitude - longitude) / 2) ** 2;
    const distance = 6371000 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return distance <= place.radius_m + Math.min(accuracy, 100);
  });
  // An overlapping or poor-quality location is not a declaration of context.
  return candidates.length === 1 ? candidates[0] : null;
}
