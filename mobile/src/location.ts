import type { Place } from "./types";

export function metersBetween(fromLat: number, fromLon: number, toLat: number, toLon: number) {
  const rad = (value: number) => (value * Math.PI) / 180;
  const inner =
    Math.sin(rad(toLat - fromLat) / 2) ** 2 +
    Math.cos(rad(fromLat)) *
      Math.cos(rad(toLat)) *
      Math.sin(rad(toLon - fromLon) / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(inner)));
}

export function nearestPlace(
  places: Place[],
  latitude: number,
  longitude: number,
  accuracy = 0,
) {
  const tolerance = Math.min(Math.max(accuracy, 0), 100);
  let best: { place: Place; meters: number } | null = null;
  for (const place of places) {
    if (place.latitude === null || place.longitude === null) continue;
    const meters = metersBetween(latitude, longitude, place.latitude, place.longitude);
    if (meters > place.radius_m + tolerance) continue;
    if (!best || meters < best.meters) best = { place, meters };
  }
  return best;
}

export function isTagged(place: Place) {
  return place.latitude !== null && place.longitude !== null;
}
