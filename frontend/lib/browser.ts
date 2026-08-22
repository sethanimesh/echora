"use client";

import { useSyncExternalStore } from "react";

/**
 * Browser-only readings, subscribed rather than copied into state.
 *
 * Both of these were tempting `useEffect` + `setState` pairs, which React 19
 * rejects: setting state synchronously from an effect cascades renders. Each is
 * really an external store, so it is read as one, and each has a server snapshot
 * so the first client paint agrees with the server.
 */

const REDUCE = "(prefers-reduced-motion: reduce)";

function subscribeMotion(onChange: () => void) {
  const query = window.matchMedia(REDUCE);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/**
 * True when the speaker has asked the system for reduced motion.
 *
 * The stage already honours this in CSS and in the analyser loop, but the orb
 * animates in WebGL where a media query cannot reach it. This is what lets it be
 * skipped outright -- which also means three.js is never downloaded.
 */
export function useReducedMotion() {
  return useSyncExternalStore(
    subscribeMotion,
    () => window.matchMedia(REDUCE).matches,
    () => false,
  );
}

const never = () => () => undefined;

/** False on the server and during hydration; true once the client has taken over. */
export function useHydrated() {
  return useSyncExternalStore(
    never,
    () => true,
    () => false,
  );
}
