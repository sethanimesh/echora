/* eslint-disable react/react-compiler -- Hydrates device-local browser preferences after SSR and reports storage failures; no shared application data is mutated. */
'use client';
import { useEffect, useState } from 'react';
import {
  defaults,
  normalizePreferences,
  type AccessPreferences,
} from './preferences';
const KEY = 'echora-accessibility-v1';
export function useAccessPreferences() {
  const [preferences, setPreferences] = useState(defaults);
  const [ready, setReady] = useState(false);
  const [storageError, setStorageError] = useState('');
  useEffect(() => {
    try {
      setPreferences(
        normalizePreferences(JSON.parse(localStorage.getItem(KEY) || '{}')),
      );
    } catch {
      setStorageError(
        'Settings work for this visit, but could not be loaded from this browser.',
      );
    }
    setReady(true);
  }, []);
  useEffect(() => {
    if (!ready) return;
    const root = document.documentElement;
    root.dataset.accessText = preferences.textSize;
    root.dataset.accessContrast = String(preferences.contrast);
    root.dataset.accessTargets = String(preferences.largeTargets);
    root.dataset.accessMotion = String(preferences.reduceMotion);
    root.dataset.accessSimple = String(preferences.simpleView);
    try {
      localStorage.setItem(KEY, JSON.stringify(preferences));
    } catch {
      setStorageError(
        'Settings work for this visit. This browser is not saving them.',
      );
    }
  }, [preferences, ready]);
  function update(patch: Partial<AccessPreferences>) {
    setPreferences((p) => normalizePreferences({ ...p, ...patch }));
  }
  return {
    preferences,
    update,
    storageError,
    reset: () => setPreferences(defaults),
  };
}
