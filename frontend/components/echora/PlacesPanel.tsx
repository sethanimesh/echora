'use client';
import { useEffect, useRef, useState } from 'react';
import { api } from './types';
import { startupRead } from './startupRead';
import type { ContextSelection, CoreContext } from './contextTypes';
import {
  matchPlace,
  placeSelection,
  type Place,
  type PlaceSettings,
} from './places';

type Props = {
  selection: ContextSelection;
  disabled: boolean;
  idle: boolean;
  onChange: (next: ContextSelection) => void;
  manualOverride?: boolean;
  onResume?: () => void;
};
export default function PlacesPanel({
  selection,
  disabled,
  idle,
  onChange,
  manualOverride = false,
  onResume,
}: Props) {
  const [settings, setSettings] = useState<PlaceSettings | null>(null);
  const [message, setMessage] = useState('');
  const [saving, setSaving] = useState(false);
  const [manual, setManual] = useState(false);
  const [name, setName] = useState('');
  const latest = useRef({ selection, idle, disabled, onChange, manual });
  useEffect(() => {
    latest.current = {
      selection,
      idle,
      disabled,
      onChange,
      manual: manual || manualOverride,
    };
  });
  useEffect(() => {
    let active = true;
    const bootstrap = new AbortController();
    void startupRead<PlaceSettings>('/v1/places', bootstrap.signal)
      .then((data) => {
        if (active) setSettings(data);
      })
      .catch(() => {
        if (active)
          setMessage('Saved places are unavailable. Choose a setting below.');
      });
    return () => {
      active = false;
      bootstrap.abort();
    };
  }, []);
  useEffect(() => {
    if (!settings?.auto_detect || !idle || disabled || manual || manualOverride)
      return;
    if (!navigator.geolocation) return;
    const watch = navigator.geolocation.watchPosition(
      (position) => {
        const current = latest.current;
        if (!current.idle || current.disabled || current.manual) return;
        const place = matchPlace(
          settings.places,
          position.coords.latitude,
          position.coords.longitude,
          position.coords.accuracy,
        );
        if (!place) {
          setMessage(
            'No clear saved-place match. Your selected setting is unchanged.',
          );
          return;
        }
        setMessage(
          `Detected ${place.label}. You can choose a different place.`,
        );
        if (current.selection.place_id !== place.id)
          current.onChange(placeSelection(current.selection, place));
      },
      () => setMessage('Location could not be read. Choose a place yourself.'),
      { enableHighAccuracy: false, maximumAge: 30000, timeout: 10000 },
    );
    return () => navigator.geolocation.clearWatch(watch);
  }, [settings, idle, disabled, manual, manualOverride]);
  async function save(next: PlaceSettings) {
    setSaving(true);
    try {
      const saved = await api<PlaceSettings>('/v1/places', next);
      setSettings(saved);
      setMessage('Places saved on this computer.');
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : 'Places could not be saved.',
      );
    } finally {
      setSaving(false);
    }
  }
  function update(place: Place, patch: Partial<Place>) {
    if (settings)
      void save({
        ...settings,
        places: settings.places.map((item) =>
          item.id === place.id ? { ...item, ...patch } : item,
        ),
      });
  }
  function tag(place: Place) {
    if (!navigator.geolocation) {
      setMessage('Location is unavailable.');
      return;
    }
    setMessage('Reading your location for this place…');
    navigator.geolocation.getCurrentPosition(
      (position) =>
        update(place, {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          tagged_at: new Date().toISOString(),
        }),
      () =>
        setMessage(
          'Location permission was not available. The place is still usable by name.',
        ),
      { timeout: 10000 },
    );
  }
  if (!settings)
    return message ? <p className="context-hint">{message}</p> : null;
  return (
    <section className="places-panel" aria-label="Named places">
      <div className="place-choice">
        <label>
          Place{' '}
          <select
            disabled={disabled || saving}
            value={selection.place_id ?? ''}
            onChange={(event) => {
              setManual(true);
              const place = settings.places.find(
                (item) => item.id === event.target.value,
              );
              onChange(
                place
                  ? placeSelection(selection, place)
                  : {
                      ...selection,
                      place_id: '',
                      declared_listener: null,
                      core_context: null,
                    },
              );
              setMessage('Your chosen place stays active for this visit.');
            }}
          >
            <option value="">Choose a setting below</option>
            {settings.places.map((place) => (
              <option key={place.id} value={place.id}>
                {place.label}
              </option>
            ))}
          </select>
        </label>
        <details className="place-settings">
          <summary>Manage places</summary>
          <fieldset disabled={disabled || saving}>
            <label className="context-check">
              <input
                type="checkbox"
                checked={settings.auto_detect}
                onChange={(event) => {
                  setManual(false);
                  onResume?.();
                  void save({ ...settings, auto_detect: event.target.checked });
                }}
              />{' '}
              Detect my saved places while idle
            </label>
            <p>
              Matching happens in your browser. Tagged coordinates are stored by
              your Echora service. Recording and message review keep their
              current context.
            </p>
            {(manual || manualOverride) && settings.auto_detect && (
              <button
                className="context-link"
                onClick={() => {
                  setManual(false);
                  onResume?.();
                }}
              >
                Resume automatic detection
              </button>
            )}
            {settings.places.map((place) => (
              <div className="place-editor" key={place.id}>
                <strong>{place.label}</strong>
                <label>
                  Setting{' '}
                  <select
                    value={place.context}
                    disabled={place.builtin}
                    onChange={(e) =>
                      update(place, { context: e.target.value as CoreContext })
                    }
                  >
                    {['general', 'home', 'care', 'outdoors'].map((value) => (
                      <option value={value} key={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  People here{' '}
                  <select
                    value={place.listener ?? ''}
                    onChange={(e) =>
                      update(place, {
                        listener: (e.target.value || null) as Place['listener'],
                      })
                    }
                  >
                    <option value="">Use my profile setting</option>
                    <option value="familiar">Know me</option>
                    <option value="unfamiliar">Are unfamiliar</option>
                  </select>
                </label>
                <button className="context-link" onClick={() => tag(place)}>
                  {place.latitude == null
                    ? 'Tag where I am'
                    : 'Update location tag'}
                </button>
                {place.latitude != null && (
                  <button
                    className="context-link"
                    onClick={() =>
                      update(place, {
                        latitude: null,
                        longitude: null,
                        tagged_at: null,
                      })
                    }
                  >
                    Remove location tag
                  </button>
                )}
                {!place.builtin && (
                  <button
                    className="context-link"
                    onClick={() => {
                      if (selection.place_id === place.id)
                        onChange({
                          ...selection,
                          place_id: '',
                          core_context: null,
                          declared_listener: null,
                        });
                      void save({
                        ...settings,
                        places: settings.places.filter(
                          (p) => p.id !== place.id,
                        ),
                      });
                    }}
                  >
                    Remove place
                  </button>
                )}
              </div>
            ))}
            <label>
              New place name{' '}
              <input
                maxLength={60}
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="For example, the local café"
              />
            </label>
            <button
              className="context-save"
              disabled={!name.trim()}
              onClick={() => {
                void save({
                  ...settings,
                  places: [
                    ...settings.places,
                    {
                      id: `place-${crypto.randomUUID()}`,
                      label: name.trim(),
                      context: 'general',
                      listener: null,
                      builtin: false,
                      latitude: null,
                      longitude: null,
                      radius_m: 100,
                      tagged_at: null,
                    },
                  ],
                });
                setName('');
              }}
            >
              Add place
            </button>
          </fieldset>
        </details>
      </div>
      {message && (
        <output className="context-hint" aria-live="polite">
          {message}
        </output>
      )}
    </section>
  );
}
