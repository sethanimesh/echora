'use client';
import { useState } from 'react';
import { ChevronDown, Plus, X, UserRound } from 'lucide-react';
import PersonalProfileFields from './PersonalProfileFields';
import RecognitionProfileFields from './RecognitionProfileFields';
import {
  scenarios,
  emptyProfile,
  type ContextSelection,
  type Profile,
  type Scenario,
} from './contextTypes';

type Props = {
  profiles: Profile[];
  selection: ContextSelection;
  disabled: boolean;
  ready: boolean;
  onChange: (next: ContextSelection) => void;
  onSave: (profile: Profile) => void;
  onTry: (words: string) => void;
};
export default function ContextPanel({
  profiles,
  selection,
  disabled,
  ready,
  onChange,
  onSave,
  onTry,
}: Props) {
  const profile = profiles.find((p) => p.id === selection.profile_id);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<Profile>(emptyProfile);
  const [situation, setSituation] = useState(selection.situation);
  const [details, setDetails] = useState(false);
  const change = (fields: Partial<ContextSelection>) =>
    onChange({ ...selection, ...fields });
  const patch = (fields: Partial<Profile>) => setDraft({ ...draft, ...fields });
  return (
    <section className="context-panel" aria-label="Communication context">
      <fieldset disabled={disabled || !ready}>
        <div className="context-top">
          <label className="profile-picker">
            <UserRound size={16} aria-hidden="true" />
            <span className="sr-only">Communication profile</span>
            <select
              value={selection.profile_id}
              onChange={(e) => {
                const p = profiles.find((p) => p.id === e.target.value);
                change({
                  profile_id: p?.id ?? '',
                  profile_revision: p?.revision ?? 0,
                  recipient: '',
                  audience_id: '',
                  listener: 'unspecified',
                  situation: '',
                  moment_id: '',
                });
              }}
            >
              <option value="">No profile</option>
              {profiles.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <button
            className="context-link"
            onClick={() => {
              setDraft(structuredClone(emptyProfile));
              setEditing(true);
            }}
          >
            <Plus size={14} /> New profile
          </button>
          {(profile || editing) && (
            <button
              className="context-link"
              onClick={() => {
                setDraft(structuredClone(profile ?? emptyProfile));
                setEditing(!editing);
              }}
            >
              {editing
                ? 'Close editor'
                : profile
                  ? 'Edit profile'
                  : 'Create profile'}
            </button>
          )}
        </div>
        {!profile && !editing && (
          <div className="profile-discovery">
            <div>
              <h2>Create your personal profile</h2>
              <p>
                Create your own profile, or explore a fictional life and make it
                yours.
              </p>
            </div>
            <div className="profile-story-grid">
              {profiles
                .filter((p) => p.sample)
                .map((p) => (
                  <button
                    key={p.id}
                    className="profile-story"
                    onClick={() =>
                      change({
                        profile_id: p.id,
                        profile_revision: p.revision,
                        scenario: 'general',
                        recipient: '',
                        audience_id: '',
                        listener: 'unspecified',
                        situation: '',
                        moment_id: '',
                      })
                    }
                  >
                    <span className="profile-initial">{p.label[0]}</span>
                    <strong>{p.label.replace(' · sample', '')}</strong>
                    <span>{p.about}</span>
                    <small>
                      {p.moments?.length ?? 0} personal moments · Explore
                      profile
                    </small>
                  </button>
                ))}
            </div>
          </div>
        )}
        {profile && !editing && (
          <div className="profile-identity">
            <span className="profile-initial">{profile.label[0]}</span>
            <div>
              <h2>{profile.label.replace(' · sample', '')}</h2>
              <p>
                {profile.about ||
                  'Your people, familiar moments and words that feel like you.'}
              </p>
              <span className="profile-manner">
                {profile.manner === 'warm'
                  ? 'Warm and friendly'
                  : profile.manner === 'direct'
                    ? 'Clear and direct'
                    : 'Natural and neutral'}{' '}
                ·{' '}
                {profile.language === 'original'
                  ? 'Your language'
                  : profile.language}
              </span>
            </div>
          </div>
        )}
        <div className="scenario-chips" aria-label="Current setting">
          {Object.entries(scenarios).map(([value, label]) => (
            <button
              key={value}
              aria-pressed={selection.scenario === value}
              onClick={() =>
                change({
                  scenario: value as Scenario,
                  moment_id: '',
                  place_id: '',
                  declared_listener: null,
                  core_context: null,
                })
              }
            >
              {label}
            </button>
          ))}
        </div>
        {!!profile?.moments?.length && !editing && (
          <div className="personal-moments">
            <div className="moments-heading">
              <span>My moments</span>
              {selection.moment_id && (
                <button
                  className="context-link"
                  onClick={() => change({ moment_id: '' })}
                >
                  Clear moment
                </button>
              )}
            </div>
            <div className="moment-grid">
              {profile.moments.map((m) => (
                <button
                  key={m.id}
                  className="moment-card"
                  aria-pressed={selection.moment_id === m.id}
                  onClick={() =>
                    change({
                      scenario: m.scenario,
                      moment_id: m.id,
                      place_id: '',
                      declared_listener: null,
                      core_context: null,
                    })
                  }
                >
                  <small>{scenarios[m.scenario]}</small>
                  <strong>{m.title}</strong>
                  <span>“{m.cue}”</span>
                </button>
              ))}
            </div>
            {profile.moments
              .filter((m) => m.id === selection.moment_id)
              .map((m) => (
                <div className="moment-ready" key={m.id}>
                  <p>
                    “{m.cue}” means <strong>“{m.message}”</strong>
                  </p>
                  <button className="context-link" onClick={() => onTry(m.cue)}>
                    Try “{m.cue}” <span aria-hidden="true">→</span>
                  </button>
                </div>
              ))}
          </div>
        )}
        <div className="context-people">
          {!!profile?.audiences?.length && (
            <label>
              Named listener
              <select
                value={selection.audience_id ?? ''}
                onChange={(e) => {
                  const audience = profile.audiences?.find(
                    (a) => a.id === e.target.value,
                  );
                  change({
                    audience_id: e.target.value,
                    recipient: audience?.label ?? '',
                    listener: 'unspecified',
                  });
                }}
              >
                <option value="">Use recipient and place settings</option>
                {profile.audiences
                  .filter((audience) => {
                    const core =
                      selection.core_context ??
                      (selection.scenario === 'outside' ||
                      selection.scenario === 'shopping' ||
                      selection.scenario === 'cafe'
                        ? 'outdoors'
                        : selection.scenario);
                    return (
                      audience.id === selection.audience_id ||
                      ((!audience.visible_in_settings?.length ||
                        audience.visible_in_settings.includes(core)) &&
                        (!audience.visible_in_places?.length ||
                          audience.visible_in_places.includes(
                            selection.place_id ?? '',
                          )))
                    );
                  })
                  .map((audience) => (
                    <option key={audience.id} value={audience.id}>
                      {audience.label}
                      {audience.relationship
                        ? ` · ${audience.relationship}`
                        : ''}
                    </option>
                  ))}
              </select>
            </label>
          )}
          <label>
            Talking to
            <select
              value={selection.recipient}
              onChange={(e) => {
                const person = profile?.people.find(
                  (p) => p.name === e.target.value,
                );
                change({
                  recipient: e.target.value,
                  audience_id: '',
                  listener: person
                    ? person.familiar
                      ? 'familiar'
                      : 'new'
                    : 'unspecified',
                });
              }}
            >
              <option value="">No one selected</option>
              {selection.recipient &&
                !profile?.people.some(
                  (p) => p.name === selection.recipient,
                ) && (
                  <option value={selection.recipient}>
                    {selection.recipient}
                  </option>
                )}
              {profile?.people.map((p, i) => (
                <option key={i} value={p.name}>
                  {p.name}
                  {p.relationship ? ` · ${p.relationship}` : ''}
                </option>
              ))}
            </select>
          </label>
          <label>
            Listener
            <select
              value={selection.listener}
              onChange={(e) =>
                change({
                  listener: e.target.value as ContextSelection['listener'],
                  audience_id: '',
                })
              }
            >
              <option value="unspecified">Not specified</option>
              <option value="familiar">Someone familiar</option>
              <option value="new">Someone new / staff</option>
            </select>
          </label>
          <button
            className="context-link"
            aria-expanded={details}
            onClick={() => setDetails(!details)}
          >
            More context <ChevronDown size={14} />
          </button>
        </div>
        {details && (
          <div className="context-extra">
            <label>
              Current situation <span className="optional-label">optional</span>
              <input
                value={situation}
                maxLength={300}
                placeholder="For example, ordering a drink at a café"
                onChange={(e) => setSituation(e.target.value)}
              />
            </label>
            <button
              className="context-save"
              disabled={situation.trim() === selection.situation}
              onClick={() => change({ situation: situation.trim() })}
            >
              Apply situation
            </button>
            <label className="context-check">
              <input
                type="checkbox"
                checked={selection.use_personal_wording}
                onChange={(e) =>
                  change({ use_personal_wording: e.target.checked })
                }
              />{' '}
              Use my saved personal wording
            </label>
            <p>
              This context stays active until you change it. Saved-place
              detection is optional and only runs while idle.
            </p>
          </div>
        )}
        {profile?.sample && !editing && (
          <p className="context-hint">
            Fictional sample for trying scenarios. Edit it to save your own
            copy.
          </p>
        )}
        {editing && (
          <form
            className="profile-editor"
            onSubmit={(e) => {
              e.preventDefault();
              onSave({
                ...draft,
                protected_terms: (draft.protected_terms ?? [])
                  .map((term) => term.trim())
                  .filter(Boolean),
                ...(draft.sample
                  ? {
                      id: '',
                      revision: 0,
                      label:
                        draft.label.replace(' · sample', '').slice(0, 48) +
                        ' · my copy',
                    }
                  : {}),
                sample: false,
              });
            }}
          >
            <div className="profile-fields">
              <label>
                Profile name
                <input
                  required
                  value={draft.label}
                  maxLength={60}
                  onChange={(e) => patch({ label: e.target.value })}
                />
              </label>
              <label>
                Message language
                <select
                  value={draft.language}
                  onChange={(e) =>
                    patch({ language: e.target.value as Profile['language'] })
                  }
                >
                  <option value="original">Keep my language</option>
                  <option value="English">English</option>
                  <option value="Hindi/Hinglish">Hindi / Hinglish</option>
                </select>
              </label>
              <label>
                Wording
                <select
                  value={draft.style}
                  onChange={(e) =>
                    patch({ style: e.target.value as Profile['style'] })
                  }
                >
                  <option value="natural">Natural</option>
                  <option value="concise">Short and complete</option>
                </select>
              </label>
            </div>
            <PersonalProfileFields draft={draft} patch={patch} />
            <RecognitionProfileFields draft={draft} patch={patch} />
            <label>
              Names and brands to keep exactly
              <textarea
                value={(draft.protected_terms ?? []).join('\n')}
                maxLength={1944}
                placeholder={
                  'One per line, for example:\nMaya\nLipton\nHDFC Bank'
                }
                onChange={(e) =>
                  patch({
                    protected_terms: e.target.value.split('\n').slice(0, 24),
                  })
                }
              />
            </label>
            <p>
              People listed below are protected too. This preserves spelling in
              wording and pronunciation preparation; it does not clone or train
              a voice.
            </p>
            <h2>People</h2>
            {draft.people.map((p, i) => (
              <div className="person-editor" key={i}>
                <label>
                  Name
                  <input
                    required
                    maxLength={60}
                    value={p.name}
                    onChange={(e) =>
                      patch({
                        people: draft.people.map((v, n) =>
                          n === i ? { ...v, name: e.target.value } : v,
                        ),
                      })
                    }
                  />
                </label>
                <label>
                  Relationship
                  <input
                    maxLength={60}
                    value={p.relationship}
                    placeholder="Friend, caregiver…"
                    onChange={(e) =>
                      patch({
                        people: draft.people.map((v, n) =>
                          n === i ? { ...v, relationship: e.target.value } : v,
                        ),
                      })
                    }
                  />
                </label>
                <label className="context-check">
                  <input
                    type="checkbox"
                    checked={p.familiar}
                    onChange={(e) =>
                      patch({
                        people: draft.people.map((v, n) =>
                          n === i ? { ...v, familiar: e.target.checked } : v,
                        ),
                      })
                    }
                  />{' '}
                  Knows me
                </label>
                <button
                  type="button"
                  className="context-remove"
                  aria-label={`Remove person ${i + 1}`}
                  onClick={() =>
                    patch({ people: draft.people.filter((_, n) => n !== i) })
                  }
                >
                  <X size={16} />
                </button>
              </div>
            ))}
            <button
              type="button"
              className="context-link"
              disabled={draft.people.length >= 12}
              onClick={() =>
                patch({
                  people: [
                    ...draft.people,
                    { name: '', relationship: '', familiar: true },
                  ],
                })
              }
            >
              <Plus size={14} /> Add a person
            </button>
            <h2>My words in different settings</h2>
            <p>
              Choose where a detail belongs. Include your original word in the
              fuller wording, such as “tea” → “Lipton green tea”.
            </p>
            {draft.rules.map((r, i) => (
              <div className="rule-editor" key={i}>
                <div className="rule-fields">
                  <label>
                    When I say
                    <input
                      required
                      maxLength={60}
                      value={r.anchor}
                      placeholder="tea"
                      onChange={(e) =>
                        patch({
                          rules: draft.rules.map((v, n) =>
                            n === i ? { ...v, anchor: e.target.value } : v,
                          ),
                        })
                      }
                    />
                  </label>
                  <label>
                    My fuller wording
                    <input
                      required
                      maxLength={150}
                      value={r.wording}
                      placeholder="Lipton green tea"
                      onChange={(e) =>
                        patch({
                          rules: draft.rules.map((v, n) =>
                            n === i ? { ...v, wording: e.target.value } : v,
                          ),
                        })
                      }
                    />
                  </label>
                  <button
                    type="button"
                    className="context-remove"
                    aria-label={`Remove wording rule ${i + 1}`}
                    onClick={() =>
                      patch({ rules: draft.rules.filter((_, n) => n !== i) })
                    }
                  >
                    <X size={16} />
                  </button>
                </div>
                <fieldset
                  className="rule-settings"
                  aria-label={`Settings for rule ${i + 1}`}
                >
                  {Object.entries(scenarios).map(([key, label]) => (
                    <label className="context-check" key={key}>
                      <input
                        type="checkbox"
                        checked={r.scenarios.includes(key as Scenario)}
                        onChange={(e) =>
                          patch({
                            rules: draft.rules.map((v, n) =>
                              n === i
                                ? {
                                    ...v,
                                    scenarios: e.target.checked
                                      ? [...v.scenarios, key as Scenario]
                                      : v.scenarios.filter((s) => s !== key),
                                  }
                                : v,
                            ),
                          })
                        }
                      />
                      {label}
                    </label>
                  ))}
                </fieldset>
                <label>
                  Use this detail
                  <select
                    value={r.mode}
                    onChange={(e) =>
                      patch({
                        rules: draft.rules.map((v, n) =>
                          n === i
                            ? { ...v, mode: e.target.value as 'use' | 'ask' }
                            : v,
                        ),
                      })
                    }
                  >
                    <option value="use">In my draft automatically</option>
                    <option value="ask">Ask me first</option>
                  </select>
                </label>
              </div>
            ))}
            <button
              type="button"
              className="context-link"
              disabled={draft.rules.length >= 16}
              onClick={() =>
                patch({
                  rules: [
                    ...draft.rules,
                    {
                      anchor: '',
                      wording: '',
                      scenarios: [selection.scenario],
                      mode: 'use',
                    },
                  ],
                })
              }
            >
              <Plus size={14} /> Add personal wording
            </button>
            <div className="profile-save">
              <button className="context-save" type="submit">
                {draft.sample ? 'Save my copy' : 'Save profile'}
              </button>
              <span>
                Saved on this computer. Relevant context goes to the wording
                model when requested.
              </span>
            </div>
          </form>
        )}
      </fieldset>
    </section>
  );
}
