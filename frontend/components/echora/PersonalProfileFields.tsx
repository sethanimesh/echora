'use client';
import { Plus, X } from 'lucide-react';
import { scenarios, type Profile, type Scenario } from './contextTypes';
export default function PersonalProfileFields({
  draft,
  patch,
}: {
  draft: Profile;
  patch: (fields: Partial<Profile>) => void;
}) {
  const moments = draft.moments ?? [];
  return (
    <>
      <div className="personal-intro-fields">
        <label>
          A little about me
          <textarea
            maxLength={400}
            rows={3}
            value={draft.about ?? ''}
            placeholder="What makes this profile yours? Your interests, habits, the way you like to spend your day…"
            onChange={(e) => patch({ about: e.target.value })}
          />
        </label>
        <label>
          How my words should sound
          <select
            value={draft.manner ?? 'neutral'}
            onChange={(e) =>
              patch({ manner: e.target.value as Profile['manner'] })
            }
          >
            <option value="neutral">Natural and neutral</option>
            <option value="warm">Warm and friendly</option>
            <option value="direct">Clear and direct</option>
          </select>
          <span className="optional-label">
            Shapes the wording, not the audio voice.
          </span>
        </label>
      </div>
      <h2>Little moments, in my own words</h2>
      <p>
        Give a familiar situation a name, then save a short cue and the complete
        message it means. Selecting a moment alone never speaks or creates a
        request.
      </p>
      <div className="moment-inspirations">
        <span>Ideas to make your own</span>
        <button
          type="button"
          onClick={() =>
            patch({
              moments: [
                ...moments,
                {
                  id: crypto.randomUUID(),
                  title: 'My usual order',
                  scenario: 'cafe',
                  cue: 'usual',
                  message: '',
                },
              ],
            })
          }
          disabled={moments.length >= 12}
        >
          My usual order
        </button>
        <button
          type="button"
          onClick={() =>
            patch({
              moments: [
                ...moments,
                {
                  id: crypto.randomUUID(),
                  title: 'Time to reply',
                  scenario: 'outside',
                  cue: 'a moment',
                  message: 'Please give me a moment to finish my reply.',
                },
              ],
            })
          }
          disabled={moments.length >= 12}
        >
          Time to reply
        </button>
        <button
          type="button"
          onClick={() =>
            patch({
              moments: [
                ...moments,
                {
                  id: crypto.randomUUID(),
                  title: 'Getting comfortable',
                  scenario: 'home',
                  cue: 'comfortable',
                  message: '',
                },
              ],
            })
          }
          disabled={moments.length >= 12}
        >
          Getting comfortable
        </button>
      </div>
      {moments.map((m, i) => {
        const change = (fields: Partial<typeof m>) =>
          patch({
            moments: moments.map((v, n) => (n === i ? { ...v, ...fields } : v)),
          });
        return (
          <div className="personal-moment-editor" key={m.id}>
            <div className="moment-editor-heading">
              <span>Moment {i + 1}</span>
              <button
                type="button"
                className="context-remove"
                aria-label={`Remove moment ${i + 1}`}
                onClick={() =>
                  patch({ moments: moments.filter((_, n) => n !== i) })
                }
              >
                <X size={16} />
              </button>
            </div>
            <div className="profile-fields">
              <label>
                Give it a name
                <input
                  required
                  value={m.title}
                  maxLength={60}
                  onChange={(e) => change({ title: e.target.value })}
                />
              </label>
              <label>
                Where it belongs
                <select
                  value={m.scenario}
                  onChange={(e) =>
                    change({ scenario: e.target.value as Scenario })
                  }
                >
                  {Object.entries(scenarios).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                My short cue
                <input
                  required
                  value={m.cue}
                  maxLength={80}
                  placeholder="usual"
                  onChange={(e) => change({ cue: e.target.value })}
                />
              </label>
            </div>
            <label>
              What I mean
              <textarea
                required
                value={m.message}
                rows={2}
                maxLength={500}
                placeholder="Write the exact message you want this cue to mean."
                onChange={(e) => change({ message: e.target.value })}
              />
            </label>
            {m.cue && m.message && (
              <p className="moment-preview">
                “{m.cue}” <span>→</span> “{m.message}”
              </p>
            )}
          </div>
        );
      })}
      <button
        type="button"
        className="context-link"
        disabled={moments.length >= 12}
        onClick={() =>
          patch({
            moments: [
              ...moments,
              {
                id: crypto.randomUUID(),
                title: '',
                scenario: 'home',
                cue: '',
                message: '',
              },
            ],
          })
        }
      >
        <Plus size={14} /> Add my own moment
      </button>
    </>
  );
}
