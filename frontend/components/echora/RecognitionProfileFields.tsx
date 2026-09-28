'use client';
import type {
  Audience,
  CommunicationStyle,
  CoreContext,
  Profile,
} from './contextTypes';
const settings: CoreContext[] = ['general', 'home', 'care', 'outdoors'];
const defaultStyle: CommunicationStyle = {
  brevity: 'natural',
  courtesy: 'plain',
  formality: 'neutral',
};
function StyleFields({
  value,
  onChange,
}: {
  value: CommunicationStyle;
  onChange: (style: CommunicationStyle) => void;
}) {
  return (
    <div className="profile-fields">
      <label>
        Length
        <select
          value={value.brevity}
          onChange={(e) =>
            onChange({
              ...value,
              brevity: e.target.value as CommunicationStyle['brevity'],
            })
          }
        >
          <option value="short">Short</option>
          <option value="natural">Natural</option>
          <option value="complete">Complete</option>
        </select>
      </label>
      <label>
        Courtesy
        <select
          value={value.courtesy}
          onChange={(e) =>
            onChange({
              ...value,
              courtesy: e.target.value as CommunicationStyle['courtesy'],
            })
          }
        >
          <option value="plain">Plain words</option>
          <option value="please">Use please</option>
        </select>
      </label>
      <label>
        Formality
        <select
          value={value.formality}
          onChange={(e) =>
            onChange({
              ...value,
              formality: e.target.value as CommunicationStyle['formality'],
            })
          }
        >
          <option value="informal">Informal</option>
          <option value="neutral">Neutral</option>
          <option value="formal">Formal</option>
        </select>
      </label>
    </div>
  );
}
function Scope({
  value,
  onChange,
}: {
  value: CoreContext[];
  onChange: (scope: CoreContext[]) => void;
}) {
  return (
    <div className="rule-settings">
      <small>Where this applies (none selected means everywhere)</small>
      {settings.map((setting) => (
        <label className="context-check" key={setting}>
          <input
            type="checkbox"
            checked={value.includes(setting)}
            onChange={(e) =>
              onChange(
                e.target.checked
                  ? [...value, setting]
                  : value.filter((s) => s !== setting),
              )
            }
          />
          {setting}
        </label>
      ))}
    </div>
  );
}
export default function RecognitionProfileFields({
  draft,
  patch,
}: {
  draft: Profile;
  patch: (fields: Partial<Profile>) => void;
}) {
  const audiences = draft.audiences ?? [];
  const lexicon = draft.lexicon ?? [];
  const details = draft.specializations ?? [];
  const updateAudience = (index: number, fields: Partial<Audience>) =>
    patch({
      audiences: audiences.map((audience, i) =>
        i === index ? { ...audience, ...fields } : audience,
      ),
    });
  return (
    <details className="recognition-profile">
      <summary>My listeners, familiar words and speech wording</summary>
      <p>
        These settings also support adapted speech recognition. Current words
        take priority over saved preferences.
      </p>
      <h3>My usual wording</h3>
      <StyleFields
        value={draft.communication_style ?? defaultStyle}
        onChange={(communication_style) => patch({ communication_style })}
      />
      <h3>People in each setting</h3>
      <div className="profile-fields">
        {settings.map((setting) => (
          <label key={setting}>
            {setting}
            <select
              value={draft.listener_by_setting?.[setting] ?? ''}
              onChange={(e) => {
                const mapping = { ...draft.listener_by_setting };
                if (e.target.value)
                  mapping[setting] = e.target.value as
                    | 'familiar'
                    | 'unfamiliar';
                else delete mapping[setting];
                patch({ listener_by_setting: mapping });
              }}
            >
              <option value="">Use the setting default</option>
              <option value="familiar">Usually know me</option>
              <option value="unfamiliar">Usually unfamiliar</option>
            </select>
          </label>
        ))}
      </div>
      <h3>Named listeners</h3>
      {audiences.map((audience, i) => (
        <div className="rule-editor" key={audience.id}>
          <div className="profile-fields">
            <label>
              Name
              <input
                required
                maxLength={60}
                value={audience.label}
                onChange={(e) => updateAudience(i, { label: e.target.value })}
              />
            </label>
            <label>
              Relationship
              <input
                maxLength={60}
                value={audience.relationship}
                onChange={(e) =>
                  updateAudience(i, { relationship: e.target.value })
                }
              />
            </label>
            <label>
              Familiarity
              <select
                value={audience.listener}
                onChange={(e) =>
                  updateAudience(i, {
                    listener: e.target.value as Audience['listener'],
                  })
                }
              >
                <option value="familiar">Knows me</option>
                <option value="unfamiliar">Unfamiliar / staff</option>
              </select>
            </label>
          </div>
          <StyleFields
            value={audience.style ?? defaultStyle}
            onChange={(style) => updateAudience(i, { style })}
          />
          <Scope
            value={audience.visible_in_settings ?? []}
            onChange={(visible_in_settings) =>
              updateAudience(i, { visible_in_settings })
            }
          />
          {!!audience.style_by_setting &&
            Object.keys(audience.style_by_setting).length > 0 && (
              <details>
                <summary>Wording by setting</summary>
                {settings
                  .filter((setting) => audience.style_by_setting?.[setting])
                  .map((setting) => (
                    <div key={setting}>
                      <strong>{setting}</strong>
                      <StyleFields
                        value={audience.style_by_setting![setting]!}
                        onChange={(style) =>
                          updateAudience(i, {
                            style_by_setting: {
                              ...audience.style_by_setting,
                              [setting]: style,
                            },
                          })
                        }
                      />
                    </div>
                  ))}
              </details>
            )}
          {!!details.length && (
            <details>
              <summary>Details this listener already knows</summary>
              {details.map((detail) => (
                <label className="context-check" key={detail.id}>
                  <input
                    type="checkbox"
                    checked={
                      audience.known_detail_ids?.includes(detail.id) ?? false
                    }
                    onChange={(e) =>
                      updateAudience(i, {
                        known_detail_ids: e.target.checked
                          ? [...(audience.known_detail_ids ?? []), detail.id]
                          : (audience.known_detail_ids ?? []).filter(
                              (id) => id !== detail.id,
                            ),
                      })
                    }
                  />
                  {detail.surface}
                </label>
              ))}
            </details>
          )}
          <button
            type="button"
            className="context-link"
            onClick={() =>
              patch({ audiences: audiences.filter((_, n) => n !== i) })
            }
          >
            Remove listener
          </button>
        </div>
      ))}
      <button
        type="button"
        className="context-link"
        onClick={() =>
          patch({
            audiences: [
              ...audiences,
              {
                id: crypto.randomUUID(),
                label: '',
                relationship: '',
                listener: 'familiar',
                style: { ...defaultStyle },
              },
            ],
          })
        }
      >
        Add a named listener
      </button>
      <h3>Familiar words</h3>
      <p>
        Words and alternative spellings help interpret recognition evidence.
      </p>
      {lexicon.map((entry, i) => (
        <div className="rule-editor" key={entry.id}>
          <div className="profile-fields">
            <label>
              Word
              <input
                required
                value={entry.word}
                maxLength={80}
                onChange={(e) =>
                  patch({
                    lexicon: lexicon.map((v, n) =>
                      n === i ? { ...v, word: e.target.value } : v,
                    ),
                  })
                }
              />
            </label>
            <label>
              Display spelling
              <input
                required
                value={entry.display}
                maxLength={80}
                onChange={(e) =>
                  patch({
                    lexicon: lexicon.map((v, n) =>
                      n === i ? { ...v, display: e.target.value } : v,
                    ),
                  })
                }
              />
            </label>
            <label>
              Other spellings (comma separated)
              <input
                value={entry.aliases.join(', ')}
                onChange={(e) =>
                  patch({
                    lexicon: lexicon.map((v, n) =>
                      n === i
                        ? {
                            ...v,
                            aliases: e.target.value
                              .split(',')
                              .map((word) => word.trim())
                              .filter(Boolean),
                          }
                        : v,
                    ),
                  })
                }
              />
            </label>
          </div>
          <Scope
            value={entry.settings}
            onChange={(scope) =>
              patch({
                lexicon: lexicon.map((v, n) =>
                  n === i ? { ...v, settings: scope } : v,
                ),
              })
            }
          />
          <button
            type="button"
            className="context-link"
            onClick={() =>
              patch({ lexicon: lexicon.filter((_, n) => n !== i) })
            }
          >
            Remove word
          </button>
        </div>
      ))}
      <button
        type="button"
        className="context-link"
        onClick={() =>
          patch({
            lexicon: [
              ...lexicon,
              {
                id: crypto.randomUUID(),
                word: '',
                display: '',
                aliases: [],
                kind: 'object',
                settings: [],
              },
            ],
          })
        }
      >
        Add a familiar word
      </button>
      <h3>My specific wording</h3>
      {details.map((detail, i) => (
        <div className="rule-editor" key={detail.id}>
          <div className="profile-fields">
            {(['anchor', 'plain', 'surface'] as const).map((field) => (
              <label key={field}>
                {field === 'anchor'
                  ? 'When heard'
                  : field === 'plain'
                    ? 'Plain wording'
                    : 'My fuller wording'}
                <input
                  required
                  value={detail[field]}
                  maxLength={150}
                  onChange={(e) =>
                    patch({
                      specializations: details.map((v, n) =>
                        n === i ? { ...v, [field]: e.target.value } : v,
                      ),
                    })
                  }
                />
              </label>
            ))}
          </div>
          <Scope
            value={detail.settings}
            onChange={(scope) =>
              patch({
                specializations: details.map((v, n) =>
                  n === i ? { ...v, settings: scope } : v,
                ),
              })
            }
          />
          <button
            type="button"
            className="context-link"
            onClick={() =>
              patch({ specializations: details.filter((_, n) => n !== i) })
            }
          >
            Remove specific wording
          </button>
        </div>
      ))}
      <button
        type="button"
        className="context-link"
        onClick={() =>
          patch({
            specializations: [
              ...details,
              {
                id: crypto.randomUUID(),
                anchor: '',
                plain: '',
                surface: '',
                kind: 'object',
                settings: [],
              },
            ],
          })
        }
      >
        Add specific wording
      </button>
    </details>
  );
}
