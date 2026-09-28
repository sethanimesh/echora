export const scenarios = {
  general: 'General',
  home: 'Home',
  cafe: 'Café',
  shopping: 'Shopping',
  care: 'Care',
  outside: 'Outside',
} as const;
export type Scenario = keyof typeof scenarios;
export type CoreContext = 'general' | 'home' | 'care' | 'outdoors';
export type CommunicationStyle = {
  brevity: 'short' | 'natural' | 'complete';
  courtesy: 'plain' | 'please';
  formality: 'informal' | 'neutral' | 'formal';
};
export type Audience = {
  id: string;
  label: string;
  relationship: string;
  listener: 'familiar' | 'unfamiliar';
  style: CommunicationStyle;
  style_by_setting?: Partial<Record<CoreContext, CommunicationStyle>>;
  visible_in_settings?: CoreContext[];
  visible_in_places?: string[];
  known_detail_ids?: string[];
  [key: string]: unknown;
};
export type ContextSelection = {
  place_id?: string;
  audience_id?: string;
  declared_listener?: 'familiar' | 'unfamiliar' | null;
  core_context?: CoreContext | null;
  profile_id: string;
  profile_revision: number;
  scenario: Scenario;
  listener: 'unspecified' | 'familiar' | 'new';
  recipient: string;
  situation: string;
  moment_id: string;
  use_personal_wording: boolean;
};
export type Profile = {
  schema_version?: number;
  provenance?: Record<string, unknown>;
  context_default?: CoreContext;
  communication_style?: CommunicationStyle;
  style_by_setting?: Partial<Record<CoreContext, CommunicationStyle>>;
  listener_by_setting?: Partial<Record<CoreContext, 'familiar' | 'unfamiliar'>>;
  audiences?: Audience[];
  lexicon?: {
    id: string;
    word: string;
    display: string;
    aliases: string[];
    kind: string;
    settings: CoreContext[];
    [key: string]: unknown;
  }[];
  specializations?: {
    id: string;
    anchor: string;
    plain: string;
    surface: string;
    kind: string;
    settings: CoreContext[];
    [key: string]: unknown;
  }[];
  id: string;
  revision: number;
  label: string;
  sample: boolean;
  language: 'original' | 'English' | 'Hindi/Hinglish';
  style: 'natural' | 'concise';
  about: string;
  manner: 'neutral' | 'warm' | 'direct';
  moments: {
    id: string;
    title: string;
    scenario: Scenario;
    cue: string;
    message: string;
  }[];
  people: { name: string; relationship: string; familiar: boolean }[];
  protected_terms?: string[];
  rules: {
    anchor: string;
    wording: string;
    scenarios: Scenario[];
    mode: 'use' | 'ask';
  }[];
};
export const emptyContext: ContextSelection = {
  place_id: '',
  audience_id: '',
  declared_listener: null,
  core_context: null,
  profile_id: '',
  profile_revision: 0,
  scenario: 'general',
  listener: 'unspecified',
  recipient: '',
  situation: '',
  moment_id: '',
  use_personal_wording: true,
};
export const emptyProfile: Profile = {
  id: '',
  revision: 0,
  label: 'My profile',
  sample: false,
  language: 'original',
  style: 'natural',
  about: '',
  manner: 'neutral',
  moments: [],
  people: [],
  rules: [],
};
