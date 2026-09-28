"""Reviewed local profiles and bounded, message-specific context snapshots."""
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
from contextlib import contextmanager
from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from app.schemas import CommunicationContext, CommunicationRegister
from app.personal.profile import AudienceProfile, LexiconEntry, SpecializationRule, valid_rule

Scenario = Literal['general', 'home', 'cafe', 'shopping', 'care', 'outside']
Listener = Literal['unspecified', 'familiar', 'new']
SCENARIOS = {'general': 'General', 'home': 'Home', 'cafe': 'Café', 'shopping': 'Shopping', 'care': 'Care', 'outside': 'Outside'}

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

class Person(StrictModel):
    name: str = Field(min_length=1, max_length=60)
    relationship: str = Field(default='', max_length=60)
    familiar: bool = True

class Rule(StrictModel):
    anchor: str = Field(min_length=1, max_length=60)
    wording: str = Field(min_length=1, max_length=150)
    scenarios: list[Scenario] = Field(min_length=1, max_length=6)
    mode: Literal['use', 'ask'] = 'use'
    @model_validator(mode='after')
    def meaningful(self):
        if not re.search(r'(?<!\w)' + re.escape(self.anchor) + r'(?!\w)', self.wording, re.I):
            raise ValueError('Personal wording must include the original word or phrase.')
        return self

class Moment(StrictModel):
    id: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=60)
    scenario: Scenario = 'home'
    cue: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=500)

class Profile(StrictModel):
    schema_version: Literal[2] = 2
    id: str = Field(default='', max_length=50)
    revision: int = Field(default=0, ge=0)
    label: str = Field(min_length=1, max_length=200)
    sample: bool = False
    language: Literal['original', 'English', 'Hindi/Hinglish'] = 'original'
    style: Literal['natural', 'concise'] = 'natural'
    about: str = Field(default='', max_length=400)
    manner: Literal['neutral', 'warm', 'direct'] = 'neutral'
    moments: list[Moment] = Field(default_factory=list, max_length=12)
    people: list[Person] = Field(default_factory=list, max_length=12)
    protected_terms: list[str] = Field(default_factory=list, max_length=24)
    rules: list[Rule] = Field(default_factory=list, max_length=16)
    # One versioned document keeps both clients' explicit knowledge. The older
    # `style` field stays intact; the bounded register is a separate concept.
    communication_style: CommunicationRegister | None = None
    context_default: CommunicationContext = 'general'
    listener_by_setting: dict[CommunicationContext, Literal['familiar', 'unfamiliar']] = Field(default_factory=dict)
    style_by_setting: dict[CommunicationContext, CommunicationRegister] = Field(default_factory=dict)
    audiences: list[AudienceProfile] = Field(default_factory=list)
    lexicon: list[LexiconEntry] = Field(default_factory=list)
    specializations: list[SpecializationRule] = Field(default_factory=list)
    speaker_note: str = ''
    icon: str = 'circle'
    provenance: dict = Field(default_factory=dict)
    @model_validator(mode='after')
    def unique_scopes(self):
        if any(not term.strip() or len(term) > 80 for term in self.protected_terms):
            raise ValueError('Use nonempty protected names or brands of at most 80 characters.')
        self.protected_terms = list(dict.fromkeys(term.strip() for term in self.protected_terms))
        if len({m.id for m in self.moments}) != len(self.moments):
            raise ValueError('Each personal moment needs a unique identifier.')
        for collection in (self.audiences, self.lexicon, self.specializations):
            if len({entry.id for entry in collection}) != len(collection):
                raise ValueError('Each audience, word, and detail needs a unique identifier.')
        if any(not valid_rule(rule) for rule in self.specializations):
            raise ValueError('A personal detail must preserve its anchor and reversible plain wording.')
        seen = set()
        for rule in self.rules:
            for setting in set(rule.scenarios):
                key = (rule.anchor.casefold(), setting)
                if key in seen:
                    raise ValueError('Use only one rule per word and scenario.')
                seen.add(key)
        return self

class Selection(StrictModel):
    profile_id: str = Field(default='', max_length=50)
    profile_revision: int = Field(default=0, ge=0)
    scenario: Scenario = 'general'
    listener: Listener = 'unspecified'
    recipient: str = Field(default='', max_length=60)
    situation: str = Field(default='', max_length=300)
    moment_id: str = Field(default='', max_length=50)
    use_personal_wording: bool = True
    place_id: str = Field(default='', max_length=100)
    audience_id: str = Field(default='', max_length=100)
    declared_listener: Literal['familiar', 'unfamiliar'] | None = None
    core_context: CommunicationContext | None = None

SAMPLES = [
    Profile(id='sample-tea', revision=1, sample=True, label='Asha · sample', language='English', manner='warm',
        about='A balcony gardener who enjoys a morning newspaper, a familiar tea order and unhurried outings. Fictional profile for exploration.',
        moments=[Moment(id='balcony', title='A little fresh air', scenario='home', cue='balcony', message='I would like to sit on the balcony for a while.'),
                 Moment(id='tea-stop', title='My usual tea stop', scenario='cafe', cue='usual', message='Could I have Lipton green tea, please?'),
                 Moment(id='quiet', title='A quieter outing', scenario='outside', cue='quiet spot', message='Could we find somewhere quieter to sit?')],
        people=[Person(name='Meena', relationship='Caregiver'), Person(name='Arjun', relationship='Friend')],
        rules=[Rule(anchor='tea', wording='Lipton green tea', scenarios=['cafe', 'outside', 'care']),
               Rule(anchor='soup', wording='tomato soup', scenarios=['home'], mode='ask')]),
    Profile(id='sample-hinglish', revision=1, sample=True, label='Kabir · sample', language='Hindi/Hinglish', manner='direct',
        about='A music lover and student who switches between Hindi and English. Likes direct requests and short breaks. Fictional profile for exploration.',
        moments=[Moment(id='study-break', title='Between study sessions', scenario='home', cue='break', message='Mujhe thoda break chahiye, please.'),
                 Moment(id='chai-stop', title='Chai with friends', scenario='cafe', cue='usual', message='Mujhe adrak wali chai chahiye, please.'),
                 Moment(id='heading-home', title='Heading home', scenario='outside', cue='ride', message='Mujhe ghar jaane ke liye cab book karni hai.')],
        people=[Person(name='Meena', relationship='Caregiver')],
        rules=[Rule(anchor='chai', wording='adrak wali chai', scenarios=['home', 'cafe'])]),
    Profile(id='sample-concise', revision=1, sample=True, label='Leela · sample', style='concise', language='English', manner='direct',
        about='A reader and writer who prefers short, complete sentences and space to finish a thought. Fictional profile for exploration.',
        moments=[Moment(id='reading', title='Lost in a book', scenario='home', cue='page', message='Please turn the page for me.'),
                 Moment(id='conversation', title='Time to reply', scenario='outside', cue='moment', message='Please give me a moment to type my reply.'),
                 Moment(id='appointment', title='At an appointment', scenario='care', cue='explain', message='Please explain one step at a time.')],
        rules=[Rule(anchor='blanket', wording='blue blanket', scenarios=['home'])]),
]

@contextmanager
def db():
    path = Path(os.getenv('ECHORA_PROFILE_DB', str(Path(__file__).resolve().parents[2] / 'data' / 'personal' / 'profiles.sqlite3')))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute('PRAGMA foreign_keys=ON')
    try:
        with conn:
            conn.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL)')
            from .memory import ensure_schema
            ensure_schema(conn)
            yield conn
    finally:
        conn.close()

def list_profiles():
    with db() as conn:
        custom = [Profile.model_validate_json(row[0]) for row in conn.execute('SELECT body FROM profiles ORDER BY id')]
    return [p.model_dump() for p in SAMPLES + custom]

def get_profile(profile_id):
    for p in SAMPLES:
        if p.id == profile_id: return p
    with db() as conn:
        row = conn.execute('SELECT body FROM profiles WHERE id=?', (profile_id,)).fetchone()
    if not row: raise HTTPException(404, 'This profile is unavailable. Choose another profile.')
    return Profile.model_validate_json(row[0])

def save_profile(profile):
    if profile.sample or profile.id.startswith('sample-'):
        raise HTTPException(422, 'Save a personal copy of a sample profile.')
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if profile.id:
            row = conn.execute('SELECT revision, body FROM profiles WHERE id=?', (profile.id,)).fetchone()
            if not row or row[0] != profile.revision:
                raise HTTPException(409, 'This profile changed. Reload the page before saving again.')
            # Import identity is immutable metadata, even if an older editor
            # omits it. Re-importing must never replace this person's edits.
            profile = profile.model_copy(update={'provenance': json.loads(row[1]).get('provenance', {})})
        elif conn.execute('SELECT count(*) FROM profiles').fetchone()[0] >= 20:
            raise HTTPException(422, 'The local limit of 20 profiles has been reached.')
        saved = profile.model_copy(update={'id': profile.id or secrets.token_hex(12), 'revision': profile.revision + 1})
        # REPLACE deletes the old row, which would cascade explicit memories.
        conn.execute('''INSERT INTO profiles VALUES (?,?,?) ON CONFLICT(id) DO UPDATE
                        SET revision=excluded.revision,body=excluded.body''', (saved.id, saved.revision, saved.model_dump_json()))
        from .memory import sync_profile_sources, invalidate
        changed = sync_profile_sources(conn, saved.model_dump())
    # Audience/recipient/scope edits can change eligibility even when no wording
    # changed. Signal the profile revision as well as individual source changes.
    invalidate(saved.id, changed | {'profile/' + saved.id})
    return saved.model_dump()

def composition_snapshot(snapshot, runtime=None):
    """Keep explicit session choices while honoring the profile-use switch."""
    settings = getattr(runtime, 'settings', runtime)
    if not snapshot or getattr(settings, 'personal_enabled', True):
        return snapshot
    from .profile_adapter import resolve_snapshot
    selection = {**snapshot['selection'], 'audience_id': ''}
    return resolve_snapshot(selection, None)


def snapshot(selection, runtime=None):
    profile = get_profile(selection.profile_id) if selection.profile_id else None
    if profile and selection.profile_revision != profile.revision:
        raise HTTPException(409, 'Your profile changed. Reload the page to use its latest version.')
    if selection.moment_id:
        moment = next((m for m in profile.moments if m.id == selection.moment_id), None) if profile else None
        if not moment or moment.scenario != selection.scenario:
            raise HTTPException(422, 'Choose a personal moment that belongs to this setting.')
    from .profile_adapter import resolve_snapshot
    return composition_snapshot(resolve_snapshot(selection.model_dump(), profile.model_dump() if profile else None), runtime)

def brief(snapshot, text, answers, runtime=None):
    snapshot = composition_snapshot(snapshot, runtime)
    if not snapshot: return {}
    settings = getattr(runtime, 'settings', runtime)
    selection = snapshot['selection']
    profile = snapshot['profile'] or {}
    # Only unqualified requests receive substitutions. Explicit qualifiers,
    # negation and clarification answers suppress saved wording automatically.
    normalized = re.sub(r'[^\w\s]', ' ', text.casefold()).strip()
    recipient = selection['recipient'].casefold()
    if recipient:
        normalized = re.sub(r'(?<!\w)' + re.escape(recipient) + r'(?!\w)', ' ', normalized)
    scaffold = {'please', 'bring', 'me', 'my', 'some', 'a', 'the', 'i', 'want', 'need', 'get', 'can', 'could', 'you', 'have'}
    words = normalized.split()
    content = ' '.join(w for w in words if w not in scaffold)
    if 'have' in words and not ({'can', 'could'} & set(words)):
        content = ''  # “I have tea” is a statement, not a usual-order shortcut.
    eligible = []
    if selection['use_personal_wording'] and not answers and getattr(settings, 'personal_specializations', True):
        for rule in profile.get('rules', []):
            if selection['scenario'] in rule['scenarios'] and content == rule['anchor'].casefold():
                eligible.append(rule)
        audience = next((item for item in profile.get('audiences', []) if item['id'] == selection.get('audience_id')), {})
        known = set(audience.get('known_detail_ids', []))
        for rule in profile.get('specializations', []):
            if (rule['id'] not in known and (not rule['settings'] or snapshot['core_context'] in rule['settings'])
                    and content == rule['anchor'].casefold()
                    and not any(item['anchor'].casefold() == content for item in eligible)):
                eligible.append({'anchor': rule['anchor'], 'wording': rule['surface'], 'plain': rule['plain'],
                                 'scenarios': [selection['scenario']], 'mode': 'use', 'source_id': rule['id']})
    moment = next((m for m in profile.get('moments', []) if m['id'] == selection.get('moment_id') and m['scenario'] == selection['scenario']), None)
    # A selected routine is not itself a request. Only its exact, explicitly
    # saved cue can activate the complete message; no fuzzy/semantic guessing.
    def normalized_cue(value):
        return ' '.join(re.sub(r'[^\w\s]', ' ', value.casefold()).split())
    approved = moment if (moment and selection['use_personal_wording'] and not answers
                          and normalized_cue(text) == normalized_cue(moment['cue'])) else None
    resolved = snapshot.get('resolved_audience') or {}
    return {'scenario': selection['scenario'], 'listener': ('new' if resolved.get('listener') == 'unfamiliar' else 'familiar') if resolved else selection['listener'],
            'core_context': snapshot.get('core_context', 'general'),
            'resolved_audience': resolved,
            'recipient': selection['recipient'], 'situation': selection['situation'],
            'style': profile.get('style', 'natural'), 'manner': profile.get('manner', 'neutral'),
            'personal_wording': [] if approved else eligible,
            'approved_message': approved['message'] if approved else '',
            'moment_title': moment['title'] if moment else '',
            'moment_cue': moment['cue'] if approved else ''}

def recognition_context(snapshot, hypotheses, runtime=None, *, text='', answers=None, reference=None):
    """Adapt only a frozen document; never reopen a live profile mid-utterance."""
    from .profile_adapter import evidence_context
    settings = getattr(runtime, 'settings', runtime)
    snapshot = composition_snapshot(snapshot, runtime)
    personal_brief, audience = evidence_context(
        snapshot, hypotheses, text=text, answers=answers, reference=reference,
        anchor_share=getattr(settings, 'personal_anchor_share', 0.75),
        max_hints=getattr(settings, 'personal_max_hints', 8),
    )
    if personal_brief and not getattr(settings, 'personal_specializations', True):
        personal_brief = personal_brief.model_copy(update={'specializations': []})
    wording = brief(snapshot, text, answers or {}, runtime).get('personal_wording', [])
    if reference:
        wording = []
    return {'context': (snapshot or {}).get('core_context', 'general'),
            'listener': audience.listener, 'register': audience.style,
            'brief': personal_brief, 'audience': audience,
            'personal_wording': wording,
            'requires_personal_choice': any(rule['mode'] == 'ask' for rule in wording)}

def finish(result, brief):
    result = dict(result)
    applied = []
    for rule in brief.get('personal_wording', []):
        if rule['mode'] == 'ask' and not result['question']:
            raise ValueError('A preference requiring a choice was not clarified')
        if result['question']: continue
        wording = rule['wording']
        # Fill only the anchor already present in a completed draft. Never
        # append a detached preference or apply a rule while clarifying.
        if rule['mode'] == 'use' and wording.casefold() not in result['text'].casefold():
            plain = rule.get('plain', rule['anchor'])
            anchor = plain if re.search(r'(?<!\w)' + re.escape(plain) + r'(?!\w)', result['text'], re.I) else rule['anchor']
            result['text'] = re.sub(r'(?<!\w)' + re.escape(anchor) + r'(?!\w)', lambda _: wording,
                                    result['text'], count=1, flags=re.I)
        if wording.casefold() in result['text'].casefold():
            applied.append({'anchor': rule['anchor'], 'wording': wording})
    if len(result['text']) > 2000:
        raise ValueError('Personal wording exceeds message limit')
    return result, applied
