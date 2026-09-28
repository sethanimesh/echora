"""Lossless context resolution and conservative adaptation to literal evidence.

The versioned communication profile is authoritative. This module never opens
either project's stores and never lets a subsequent edit change a snapshot.
"""
from copy import deepcopy
import hashlib
import re

from fastapi import HTTPException

from app.schemas import CommunicationRegister, PersonalBrief, PersonaSummary, ResolvedAudience, resolve_listener
from app.personal.profile import PersonaProfile, SpecializationRule, valid_rule
from app.personal.lexicon import lexicon_hints, specialization_offers
from app.messaging.alignment import _slot_alignment, _tokens

CORE_CONTEXTS = {'general': 'general', 'home': 'home', 'care': 'care',
                 'cafe': 'outdoors', 'shopping': 'outdoors', 'outside': 'outdoors'}


def core_context(selection):
    return selection.get('core_context') or CORE_CONTEXTS.get(selection.get('scenario'), 'general')


def resolve_snapshot(selection, profile=None):
    selection, profile = deepcopy(selection), deepcopy(profile)
    document = profile or {}
    setting = core_context(selection)
    audience_id = selection.get('audience_id', '')
    audience = next((item for item in document.get('audiences', []) if item['id'] == audience_id), None)
    if audience_id and audience is None:
        raise HTTPException(422, 'Choose an audience from the selected profile.')
    if audience:
        visible_settings, visible_places = audience.get('visible_in_settings', []), audience.get('visible_in_places', [])
        if (visible_settings or visible_places) and setting not in visible_settings and selection.get('place_id') not in visible_places:
            raise HTTPException(422, 'This audience is not available in the selected place or setting.')
    explicit = {'familiar': 'familiar', 'new': 'unfamiliar'}.get(selection.get('listener'))
    person = next((item for item in document.get('people', [])
                   if item['name'].casefold() == selection.get('recipient', '').casefold()), None)
    # A chosen named audience is strongest. Preserve B's explicit listener and
    # named recipient as session choices before consulting place/profile defaults.
    chosen = audience['listener'] if audience else explicit or (
        ('familiar' if person.get('familiar', True) else 'unfamiliar') if person else None)
    declared = selection.get('declared_listener')
    by_profile = document.get('listener_by_setting', {}).get(setting)
    listener = resolve_listener(setting, by_audience=chosen, declared=declared, by_profile=by_profile)
    default_style = CommunicationRegister(brevity='short' if document.get('style') == 'concise' else 'natural').model_dump()
    if audience:
        style = audience.get('style_by_setting', {}).get(setting, audience.get('style', default_style))
        style_source = 'audience_setting' if setting in audience.get('style_by_setting', {}) else 'audience'
    else:
        style = document.get('style_by_setting', {}).get(setting, document.get('communication_style') or default_style)
        style_source = 'profile_setting' if setting in document.get('style_by_setting', {}) else 'profile' if profile else 'default'
    resolved = ResolvedAudience(
        audience_id=audience['id'] if audience else None,
        audience_label=audience['label'] if audience else selection.get('recipient') or ('Someone who knows you' if listener == 'familiar' else 'Someone new'),
        audience_kind=audience.get('kind', 'person') if audience else 'person' if person else 'generic',
        listener=listener, style=CommunicationRegister.model_validate(style),
        listener_source='audience' if chosen else 'place' if declared else 'profile' if by_profile else 'setting',
        style_source=style_source,
    )
    return {'selection': selection, 'profile': profile, 'core_context': setting,
            'resolved_audience': resolved.model_dump()}


def _unqualified(text, recipient=''):
    normalized = re.sub(r'[^\w\s]', ' ', text.casefold()).strip()
    if recipient:
        normalized = re.sub(r'(?<!\w)' + re.escape(recipient.casefold()) + r'(?!\w)', ' ', normalized)
    words = normalized.split()
    if 'have' in words and not ({'can', 'could'} & set(words)):
        return ''
    scaffold = {'please', 'bring', 'me', 'my', 'some', 'a', 'the', 'i', 'want', 'need', 'get', 'can', 'could', 'you', 'have'}
    return ' '.join(word for word in words if word not in scaffold)


def _rule_id(rule):
    return 'rule/' + hashlib.sha256((rule['anchor'] + '\0' + rule['wording']).encode()).hexdigest()[:16]


def evidence_context(snapshot, hypotheses, *, text='', answers=None, reference=None, anchor_share=0.75, max_hints=8):
    """Return (PersonalBrief | None, ResolvedAudience) from frozen inputs only.

    Original scoped rules stay in the profile; only a use rule in the selected
    six-value scenario can become an A detail. Ask rules never auto-apply.
    """
    snap = snapshot or resolve_snapshot({'scenario': 'general'})
    resolved = ResolvedAudience.model_validate(snap.get('resolved_audience') or resolve_snapshot(snap['selection'], snap.get('profile'))['resolved_audience'])
    document = snap.get('profile')
    if not document:
        return None, resolved
    selection, setting = snap['selection'], snap.get('core_context') or core_context(snap['selection'])
    details = deepcopy(document.get('specializations', []))
    # Keep A's exact IDs so known_detail_ids remains meaningful after importing.
    occupied_anchors = {rule['anchor'].casefold() for rule in details}
    for rule in document.get('rules', []):
        if selection.get('scenario') not in rule['scenarios'] or rule['mode'] != 'use' or rule['anchor'].casefold() in occupied_anchors:
            continue
        candidate = SpecializationRule(id=_rule_id(rule), anchor=rule['anchor'], plain=rule['anchor'], surface=rule['wording'], kind='object')
        if valid_rule(candidate):
            details.append(candidate.model_dump())
            occupied_anchors.add(rule['anchor'].casefold())
    lexicon = deepcopy(document.get('lexicon', []))
    for index, term in enumerate([p['name'] for p in document.get('people', [])] + document.get('protected_terms', [])):
        if len(_tokens(term)) == 1 and not any(entry['word'].casefold() == term.casefold() for entry in lexicon):
            lexicon.append({'id': f'legacy-word/{index}', 'word': term, 'display': term, 'kind': 'person', 'aliases': [], 'settings': []})
    profile = PersonaProfile(id=document['id'], label=document['label'], lexicon=lexicon,
                             specializations=details, speaker_note=document.get('speaker_note', ''))
    slots = _slot_alignment(hypotheses)
    offers = specialization_offers(profile, slots, anchor_share, setting)
    selected_audience = next((a for a in document.get('audiences', []) if a['id'] == selection.get('audience_id')), {})
    known = set(selected_audience.get('known_detail_ids', []))
    source = text or (hypotheses[0].literal_text if hypotheses else '')
    content = _unqualified(source, selection.get('recipient', ''))
    # Explicit qualifications, negation, alternative requests and clarification
    # answers always take precedence over a saved usual wording.
    offers = [offer for offer in offers if offer.source not in known and content in {offer.anchor, *offer.matches}]
    if answers or reference or not selection.get('use_personal_wording', True):
        offers = []
    audit = set()
    for entry in lexicon:
        for value in [entry['word'], entry['display'], *entry.get('aliases', [])]:
            audit.update(_tokens(value))
    for rule in document.get('specializations', []):
        audit.update(set(_tokens(rule['surface'])) - set(_tokens(rule['plain'])))
    for rule in document.get('rules', []):
        audit.update(set(_tokens(rule['wording'])) - set(_tokens(rule['anchor'])))
    brief = PersonalBrief(profile_id=profile.id, profile_label=profile.label,
                          speaker_note=profile.speaker_note,
                          lexicon=lexicon_hints(profile, slots, max_hints, setting),
                          specializations=offers, audit_vocabulary=sorted(audit))
    return (None if brief.is_empty() and not brief.audit_vocabulary else brief), resolved


def as_legacy_profile(document):
    """Present a unified profile to the existing mobile profile contract.

    The complete B rules/moments remain in the authoritative document. Its ask
    semantics and six scenario scopes cannot be losslessly expressed by A's
    four-setting specialization format, so this view exposes A-compatible
    fields and never rewrites or discards the rest.
    """
    data = document.model_dump() if hasattr(document, 'model_dump') else deepcopy(document)
    default_style = CommunicationRegister(brevity='short' if data.get('style') == 'concise' else 'natural')
    return PersonaProfile(
        id=data['id'], label=data['label'], blurb=data.get('about', ''),
        icon=data.get('icon', 'circle'), baseline=data.get('sample', False),
        context_default=data.get('context_default', 'general'),
        listener_by_setting=data.get('listener_by_setting', {}),
        style=data.get('communication_style') or default_style,
        style_by_setting=data.get('style_by_setting', {}),
        speaker_note=data.get('speaker_note', ''), lexicon=data.get('lexicon', []),
        specializations=data.get('specializations', []), audiences=data.get('audiences', []),
        created_at=data.get('provenance', {}).get('source_created_at', ''),
    )


def legacy_summary(document):
    data = document.model_dump() if hasattr(document, 'model_dump') else document
    profile = as_legacy_profile(data)
    return PersonaSummary(id=profile.id, label=profile.label, blurb=profile.blurb,
        icon=profile.icon, context_default=profile.context_default,
        listener_by_setting=profile.listener_by_setting,
        lexicon_size=len(profile.lexicon), specialization_size=len(profile.specializations),
        audience_size=len(profile.audiences), baseline=data.get('sample', False))


def merge_legacy_profile(document, changes):
    """Merge an explicit mobile edit without deleting B rules, cues, or language.

    Returns a validated unified Profile with its existing ID/revision; the
    caller must use context.save_profile for optimistic concurrency and storage.
    """
    from .context import Profile
    data = document.model_dump() if hasattr(document, 'model_dump') else deepcopy(document)
    values = changes.model_dump() if hasattr(changes, 'model_dump') else dict(changes)
    mappings = {'blurb': 'about', 'style': 'communication_style'}
    for key in ('label', 'blurb', 'context_default', 'listener_by_setting', 'style',
                'style_by_setting', 'lexicon', 'specializations', 'audiences'):
        if key in values:
            data[mappings.get(key, key)] = values[key]
    return Profile.model_validate(data)
