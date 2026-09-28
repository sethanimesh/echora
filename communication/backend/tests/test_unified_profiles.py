"""Offline migration, frozen-context, and explicit-wording contract checks."""
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from communication.backend import context
from communication.backend.profile_adapter import evidence_context, resolve_snapshot
from communication.backend.profile_migration import adapt_a, adapt_b, candidates, import_profiles
from app.schemas import Hypothesis


def a_profile():
    return {'id': 'speaker', 'label': 'Demo speaker', 'context_default': 'home',
            'listener_by_setting': {'outdoors': 'familiar'},
            'style': {'brevity': 'complete', 'courtesy': 'please', 'formality': 'formal'},
            'lexicon': [{'id': 'speaker/word/dawn', 'word': 'dawn', 'aliases': ['dorn'], 'display': 'Dawn', 'kind': 'person', 'settings': ['home']}],
            'specializations': [{'id': 'speaker/detail/tea', 'anchor': 'tea', 'plain': 'tea', 'surface': 'Lipton tea', 'kind': 'food', 'settings': ['home', 'outdoors']}],
            'audiences': [{'id': 'carer', 'label': 'Dawn', 'listener': 'familiar', 'known_detail_ids': ['speaker/detail/tea'],
                           'style': {'brevity': 'short', 'courtesy': 'plain', 'formality': 'informal'},
                           'style_by_setting': {'care': {'brevity': 'complete', 'courtesy': 'please', 'formality': 'formal'}}}]}


def beams(*texts):
    return [Hypothesis(id=f'h{i}', literal_text=text, sequence_score=-i, search_weight=1 / len(texts)) for i, text in enumerate(texts, 1)]


def freeze(document=None, **values):
    selection = context.Selection(**values).model_dump()
    return resolve_snapshot(selection, document)


def test_dry_run_is_read_only_and_import_preserves_a_fields(tmp_path):
    original = a_profile()
    profile = adapt_a(original)
    target = tmp_path / 'new' / 'profiles.sqlite3'
    report = import_profiles([(profile, original)], target)
    assert report['counts'] == {'imported': 1, 'skipped': 0, 'conflicts': 0}
    assert not target.parent.exists()
    assert 'Demo speaker' not in json.dumps(report) and 'Lipton' not in json.dumps(report)
    import_profiles([(profile, original)], target, apply=True)
    with sqlite3.connect(target) as conn:
        stored = context.Profile.model_validate_json(conn.execute('SELECT body FROM profiles').fetchone()[0])
    assert stored.schema_version == 2
    assert stored.lexicon[0].aliases == ['dorn']
    assert stored.specializations[0].id == 'speaker/detail/tea'
    assert stored.audiences[0].known_detail_ids == ['speaker/detail/tea']
    assert stored.communication_style.formality == 'formal'
    assert stored.provenance['source_id'] == 'speaker'


def test_reimport_never_overwrites_explicit_edits(tmp_path, monkeypatch):
    destination = tmp_path / 'profiles.sqlite3'
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(destination))
    raw = a_profile()
    first = import_profiles([(adapt_a(raw), raw)], destination, apply=True)
    target_id = first['imported'][0]
    loaded = context.get_profile(target_id)
    edited = loaded.model_copy(update={'label': 'My explicit edit', 'provenance': {}})
    context.save_profile(edited)
    changed_source = {**raw, 'label': 'Upstream changed'}
    second = import_profiles([(adapt_a(changed_source), changed_source)], destination, apply=True)
    assert second['skipped'] == [target_id]
    assert context.get_profile(target_id).label == 'My explicit edit'
    assert context.get_profile(target_id).revision == 2
    assert context.get_profile(target_id).provenance['source_id'] == raw['id']


def test_duplicate_source_ids_with_conflicting_data_are_not_imported(tmp_path):
    raw = a_profile()
    other = {**raw, 'label': 'Distinct source with same ID'}
    destination = tmp_path / 'profiles.sqlite3'
    report = import_profiles([(adapt_a(raw), raw), (adapt_a(other), other)], destination, apply=True)
    assert report['counts'] == {'imported': 0, 'skipped': 0, 'conflicts': 1}
    assert not destination.exists()


def test_original_b_sqlite_is_untouched_and_rules_cues_preserved(tmp_path):
    source, destination = tmp_path / 'old.sqlite3', tmp_path / 'new.sqlite3'
    original = context.Profile(id='old', revision=7, label='Existing B', language='Hindi/Hinglish',
        rules=[context.Rule(anchor='tea', wording='green tea', scenarios=['cafe'], mode='ask')],
        moments=[context.Moment(id='usual', title='Tea stop', scenario='cafe', cue='my usual', message='Meri chai please.')])
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE TABLE profiles (id TEXT PRIMARY KEY, revision INTEGER, body TEXT)')
        conn.execute('INSERT INTO profiles VALUES (?,?,?)', (original.id, original.revision, original.model_dump_json()))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    report = import_profiles(candidates(b_database=source), destination, apply=True)
    assert len(report['imported']) == 1
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with sqlite3.connect(destination) as conn:
        stored = context.Profile.model_validate_json(conn.execute('SELECT body FROM profiles').fetchone()[0])
    assert stored.rules == original.rules and stored.moments == original.moments
    assert stored.provenance['source_revision'] == 7
    assert stored.language == 'Hindi/Hinglish'


def test_explicit_audience_place_profile_default_precedence_and_abstention():
    document = adapt_a(a_profile()).model_copy(update={'id': 'speaker'}).model_dump()
    assert freeze(document, scenario='outside')['resolved_audience']['listener_source'] == 'profile'
    placed = freeze(document, scenario='outside', declared_listener='unfamiliar')
    assert placed['resolved_audience']['listener'] == 'unfamiliar'
    chosen = freeze(document, scenario='outside', declared_listener='unfamiliar', audience_id='carer')
    assert chosen['resolved_audience']['listener_source'] == 'audience'
    assert chosen['resolved_audience']['listener'] == 'familiar'
    assert chosen['resolved_audience']['style']['brevity'] == 'short'
    assert freeze(scenario='shopping')['resolved_audience']['listener'] == 'unfamiliar'
    assert freeze(document, scenario='care', audience_id='carer')['resolved_audience']['style_source'] == 'audience_setting'


def test_snapshot_holds_whole_profile_and_survives_later_edits(tmp_path, monkeypatch):
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path / 'profiles.sqlite3'))
    saved = context.save_profile(adapt_a(a_profile()))
    snap = context.snapshot(context.Selection(profile_id=saved['id'], profile_revision=saved['revision'], scenario='home', audience_id='carer'))
    current = context.get_profile(saved['id'])
    current.audiences[0].label = 'Changed after recording'
    context.save_profile(current)
    _, audience = evidence_context(snap, beams('tea'))
    assert audience.audience_label == 'Dawn'
    assert snap['profile']['audiences'][0]['label'] == 'Dawn'
    with pytest.raises(Exception) as error:
        context.snapshot(context.Selection(profile_id=saved['id'], profile_revision=1))
    assert error.value.status_code == 409


def test_audience_scopes_do_not_guess_someone_at_a_location():
    raw = a_profile()
    raw['audiences'][0]['visible_in_places'] = ['my-home']
    document = adapt_a(raw).model_copy(update={'id': 'speaker'}).model_dump()
    assert freeze(document, place_id='my-home')['resolved_audience']['audience_id'] is None
    with pytest.raises(Exception) as error:
        freeze(document, place_id='elsewhere', audience_id='carer')
    assert error.value.status_code == 422
    assert freeze(document, place_id='my-home', audience_id='carer')['resolved_audience']['audience_id'] == 'carer'


def test_anchor_gate_explicit_words_answers_and_known_details():
    document = adapt_a(a_profile()).model_copy(update={'id': 'speaker'}).model_dump()
    snap = freeze(document, scenario='home')
    personal, _ = evidence_context(snap, beams('tea'))
    assert personal.specializations[0].surface == 'Lipton tea'
    for explicit in ['black tea', 'no tea', 'I do not want tea', 'tea or coffee']:
        personal, _ = evidence_context(snap, beams(explicit))
        assert personal is None or not personal.specializations
    personal, _ = evidence_context(snap, beams('tea'), answers=[{'answer': 'black tea'}])
    assert personal is None or not personal.specializations
    personal, _ = evidence_context(snap, beams('tea', 'coffee'))
    assert personal is None or not personal.specializations
    personal, _ = evidence_context(freeze(document, scenario='home', audience_id='carer'), beams('tea'))
    assert personal is None or not personal.specializations


def test_six_b_scenarios_remain_distinct_and_ask_never_auto_applies():
    profile = context.Profile(id='b', label='B', rules=[context.Rule(anchor='tea', wording='green tea', scenarios=['cafe'], mode='ask')])
    cafe = freeze(profile.model_dump(), scenario='cafe')
    result = context.recognition_context(cafe, beams('tea'), text='tea')
    assert result['requires_personal_choice'] is True
    assert result['brief'] is None or not result['brief'].specializations
    shopping = context.recognition_context(freeze(profile.model_dump(), scenario='shopping'), beams('tea'), text='tea')
    assert cafe['core_context'] == shopping['context'] == 'outdoors'
    assert not shopping['requires_personal_choice']


def test_text_path_retains_imported_details_with_explicit_overrides():
    document = adapt_a(a_profile()).model_copy(update={'id': 'speaker'}).model_dump()
    snap = freeze(document, scenario='home')
    brief = context.brief(snap, 'tea', [])
    assert brief['personal_wording'][0]['source_id'] == 'speaker/detail/tea'
    result, applied = context.finish({'text': 'Please bring tea.', 'question': '', 'options': []}, brief)
    assert result['text'] == 'Please bring Lipton tea.' and applied
    assert not context.brief(snap, 'black tea', [])['personal_wording']
    assert not context.brief(snap, 'tea', [{'answer': 'black tea'}])['personal_wording']


def test_import_ids_are_namespaced_and_legacy_concise_style_is_kept(tmp_path):
    raw = a_profile()
    b = {'id': raw['id'], 'revision': 1, 'label': 'B', 'style': 'concise'}
    report = import_profiles([(adapt_a(raw), raw), (adapt_b(b), b)], tmp_path / 'profiles.sqlite3')
    assert len(set(report['imported'])) == 2
    assert freeze(context.Profile.model_validate(b).model_dump())['resolved_audience']['style']['brevity'] == 'short'


def test_runtime_profile_disable_keeps_session_place_and_explicit_listener():
    document = adapt_a(a_profile()).model_copy(update={'id': 'speaker'}).model_dump()
    snap = freeze(document, scenario='outside', place_id='cafe', declared_listener='unfamiliar', audience_id='carer')
    runtime = SimpleNamespace(settings=SimpleNamespace(personal_enabled=False))
    resolved = context.recognition_context(snap, beams('tea'), runtime, text='tea')
    assert resolved['brief'] is None and resolved['personal_wording'] == []
    assert resolved['listener'] == 'unfamiliar'
    assert not context.brief(snap, 'tea', [], runtime)['personal_wording']
    assert snap['profile'] == document and snap['selection']['audience_id'] == 'carer'
    explicit = freeze(document, scenario='outside', listener='familiar', declared_listener='unfamiliar')
    assert context.recognition_context(explicit, beams('tea'), runtime, text='tea')['listener'] == 'familiar'


def test_specialization_disable_preserves_lexicon_and_audit_without_substitutions():
    document = adapt_a(a_profile()).model_copy(update={'id': 'speaker'}).model_dump()
    snap = freeze(document, scenario='home')
    runtime = SimpleNamespace(settings=SimpleNamespace(personal_enabled=True, personal_specializations=False))
    resolved = context.recognition_context(snap, beams('tea'), runtime, text='tea')
    assert not resolved['brief'].specializations
    assert 'lipton' in resolved['brief'].audit_vocabulary
    assert not resolved['personal_wording'] and not resolved['requires_personal_choice']
    assert not context.brief(snap, 'tea', [], runtime)['personal_wording']
    named = context.recognition_context(snap, beams('dawn'), runtime, text='dawn')
    assert named['brief'].lexicon


def test_disabled_wording_and_followup_do_not_reactivate_ask_preference():
    profile = context.Profile(id='b', label='B', rules=[context.Rule(anchor='tea', wording='green tea', scenarios=['cafe'], mode='ask')])
    for runtime, values, reference in [
        (SimpleNamespace(personal_specializations=False), {}, None),
        (None, {'use_personal_wording': False}, None),
        (None, {}, {'text': 'Tea with sugar.', 'job_id': 'old'}),
    ]:
        snap = freeze(profile.model_dump(), scenario='cafe', **values)
        resolved = context.recognition_context(snap, beams('tea'), runtime, text='tea', reference=reference)
        assert not resolved['personal_wording'] and not resolved['requires_personal_choice']
