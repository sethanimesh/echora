"""Learned selection is separate from raw evidence, wording and speech consent."""
from types import SimpleNamespace
import json
import pytest

from communication.backend.tests.test_unified_recognition import (
    unified, raw, message, composed, upload, ready, post,
)


def verifier(runtime, selected=None, status='ready'):
    def rank(result, scores, retrieval):
        return {'status': status, 'decision': 'selected' if selected else 'ambiguous',
                'selected_hypothesis_id': selected,
                'ordered_hypothesis_ids': [h.id for h in result.hypotheses],
                'scores': [], 'artifacts': {'test': True}, 'reason': 'test decision'}
    runtime.verification = SimpleNamespace(available=True, rank=rank)


def test_selected_fourth_beam_speaks_its_completion_and_keeps_all_literals(unified):
    client, runtime, calls = unified
    runtime.asr.result = raw('tea', 'toast', 'soup', 'water', 'coffee')
    runtime.message_chain.result = composed(message('m4', 'I need water.', 'h4', 'water'))
    verifier(runtime, 'h4')
    upload(client)
    job = ready(client)
    assert job['selected_candidate_id'] == 'm4'
    assert job['text'] == 'I need water.'
    assert job['auto_speak_revision'] == job['revision']
    assert job['evidence']['hypotheses'] == [h.model_dump() for h in runtime.asr.result.hypotheses]
    assert {c['id'] for c in job['candidates']} >= {f'raw-h{i}' for i in range(1, 6)}
    assert calls[-1][3]['communication_context']['allowed_readings'] == ['water']
    chosen = post(client, job, 'choose', candidate_id='raw-h5').json()
    assert chosen['text'] == 'coffee' and chosen['decision_source'] == 'user'
    assert chosen['ranking'] == job['ranking']
    assert chosen['auto_speak_revision'] == chosen['revision']
    assert len([call for call in calls if call[0] == 'chain']) == 1


def test_single_generated_option_cannot_resolve_uncertain_acoustics(unified):
    client, runtime, _ = unified
    verifier(runtime)
    upload(client)
    job = ready(client)
    assert job['question'] and job['auto_speak_revision'] is None
    assert not job.get('selected_candidate_id')
    assert post(client, job, 'confirm').status_code == 409
    selected = post(client, job, 'choose', candidate_id='m1').json()
    assert selected['text'] == 'I need water.' and not selected['question']


def test_rejected_selected_reading_does_not_speak_valid_rival(unified):
    client, runtime, _ = unified
    verifier(runtime, 'h2')
    upload(client)
    job = ready(client)
    assert job['auto_speak_revision'] is None and job['question']
    assert all(c.get('literal_fallback') for c in job['candidates'])


def test_missing_artifact_is_advisory_not_legacy_confidence(unified):
    client, runtime, _ = unified
    verifier(runtime, status='unavailable')
    upload(client)
    job = ready(client)
    assert job['ranking']['route'] == 'verified'
    assert job['auto_speak_revision'] is None and job['question']


def test_remote_route_keeps_legacy_behavior_without_fabricated_scores(unified):
    client, runtime, _ = unified
    verifier(runtime)
    runtime.asr.result = runtime.asr.result.model_copy(update={'backend': 'pod'})
    upload(client)
    job = ready(client)
    assert job['ranking']['route'] == 'legacy' and job['ranking']['scores'] == []
    assert job['auto_speak_revision'] == job['revision']


def test_all_five_raw_fallbacks_survive_provider_failure(unified):
    client, runtime, _ = unified
    runtime.asr.result = raw('tea', 'toast', 'soup', 'water', 'coffee')
    runtime.message_chain.result = composed(message(text='tea', literal='tea'), available=False)
    verifier(runtime)
    upload(client)
    job = ready(client)
    assert job['auto_speak_revision'] is None
    assert {h for c in job['candidates'] for h in c['source_hypothesis_ids']} == {f'h{i}' for i in range(1, 6)}


def test_autoplay_disabled_does_not_change_machine_selection(unified):
    client, runtime, _ = unified
    verifier(runtime, 'h1')
    runtime.settings.speech_autoplay = False
    upload(client)
    job = ready(client)
    assert job['ranking']['decision'] == 'selected'
    assert job['selected_candidate_id'] == 'm1' and job['auto_speak_revision'] is None
    chosen = post(client, job, 'choose', candidate_id='raw-h1').json()
    assert chosen['auto_speak_revision'] == chosen['revision']


@pytest.mark.parametrize('output,accepted', [('Please bring water.', False), ('Maya, please bring water.', True)])
def test_normalized_reading_cannot_erase_original_protected_name(unified, monkeypatch, output, accepted):
    from communication.backend import recognition
    client, runtime, _ = unified
    monkeypatch.setattr(recognition.wording_fidelity, 'terms', lambda snapshot: ['Maya'])
    runtime.asr.result = raw('Maya bring water', 'Maya bring tea')
    runtime.message_chain.result = composed(message('m1', output, 'h1', 'maya bring water'))
    verifier(runtime, 'h1')
    upload(client)
    job = ready(client)
    assert bool(job['auto_speak_revision']) is accepted
    if not accepted:
        assert job['error']['code'] == 'wording_fidelity'
        assert job['text'] == 'Maya bring water'


def test_explicit_redraft_keeps_fifth_literal_selectable(unified, monkeypatch):
    from communication.backend import providers
    client, runtime, calls = unified
    runtime.asr.result = raw('tea', 'toast', 'soup', 'water', 'coffee')
    runtime.message_chain.result = composed(message('m4', 'I need water.', 'h4', 'water'))
    verifier(runtime, 'h4')
    async def draft(*args):
        return {'text': 'Please bring water.', 'question': '', 'options': []}, {'model': 'test'}
    monkeypatch.setattr(providers, 'draft', draft)
    upload(client)
    job = ready(client)
    assert post(client, job, 'draft').status_code == 200
    rewritten = ready(client)
    assert rewritten['selected_candidate_id'].startswith('draft-')
    assert {item['id'] for item in rewritten['candidates']} >= {f'raw-h{i}' for i in range(1, 6)}
    chosen = post(client, rewritten, 'choose', candidate_id='raw-h5').json()
    assert chosen['text'] == 'coffee' and chosen['auto_speak_revision'] == chosen['revision']


@pytest.mark.parametrize('action,body', [
    ('context', {'context': {'scenario': 'outside'}}),
    ('conversation', {'action': 'forget'}),
    ('draft', {'context': {'scenario': 'outside'}}),
])
def test_context_change_revokes_machine_choice_and_keeps_raw_options(unified, action, body):
    client, runtime, calls = unified
    runtime.asr.result = raw('water', 'tea', 'coffee', 'soup', 'toast')
    verifier(runtime, 'h1')
    upload(client)
    job = ready(client)
    changed = client.post(f"/api/messages/{job['id']}/{action}",
                          json={'revision': job['revision'], **body}, headers={'X-Echora-Client': '1'}).json()
    assert changed['ranking']['decision'] == 'ambiguous'
    assert changed['question'] and changed['auto_speak_revision'] is None
    assert {item['id'] for item in changed['candidates']} == {f'raw-h{i}' for i in range(1, 6)}
    assert post(client, changed, 'confirm').status_code == 409
    assert post(client, changed, 'draft').status_code == 409
    chosen = post(client, changed, 'choose', candidate_id='raw-h5').json()
    assert chosen['text'] == 'toast' and chosen['auto_speak_revision'] == chosen['revision']
    assert len([call for call in calls if call[0] == 'chain']) == 1


def test_server_gate_does_not_infer_resolution_from_an_empty_question(unified):
    from communication.backend import app as shared
    client, runtime, _ = unified
    verifier(runtime)
    upload(client)
    ready(client)
    job = next(iter(shared.sessions.values())).job
    job['question'] = ''
    shared.mark_for_speech(job)
    assert job['auto_speak_revision'] is None
    assert post(client, job, 'confirm').status_code == 409


def test_manual_edit_keeps_literal_choices_without_claiming_acoustic_support_or_speaking(unified, monkeypatch):
    from communication.backend import context, memory, providers
    client, runtime, calls = unified
    profile = context.save_profile(context.Profile(label='Speaker', language='English'))
    runtime.asr.result = raw('water', 'tea', 'coffee', 'soup', 'toast')
    verifier(runtime, 'h1')
    upload(client, context=json.dumps({'profile_id': profile['id'], 'profile_revision': profile['revision']}))
    original = ready(client)
    edited = post(client, original, 'edit', text='Please call Maya at five.').json()
    authored = next(item for item in edited['candidates'] if item['id'] == edited['selected_candidate_id'])
    assert authored['based_on_user_edit'] and authored['text'] == edited['text']
    assert authored['source_hypothesis_ids'] == authored['source_literals'] == []
    assert edited['evidence'] == original['evidence'] and edited['ranking'] == original['ranking']
    assert edited['auto_speak_revision'] is None and edited['confirmed'] is None
    assert {item['id'] for item in edited['candidates']} >= {f'raw-h{i}' for i in range(1, 6)}
    assert memory.list_memories(profile['id'])['memories'] == []
    remembered = post(client, edited, 'remember', memory_revision=0)
    assert remembered.status_code == 200
    assert remembered.json()['memory']['message'] == edited['text']
    assert remembered.json()['memory']['source']['source_hypothesis_ids'] == []
    chosen = post(client, edited, 'choose', candidate_id='raw-h5').json()
    assert chosen['text'] == 'toast' and chosen['auto_speak_revision'] == chosen['revision']
    assert len([call for call in calls if call[0] == 'chain']) == 1

    draft_inputs = []
    async def draft(text, *args):
        draft_inputs.append(text)
        return {'text': 'Maya, please call me at five.', 'question': '', 'options': []}, {'model': 'test'}
    monkeypatch.setattr(providers, 'draft', draft)
    edited_again = post(client, chosen, 'edit', text='Maya call me at five.').json()
    assert post(client, edited_again, 'draft').status_code == 200
    rewritten = ready(client)
    assert draft_inputs == ['Maya call me at five.']
    assert {item['id'] for item in rewritten['candidates']} >= {f'raw-h{i}' for i in range(1, 6)}
