"""Shared lifecycle checks for explicit memory, native access and deletion races."""
import asyncio
import copy
import json
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.native import router as native_router
from app.schemas import AudioQuality, Hypothesis, RawAsrResult
from communication.backend import app as shared, context, conversation, memory, recognition
from communication.backend.profile_adapter import resolve_snapshot

HEADERS = {'X-Echora-Client': '1'}


@pytest.fixture(autouse=True)
def lifecycle_store(tmp_path, monkeypatch):
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path / 'profiles.sqlite3'))
    monkeypatch.setenv('GROQ_API_KEY', 'offline-test-only')
    monkeypatch.setattr(shared, 'calls_used', 0)
    shared.sessions.clear()
    old = memory._invalidator
    memory.set_invalidator(shared.invalidate_memory_context)
    yield
    shared.sessions.clear()
    memory.set_invalidator(old)


def new_profile(label='Speaker'):
    return context.save_profile(context.Profile(label=label, language='English'))


def frozen(profile):
    return resolve_snapshot(context.Selection(profile_id=profile['id'], profile_revision=profile['revision'], scenario='home').model_dump(), profile)


def seed(profile):
    return memory.remember({'id': 'saved-job', 'revision': 1, 'status': 'review', 'text': 'Please bring water.',
        'context': frozen(profile), 'language': 'en', 'output_language': 'English', 'candidates': [],
        'evidence': {'recognizer': 'adapted', 'model': 'test'}}, 0)


def dependencies(profile):
    source = memory.source_snapshot(frozen(profile))
    return {'profile_id': profile['id'], 'profile_revision': profile['revision'], 'memory_revision': source['memory_revision'],
            'dependencies': [{key: entry[key] for key in ('source_id', 'source_revision', 'content_hash')} for entry in source['entries']],
            'candidates': {}}


def install_audio_job(session, profile, retrieved=None):
    j = shared.begin(session, 'water', 'audio')
    j.update(context=frozen(profile), original='water', source_text='water', text='Please bring water.',
             evidence={'kind': 'audio', 'recognizer': 'adapted', 'backend': 'local', 'model': 'test',
                       'hypotheses': [{'id': 'h1', 'literal_text': 'water'}]},
             candidates=[{'id': 'm1', 'text': 'Please bring water.', 'reading': 'water', 'available': True,
                          'source_hypothesis_ids': ['h1'], 'source_literals': ['water']}],
             selected_candidate_id='m1', retrieval=retrieved,
             ranking={'route': 'verified', 'decision': 'selected', 'selected_hypothesis_id': 'h1'})
    shared.mark_for_speech(j)
    return j


def result():
    return SimpleNamespace(warnings=[], ranking_seconds=0,
        ranker=SimpleNamespace(source='groq', reason='', assistant_model='test'))


def raw():
    return RawAsrResult(backend='local', device='test', decode_seconds=0,
        audio_quality=AudioQuality(seconds=1, peak_dbfs=-3, rms_dbfs=-10, clipped_samples=0, low_level_warning=False),
        hypotheses=[Hypothesis(id='h1', literal_text='water', sequence_score=-1, search_weight=1)])


def test_arrival_confirm_and_session_restore_never_store_automatically(monkeypatch):
    p = new_profile()
    runtime = SimpleNamespace(asr=object(), settings=SimpleNamespace(beams=5, max_audio_seconds=60, backend='local'))
    monkeypatch.setattr(shared.app.state, 'runtime', runtime, raising=False)

    async def transcribe(*args): return raw(), {'model': 'test', 'elapsed_ms': 1}
    async def rank(*args): return {'route': 'legacy', 'decision': 'ambiguous'}, None
    async def compose(*args, **kwargs):
        return [{'id': 'm1', 'text': 'Please bring water.', 'reading': 'water', 'available': True,
                 'source_hypothesis_ids': ['h1'], 'source_literals': ['water']}], result(), {}
    monkeypatch.setattr(recognition, 'transcribe', transcribe)
    monkeypatch.setattr(recognition, 'rank', rank)
    monkeypatch.setattr(recognition, 'compose', compose)
    with TestClient(shared.app) as client:
        client.get('/api/session')
        response = client.post('/api/audio', files={'file': ('sample.wav', b'test', 'audio/wav')},
            data={'context': json.dumps(frozen(p)['selection']), 'recognition_backend': 'adapted'}, headers=HEADERS)
        assert response.status_code == 200
        until = time.monotonic() + 2
        while time.monotonic() < until:
            j = client.get('/api/session').json()['job']
            if j['status'] == 'review': break
            time.sleep(.01)
        assert j['auto_speak_revision'] == j['revision']
        assert memory.list_memories(p['id'])['memories'] == []
        confirmed = client.post(f"/api/messages/{j['id']}/confirm", json={'revision': j['revision']}, headers=HEADERS).json()
        assert confirmed['confirmed']['text'] == 'Please bring water.'
        client.get('/api/session')
        assert memory.list_memories(p['id'])['memories'] == []
        before = copy.deepcopy(next(iter(shared.sessions.values())).job)
        saved = client.post(f"/api/messages/{j['id']}/remember", json={'revision': confirmed['revision'], 'memory_revision': 0}, headers=HEADERS)
        assert saved.status_code == 200
        assert next(iter(shared.sessions.values())).job == before
        assert memory.list_memories(p['id'])['memories'][0]['message'] == before['text']


def test_delete_revokes_actual_prepared_audio_and_session_references():
    p = new_profile()
    saved = seed(p)
    with TestClient(shared.app) as client:
        client.get('/api/session')
        s = next(iter(shared.sessions.values()))
        j = install_audio_job(s, p, dependencies(p))
        confirmation = client.post(f"/api/messages/{j['id']}/confirm", json={'revision': j['revision']}, headers=HEADERS).json()
        old_revision, old_confirmation = confirmation['revision'], confirmation['confirmed']['id']
        j['prepared_speech'] = {'revision': old_revision, 'display_text': j['text'], 'speech_text': j['text']}
        s.audio = {'key': ('cached',), 'data': b'old-audio'}
        s.memory_candidate = copy.deepcopy(s.memory)
        assert s.memory and s.memory_candidate
        deleted = client.post(f"/api/profiles/{p['id']}/memories/{saved['memory_id']}/delete", json={'memory_revision': 1}, headers=HEADERS)
        assert deleted.status_code == 200
        assert j['revision'] > old_revision and j['text'] == 'water'
        assert j['confirmed'] is None and j['prepared_speech'] is None and j['auto_speak_revision'] is None
        assert s.audio is None and s.memory is None and s.memory_candidate is None
        assert client.post(f"/api/messages/{j['id']}/audio", json={'revision': old_revision, 'confirmation_id': old_confirmation}, headers=HEADERS).status_code == 409


def test_deletion_defeats_a_provider_that_finishes_after_cancellation(monkeypatch):
    p = new_profile()
    saved = seed(p)
    lookup = dependencies(p)

    async def run():
        started = asyncio.Event()
        session = shared.Session()
        shared.sessions['late-compose'] = session
        j = install_audio_job(session, p)
        j.update(status='transcribing', confirmed=None, auto_speak_revision=None)
        async def transcribe(*args): return raw(), {'model': 'test', 'elapsed_ms': 1}
        async def rank(*args): return {'route': 'verified', 'decision': 'selected', 'selected_hypothesis_id': 'h1'}, lookup
        async def delayed(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # A transport can return buffered data despite cancellation.
                return [{'id': 'late', 'text': 'Please bring water.', 'available': True,
                         'source_hypothesis_ids': ['h1'], 'source_literals': ['water']}], result(), {}
        monkeypatch.setattr(recognition, 'transcribe', transcribe)
        monkeypatch.setattr(recognition, 'rank', rank)
        monkeypatch.setattr(recognition, 'compose', delayed)
        session.task = asyncio.create_task(shared.adapted_job(session, j['id'], b'test', 'sample.wav', SimpleNamespace()))
        await started.wait()
        memory.delete_memories(p['id'], 1, saved['memory_id'])
        await session.task
        assert j['status'] == 'review' and j['text'] == 'water'
        assert [item['id'] for item in j['candidates']] == ['raw-h1']
        assert not j['confirmed'] and not j['auto_speak_revision']
        assert all(event['type'] != 'draft_ready' for event in session.events)
    asyncio.run(run())


def test_native_wrappers_isolate_speakers_persist_only_memories_and_check_owner():
    p, other = new_profile(), new_profile('Other speaker')
    application = FastAPI()
    application.include_router(native_router)
    with TestClient(application) as client:
        token = client.get('/api/v1/communication/session').json()['session_token']
        headers = {**HEADERS, 'X-Echora-Session': token}
        s = shared.sessions[token]
        j = install_audio_job(s, p)
        endpoint = f"/api/v1/communication/messages/{j['id']}/remember"
        stranger = client.get('/api/v1/communication/session').json()['session_token']
        assert client.post(endpoint, headers={**HEADERS, 'X-Echora-Session': stranger}, json={'revision': j['revision'], 'memory_revision': 0}).status_code == 404
        saved = client.post(endpoint, headers=headers, json={'revision': j['revision'], 'memory_revision': 0})
        assert saved.status_code == 200
        assert client.get(f"/api/v1/communication/profiles/{other['id']}/memories", headers=headers).json()['memories'] == []
        shared.sessions.clear()  # same lifecycle as process/session restart; disk is retained
        token = client.get('/api/v1/communication/session').json()['session_token']
        headers = {**HEADERS, 'X-Echora-Session': token}
        assert client.get('/api/v1/communication/state', headers=headers).json()['job'] is None
        remembered = client.get(f"/api/v1/communication/profiles/{p['id']}/memories", headers=headers).json()
        assert remembered['memories'][0]['id'] == saved.json()['memory_id']
        assert shared.sessions[token].memory is None


def test_cross_process_delete_during_redraft_cannot_publish_stale_wording(monkeypatch):
    p = new_profile()
    saved = seed(p)

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        session = shared.Session()
        shared.sessions['late-redraft'] = session
        j = install_audio_job(session, p, dependencies(p))
        j['status'] = 'drafting'
        async def delayed(*args):
            started.set()
            await release.wait()
            return {'text': 'Please bring the old water.', 'question': '', 'options': []}, {'model': 'test'}
        monkeypatch.setattr(shared.providers, 'draft', delayed)
        session.task = asyncio.create_task(shared.draft_job(session, j['id']))
        await started.wait()
        memory.set_invalidator(None)
        memory.delete_memories(p['id'], 1, saved['memory_id'])
        memory.set_invalidator(shared.invalidate_memory_context)
        release.set()
        # The dependency check revokes and cancels this same in-flight task.
        await asyncio.gather(session.task, return_exceptions=True)
        assert j['text'] == 'water' and j['status'] == 'review'
        assert j['confirmed'] is None and j['auto_speak_revision'] is None
        assert [item['id'] for item in j['candidates']] == ['raw-h1']
    asyncio.run(run())


def test_profile_scope_edit_revokes_pending_audio_without_deleting_memories():
    p = new_profile()
    seed(p)
    session = shared.Session()
    shared.sessions['profile-edit'] = session
    j = install_audio_job(session, p, dependencies(p))
    j['confirmed'] = {'id': 'old', 'context': frozen(p), 'text': j['text']}
    session.memory = conversation.remember(j)
    j['prepared_speech'] = {'speech_text': j['text']}
    session.audio = {'data': b'old'}
    updated = context.Profile.model_validate(p)
    updated.listener_by_setting = {'home': 'unfamiliar'}
    context.save_profile(updated)
    assert j['confirmed'] is None and j['prepared_speech'] is None and session.audio is None
    assert j['text'] == 'water' and j['question']
    assert session.memory is None
    assert len(memory.list_memories(p['id'])['memories']) == 1


def test_profile_edit_detaches_stale_snapshot_and_user_words_can_speak():
    p = new_profile()
    with TestClient(shared.app) as client:
        client.get('/api/session')
        session = next(iter(shared.sessions.values()))
        j = install_audio_job(session, p)
        revised = context.Profile.model_validate(p)
        revised.label = 'Updated speaker'
        context.save_profile(revised)
        assert j['context']['profile'] is None
        assert j['context']['selection']['profile_id'] == ''
        edited = client.post(f"/api/messages/{j['id']}/edit", json={
            'revision': j['revision'], 'text': 'Please bring water.'}, headers=HEADERS)
        assert edited.status_code == 200
        confirmed = client.post(f"/api/messages/{j['id']}/confirm", json={
            'revision': edited.json()['revision']}, headers=HEADERS)
        assert confirmed.status_code == 200
        assert confirmed.json()['confirmed']['text'] == 'Please bring water.'


def test_cross_process_delete_is_caught_before_remember_or_audio():
    p = new_profile()
    saved = seed(p)
    session = shared.Session()
    shared.sessions['cross-process'] = session
    j = install_audio_job(session, p, dependencies(p))
    j['confirmed'] = {'id': 'old', 'text': j['text'], 'speech_text': j['text']}
    memory.set_invalidator(None)  # another backend process cannot call this process's callback
    memory.delete_memories(p['id'], 1, saved['memory_id'])
    memory.set_invalidator(shared.invalidate_memory_context)
    with pytest.raises(Exception) as save_error:
        memory.remember(j, 2)
    assert save_error.value.status_code == 409
    with pytest.raises(Exception) as audio_error:
        shared.confirmed_audio(session, j['id'], shared.AudioInput(revision=j['revision'], confirmation_id='old'))
    assert audio_error.value.status_code == 409
    assert j['confirmed'] is None and j['auto_speak_revision'] is None


def test_cross_process_profile_delete_guards_typed_path_without_retrieval():
    p = new_profile()
    session = shared.Session()
    shared.sessions['typed-profile-deleted'] = session
    j = shared.begin(session, 'Please bring water.', 'text')
    j.update(context=frozen(p), confirmed={'id': 'old', 'text': j['text']})
    memory.set_invalidator(None)
    memory.delete_profile(p['id'], p['revision'], 0)
    memory.set_invalidator(shared.invalidate_memory_context)
    with pytest.raises(Exception) as error:
        shared.confirmed_audio(session, j['id'], shared.AudioInput(revision=j['revision'], confirmation_id='old'))
    assert error.value.status_code == 409
    assert j['confirmed'] is None and j['context']['profile'] is None
    assert j['text'] == 'Please bring water.'  # independently typed wording remains available


def test_clear_revokes_followup_reference_and_deferred_arrival_without_vector_hits():
    p = new_profile()
    seed(p)
    session = shared.Session()
    shared.sessions['pending-followup'] = session
    j = shared.begin(session, 'without sugar', 'audio')
    j.update(context=frozen(p), text='Please bring water without sugar.',
             conversation={'job_id': 'saved-job', 'text': 'Please bring water.'},
             _speech_when_ready=True, _explicit_speech_when_ready=True, _defer_auto_speak=True,
             auto_speak_revision=j['revision'])
    memory.delete_memories(p['id'], 1)
    assert j['text'] == 'without sugar' and j['conversation'] is None
    assert j['auto_speak_revision'] is None
    assert not any(key in j for key in ('_speech_when_ready', '_explicit_speech_when_ready', '_defer_auto_speak'))
    shared.finish_optional_analysis(j)
    assert j['auto_speak_revision'] is None


def test_profile_edit_revokes_typed_generated_wording_but_keeps_literal():
    p = new_profile()
    session = shared.Session()
    shared.sessions['typed-generated'] = session
    j = shared.begin(session, 'tea', 'text')
    j.update(context=frozen(p), text='Please bring green tea.', metadata={'wording': {'model': 'test'}},
             auto_speak_revision=j['revision'], confirmed={'id': 'old', 'text': 'Please bring green tea.'})
    updated = context.Profile.model_validate(p)
    updated.listener_by_setting = {'home': 'unfamiliar'}
    context.save_profile(updated)
    assert j['text'] == 'tea' and j['question']
    assert j['confirmed'] is None and j['auto_speak_revision'] is None
