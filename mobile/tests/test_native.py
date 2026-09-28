"""Native transport uses the web session decisions; no models or network."""
from types import SimpleNamespace
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.native import router
from app.schemas import AudioQuality, Hypothesis, RawAsrResult
from communication.backend import app as shared, context, conversation


@pytest.fixture
def client(monkeypatch, tmp_path):
    shared.sessions.clear()
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path / 'profiles.sqlite3'))
    application = FastAPI()
    application.include_router(router)
    async def speech(text): return None
    application.state.speech = SimpleNamespace(synthesize=speech)
    with TestClient(application, base_url='http://192.0.2.10:8000') as client:
        yield client
    shared.sessions.clear()


def headers(client):
    result = client.get('/api/v1/communication/session')
    assert result.status_code == 200
    assert result.headers['cache-control'] == 'no-store'
    assert result.json()['reference_ttl_seconds'] == 600
    return {'X-Echora-Session': result.json()['session_token'], 'X-Echora-Client': '1'}


def message(client, identity, text='Please bring tea.'):
    result = client.post('/api/v1/communication/messages', headers=identity, json={'text': text})
    assert result.status_code == 200
    return result.json()


def test_native_boundaries_and_session_ownership(client):
    one, other = headers(client), headers(client)
    assert one['X-Echora-Session'] != other['X-Echora-Session']
    assert client.get('/api/v1/communication/session', headers={'Origin': 'https://untrusted.example'}).status_code == 403
    assert client.post('/api/v1/communication/messages', headers={'X-Echora-Session': one['X-Echora-Session']}, json={'text': 'tea'}).status_code == 403
    job = message(client, one)
    endpoint = f"/api/v1/communication/messages/{job['id']}/confirm"
    assert client.post(endpoint, headers=other, json={'revision': job['revision']}).status_code == 404
    assert client.post(endpoint, headers=one, json={'revision': job['revision'] + 1}).status_code == 409


def test_speech_requires_exact_confirmation_and_remembers_only_that_revision(client):
    identity = headers(client)
    job = message(client, identity)
    path = f"/api/v1/communication/messages/{job['id']}"
    assert client.post(path + '/speech', headers=identity, json={'revision': job['revision'], 'confirmation_id': 'not-confirmed'}).status_code == 409
    approved = client.post(path + '/confirm', headers=identity, json={'revision': job['revision']}).json()
    spoken = client.post(path + '/speech', headers=identity, json={'revision': approved['revision'], 'confirmation_id': approved['confirmed']['id']})
    assert spoken.status_code == 200
    assert spoken.json()['speech_text'] == 'Please bring tea.'
    assert spoken.json()['speech'] is None
    session = shared.sessions[identity['X-Echora-Session']]
    assert session.memory['text'] == approved['text']
    edited = client.post(path + '/edit', headers=identity, json={'revision': approved['revision'], 'text': 'Please bring coffee.'}).json()
    assert edited['evidence']['hypotheses'][0]['literal_text'] == 'Please bring tea.'
    assert edited['confirmed'] is None
    assert client.post(path + '/speech', headers=identity, json={'revision': approved['revision'], 'confirmation_id': approved['confirmed']['id']}).status_code == 409


def test_native_explicit_raw_choice_authorizes_exact_evidence_revision(client):
    identity = headers(client)
    job = message(client, identity, 'tea')
    active = shared.sessions[identity['X-Echora-Session']].job
    active.update(candidates=[{'id': 'raw-h1', 'text': 'tea', 'available': False}],
        error={'code': 'wording_unavailable', 'message': 'Wording unavailable.'})
    assert active['auto_speak_revision'] is None
    path = f"/api/v1/communication/messages/{job['id']}"
    chosen = client.post(path + '/choose', headers=identity,
        json={'revision': job['revision'], 'candidate_id': 'raw-h1'}).json()
    assert chosen['text'] == 'tea'
    assert chosen['evidence'] == job['evidence']
    assert chosen['auto_speak_revision'] == chosen['revision']
    assert chosen['error'] is None
    approved = client.post(path + '/confirm', headers=identity, json={'revision': chosen['revision']}).json()
    assert approved['revision'] == chosen['revision']
    spoken = client.post(path + '/speech', headers=identity,
        json={'revision': approved['revision'], 'confirmation_id': approved['confirmed']['id']}).json()
    assert spoken['speech_text'] == 'tea'


def test_full_length_device_fallback_and_ten_minute_reference(client):
    identity = headers(client)
    text = 'Please bring tea. ' * 20
    job = message(client, identity, text)
    path = f"/api/v1/communication/messages/{job['id']}"
    approved = client.post(path + '/confirm', headers=identity, json={'revision': job['revision']}).json()
    spoken = client.post(path + '/speech', headers=identity, json={'revision': approved['revision'], 'confirmation_id': approved['confirmed']['id']}).json()
    assert spoken['speech_text'] == text.strip()
    session = shared.sessions[identity['X-Echora-Session']]
    message(client, identity, 'without sugar')
    assert conversation.reference(session.memory_candidate, 'without sugar', None)['text'] == text.strip()
    session.memory_candidate['at'] -= 601
    assert conversation.reference(session.memory_candidate, 'without sugar', None) is None


def test_cancel_during_synthesis_rejects_late_voice(client):
    identity = headers(client)
    job = message(client, identity)
    path = f"/api/v1/communication/messages/{job['id']}"
    approved = client.post(path + '/confirm', headers=identity, json={'revision': job['revision']}).json()
    async def delayed(text):
        active = shared.sessions[identity['X-Echora-Session']].job
        active['revision'] += 1
        active['confirmed'] = None
        return None
    client.app.state.speech = SimpleNamespace(synthesize=delayed)
    assert client.post(path + '/speech', headers=identity, json={'revision': approved['revision'], 'confirmation_id': approved['confirmed']['id']}).status_code == 409


def test_native_profile_edit_retains_b_rules_moments_and_revision(client):
    identity = headers(client)
    source = context.Profile(label='Unified person', rules=[context.Rule(anchor='soup', wording='tomato soup', scenarios=['home'], mode='ask')],
        moments=[context.Moment(id='rest', title='Rest', cue='rest', message='I need a break.')])
    saved = context.save_profile(source)
    path = f"/api/v1/communication/profiles/{saved['id']}"
    legacy = client.get(path, headers=identity).json()
    changed = {key: value for key, value in legacy.items() if key not in {'id', 'revision', 'icon', 'baseline', 'speaker_note', 'created_at'}}
    changed['label'] = 'Updated from native'
    response = client.put(path, headers=identity, json={'revision': 1, 'profile': changed})
    assert response.status_code == 200
    current = context.get_profile(saved['id'])
    assert current.revision == 2 and current.rules == source.rules and current.moments == source.moments
    assert client.put(path, headers=identity, json={'revision': 1, 'profile': changed}).status_code == 409


def test_native_audio_uses_shared_job_and_freezes_selection(client, monkeypatch):
    identity = headers(client)
    async def fake_audio(request, **kwargs):
        assert kwargs['recognition_backend'] == 'adapted'
        assert kwargs['output_language'] == 'original'
        selection = context.Selection.model_validate_json(kwargs['context'])
        session = shared.session(request)
        job = shared.begin(session, 'tea', 'audio')
        job['context'] = context.snapshot(selection)
        return job
    monkeypatch.setattr(shared, 'audio_message', fake_audio)
    selection = context.Selection(scenario='outside', core_context='outdoors', declared_listener=None).model_dump_json()
    result = client.post('/api/v1/communication/audio', headers=identity, files={'file': ('speech.wav', b'fake', 'audio/wav')}, data={'selection': selection})
    assert result.status_code == 200
    assert result.json()['context']['core_context'] == 'outdoors'
    assert result.json()['context']['selection']['declared_listener'] is None
    assert client.get('/api/v1/communication/state', headers=identity).json()['job']['id'] == result.json()['id']


def test_native_audio_uses_profile_language_and_rejects_stale_revision(client, monkeypatch):
    identity = headers(client)
    profile = context.save_profile(context.Profile(label='Shared speaker', language='Hindi/Hinglish'))
    seen = []
    async def fake_audio(request, **kwargs):
        seen.append(kwargs['output_language'])
        return shared.begin(shared.session(request), 'chai', 'audio')
    monkeypatch.setattr(shared, 'audio_message', fake_audio)
    selection = context.Selection(profile_id=profile['id'], profile_revision=profile['revision'])
    endpoint = '/api/v1/communication/audio'
    files = {'file': ('speech.wav', b'fake', 'audio/wav')}
    response = client.post(endpoint, headers=identity, files=files, data={'selection': selection.model_dump_json()})
    assert response.status_code == 200
    assert seen == ['Hindi/Hinglish']
    context.save_profile(context.Profile.model_validate({**profile, 'language': 'English'}))
    response = client.post(endpoint, headers=identity, files=files, data={'selection': selection.model_dump_json()})
    assert response.status_code == 409
    assert seen == ['Hindi/Hinglish']


def test_native_audio_rejects_unreadable_selection(client, monkeypatch):
    identity = headers(client)
    async def unexpected_audio(*args, **kwargs):
        pytest.fail('Invalid selection must not start recognition')
    monkeypatch.setattr(shared, 'audio_message', unexpected_audio)
    response = client.post('/api/v1/communication/audio', headers=identity,
        files={'file': ('speech.wav', b'fake', 'audio/wav')}, data={'selection': 'not-json'})
    assert response.status_code == 422


@pytest.mark.parametrize('next_context,expired,linked', [('home', False, True), ('home', True, False), ('outdoors', False, False)])
def test_native_recordings_apply_only_current_scoped_followup(client, monkeypatch, next_context, expired, linked):
    identity = headers(client)
    monkeypatch.setattr(shared.app.state, 'runtime', SimpleNamespace(asr=object()), raising=False)
    references = []
    async def transcribe(runtime, audio, filename):
        raw = RawAsrResult(backend='local', device='test', decode_seconds=0,
            audio_quality=AudioQuality(seconds=1, peak_dbfs=None, rms_dbfs=None, clipped_samples=0, low_level_warning=False),
            hypotheses=[Hypothesis(id='h1', literal_text=audio.decode(), sequence_score=0, search_weight=1)])
        return raw, {'language': 'en'}
    async def compose(runtime, raw, job, reference, **kwargs):
        references.append(reference)
        text = 'Please bring tea without sugar.' if reference else 'Please bring tea.'
        literal = raw.hypotheses[0].literal_text
        return [{'id': 'm1', 'text': text, 'reading': literal, 'available': True,
                 'source_hypothesis_ids': ['h1'], 'source_literals': [literal],
                 'specializations': [], 'plain_text': None}], SimpleNamespace(warnings=[], ranking_seconds=0,
            ranker=SimpleNamespace(source='fake', assistant_model='fake')), {}
    monkeypatch.setattr(shared.recognition, 'transcribe', transcribe)
    monkeypatch.setattr(shared.recognition, 'compose', compose)

    def upload(text, setting):
        selected = context.Selection(scenario='outside' if setting == 'outdoors' else setting, core_context=setting)
        response = client.post('/api/v1/communication/audio', headers=identity,
            files={'file': ('speech.wav', text.encode(), 'audio/wav')}, data={'selection': selected.model_dump_json()})
        assert response.status_code == 200
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            job = client.get('/api/v1/communication/state', headers=identity).json()['job']
            if job['status'] not in shared.BUSY:
                return job
            time.sleep(.01)
        pytest.fail('Native recording did not become ready')

    first = upload('tea', 'home')
    assert first['auto_speak_revision'] == first['revision']
    approved = client.post(f"/api/v1/communication/messages/{first['id']}/confirm", headers=identity,
        json={'revision': first['revision']})
    assert approved.status_code == 200
    if expired:
        shared.sessions[identity['X-Echora-Session']].memory['at'] -= 601
    second = upload('without sugar', next_context)
    assert second['evidence']['hypotheses'][0]['literal_text'] == 'without sugar'
    if linked:
        assert second['conversation']['text'] == first['text']
        assert references[-1]['text'] == first['text']
        assert second['auto_speak_revision'] == second['revision']
    else:
        assert second['conversation'] is None
        assert second['question']
        assert second['auto_speak_revision'] is None
        assert len(references) == 1
