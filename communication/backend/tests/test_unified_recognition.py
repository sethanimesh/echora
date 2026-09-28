"""Unified contracts, using fake recognition/model services and no hardware."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from communication.backend import app as module, recognition, providers
from app.schemas import AudioQuality, CommunicationRegister, Hypothesis, MessageCandidate, RawAsrResult

HEADERS = {'X-Echora-Client': '1'}


def raw(*texts):
    return RawAsrResult(backend='local', device='test', decode_seconds=.01,
        audio_quality=AudioQuality(seconds=1, peak_dbfs=-3, rms_dbfs=-10, clipped_samples=0, low_level_warning=False),
        hypotheses=[Hypothesis(id=f'h{i}', literal_text=text, sequence_score=-i,
                               search_weight=1 / len(texts)) for i, text in enumerate(texts, 1)])


def message(ident='m1', text='I need water.', source='h1', literal='water'):
    return MessageCandidate(message_id=ident, hypothesis_id=source, source_hypothesis_ids=[source],
        source_literals=[literal], literal_text=literal, interpreted_intent=literal,
        corrected_text=text, repair_status='corrected', repair_note='test')


def composed(*messages, available=True):
    return SimpleNamespace(messages=list(messages), warnings=[] if available else ['Unavailable'], ranking_seconds=.01,
        ranker=SimpleNamespace(source='groq' if available else 'unavailable', reason='Unavailable', assistant_model='test-model'))


@pytest.fixture
def unified(monkeypatch, tmp_path):
    module.sessions.clear()
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path / 'profiles.sqlite3'))
    monkeypatch.setattr(module, 'calls_used', 0)
    calls = []
    recognizer = SimpleNamespace(result=raw('water', 'warmer'))
    chain = SimpleNamespace(result=composed(message()))
    async def transcribe(waveform, beams):
        calls.append(('asr', waveform, beams))
        return recognizer.result
    async def run(hypotheses, context, **kwargs):
        calls.append(('chain', hypotheses, context, kwargs))
        return chain.result
    async def decode(data, filename, max_seconds):
        return b'decoded'
    recognizer.transcribe = transcribe
    chain.run = run
    runtime = SimpleNamespace(asr=recognizer, message_chain=chain,
                              settings=SimpleNamespace(beams=5, max_audio_seconds=60, backend='local'))
    monkeypatch.setattr(module.app.state, 'runtime', runtime, raising=False)
    monkeypatch.setattr(recognition, 'decode', decode)
    monkeypatch.setattr(module.personal, 'recognition_context', lambda snapshot, hypotheses, runtime, **kwargs: {
        'context': ((snapshot or {}).get('selection') or {}).get('scenario', 'general'),
        'listener': 'familiar', 'register': CommunicationRegister(), 'brief': None,
    }, raising=False)
    with TestClient(module.app) as client:
        client.get('/api/session')
        yield client, runtime, calls


def upload(client, **fields):
    return client.post('/api/audio', files={'file': ('clip.wav', b'fake-recording', 'audio/wav')},
                       data=fields, headers=HEADERS)


def ready(client):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        job = client.get('/api/session').json()['job']
        if job['status'] not in module.BUSY: return job
        time.sleep(.01)
    raise AssertionError('Message did not become ready')


def post(client, job, action, **body):
    return client.post(f"/api/messages/{job['id']}/{action}", json={'revision': job['revision'], **body}, headers=HEADERS)


def test_adapted_single_candidate_preserves_all_evidence_and_composes_once(unified):
    client, runtime, calls = unified
    session = client.get('/api/session').json()
    assert session['recognition']['default_backend'] == 'adapted'
    assert session['recognition']['options'][0]['ready']
    assert upload(client, context=json.dumps({'scenario': 'home'})).status_code == 200
    job = ready(client)
    assert job['text'] == 'I need water.' and job['original'] == 'water'
    assert job['evidence']['hypotheses'] == [item.model_dump() for item in runtime.asr.result.hypotheses]
    assert job['candidates'][0]['source_hypothesis_ids'] == ['h1']
    assert job['auto_speak_revision'] == job['revision']
    assert job['confirmed'] is None
    assert next(iter(module.sessions.values())).memory is None
    assert [call[0] for call in calls] == ['asr', 'chain']
    assert calls[1][2] == 'home'
    assert job['context']['selection']['scenario'] == 'home'


def test_ambiguous_choice_authorizes_only_current_candidate_and_keeps_provenance(unified):
    client, runtime, calls = unified
    runtime.message_chain.result = composed(message(), message('m2', 'Make it warmer.', 'h2', 'warmer'))
    upload(client)
    job = ready(client)
    assert job['auto_speak_revision'] is None and job['question']
    chosen = post(client, job, 'choose', candidate_id='m2').json()
    assert chosen['text'] == 'Make it warmer.'
    assert chosen['auto_speak_revision'] == chosen['revision']
    assert chosen['evidence'] == job['evidence'] and chosen['original'] == 'water'
    assert chosen['candidates'][1]['source_hypothesis_ids'] == ['h2']
    assert post(client, job, 'choose', candidate_id='m1').status_code == 409


def test_raw_failure_alternatives_stay_silent_until_explicitly_selected(unified):
    client, runtime, calls = unified
    runtime.message_chain.result = composed(message(text='water'), available=False)
    upload(client)
    job = ready(client)
    assert job['error']['code'] == 'wording_unavailable' and job['auto_speak_revision'] is None
    chosen = post(client, job, 'choose', candidate_id='m1').json()
    assert chosen['auto_speak_revision'] == chosen['revision']
    assert chosen['text'] == 'water' and chosen['error'] is None
    assert chosen['warnings'] == job['warnings'] and chosen['evidence'] == job['evidence']


def test_chain_exception_retains_raw_words_and_does_not_speak(unified):
    client, runtime, calls = unified
    async def fail(*args, **kwargs): raise RuntimeError('failed')
    runtime.message_chain.run = fail
    upload(client)
    job = ready(client)
    assert job['original'] == job['text'] == 'water'
    assert len(job['evidence']['hypotheses']) == 2
    assert job['auto_speak_revision'] is None and job['error']


def test_explicit_whisper_has_no_invented_scores_or_adapted_call(unified, monkeypatch):
    client, runtime, calls = unified
    async def transcribe(*args): return 'water', {'model': providers.ASR_MODEL, 'elapsed_ms': 1}
    async def draft(*args):
        calls.append(('draft',))
        return {'text': 'Please bring me water.', 'question': '', 'options': []}, {'model': 'test', 'elapsed_ms': 1}
    monkeypatch.setattr(providers, 'transcribe', transcribe)
    monkeypatch.setattr(providers, 'draft', draft)
    upload(client, recognition_backend='whisper')
    job = ready(client)
    assert job['evidence']['backend'] == 'whisper'
    assert job['evidence']['hypotheses'] == [{'id': 'whisper-1', 'literal_text': 'water'}]
    assert job['auto_speak_revision'] == job['revision']
    assert calls == [('draft',)]


def test_unavailable_adapted_requires_explicit_provider_change(unified):
    client, runtime, calls = unified
    runtime.asr = None
    assert upload(client).status_code == 503
    assert calls == []


def test_edit_invalidates_auto_marker_but_preserves_immutable_audio(unified):
    client, runtime, calls = unified
    upload(client)
    job = ready(client)
    edited = post(client, job, 'edit', text='No water.').json()
    assert edited['auto_speak_revision'] is None
    authored = next(item for item in edited['candidates'] if item['id'] == edited['selected_candidate_id'])
    assert authored['text'] == 'No water.' and authored['source_hypothesis_ids'] == []
    assert {item['id'] for item in edited['candidates']} >= {'raw-h1', 'raw-h2'}
    assert edited['evidence'] == job['evidence'] and edited['original'] == 'water'


def test_typed_phrase_evidence_and_only_server_clarification_selection_autoplays(unified):
    client, runtime, calls = unified
    job = client.post('/api/messages', json={'text': 'Help.', 'kind': 'phrase'}, headers=HEADERS).json()
    assert job['evidence']['kind'] == 'phrase'
    assert job['evidence']['hypotheses'] == [{'id': 'input', 'literal_text': 'Help.'}]
    assert job['auto_speak_revision'] is None
    assert post(client, job, 'edit', text='I need help.', selection=True).status_code == 409
    server_job = next(iter(module.sessions.values())).job
    server_job.update(question='What help?', options=['Please open the door.', 'Please close the door.'])
    chosen = post(client, job, 'edit', text='Please open the door.', selection=True).json()
    assert chosen['auto_speak_revision'] == chosen['revision']


def test_confirm_authorizes_scoped_followup_but_arrival_does_not(unified):
    client, runtime, calls = unified
    runtime.asr.result = raw('tea with sugar')
    runtime.message_chain.result = composed(message(text='Please bring tea with sugar.', literal='tea with sugar'))
    upload(client)
    first = ready(client)
    assert post(client, first, 'confirm', pronunciation='original').status_code == 200
    runtime.asr.result = raw('without sugar')
    runtime.message_chain.result = composed(message(text='Please bring tea without sugar.', literal='without sugar'))
    upload(client)
    second = ready(client)
    assert second['conversation']['text'] == first['text']
    assert calls[-1][3]['conversation_reference']['text'] == first['text']


def test_cancellation_discards_delayed_recognition(unified):
    client, runtime, calls = unified
    async def delayed(*args):
        await asyncio.sleep(5)
        return raw('water')
    runtime.asr.transcribe = delayed
    running = upload(client).json()
    assert post(client, running, 'cancel').status_code == 200
    job = ready(client)
    assert job['status'] == 'cancelled' and job['auto_speak_revision'] is None
    assert job['original'] == ''


def test_optional_delivery_finishes_before_speech_marker(unified, monkeypatch):
    client, runtime, calls = unified
    async def suggest(audio):
        await asyncio.sleep(.05)
        return {'tone': 'warm', 'rate': .9}
    monkeypatch.setattr(module.delivery_cues, 'status', lambda: {'configured': True})
    monkeypatch.setattr(module.delivery_cues, 'suggest', suggest)
    monkeypatch.setattr(module, 'cue_calls_used', 0)
    upload(client, suggest_delivery='true')
    job = ready(client)
    assert job['delivery_suggestion']['state'] == 'ready'
    assert job['auto_speak_revision'] == job['revision']
    events = next(iter(module.sessions.values())).events
    assert all(event['job']['auto_speak_revision'] is None for event in events
               if event['type'] in {'draft_ready', 'delivery_suggestion_started'})


def test_ask_preference_remains_explicit_choice(unified, monkeypatch):
    client, runtime, calls = unified
    runtime.asr.result = raw('soup')
    runtime.message_chain.result = composed(message(text='I need soup.', literal='soup'))
    monkeypatch.setattr(module.personal, 'recognition_context', lambda *args, **kwargs: {
        'context': 'home', 'listener': 'familiar', 'register': CommunicationRegister(), 'brief': None,
        'requires_personal_choice': True,
        'personal_wording': [{'mode': 'ask', 'anchor': 'soup', 'wording': 'tomato soup'}],
    })
    upload(client)
    job = ready(client)
    assert job['question'] and job['auto_speak_revision'] is None
    assert job['options'] == ['I need tomato soup.', 'I need soup.']
    assert post(client, job, 'choose', candidate_id='m1').status_code == 409


def test_followup_scope_includes_named_place_and_audience():
    from communication.backend import conversation
    original = {'selection': {'scenario': 'home', 'place_id': 'home', 'audience_id': 'carer'},
                'core_context': 'home', 'resolved_audience': {'listener': 'familiar', 'style': {'brevity': 'short'}}}
    saved = {'text': 'Tea with sugar.', 'job_id': 'old', 'at': time.monotonic(), 'scope': conversation.scope(original)}
    assert conversation.reference(saved, 'without sugar', original)
    for change in ({'place_id': 'office'}, {'audience_id': 'friend'}, {'declared_listener': 'unfamiliar'}):
        changed = {**original, 'selection': {**original['selection'], **change}}
        assert conversation.reference(saved, 'without sugar', changed) is None


def test_exact_approved_moment_keeps_zero_model_call_route(unified):
    client, runtime, calls = unified
    runtime.asr.result = raw('usual')
    upload(client, context=json.dumps({'profile_id': 'sample-tea', 'profile_revision': 1,
                                      'scenario': 'cafe', 'moment_id': 'tea-stop'}), output_language='English')
    job = ready(client)
    assert job['text'] == 'Could I have Lipton green tea, please?'
    assert job['candidates'][0]['saved_message']
    assert job['auto_speak_revision'] == job['revision']
    assert [call[0] for call in calls] == ['asr']
    assert module.calls_used == 0


def test_alternative_names_are_protected_per_literal_reading(unified, monkeypatch):
    client, runtime, calls = unified
    runtime.asr.result = raw('Maya water', 'Kiran water')
    runtime.message_chain.result = composed(message(text='Maya, please bring water.', literal='Maya water'),
                                             message('m2', 'Kiran, please bring water.', 'h2', 'Kiran water'))
    monkeypatch.setattr(module.wording_fidelity, 'terms', lambda snapshot: ['Maya', 'Kiran'])
    upload(client)
    job = ready(client)
    assert len([item for item in job['candidates'] if item['available']]) == 2 and job['error'] is None
    assert {item['id'] for item in job['candidates']} >= {'raw-h1', 'raw-h2'}
    assert calls[-1][3]['communication_context']['fidelity']['protected_terms'] == ['Maya', 'Kiran']


def test_autoplay_setting_suppresses_arrival_but_not_explicit_choice(unified):
    client, runtime, calls = unified
    runtime.settings.speech_autoplay = False
    upload(client)
    job = ready(client)
    assert job['auto_speak_revision'] is None
    chosen = post(client, job, 'choose', candidate_id='m1').json()
    assert chosen['auto_speak_revision'] == chosen['revision']


@pytest.mark.parametrize('text, original, script, names, options', [
    ('मुझे चाय चाहिए।', 'chai', 'latin', [], ['मुझे चाय चाहिए।', 'Paani please.']),
    ('Mujhe chai chahiye.', 'chai', 'devanagari', [], ['Mujhe chai chahiye.', 'मुझे पानी चाहिए।']),
    ('maya', 'Maya or Kiran', 'latin', ['Maya', 'Kiran'], ['maya', 'Kiran']),
    ('माया, चाय दीजिए।', 'Maya chai', 'devanagari', ['Maya'], ['माया, चाय दीजिए।', 'Maya, पानी दीजिए।']),
])
def test_selected_question_option_must_pass_final_fidelity_before_speech(unified, monkeypatch, text, original, script, names, options):
    client, runtime, calls = unified
    job = client.post('/api/messages', json={'text': original}, headers=HEADERS).json()
    server_job = next(iter(module.sessions.values())).job
    server_job.update(question='Which message?', options=options, output_language='Hindi/Hinglish', output_script=script)
    monkeypatch.setattr(module.wording_fidelity, 'terms', lambda snapshot: names)
    response = post(client, job, 'edit', text=text, selection=True)
    assert response.status_code == 422
    assert server_job['revision'] == job['revision'] and server_job['question']
    assert server_job['text'] == original and server_job['auto_speak_revision'] is None


def test_selected_short_name_does_not_require_competing_recipient(unified, monkeypatch):
    client, runtime, calls = unified
    runtime.settings.speech_autoplay = False
    job = client.post('/api/messages', json={'text': 'Maya or Kiran'}, headers=HEADERS).json()
    server_job = next(iter(module.sessions.values())).job
    server_job.update(question='Who?', options=['Maya', 'Kiran'], output_language='Hindi/Hinglish', output_script='devanagari')
    monkeypatch.setattr(module.wording_fidelity, 'terms', lambda snapshot: ['Maya', 'Kiran'])
    response = post(client, job, 'edit', text='Maya', selection=True)
    assert response.status_code == 200
    chosen = response.json()
    assert chosen['text'] == 'Maya' and chosen['auto_speak_revision'] == chosen['revision']


def test_cancel_clears_deferred_speech_before_later_edit(unified):
    client, runtime, calls = unified
    job = client.post('/api/messages', json={'text': 'water'}, headers=HEADERS).json()
    server_job = next(iter(module.sessions.values())).job
    server_job.update(_speech_when_ready=True, _explicit_speech_when_ready=True, _defer_auto_speak=True)
    cancelled = post(client, job, 'cancel').json()
    assert not any(key in cancelled for key in ('_speech_when_ready', '_explicit_speech_when_ready', '_defer_auto_speak'))
