import asyncio
import time
import pytest
from fastapi.testclient import TestClient
from communication.backend import app as module, hinglish, providers
from communication.backend.hinglish_core.classify.port import SpanDecision, SpanQuery
from communication.backend.hinglish_core.core.model import Label

HEADERS = {'X-Echora-Client': '1'}

@pytest.fixture
def client(monkeypatch, tmp_path):
    module.sessions.clear()
    monkeypatch.setattr(module, 'calls_used', 0)
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    monkeypatch.setenv('ECHORA_PROFILE_DB', str(tmp_path/'profiles.sqlite3'))
    with TestClient(module.app) as c:
        c.get('/api/session'); yield c

def ready(c):
    for _ in range(100):
        j = c.get('/api/session').json()['job']
        if j['status'] not in module.BUSY: return j
        time.sleep(.01)
    raise AssertionError('Preparation did not finish')

def create(c, text='Main ghar mein hoon.'):
    j = c.post('/api/messages', json={'text': text}, headers=HEADERS).json()
    return j, f"/api/messages/{j['id']}"

async def prepared(text, protected, charge):
    return {'display_text': text, 'pronunciation_text': 'मैं घर में हूँ.', 'speech_text': 'मैं घर में हूँ.', 'changes': [], 'metadata': {'model': providers.TEXT_MODEL, 'elapsed_ms': 5, 'ambiguous_spans': 1}}

def test_preparation_binds_separate_speech_and_edit_invalidates(client, monkeypatch):
    monkeypatch.setattr(hinglish, 'prepare', prepared)
    j, path = create(client)
    client.post(path+'/speech', json={'revision': 1}, headers=HEADERS)
    j = ready(client)
    assert j['text'] == j['original'] == 'Main ghar mein hoon.'
    assert j['confirmed'] is None
    confirmed = client.post(path+'/confirm', json={'revision': j['revision']}, headers=HEADERS).json()
    assert confirmed['confirmed']['text'] == 'Main ghar mein hoon.'
    assert confirmed['confirmed']['speech_text'] == 'मैं घर में हूँ.'
    edited = client.post(path+'/edit', json={'revision': j['revision'], 'text': 'Main office mein hoon.'}, headers=HEADERS).json()
    assert edited['prepared_speech'] is None and edited['confirmed'] is None
    assert client.post(path+'/confirm', json={'revision': j['revision']}, headers=HEADERS).status_code == 409

def test_context_change_and_reset_remove_prepared_speech(client, monkeypatch):
    monkeypatch.setattr(hinglish, 'prepare', prepared)
    j,path=create(client)
    client.post(path+'/speech',json={'revision':1},headers=HEADERS);j=ready(client)
    reset=client.post(path+'/speech/reset',json={'revision':j['revision']},headers=HEADERS).json()
    assert reset['prepared_speech'] is None and reset['text']==j['text']
    client.post(path+'/speech',json={'revision':reset['revision']},headers=HEADERS);j=ready(client)
    changed=client.post(path+'/context',json={'revision':j['revision'],'context':{'scenario':'home'}},headers=HEADERS).json()
    assert changed['prepared_speech'] is None

def test_failure_explicit_and_cancellation_blocks_late_output(client, monkeypatch):
    async def fail(*args): raise providers.ProviderFailure('rate_limit','Groq limit reached. No other model was used.')
    monkeypatch.setattr(hinglish,'prepare',fail)
    j,path=create(client)
    client.post(path+'/speech',json={'revision':1},headers=HEADERS);j=ready(client)
    assert j['error']['code']=='rate_limit' and j['prepared_speech'] is None
    async def delayed(*args): await asyncio.sleep(5); return await prepared(args[0],set(),lambda:None)
    monkeypatch.setattr(hinglish,'prepare',delayed)
    running=client.post(path+'/speech',json={'revision':j['revision']},headers=HEADERS).json()
    assert client.post(path+'/confirm',json={'revision':running['revision']},headers=HEADERS).status_code==409
    client.post(path+'/cancel',headers=HEADERS)
    assert client.get('/api/session').json()['job']['status']=='cancelled'

def test_mixed_script_preserves_names_english_and_urls(monkeypatch):
    async def classify(text, queries):
        result=[]
        for q in queries:
            reading={'main':'मैं','mein':'में','hoon':'हूँ'}.get(q.text.lower())
            index=q.candidates.index(reading) if reading in q.candidates else None
            result.append(SpanDecision(q.span_id,Label.HI if reading else Label.EN,candidate_index=index))
        return result, None
    monkeypatch.setattr(hinglish,'classify',classify)
    source='Main office mein hoon. Aman https://example.com/API'
    output=asyncio.run(hinglish.prepare(source,{'Aman'},lambda:None))
    assert output['display_text']==source
    assert 'मैं' in output['pronunciation_text'] and 'में' in output['pronunciation_text']
    assert 'office' in output['pronunciation_text'] and 'Aman' in output['pronunciation_text']
    assert 'https://example.com/API' in output['speech_text']

def test_already_devanagari_needs_no_cloud(monkeypatch):
    def forbidden(): raise AssertionError('Should not need inference')
    source='नमस्ते 😊 https://example.com/API'
    output=asyncio.run(hinglish.prepare(source,set(),forbidden))
    assert output['display_text']==source and output['speech_text']==source
    assert output['metadata']['ambiguous_spans']==0

def test_duplicate_or_missing_cloud_labels_are_explicit_errors(monkeypatch):
    async def invalid(path,**kwargs):
        assert kwargs['json']['model']=='openai/gpt-oss-120b'
        return {'choices':[{'finish_reason':'stop','message':{'content':'{"spans":[]}'}}]}
    monkeypatch.setattr(providers,'request',invalid)
    with pytest.raises(providers.ProviderFailure) as error:
        asyncio.run(hinglish.classify('main',[SpanQuery(0,'main',0,4)]))
    assert error.value.code=='hinglish_response'

def test_selected_profile_names_are_protected_before_drafting(client,monkeypatch):
    async def check(text,protected,charge):
        assert {'Meena','Arjun'} <= protected
        return await prepared(text,protected,charge)
    monkeypatch.setattr(hinglish,'prepare',check)
    j,path=create(client,'Meena kal aayegi.')
    r=client.post(path+'/speech',json={'revision':1,'context':{'profile_id':'sample-tea','profile_revision':1}},headers=HEADERS)
    assert r.status_code==200 and ready(client)['prepared_speech']


@pytest.mark.parametrize('language,question,should_prepare', [
    ('Hindi/Hinglish', '', True), ('English', '', False),
    ('Hindi/Hinglish', 'Which drink?', False),
])
def test_drafting_chains_preparation_only_after_resolved_hinglish(client, monkeypatch, language, question, should_prepare):
    seen = []
    async def draft(*args):
        return {'text': 'Mujhe chai chahiye.', 'question': question, 'options': []}, {'model': 'test', 'elapsed_ms': 0}
    async def prepare(text, protected, charge):
        seen.append(text)
        return await prepared(text, protected, charge)
    monkeypatch.setattr(providers, 'draft', draft)
    monkeypatch.setattr(hinglish, 'prepare', prepare)
    j, path = create(client, 'chai')
    assert client.post(path+'/draft', json={'revision': 1, 'output_language': language}, headers=HEADERS).status_code == 200
    j = ready(client)
    assert bool(j['prepared_speech']) == should_prepare
    assert seen == (['Mujhe chai chahiye.'] if should_prepare else [])
    assert j['confirmed'] is None and j['original'] == 'chai'
    if question: assert j['text'] == 'chai'
    events = next(iter(module.sessions.values())).events
    if should_prepare:
        assert not any(e['type'] == 'draft_ready' for e in events)


def test_direct_confirm_prepares_current_edit_and_reuses_on_repeat(client, monkeypatch):
    seen = []
    async def prepare(text, protected, charge):
        seen.append(text)
        return {**await prepared(text, protected, charge), 'speech_text': 'prepared: '+text}
    monkeypatch.setattr(hinglish, 'prepare', prepare)
    j, path = create(client)
    first = client.post(path+'/confirm', json={'revision': 1, 'output_language': 'Hindi/Hinglish'}, headers=HEADERS).json()
    assert first['confirmed']['text'] == j['text']
    assert first['confirmed']['speech_text'] == 'prepared: '+j['text']
    again = client.post(path+'/confirm', json={'revision': first['revision']}, headers=HEADERS).json()
    assert again['confirmed']['id'] == first['confirmed']['id'] and len(seen) == 1
    edited = client.post(path+'/edit', json={'revision': first['revision'], 'text': 'Mujhe paani chahiye.'}, headers=HEADERS).json()
    final = client.post(path+'/confirm', json={'revision': edited['revision']}, headers=HEADERS).json()
    assert seen[-1] == final['confirmed']['text'] == 'Mujhe paani chahiye.'
    assert final['confirmed']['speech_text'] == 'prepared: Mujhe paani chahiye.'
    assert len(seen) == 2


def test_auto_preparation_failure_never_confirms_and_original_is_explicit(client, monkeypatch):
    async def fail(*args): raise providers.ProviderFailure('rate_limit', 'Groq limit reached. No other model was used.')
    monkeypatch.setattr(hinglish, 'prepare', fail)
    j, path = create(client)
    response = client.post(path+'/confirm', json={'revision': 1, 'output_language': 'Hindi/Hinglish'}, headers=HEADERS)
    assert response.status_code == 503
    j = ready(client)
    assert j['confirmed'] is None and j['text'] == j['original']
    assert j['error']['stage'] == 'pronunciation'
    fallback = client.post(path+'/confirm', json={'revision': j['revision'], 'pronunciation': 'original'}, headers=HEADERS).json()
    assert fallback['confirmed']['speech_text'] == j['text']


def test_cancel_during_confirm_prevents_confirmation(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    async def delayed(*args):
        await asyncio.sleep(5)
        return await prepared(args[0], set(), lambda: None)
    monkeypatch.setattr(hinglish, 'prepare', delayed)
    j, path = create(client)
    with ThreadPoolExecutor(max_workers=1) as pool:
        confirmation = pool.submit(client.post, path+'/confirm', json={'revision': 1, 'output_language': 'Hindi/Hinglish'}, headers=HEADERS)
        for _ in range(100):
            if client.get('/api/session').json()['job']['status'] == 'preparing': break
            time.sleep(.01)
        else: raise AssertionError('Preparation did not start')
        assert client.post(path+'/cancel', headers=HEADERS).status_code == 200
        assert confirmation.result(timeout=2).status_code == 409
    j = client.get('/api/session').json()['job']
    assert j['status'] == 'cancelled' and j['confirmed'] is None
