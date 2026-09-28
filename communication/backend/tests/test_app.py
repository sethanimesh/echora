import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from communication.backend import app as module, providers

HEADERS={'X-Echora-Client':'1'}

@pytest.fixture
def client(monkeypatch):
    module.sessions.clear()
    monkeypatch.setattr(module,'calls_used',0)
    monkeypatch.setenv('GROQ_API_KEY','test-only-key')
    with TestClient(module.app) as c:
        c.get('/api/session')
        yield c

def create(c,text='Please do not close the window.'):
    r=c.post('/api/messages',json={'text':text},headers=HEADERS)
    assert r.status_code==200
    return r.json()

def post(c,j,action,**body):
    return c.post(f"/api/messages/{j['id']}/{action}",json=body,headers=HEADERS)

def test_confirm_version_and_edit_invalidation(client):
    j=create(client)
    confirmed=post(client,j,'confirm',revision=j['revision']).json()
    assert confirmed['confirmed']['text']=='Please do not close the window.'
    again=post(client,j,'confirm',revision=j['revision']).json()
    assert again['confirmed']['id']==confirmed['confirmed']['id']
    edited=post(client,j,'edit',revision=j['revision'],text='Please close the window.').json()
    assert edited['confirmed'] is None
    assert edited['original']=='Please do not close the window.'
    assert edited['source_text']=='Please close the window.'
    assert post(client,j,'confirm',revision=j['revision']).status_code==409

def test_cross_session_cannot_access_message(client):
    j=create(client)
    with TestClient(module.app) as stranger:
        stranger.get('/api/session')
        assert post(stranger,j,'confirm',revision=1).status_code==404

def test_cross_origin_and_missing_header_rejected(client):
    assert client.post('/api/messages',json={'text':'hello'}).status_code==403
    assert client.post('/api/messages',json={'text':'hello'},headers={**HEADERS,'Origin':'https://untrusted.example'}).status_code==403
    assert client.get('/api/session',headers={'Host':'untrusted.example'}).status_code==403

def test_cancel_prevents_confirmation(client):
    j=create(client)
    cancelled=post(client,j,'cancel').json()
    assert cancelled['status']=='cancelled'
    assert post(client,j,'confirm',revision=cancelled['revision']).status_code==409

def test_new_message_replaces_old(client):
    old=create(client)
    new=create(client,'I would like company.')
    assert new['id']!=old['id']
    assert post(client,old,'confirm',revision=1).status_code==404

def test_audio_upload_preserves_transcript_and_metadata(client,monkeypatch):
    async def fake(audio,filename,content_type,language):
        assert audio==b'example-audio'
        assert language=='auto'
        return 'My left leg hurts.',{'model':providers.ASR_MODEL,'elapsed_ms':25}
    monkeypatch.setattr(providers,'transcribe',fake)
    r=client.post('/api/audio',files={'file':('sample.wav',b'example-audio','audio/wav')},data={'language':'auto'},headers=HEADERS)
    assert r.status_code==200
    j=client.get('/api/session').json()['job']
    assert j['original']=='My left leg hurts.'
    assert j['confirmed'] is None
    assert j['metadata']['transcription']['model']==providers.ASR_MODEL

def test_audio_failure_is_explicit_no_fallback(client,monkeypatch):
    async def fake(*args):raise providers.ProviderFailure('rate_limit','Groq limit reached.')
    monkeypatch.setattr(providers,'transcribe',fake)
    client.post('/api/audio',files={'file':('sample.wav',b'example-audio','audio/wav')},headers=HEADERS)
    j=client.get('/api/session').json()['job']
    assert j['status']=='review'
    assert j['error']['code']=='rate_limit'
    assert j['original']==''
    assert module.calls_used==1

def test_upload_rejects_invalid_and_large_inputs(client):
    assert client.post('/api/audio',files={'file':('a.txt',b'abc')},headers=HEADERS).status_code==422
    assert client.post('/api/audio',files={'file':('a.wav',b'x'*(module.MAX_AUDIO+1))},headers=HEADERS).status_code==413
    assert module.calls_used==0

def test_request_cap_prevents_provider_call(client,monkeypatch):
    monkeypatch.setattr(module,'calls_used',module.CALL_LIMIT)
    j=create(client)
    assert post(client,j,'draft',revision=1).status_code==429
    assert client.get('/api/session').json()['job']['status']=='review'

def test_ambiguity_requires_choice_and_keeps_evidence(client,monkeypatch):
    async def fake(*args):return {'text':'leg… arm…','question':'Which part hurts?','options':['My leg hurts.','My arm hurts.']},{'model':providers.TEXT_MODEL,'elapsed_ms':5}
    monkeypatch.setattr(providers,'draft',fake)
    j=create(client,'leg… arm…')
    post(client,j,'draft',revision=1)
    j=client.get('/api/session').json()['job']
    assert j['question']
    assert post(client,j,'confirm',revision=j['revision']).status_code==409
    selected=post(client,j,'edit',revision=j['revision'],text='My arm hurts.').json()
    assert selected['original']=='leg… arm…'
    assert post(client,selected,'confirm',revision=selected['revision']).json()['confirmed']['text']=='My arm hurts.'

def test_cancel_pending_inference_cannot_restore_draft(client,monkeypatch):
    async def delayed(*args):
        await asyncio.sleep(5)
        return {'text':'late','question':'','options':[]},{}
    monkeypatch.setattr(providers,'draft',delayed)
    j=create(client)
    processing=post(client,j,'draft',revision=1).json()
    assert processing['status']=='drafting'
    assert post(client,processing,'confirm',revision=processing['revision']).status_code==409
    post(client,processing,'cancel')
    assert client.get('/api/session').json()['job']['status']=='cancelled'

def test_sse_snapshot_replay_and_gap():
    async def run():
        s=module.Session()
        module.sessions['test-stream']=s
        module.begin(s,'Hello.','text')
        class Request:
            cookies={'echora_session':'test-stream'}
            headers={}
            async def is_disconnected(self):return False
        r=Request()
        response=await module.events(r)
        packet=await anext(response.body_iterator)
        assert '"type": "snapshot"' in packet
        await response.body_iterator.aclose()
        r.headers={'last-event-id':str(s.sequence)}
        s.job['status']='confirmed';s.emit('message_confirmed')
        response=await module.events(r)
        packet=await anext(response.body_iterator)
        assert '"type": "message_confirmed"' in packet
        assert 'event: state' in packet
        await response.body_iterator.aclose()
        for _ in range(70):s.emit('message_edited')
        r.headers={'last-event-id':'1'}
        response=await module.events(r)
        packet=await anext(response.body_iterator)
        assert '"type": "snapshot"' in packet
        await response.body_iterator.aclose()
        module.sessions.clear()
    asyncio.run(run())


def test_late_draft_failure_cannot_overwrite_edit_after_stop(monkeypatch):
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        async def draft(*args):
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                await finish.wait()
            raise providers.ProviderFailure('late_failure', 'Late failure')
        monkeypatch.setattr(providers, 'draft', draft)
        s = module.Session()
        module.sessions['late-draft'] = s
        class Request:
            cookies = {'echora_session': 'late-draft'}
        request = Request()
        job = module.begin(s, 'tea', 'text')
        job['status'] = 'drafting'
        s.task = asyncio.create_task(module.draft_job(s, job['id']))
        await started.wait()
        await module.cancel(job['id'], request)
        await module.edit(job['id'], module.EditInput(revision=job['revision'], text='No tea.'), request)
        expected = dict(job)
        finish.set()
        await s.task
        assert job == expected and job['error'] is None
        module.sessions.pop('late-draft', None)
    asyncio.run(run())
