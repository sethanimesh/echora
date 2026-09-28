import asyncio
import json
import time
import httpx
import pytest
from test_app import client, create, post, HEADERS
from communication.backend import app as module, face_cues as faces, providers
JPEG = b'\xff\xd8\xfftest-image\xff\xd9'

@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    monkeypatch.setattr(module, 'face_calls_used', 0)


def submit(client, j, images=None, revision=None):
    return client.post(f"/api/messages/{j['id']}/face-suggestion", headers=HEADERS,
        data={'revision': revision or j['revision']},
        files=[('frames', ('snapshot.jpg', image, 'image/jpeg')) for image in (images if images is not None else [JPEG]*3)])


def finish(client):
    until = time.monotonic()+2
    while time.monotonic()<until:
        j = client.get('/api/session').json()['job']
        if j['status'] != 'checking_face': return j
        time.sleep(.01)
    raise AssertionError('Check did not finish')


@pytest.mark.parametrize('visibility,cues,tone', [
    (['clear_face']*3, ['smile','smile','unclear'], 'warm'),
    (['clear_face']*3, ['broad_smile']*3, 'cheerful'),
    (['clear_face']*3, ['neutral']*3, 'neutral'),
    (['clear_face']*3, ['neutral','smile','unclear'], None),
    (['clear_face','multiple_faces','clear_face'], ['smile']*3, None),
    (['no_face']*3, ['smile']*3, None),
    (['obscured']*3, ['neutral']*3, None),
])
def test_visible_cues_require_clear_views_and_consistency(visibility, cues, tone):
    result = faces.combine(faces.FaceOutput(frames=[{'visibility': v, 'cue': c} for v,c in zip(visibility,cues)]))
    assert result['tone'] == tone and 'rate' not in result


def test_explicit_camera_action_preserves_voice_words_and_confirmation(client, monkeypatch):
    calls=[]
    async def fake(images):
        calls.append(list(images))
        return {'cue': 'smile', 'visibility': 'clear_face', 'tone': 'warm', 'model': faces.MODEL}
    monkeypatch.setattr(faces, 'suggest', fake)
    j=create(client)
    j=post(client,j,'confirm',revision=1,delivery={'tone':'firm','rate':1.2}).json()
    s=next(iter(module.sessions.values()))
    s.job['delivery_suggestion']={'state':'ready','tone':'firm','rate':1.2,'revision':1}
    assert calls == []
    assert submit(client,j).status_code == 200
    result=finish(client)
    assert result['confirmed'] == j['confirmed'] and result['text']==j['text']
    assert result['delivery_suggestion']['tone']=='firm'
    assert result['face_suggestion']['tone']=='warm'
    assert 'rate' not in result['face_suggestion']
    assert calls == [[JPEG]*3]
    assert 'test-image' not in json.dumps(list(s.events))
    edited=post(client,result,'edit',revision=1,text='New message.').json()
    assert edited['face_suggestion']['revision'] != edited['revision']
    assert submit(client,j).status_code == 409


@pytest.mark.parametrize('images,status', [([JPEG]*2,422),([JPEG]*4,422),([b'bad']*3,422),([b'x'*(faces.MAX_FRAME+1)]*3,413)])
def test_bad_uploads_never_start_inference(client,images,status):
    assert submit(client,create(client),images).status_code==status
    assert module.face_calls_used==0


def test_missing_key_and_limit_do_not_affect_message(client,monkeypatch):
    j=create(client)
    monkeypatch.setenv('GEMINI_API_KEY','')
    assert submit(client,j).status_code==503
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr(module,'face_calls_used',faces.CALL_LIMIT)
    assert submit(client,j).status_code==429
    assert client.get('/api/session').json()['job']['text']==j['text']


def test_stop_and_replacement_discard_late_camera_results(client,monkeypatch):
    async def slow(images):
        try: await asyncio.sleep(10)
        except asyncio.CancelledError: return {'tone':'warm','cue':'smile'}
    monkeypatch.setattr(faces,'suggest',slow)
    j=create(client)
    started=submit(client,j).json()
    assert submit(client,j).status_code==409
    stopped=post(client,j,'face-suggestion/stop',revision=1,suggestion_id=started['face_suggestion']['id']).json()
    assert stopped['text']==j['text'] and finish(client)['face_suggestion']['state']=='stopped'
    assert submit(client,j).status_code==200
    new=create(client,'Another message.')
    assert finish(client)['id']==new['id'] and 'face_suggestion' not in finish(client)


@pytest.mark.parametrize('status,output,valid', [
    (200, {'frames':[{'visibility':'clear_face','cue':'smile'}]*3},True),
    (200, {'frames':[{'visibility':'clear_face','cue':'angry'}]*3},False),
    (200, {'frames':[{'visibility':'clear_face','cue':'smile'}]*2},False),
    (429, {'private':'provider detail'},False),
])
def test_adapter_fixed_model_and_no_retries(monkeypatch,status,output,valid):
    calls=[]
    def handler(request):
        calls.append(request)
        assert str(request.url)==faces.ENDPOINT and faces.MODEL=='gemini-3.5-flash-lite'
        assert json.loads(request.content)['generationConfig']['thinkingConfig'] == {'thinkingLevel':'MINIMAL'}
        parts=json.loads(request.content)['contents'][0]['parts']
        assert len([p for p in parts if 'inlineData' in p])==3
        return httpx.Response(status,json={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'Non-JSON reasoning','thought':True},{'text':json.dumps(output)}]}}]})
    original=httpx.AsyncClient
    monkeypatch.setattr(httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    if valid: assert asyncio.run(faces.suggest([JPEG]*3))['tone']=='warm'
    else:
        with pytest.raises(providers.ProviderFailure) as exc: asyncio.run(faces.suggest([JPEG]*3))
        assert 'provider detail' not in exc.value.message
    assert len(calls)==1
