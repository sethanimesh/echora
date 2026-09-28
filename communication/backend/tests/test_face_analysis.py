import asyncio
import time
import pytest
from communication.backend import face_analysis as faces, providers
from test_app import client, HEADERS


def test_local_mode_never_calls_gemini(monkeypatch):
    monkeypatch.setenv('ECHORA_FACE_MODE','local')
    async def local(frames): return {'tone':'warm','cue':'positive_expression'}
    async def forbidden(frames): raise AssertionError('Unexpected cloud request')
    monkeypatch.setattr(faces.local_faces,'suggest',local)
    monkeypatch.setattr(faces.gemini,'suggest',forbidden)
    assert asyncio.run(faces.suggest([b'image']*3))['tone']=='warm'


@pytest.mark.parametrize('failed',[None,'local','gemini'])
def test_paired_calls_share_identical_frames_are_concurrent_and_keep_declared_baseline(monkeypatch,failed):
    monkeypatch.setenv('ECHORA_FACE_MODE','compare');started=set();payloads=[]
    def provider(name,cue,tone):
        async def run(frames):
            started.add(name);payloads.append(list(frames))
            while len(started)<2: await asyncio.sleep(.001)
            if failed==name: raise providers.ProviderFailure('test',name+' failed')
            return {'visibility':'clear_face','cue':cue,'tone':tone}
        return run
    monkeypatch.setattr(faces.local_faces,'suggest',provider('local','positive_expression','warm'))
    monkeypatch.setattr(faces.gemini,'suggest',provider('gemini','neutral','neutral'))
    result=asyncio.run(faces.suggest([b'a',b'b',b'c']))
    assert payloads==[[b'a',b'b',b'c']]*2
    assert result['comparison']['selected']=='gemini'
    assert result['tone']==(None if failed=='gemini' else 'neutral')
    assert result['comparison']['agreement']==('different_style' if failed is None else 'not_comparable')


def test_positive_labels_compare_as_style_not_identical_emotion(monkeypatch):
    local={'state':'ready','visibility':'clear_face','cue':'positive_expression'}
    cloud={'state':'ready','visibility':'clear_face','cue':'broad_smile'}
    assert faces.agreement(local,cloud)=='same_style'
    assert faces.agreement({**local,'visibility':'no_face'},cloud)=='not_comparable'


def test_compare_status_requires_both_and_invalid_mode_is_explicit(monkeypatch):
    monkeypatch.setenv('ECHORA_FACE_MODE','compare')
    monkeypatch.setattr(faces.local_faces,'configured',lambda:True)
    monkeypatch.setattr(faces.gemini,'configured',lambda:True)
    assert faces.status()['configured'] and faces.status()['comparison']
    monkeypatch.setattr(faces.gemini,'configured',lambda:False)
    assert not faces.configured()
    monkeypatch.setenv('ECHORA_FACE_MODE','invalid')
    assert not faces.configured()
    with pytest.raises(providers.ProviderFailure): asyncio.run(faces.suggest([b'jpeg']*3))


def test_combined_upload_exposes_paired_report_without_images_in_session(client,monkeypatch):
    monkeypatch.setenv('ECHORA_FACE_MODE','compare')
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr(faces.local_faces,'configured',lambda:True)
    from communication.backend import app
    monkeypatch.setattr(app,'face_calls_used',0);monkeypatch.setattr(app,'cue_calls_used',0)
    async def transcribe(*args): return 'Water please.',{}
    async def local(frames): return {'visibility':'clear_face','cue':'positive_expression','tone':'warm','model':'local','elapsed_ms':20}
    async def cloud(frames): return {'visibility':'clear_face','cue':'smile','tone':'warm','model':'gemini','elapsed_ms':2000}
    monkeypatch.setattr(app.providers,'transcribe',transcribe)
    monkeypatch.setattr(faces.local_faces,'suggest',local);monkeypatch.setattr(faces.gemini,'suggest',cloud)
    files=[('file',('recording.wav',b'current-audio','audio/wav'))]+[('frames',('image.jpg',b'\xff\xd8\xffprivate-pixels\xff\xd9','image/jpeg'))]*3
    response=client.post('/api/audio',headers=HEADERS,files=files)
    assert response.status_code==200
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        response=client.get('/api/session');j=response.json()['job']
        if j['status']=='review': break
        time.sleep(.01)
    assert j['face_suggestion']['comparison']['agreement']=='same_style'
    assert 'private-pixels' not in response.text and j['text']=='Water please.'


def test_failed_warmup_disables_camera_mode_without_leaking_native_error(monkeypatch):
    monkeypatch.setenv('ECHORA_FACE_MODE','compare')
    monkeypatch.setattr(faces.local_faces,'startup_error',None)
    monkeypatch.setattr(faces.local_faces,'configured',lambda:faces.local_faces.startup_error is None)
    def broken(): raise ValueError('private native detail')
    monkeypatch.setattr(faces.local_faces,'warmup',broken)
    asyncio.run(faces.prepare())
    assert 'could not load' in faces.status()['message']
    assert 'private native detail' not in faces.status()['message']
    assert not faces.status()['configured']


def test_stopped_paired_analysis_cannot_publish_late_results(client,monkeypatch):
    from communication.backend import app
    from test_app import create,post
    monkeypatch.setenv('ECHORA_FACE_MODE','compare')
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr(faces.local_faces,'configured',lambda:True)
    monkeypatch.setattr(app,'face_calls_used',0)
    async def slow(frames):
        try: await asyncio.sleep(10)
        except asyncio.CancelledError: return {'visibility':'clear_face','cue':'smile','tone':'warm'}
    monkeypatch.setattr(faces.local_faces,'suggest',slow);monkeypatch.setattr(faces.gemini,'suggest',slow)
    j=create(client)
    r=client.post(f"/api/messages/{j['id']}/face-suggestion",headers=HEADERS,data={'revision':j['revision']},files=[('frames',('image.jpg',b'\xff\xd8\xffprivate\xff\xd9','image/jpeg'))]*3)
    r.raise_for_status()
    post(client,j,'face-suggestion/stop',revision=j['revision'],suggestion_id=r.json()['face_suggestion']['id']).raise_for_status()
    result=client.get('/api/session').json()['job']
    assert result['face_suggestion']['state']=='stopped' and 'comparison' not in result['face_suggestion']


def test_gemini_only_does_not_check_load_or_infer_with_local_models(monkeypatch):
    monkeypatch.setenv('ECHORA_FACE_MODE','gemini')
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr(faces.local_faces,'startup_error','Previous local error')
    def forbidden(*args): raise AssertionError('Local model touched in Gemini mode')
    monkeypatch.setattr(faces.local_faces,'configured',forbidden)
    monkeypatch.setattr(faces.local_faces,'warmup',forbidden)
    monkeypatch.setattr(faces.local_faces,'suggest',forbidden)
    async def cloud(frames): return {'cue':'smile','tone':'warm'}
    monkeypatch.setattr(faces.gemini,'suggest',cloud)
    assert faces.status()['configured'] and faces.status()['message'] is None
    assert not faces.status()['comparison']
    asyncio.run(faces.prepare())
    assert asyncio.run(faces.suggest([b'jpeg']*3))=={'cue':'smile','tone':'warm'}
