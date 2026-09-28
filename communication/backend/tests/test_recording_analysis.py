import asyncio
import time
import pytest
from test_app import client, create, post, HEADERS
from communication.backend import app as module, providers, delivery_cues, face_cues
JPEG = b'\xff\xd8\xffcurrent-snapshot\xff\xd9'

@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','test-key')
    monkeypatch.setattr(module,'cue_calls_used',0)
    monkeypatch.setattr(module,'face_calls_used',0)
    monkeypatch.setattr(delivery_cues,'status',lambda:{'configured':True})
    async def transcribe(audio,*args):
        assert audio==b'current-audio'
        return 'Please bring water.',{'model':providers.ASR_MODEL}
    monkeypatch.setattr(providers,'transcribe',transcribe)


def submit(client, frames=True):
    files=[('file',('recording.wav',b'current-audio','audio/wav'))]
    if frames: files += [('frames',('snapshot.jpg',JPEG,'image/jpeg'))]*3
    return client.post('/api/audio',headers=HEADERS,data={'suggest_delivery':'true'},files=files)


def wait(client):
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        j=client.get('/api/session').json()['job']
        if j['status'] not in {'transcribing','analyzing_delivery','suggesting'}: return j
        time.sleep(.01)
    raise AssertionError('Analysis did not finish')


def test_one_submission_launches_independent_cue_calls_concurrently(client,monkeypatch):
    started=set()
    async def voice(audio):
        started.add('voice')
        while 'face' not in started: await asyncio.sleep(.001)
        assert audio==b'current-audio'
        return {'tone':'firm','rate':1.2}
    async def face(images):
        started.add('face')
        while 'voice' not in started: await asyncio.sleep(.001)
        assert images==[JPEG]*3
        return {'tone':'warm','cue':'smile'}
    monkeypatch.setattr(delivery_cues,'suggest',voice)
    monkeypatch.setattr(face_cues,'suggest',face)
    assert submit(client).status_code==200
    j=wait(client)
    assert started=={'voice','face'}
    assert j['text']==j['original']=='Please bring water.' and j['confirmed'] is None
    assert j['delivery_suggestion']['tone']=='firm' and j['face_suggestion']['tone']=='warm'
    assert j['delivery_suggestion']['id']==j['face_suggestion']['id']==j['analysis_id']
    assert module.cue_calls_used==module.face_calls_used==module.calls_used==1


@pytest.mark.parametrize('failed', ['voice','face'])
def test_individual_failure_keeps_other_cue_and_transcript(client,monkeypatch,failed):
    async def voice(audio):
        if failed=='voice': raise providers.ProviderFailure('test','Voice unavailable')
        return {'tone':'neutral','rate':1}
    async def face(images):
        if failed=='face': raise providers.ProviderFailure('test','Face unavailable')
        return {'tone':'warm'}
    monkeypatch.setattr(delivery_cues,'suggest',voice);monkeypatch.setattr(face_cues,'suggest',face)
    submit(client);j=wait(client)
    assert j['text']=='Please bring water.'
    assert j['delivery_suggestion']['state']==('error' if failed=='voice' else 'ready')
    assert j['face_suggestion']['state']==('error' if failed=='face' else 'ready')


def test_stop_all_checks_prevents_late_results(client,monkeypatch):
    async def slow(data):
        try: await asyncio.sleep(10)
        except asyncio.CancelledError: return {'tone':'warm'}
    monkeypatch.setattr(delivery_cues,'suggest',slow);monkeypatch.setattr(face_cues,'suggest',slow)
    submit(client)
    deadline=time.monotonic()+2
    while time.monotonic()<deadline:
        j=client.get('/api/session').json()['job']
        if j['status']=='analyzing_delivery': break
        time.sleep(.01)
    result=post(client,j,'recording-analysis/stop',revision=j['revision'],analysis_id=j['analysis_id']).json()
    assert result['text']=='Please bring water.'
    assert wait(client)['delivery_suggestion']['state']=='stopped'
    assert wait(client)['face_suggestion']['state']=='stopped'


def test_audio_only_upload_cannot_use_camera_frames_from_previous_message(client,monkeypatch):
    calls=[]
    async def voice(audio): return {'tone':'neutral','rate':1}
    async def face(images): calls.append(images);return {'tone':'warm'}
    monkeypatch.setattr(delivery_cues,'suggest',voice);monkeypatch.setattr(face_cues,'suggest',face)
    submit(client);first=wait(client)
    submit(client,False);second=wait(client)
    assert first['id']!=second['id'] and 'face_suggestion' not in second
    assert len(calls)==1


def test_face_limit_does_not_stop_voice_analysis(client,monkeypatch):
    monkeypatch.setattr(module,'face_calls_used',face_cues.CALL_LIMIT)
    async def voice(audio): return {'tone':'neutral','rate':1}
    async def forbidden(images): raise AssertionError('Over budget')
    monkeypatch.setattr(delivery_cues,'suggest',voice);monkeypatch.setattr(face_cues,'suggest',forbidden)
    submit(client);j=wait(client)
    assert j['delivery_suggestion']['state']=='ready' and j['face_suggestion']['state']=='error'
