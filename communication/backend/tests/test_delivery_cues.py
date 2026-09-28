import asyncio
import hashlib
import io
import json
import time
import wave

import httpx
import pytest
from test_app import client, create, post, HEADERS
from communication.backend import app as module, delivery_cues as cues, providers


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-gemini-key')
    monkeypatch.setattr(module, 'cue_calls_used', 0)
    monkeypatch.setattr(cues, 'status', lambda: {'configured': True, 'model': cues.MODEL})


def ready(client):
    j = create(client)
    j = post(client, j, 'confirm', revision=1, delivery={'tone': 'warm', 'rate': .9}).json()
    session = next(iter(module.sessions.values()))
    session.job['modality'] = 'audio'
    session.source_audio_hash = hashlib.sha256(b'current-recording').hexdigest()
    return j


def suggest(client, j, audio=b'current-recording', revision=None):
    return client.post(f"/api/messages/{j['id']}/delivery-suggestion", headers=HEADERS,
                       data={'revision': revision or j['revision']}, files={'file': ('audio.webm', audio, 'audio/webm')})


def finish(client):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        j = client.get('/api/session').json()['job']
        if j['status'] != 'suggesting': return j
        time.sleep(.01)
    raise AssertionError('Suggestion never completed')


def test_opt_in_bound_recording_and_words_unchanged(client, monkeypatch):
    seen = []
    async def fake(audio):
        seen.append(audio)
        return {'tone': 'firm', 'cue': 'emphatic', 'pace': 'gentle', 'rate': .9, 'model': cues.MODEL}
    monkeypatch.setattr(cues, 'suggest', fake)
    j = ready(client)
    assert seen == []  # creating and confirming do not send audio
    assert suggest(client, j, b'old-recording').status_code == 409
    assert suggest(client, j, revision=99).status_code == 409
    assert module.cue_calls_used == 0
    assert suggest(client, j).status_code == 200
    saved = finish(client)
    assert saved['delivery_suggestion']['tone'] == 'firm'
    assert saved['delivery_suggestion']['rate'] == .9
    assert saved['confirmed'] == j['confirmed']  # no automatic acceptance or speech
    assert saved['text'] == j['text'] and saved['revision'] == j['revision']
    assert saved['prepared_speech'] == j['prepared_speech']
    assert seen == [b'current-recording']
    edited = post(client, saved, 'edit', revision=1, text='Different message.').json()
    assert edited['delivery_suggestion']['revision'] != edited['revision']
    assert suggest(client, saved).status_code == 409
    assert [e['type'] for e in next(iter(module.sessions.values())).events][-3:] == [
        'delivery_suggestion_started', 'delivery_suggestion_finished', 'message_edited']


def test_cancel_preserves_confirmation_and_drops_late_result(client, monkeypatch):
    async def slow(audio):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # A provider which finishes despite cancellation cannot resurrect a result.
            return {'tone': 'firm', 'cue': 'emphatic'}
    monkeypatch.setattr(cues, 'suggest', slow)
    j = ready(client)
    started = suggest(client, j).json()
    assert suggest(client, j).status_code == 409
    token = started['delivery_suggestion']['id']
    assert post(client, j, 'delivery-suggestion/stop', revision=1, suggestion_id='old').status_code == 409
    stopped = post(client, j, 'delivery-suggestion/stop', revision=1, suggestion_id=token).json()
    assert stopped['status'] == 'confirmed'
    assert stopped['confirmed'] == j['confirmed']
    assert finish(client)['delivery_suggestion']['state'] == 'stopped'


def test_replacement_does_not_receive_old_suggestion(client, monkeypatch):
    async def slow(audio):
        try: await asyncio.sleep(10)
        except asyncio.CancelledError: return {'tone': 'firm', 'cue': 'emphatic'}
    monkeypatch.setattr(cues, 'suggest', slow)
    j = ready(client)
    assert suggest(client, j).status_code == 200
    new = create(client, 'A fresh message.')
    assert finish(client)['id'] == new['id']
    assert 'delivery_suggestion' not in finish(client)
    assert next(iter(module.sessions.values())).source_audio_hash is None


def test_provider_failure_and_budget_never_retry(client, monkeypatch):
    calls = []
    async def fail(audio):
        calls.append(audio)
        raise providers.ProviderFailure('unavailable', 'Flash-Lite unavailable. No other model was used.')
    monkeypatch.setattr(cues, 'suggest', fail)
    j = ready(client)
    assert suggest(client, j).status_code == 200
    saved = finish(client)
    assert saved['delivery_suggestion']['state'] == 'error'
    assert saved['confirmed'] == j['confirmed'] and len(calls) == 1
    monkeypatch.setattr(module, 'cue_calls_used', cues.CALL_LIMIT)
    assert suggest(client, j).status_code == 429 and len(calls) == 1


@pytest.mark.parametrize('status, result, expected, rate', [
    (200, {'cue': 'soft', 'pace': 'gentle'}, 'warm', .9),
    (200, {'audio_status': 'no_speech', 'cue': 'soft', 'pace': 'gentle'}, None, None),
    (200, {'audio_status': 'overlapping', 'cue': 'bright', 'pace': 'faster'}, None, None),
    (200, {'audio_status': 'unclear', 'cue': 'steady', 'pace': 'standard'}, None, None),
    (200, {'cue': 'unclear', 'pace': 'faster'}, None, 1.2),
    (200, {'cue': 'steady', 'pace': 'unclear'}, 'neutral', None),
    (200, {'cue': 'bright', 'pace': 'standard'}, 'cheerful', 1.0),
    (200, {'cue': 'emphatic', 'pace': 'slower'}, 'firm', .65),
    (200, {'cue': 'soft', 'pace': 'extremely_fast'}, 'error', None),
    (200, {'cue': 'soft'}, 'error', None),
    (200, {'cue': 'unclear', 'pace': 'unclear'}, None, None),
    (200, {'cue': 'angry', 'pace': 'standard'}, 'error', None),
    (200, {'cue': 'soft', 'pace': 'gentle', 'diagnosis': 'private'}, 'error', None),
    (429, {'private': 'provider detail'}, 'error', None),
])
def test_fixed_adapter_validates_and_abstains(monkeypatch, status, result, expected, rate):
    async def normalize(audio): return b'wav-sample'
    monkeypatch.setattr(cues, 'normalize', normalize)
    monkeypatch.setattr(cues, 'has_audible_signal', lambda wav: True)
    requests = []
    def handle(request):
        requests.append(request)
        assert str(request.url) == cues.ENDPOINT and 'test-gemini-key' not in str(request.url)
        body = json.loads(request.content)
        assert body['generationConfig']['thinkingConfig'] == {'thinkingLevel': 'MINIMAL'}
        assert cues.MODEL == 'gemini-3.5-flash-lite'
        assert body['contents'][0]['parts'][0]['inlineData']['mimeType'] == 'audio/wav'
        return httpx.Response(status, json={'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': 'Non-JSON reasoning', 'thought': True}, {'text': json.dumps({'audio_status': 'single_speaker', 'observation': 'A test observation.', **result})}]}}]})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    if expected == 'error':
        with pytest.raises(providers.ProviderFailure) as exc:
            asyncio.run(cues.suggest(b'audio'))
        assert 'private' not in exc.value.message
    else:
        suggestion = asyncio.run(cues.suggest(b'audio'))
        assert suggestion['tone'] == expected and suggestion['rate'] == rate
        assert 'observation' not in suggestion
    assert len(requests) == 1


def wav(seconds):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b'\x00\x00' * (16000 * seconds))
    return buffer.getvalue()


def test_local_decoder_and_duration_limit():
    decoded = asyncio.run(cues.normalize(wav(1)))
    with wave.open(io.BytesIO(decoded)) as stream:
        assert stream.getframerate() == 16000 and stream.getnframes() == 16000
    with pytest.raises(providers.ProviderFailure):
        asyncio.run(cues.normalize(b'not audio'))
    with pytest.raises(providers.ProviderFailure, match='60 seconds'):
        asyncio.run(cues.normalize(wav(61)))


def test_silence_does_not_call_cloud(monkeypatch):
    def forbidden(**kwargs): raise AssertionError('Silence must not reach the cloud')
    monkeypatch.setattr(httpx, 'AsyncClient', forbidden)
    result = asyncio.run(cues.suggest(wav(3)))
    assert result['audio_status'] == 'no_speech'
    assert result['tone'] is None and result['rate'] is None
    assert result['model'] == 'Local silence check'


def wait_audio(client):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        j = client.get('/api/session').json()['job']
        if j['status'] not in {'transcribing', 'suggesting'}: return j
        time.sleep(.01)
    raise AssertionError('Combined job did not finish')


def submit_audio(client, enabled=True):
    return client.post('/api/audio', headers=HEADERS, data={'suggest_delivery': str(enabled).lower()},
                       files={'file': ('recording.wav', b'current-recording', 'audio/wav')})


def test_transcription_can_include_delivery_in_one_action(client, monkeypatch):
    calls = []
    async def transcribe(audio, *args):
        calls.append(('words', audio))
        return 'Please bring me water.', {'model': providers.ASR_MODEL}
    async def delivery(audio):
        calls.append(('delivery', audio))
        return {'tone': 'warm', 'rate': .9, 'cue': 'soft', 'pace': 'gentle'}
    monkeypatch.setattr(providers, 'transcribe', transcribe)
    monkeypatch.setattr(cues, 'suggest', delivery)
    assert submit_audio(client).status_code == 200
    j = wait_audio(client)
    assert j['text'] == j['original'] == 'Please bring me water.'
    assert j['confirmed'] is None and j['delivery_suggestion']['tone'] == 'warm'
    assert calls == [('words', b'current-recording'), ('delivery', b'current-recording')]
    assert module.calls_used == module.cue_calls_used == 1
    assert j['metadata']['transcription']['model'] == 'whisper-large-v3-turbo'
    assert [e['type'] for e in next(iter(module.sessions.values())).events][-3:] == [
        'transcription_finished', 'delivery_suggestion_started', 'delivery_suggestion_finished']


@pytest.mark.parametrize('problem', ['quota', 'provider', 'disabled', 'empty'])
def test_delivery_issue_never_discards_transcript(client, monkeypatch, problem):
    calls = []
    async def transcribe(audio, *args):
        return '' if problem == 'empty' else 'water', {'model': providers.ASR_MODEL}
    async def delivery(audio):
        calls.append(audio)
        raise providers.ProviderFailure('test', 'Voice analysis unavailable.')
    monkeypatch.setattr(providers, 'transcribe', transcribe)
    monkeypatch.setattr(cues, 'suggest', delivery)
    if problem == 'quota': monkeypatch.setattr(module, 'cue_calls_used', cues.CALL_LIMIT)
    assert submit_audio(client, problem != 'disabled').status_code == 200
    j = wait_audio(client)
    assert j['status'] == 'review'
    assert j['text'] == ('' if problem == 'empty' else 'water')
    if problem in {'quota', 'provider'}: assert j['delivery_suggestion']['state'] == 'error'
    else: assert 'delivery_suggestion' not in j
    assert len(calls) == (1 if problem == 'provider' else 0)


def test_cancelled_transcription_cannot_start_delivery(client, monkeypatch):
    async def transcribe(audio, *args):
        try: await asyncio.sleep(10)
        except asyncio.CancelledError: return 'late words', {'model': providers.ASR_MODEL}
    async def delivery(audio): raise AssertionError('Cancelled audio cannot start analysis')
    monkeypatch.setattr(providers, 'transcribe', transcribe)
    monkeypatch.setattr(cues, 'suggest', delivery)
    j = submit_audio(client).json()
    assert post(client, j, 'cancel').status_code == 200
    j = wait_audio(client)
    assert j['status'] == 'cancelled' and not j['text']
    assert module.cue_calls_used == 0
