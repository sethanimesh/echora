import asyncio
import concurrent.futures
import time
import httpx
import pytest
from test_app import client, create, post, HEADERS
from communication.backend import app as module, fish, providers


@pytest.fixture(autouse=True)
def config(monkeypatch):
    monkeypatch.setenv('FISH_API_KEY', 'fish-test-key')
    monkeypatch.setenv('FISH_REFERENCE_ID', 'test-voice')
    monkeypatch.setattr(module, 'fish_calls_used', 0)


def ready(client, text='Please bring me water.', tone='warm'):
    j = create(client, text)
    return post(client, j, 'confirm', revision=1, delivery={'tone': tone, 'rate': 0.65}).json()


def audio(client, j):
    return post(client, j, 'audio', revision=j['revision'], confirmation_id=j['confirmed']['id'])


def test_blank_optional_file_does_not_mask_root_configuration(monkeypatch, tmp_path):
    local, root = tmp_path/'local.env', tmp_path/'root.env'
    local.write_text('FISH_API_KEY=\nFISH_REFERENCE_ID=chosen-voice\n')
    root.write_text('FISH_API_KEY=root-test-key\nFISH_REFERENCE_ID=root-voice\n')
    monkeypatch.setenv('FISH_API_KEY', '')
    monkeypatch.setenv('FISH_REFERENCE_ID', '')
    fish.load_configuration([local, root])
    assert fish.configuration() == ('root-test-key', 'chosen-voice')
    monkeypatch.setenv('FISH_API_KEY', 'process-test-key')
    fish.load_configuration([local, root])
    assert fish.configuration()[0] == 'process-test-key'


def test_confirmation_required_and_cache_bound_to_delivery(client, monkeypatch):
    seen = []
    async def synth(body):
        seen.append(body)
        return b'ID3-audio', {'model': fish.MODEL, 'elapsed_ms': 1}
    monkeypatch.setattr(fish, 'synthesize', synth)
    draft = create(client)
    assert post(client, draft, 'audio', revision=1, confirmation_id='unconfirmed').status_code == 409
    j = ready(client)
    first = audio(client, j)
    assert first.status_code == 200 and first.content == b'ID3-audio'
    assert first.headers['x-confirmation-id'] == j['confirmed']['id']
    assert first.headers['cache-control'] == 'no-store'
    assert audio(client, j).content == first.content
    assert len(seen) == 1
    changed = post(client, j, 'confirm', revision=j['revision'], delivery={'tone':'firm', 'rate':1.2}).json()
    assert audio(client, j).status_code == 409
    assert audio(client, changed).status_code == 200
    assert len(seen) == 2
    assert seen[0]['text'] == '[warm and friendly] Please bring me water.'
    assert seen[1]['prosody']['speed'] == 1.2
    edited = post(client, changed, 'edit', revision=1, text='Please bring tea.').json()
    assert edited['confirmed'] is None and audio(client, changed).status_code == 409
    events = next(iter(module.sessions.values())).events
    assert [e['type'] for e in events if e['type'] in {'synthesis_started','audio_ready'}] == ['synthesis_started','audio_ready']*2


def test_missing_key_voice_and_text_controls_never_call_provider(client, monkeypatch):
    async def forbidden(body):
        raise AssertionError('No network call allowed')
    monkeypatch.setattr(fish, 'synthesize', forbidden)
    j = ready(client)
    monkeypatch.delenv('FISH_API_KEY')
    assert audio(client, j).status_code == 503
    assert client.get('/api/session').json()['fish']['configured'] is False
    monkeypatch.setenv('FISH_API_KEY', 'fish-test-key')
    monkeypatch.setenv('FISH_REFERENCE_ID', '')
    assert audio(client, j).status_code == 503
    monkeypatch.setenv('FISH_REFERENCE_ID', 'test-voice')
    controlled = ready(client, '[laughing] Please do not call Meena.')
    assert audio(client, controlled).status_code == 503
    assert module.fish_calls_used == 0
    assert client.get('/api/session').json()['job']['text'] == '[laughing] Please do not call Meena.'


def test_failure_keeps_confirmed_words_and_does_not_retry(client, monkeypatch):
    calls = []
    async def synth(body):
        calls.append(body)
        raise providers.ProviderFailure('fish_unavailable', 'Fish Audio is unavailable.')
    monkeypatch.setattr(fish, 'synthesize', synth)
    j = ready(client)
    assert audio(client, j).status_code == 503
    saved = client.get('/api/session').json()['job']
    assert saved['status'] == 'confirmed' and saved['synthesis']['state'] == 'error'
    assert saved['confirmed'] == j['confirmed'] and len(calls) == 1
    monkeypatch.setattr(module, 'fish_calls_used', module.FISH_CALL_LIMIT)
    assert audio(client, j).status_code == 429 and len(calls) == 1


def test_stop_cancels_generation_and_preserves_message(client, monkeypatch):
    async def synth(body):
        await asyncio.sleep(10)
        raise AssertionError('Cancelled work should not finish')
    monkeypatch.setattr(fish, 'synthesize', synth)
    j = ready(client)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        request = pool.submit(audio, client, j)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if client.get('/api/session').json()['job']['status'] == 'synthesizing':
                break
            time.sleep(.01)
        assert audio(client, j).status_code == 409  # no duplicate generation
        stopped = post(client, j, 'audio/stop', revision=1, confirmation_id=j['confirmed']['id']).json()
        assert stopped['status'] == 'confirmed' and stopped['synthesis']['state'] == 'stopped'
        assert stopped['text'] == j['text']
        assert request.result(timeout=2).status_code == 409
        assert next(iter(module.sessions.values())).audio is None


def test_late_provider_failure_cannot_replace_stopped_synthesis(monkeypatch):
    async def run():
        started, finish = asyncio.Event(), asyncio.Event()
        async def synth(body):
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                await finish.wait()
            raise providers.ProviderFailure('late_fish', 'Late voice failure')
        monkeypatch.setattr(fish, 'synthesize', synth)
        s = module.Session()
        module.sessions['late-fish'] = s
        class Request:
            cookies = {'echora_session': 'late-fish'}
        job = module.begin(s, 'tea', 'text')
        job.update(status='synthesizing', confirmed={'id': 'approved'},
                   synthesis={'state': 'generating', 'confirmation_id': 'approved'})
        s.task = asyncio.create_task(module.synthesize_job(s, job['id'], 'approved', {}, ('approved',)))
        await started.wait()
        await module.stop_audio(job['id'], module.AudioInput(revision=job['revision'], confirmation_id='approved'), Request())
        finish.set()
        await s.task
        assert job['synthesis']['state'] == 'stopped' and job['status'] == 'confirmed'
        assert s.audio is None
        module.sessions.pop('late-fish', None)
    asyncio.run(run())


@pytest.mark.parametrize('status, content, content_type, fails', [
    (200, b'ID3-test-mp3', 'audio/mpeg', False),
    (200, b'{"error":"private upstream message"}', 'application/json', True),
    (200, b'not an mp3', 'audio/mpeg', True),
    (401, b'private upstream message', 'application/json', True),
    (402, b'private upstream message', 'application/json', True),
    (429, b'private upstream message', 'application/json', True),
    (503, b'private upstream message', 'application/json', True),
])
def test_adapter_pins_free_model_and_never_echoes_upstream_errors(monkeypatch, status, content, content_type, fails):
    calls = []
    def handle(request):
        calls.append(request)
        assert request.headers['model'] == 's2.1-pro-free'
        assert str(request.url) == fish.ENDPOINT
        return httpx.Response(status, content=content, headers={'Content-Type':content_type})
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    if fails:
        with pytest.raises(providers.ProviderFailure) as error:
            asyncio.run(fish.synthesize({'text':'hello'}))
        assert 'private upstream message' not in error.value.message
    else:
        data, meta = asyncio.run(fish.synthesize({'text':'hello'}))
        assert data == content and meta['model'] == fish.MODEL
    assert len(calls) == 1
