"""Stop/new-work cannot overlap two callers inside one Qwen model."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from app.asr.backends import LocalAsrBackend


@pytest.mark.asyncio
async def test_cancellation_holds_model_lock_until_worker_and_cleanup_finish():
    started, release = threading.Event(), threading.Event()
    calls = []
    def transcribe(audio, *args):
        calls.append(audio)
        if audio == 'first':
            started.set()
            assert release.wait(3)
        return audio
    backend = LocalAsrBackend.__new__(LocalAsrBackend)
    backend.engine = SimpleNamespace(device='cpu', transcribe=transcribe)
    backend._lock = asyncio.Lock()
    backend._release_unused_buffers = lambda: calls.append('cleanup')
    first = asyncio.create_task(backend.transcribe('first', 5))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        second = asyncio.create_task(backend.transcribe('second', 5))
        await asyncio.sleep(.02)
        first.cancel()  # repeated Stop/profile invalidation cannot release it
        await asyncio.sleep(.02)
        assert calls == ['first']
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        if second is not None:
            assert await second == 'second'
    assert first.cancelled()
    assert calls == ['first', 'cleanup', 'second', 'cleanup']


@pytest.mark.asyncio
async def test_failed_verification_preserves_evidence_and_releases_buffers():
    raw = SimpleNamespace(hypotheses=['literal'])
    backend = LocalAsrBackend.__new__(LocalAsrBackend)
    def failed(*args):
        raise RuntimeError('feature extraction unavailable')
    backend.engine = SimpleNamespace(device='cpu', transcribe=lambda *args: raw, extract_features=failed)
    backend._lock = asyncio.Lock()
    cleanups = []
    backend._release_unused_buffers = lambda: cleanups.append(True)
    result, scores = await backend.transcribe_verified('recording', 5, SimpleNamespace(available=True))
    assert result is raw and scores is None and cleanups == [True]
