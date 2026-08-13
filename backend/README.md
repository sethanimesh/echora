# Echora backend

The local backend foundation for Echora's ASR proof of concept. It stores audio
and transcription records locally in SQLite. Until a cloud ASR provider is
configured, uploads receive a clearly labelled deterministic fake transcript.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended)

## Run locally

```bash
cd backend
uv sync --dev
uv run uvicorn app.main:app --reload
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/healthz
```

Expected response:

```json
{"status":"ok","service":"echora-api","environment":"development"}
```

Local runtime state is created under `backend/data/` and is ignored by Git.

## Test an upload locally

With the server running, upload any small supported audio file:

```bash
curl -X POST http://127.0.0.1:8000/v1/transcriptions \
  -F "file=@/absolute/path/to/sample.wav;type=audio/wav"
```

The response has a `fake` provider and one deterministic hypothesis. Copy its
`id` to retrieve or delete the retained local record:

```bash
curl http://127.0.0.1:8000/v1/transcriptions/RECORD_ID
curl -X DELETE -i http://127.0.0.1:8000/v1/transcriptions/RECORD_ID
```

Supported extensions are WAV, FLAC, M4A, MP3, OGG, and WebM. Uploads are capped
at 25 MiB by default. Audio duration validation is added with the real ASR
worker, which will decode the media reliably.

## Provider capabilities

```bash
curl http://127.0.0.1:8000/v1/providers
```

Runpod is declared as supporting up to five raw hypotheses and timestamps;
Groq's hosted Whisper adapter is intentionally limited to one raw hypothesis.
The API rejects an incompatible request before it saves audio or makes an
upstream request:

```bash
curl -X POST http://127.0.0.1:8000/v1/transcriptions \
  -F "provider=groq" \
  -F "n_best=2" \
  -F "file=@/absolute/path/to/sample.wav;type=audio/wav"
```

## Runpod worker

The deployable Hugging Face Transformers/PyTorch worker lives in
[`runpod_worker/`](runpod_worker/). It serves a pinned Whisper Large-v3 model,
uses no `faster-whisper`, and preserves complete audio including pauses. Follow
its [deployment guide](runpod_worker/README.md) only when you are ready to
configure your own Runpod endpoint and API key.

## Checks

```bash
uv run ruff check .
uv run pytest
```
