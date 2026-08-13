# Runpod Whisper worker

This worker serves the pinned Hugging Face Transformers checkpoint
`openai/whisper-large-v3` (`1b6101d1b1f60042cfabcf6574c8852850e621c2`). It uses
PyTorch and Transformers only; `faster-whisper` is not included.

The Dockerfile intentionally builds from Runpod's CUDA 12.4/PyTorch 2.4 base
image. It does not download a separate CUDA PyTorch wheel, which prevents stale
CUDA dependency pins from breaking Runpod's remote build.

## Entry point

`worker.py` is a library module: it owns validation, decoding, and generation,
and exposes the `handler(job)` function plus the shared `runtime` object. It
does not start the serverless loop.

The process entry point is `handler.py` at the **repository root**, which calls
`runtime.load()` and then `runpod.serverless.start({"handler": handler})`.
Runpod's GitHub deploy check scans for `runpod.serverless.start()` in a
root-level module and reports "Could not find runpod.serverless.start() in your
repo" when the call is nested in a subdirectory, so the entry point is kept at
the root deliberately. Do not move it back under `backend/`.

The image mirrors the repository layout (`/worker/handler.py` and
`/worker/backend/runpod_worker/`), so `from runpod_worker.worker import ...`
resolves the same way in a local checkout and in the built container.

## Worker contract

The backend sends this Runpod queue-based endpoint input:

```json
{
  "input": {
    "audio_base64": "...",
    "filename": "utterance.wav",
    "language": "en",
    "model_id": "openai/whisper-large-v3",
    "n_best": 5
  }
}
```

The worker accepts WAV, FLAC, M4A, MP3, OGG, and WebM up to 25 MiB and 120
seconds. It decodes to 16 kHz mono PCM without silence trimming, VAD,
denoising, or source-audio mutation. It returns raw beam hypotheses, raw
sequence scores when Transformers makes them available, and Whisper timestamp
segments when timestamp tokens are present.

## Build and deploy

Build a Linux/AMD64 image and push it to your private registry. On Apple
Silicon, use a remote Linux builder or Docker Buildx with an AMD64 target:

```bash
cd /path/to/echora
docker buildx build --platform linux/amd64 \
  -f backend/runpod_worker/Dockerfile \
  -t YOUR_REGISTRY/echora-whisper-worker:0.1.0 \
  --push .
```

Alternatively, deploy directly from a GitHub repository. Connect Runpod to the
repository and set:

- Branch: `main`
- Dockerfile path: `backend/runpod_worker/Dockerfile`
- Build context: `.` (the repository root)

The build context must be the repository root, because the Dockerfile copies
`handler.py` from the root and `backend/runpod_worker/` by its full repository
path. Runpod builds and stores the worker image; no Docker registry account is
required.

In Runpod Serverless, create a **queue-based** endpoint from that image:

- GPU priority: `NVIDIA GeForce RTX 4090`, then `NVIDIA L40S`.
- Minimum workers: `0`; maximum workers: `1` for the POC.
- Execution timeout: `180` seconds.
- Mount a cache/volume at `/cache` when available so Hugging Face model files
  survive worker restarts; otherwise expect the first worker start to download
  the model before serving its first request.

After deployment, set these local backend values in `backend/.env`:

```dotenv
ECHORA_ASR_DEFAULT_PROVIDER=runpod
ECHORA_RUNPOD_ENDPOINT_ID=YOUR_ENDPOINT_ID
ECHORA_RUNPOD_API_KEY=YOUR_RUNPOD_API_KEY
```

Do not commit the `.env` file or any API key.
