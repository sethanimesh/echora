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

The process entry point is `backend/runpod_worker/handler.py`, beside the
Dockerfile selected in Runpod's GitHub deployment flow. It calls
`runtime.load()` and then `runpod.serverless.start({"handler": handler})`.
This placement lets Runpod's repository validation find the Serverless startup
call before the image is built.

The image copies this directory directly to `/worker`, so the entry point can
import `worker.py` exactly as it does when run locally with
`python backend/runpod_worker/handler.py`.

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

The endpoint must enable Runpod's cached model `openai/whisper-large-v3`. The
worker loads its pinned revision from the `/runpod-volume/huggingface-cache/hub`
mount in offline mode; it will fail clearly rather than downloading the model
during billed worker time.

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
`backend/runpod_worker/` by its repository path. Runpod builds and stores the
worker image; no Docker registry account is required.

In Runpod Serverless, create a **queue-based** endpoint from that image:

- GPU priority: `NVIDIA GeForce RTX 4090`, then `NVIDIA L40S`.
- Minimum workers: `0`; maximum workers: `1` for the POC.
- Execution timeout: `180` seconds.
- Cached model: `openai/whisper-large-v3`.
- Do not add a network volume for this POC. Runpod mounts its managed cached
  model under `/runpod-volume/huggingface-cache/hub` automatically.

After deployment, set these local backend values in `backend/.env`:

```dotenv
ECHORA_ASR_DEFAULT_PROVIDER=runpod
ECHORA_RUNPOD_ENDPOINT_ID=YOUR_ENDPOINT_ID
ECHORA_RUNPOD_API_KEY=YOUR_RUNPOD_API_KEY
```

Do not commit the `.env` file or any API key.
