# Runpod Whisper worker

This worker serves the pinned Hugging Face Transformers checkpoint
`openai/whisper-large-v3` (`1b6101d1b1f60042cfabcf6574c8852850e621c2`). It uses
PyTorch and Transformers only; `faster-whisper` is not included.

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
repository, choose the `main` branch, and use
`backend/runpod_worker/Dockerfile` as the Dockerfile path. Runpod builds and
stores the worker image; no Docker registry account is required.

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
