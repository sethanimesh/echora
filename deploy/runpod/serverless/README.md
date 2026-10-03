# RunPod Serverless inference

This worker performs raw literal ASR only. The main FastAPI app keeps the Groq key and performs candidate ranking and grammar repair locally.

Use a queue endpoint with Flex workers so the endpoint can scale to zero. Start with the 24 GB L4/A5000/3090 group. The 16 GB group is cheaper but should be enabled only after its memory smoke test passes.

## 1. Build and push from the Mac

Docker must already be signed in to the registry.

```bash
export ECHORA_IMAGE="docker.io/your-account/echora-asr:v1"
./deploy/runpod/serverless/build_and_push.sh
```

The build targets `linux/amd64`. It includes the 206 MB adapter and production code, but excludes the 4.1 GB public foundation.

## 2. Create the endpoint

In RunPod Serverless:

1. Import the pushed Docker image and choose a **Queue** endpoint.
2. Select the 24 GB L4/A5000/3090/MIG group first.
3. Set minimum workers to `0`, maximum workers to `1`, idle timeout to `5`, and FlashBoot on.
4. In the Model field, set `Qwen/Qwen3-ASR-1.7B-hf` to enable cached-model placement.
5. Add `MODEL_NAME=Qwen/Qwen3-ASR-1.7B-hf` and `ECHORA_DEVICE=cuda` as environment variables.
6. Do not attach a network volume.

The worker resolves the cached snapshot under `/runpod-volume/huggingface-cache/hub` and refuses to start if the foundation or adapter hashes differ from the verified artifacts.

## 3. Test

```bash
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"
.venv/bin/python deploy/runpod/serverless/client.py \
  /path/to/private-clip.wav
```

For SDK-style local handler testing, generate an input file and run the container with a GPU Linux host:

```bash
.venv/bin/python deploy/runpod/serverless/make_test_input.py \
  /path/to/private-clip.wav \
  deploy/runpod/serverless/test_input.json
python deploy/runpod/serverless/handler.py --test_input "$(cat deploy/runpod/serverless/test_input.json)"
```

## 4. Connect the local app

```dotenv
ECHORA_ASR_BACKEND=runpod
RUNPOD_ENDPOINT_ID=your-endpoint-id
RUNPOD_API_KEY=your-api-key
```

The FastAPI backend submits asynchronously through `/run` and polls `/status`, so a cold worker does not require changing the browser API.
