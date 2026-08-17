FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ECHORA_DEVICE=cuda \
    ECHORA_ASR_BACKEND=local \
    MODEL_NAME=Qwen/Qwen3-ASR-1.7B-hf \
    ECHORA_MODEL_ROOT=/app/models/echora-qwen3-asr-command-v3

WORKDIR /app

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
COPY deploy/runpod/serverless/requirements.txt /app/serverless-requirements.txt
RUN python -m pip install -r /app/backend/requirements.txt -r /app/serverless-requirements.txt

COPY backend /app/backend
COPY models/echora-qwen3-asr-command-v3/adapter /app/models/echora-qwen3-asr-command-v3/adapter
COPY models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json /app/models/echora-qwen3-asr-command-v3/evaluation/qwen_command_v3.json
COPY models/echora-qwen3-asr-command-v3/manifest.json /app/models/echora-qwen3-asr-command-v3/manifest.json
COPY deploy/runpod/serverless/handler.py /app/handler.py

ENV PYTHONPATH=/app/backend:/app
CMD ["python", "-u", "/app/handler.py"]
