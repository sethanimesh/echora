FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

COPY backend/runpod_worker/requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1+cu124 \
    && pip install -r requirements.txt

COPY backend/runpod_worker/worker.py ./

CMD ["python", "-u", "worker.py"]
