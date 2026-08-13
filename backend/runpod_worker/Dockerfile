# The Runpod image provides the CUDA 12.4 + PyTorch 2.4 GPU stack. Do not
# install a CUDA PyTorch wheel over it: old wheel dependency pins can disappear
# from package indexes and break the remote image build.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

# Build context is the repository root. Install dependencies before copying the
# worker source so edits to the source do not invalidate the pip layer.
COPY backend/runpod_worker/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Keep the process entry point next to this Dockerfile and its implementation.
# Runpod's GitHub validation can then discover runpod.serverless.start() before
# it begins the remote image build.
COPY backend/runpod_worker/ ./

CMD ["python", "-u", "handler.py"]
