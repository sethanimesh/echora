# The Runpod image provides the CUDA 12.4 + PyTorch 2.4 GPU stack. Do not
# install a CUDA PyTorch wheel over it: old wheel dependency pins can disappear
# from package indexes and break the remote image build.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker

# Build context is the repository root. Install dependencies before copying the
# worker source so edits to the source do not invalidate the pip layer.
COPY backend/runpod_worker/requirements.txt ./backend/runpod_worker/
RUN pip install -r backend/runpod_worker/requirements.txt

# Mirror the repository layout inside the image so handler.py resolves
# `runpod_worker.worker` the same way locally and in the built container.
COPY backend/runpod_worker/ ./backend/runpod_worker/
COPY handler.py ./

CMD ["python", "-u", "handler.py"]
