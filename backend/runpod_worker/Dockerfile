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

COPY backend/runpod_worker/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/runpod_worker/worker.py ./

CMD ["python", "-u", "worker.py"]
