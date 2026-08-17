#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/echora

if command -v apt-get >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsndfile1
fi

python3 -m venv --system-site-packages /workspace/venv-echora-inference
/workspace/venv-echora-inference/bin/python -m pip install --upgrade pip
/workspace/venv-echora-inference/bin/python -m pip install -r "$ROOT/backend/requirements.txt"
/workspace/venv-echora-inference/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable on this Pod"'

cd "$ROOT/models/echora-qwen3-asr-command-v3"
sha256sum -c CHECKSUMS.sha256
echo "ECHORA_POD_SETUP_DONE"
