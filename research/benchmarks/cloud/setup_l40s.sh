#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1

if [[ ! -x /workspace/venv-nemo/bin/python ]]; then
  python3 -m venv /workspace/venv-nemo
fi

/workspace/venv-nemo/bin/python -m pip install \
  --no-cache-dir --upgrade pip --progress-bar off -q
/workspace/venv-nemo/bin/python -m pip install \
  --no-cache-dir --progress-bar off -q 'nemo_toolkit[asr]==2.7.3'

/workspace/venv-nemo/bin/python -c \
  "import nemo, torch; print('NeMo', nemo.__version__); print('torch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0))"
df -h /workspace

