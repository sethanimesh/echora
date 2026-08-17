#!/usr/bin/env bash
set -euo pipefail

cloud_root=/workspace/echora
venv_dir=/workspace/venv-qwen-pilot-isolated
model_dir=/workspace/models/Qwen3-ASR-1.7B-hf
pip_temp_dir=/workspace/pip-qwen-pilot
model_revision=bcd2b5b7f32b480ab5790554cfa8347f246a14f3
expected_bytes=4076193080
expected_sha=2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${gpu_name}" != *L40S* ]]; then
  echo "QWEN_PILOT_SETUP_FAILED: expected an L40S, found ${gpu_name}" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1

# This venv is deliberately isolated. RunPod base images may expose an older
# system torch or only the metadata for cuDNN; inheriting either made Gate 2's
# first setup unable to import PyTorch.
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  python3 -m venv "${venv_dir}"
fi
if grep -q '^include-system-site-packages = true' "${venv_dir}/pyvenv.cfg"; then
  echo "QWEN_PILOT_SETUP_FAILED: ${venv_dir} is not isolated; use the exact pilot venv path" >&2
  exit 1
fi

mkdir -p "${pip_temp_dir}"
export TMPDIR="${pip_temp_dir}"
"${venv_dir}/bin/python" -m pip install \
  --no-cache-dir --upgrade pip --progress-bar off -q
"${venv_dir}/bin/python" -m pip install \
  --no-cache-dir --progress-bar off -q \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
"${venv_dir}/bin/python" -m pip install \
  --no-cache-dir --progress-bar off -q \
  -r "${cloud_root}/research/training/cloud/qwen_pilot_requirements.txt"

"${venv_dir}/bin/python" - <<'PY'
import torch
import transformers
import librosa
import soundfile
import safetensors

assert torch.__version__.startswith("2.6.0+cu124"), torch.__version__
assert transformers.__version__ == "5.15.0", transformers.__version__
assert safetensors.__version__ == "0.8.0", safetensors.__version__
assert torch.cuda.is_available()
x = torch.ones((1024, 1024), device="cuda")
assert float((x @ x)[0, 0]) == 1024.0
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("safetensors", safetensors.__version__)
print("GPU", torch.cuda.get_device_name(0))
print("CUDA matrix test passed")
PY

mkdir -p "${model_dir}"
export HF_HOME=/workspace/huggingface
"${venv_dir}/bin/python" -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-ASR-1.7B-hf', revision='${model_revision}', local_dir='${model_dir}')"

weight_file="${model_dir}/model.safetensors"
if [[ ! -f "${weight_file}" ]]; then
  echo "QWEN_PILOT_SETUP_FAILED: model.safetensors is missing" >&2
  exit 1
fi
actual_bytes="$(stat -c '%s' "${weight_file}")"
if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
  echo "QWEN_PILOT_SETUP_FAILED: expected ${expected_bytes} bytes, got ${actual_bytes}" >&2
  exit 1
fi
actual_sha="$(sha256sum "${weight_file}" | cut -d ' ' -f 1)"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  echo "QWEN_PILOT_SETUP_FAILED: foundation SHA-256 mismatch" >&2
  exit 1
fi

echo "QWEN_PILOT_SETUP_DONE: isolated environment and exact Qwen foundation verified on ${gpu_name}"
df -h /workspace
