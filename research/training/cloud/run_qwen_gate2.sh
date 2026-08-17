#!/usr/bin/env bash
set -euo pipefail

cloud_root=/workspace/echora
venv_python=/workspace/venv-qwen-gate2/bin/python
model_dir=/workspace/models/Qwen3-ASR-1.7B-hf
run_id=qwen3-asr-1.7b-gate2-v1
output_dir="${cloud_root}/checkpoints/${run_id}"
status_dir="${cloud_root}/status/${run_id}"

mkdir -p "${output_dir}" "${status_dir}"
exec 9>"${cloud_root}/.qwen_gate2.lock"
if ! flock -n 9; then
  echo "QWEN_GATE2_FAILED: another Gate 2 process is already running" >&2
  exit 1
fi

rm -f "${status_dir}/DONE" "${status_dir}/FAILED"
on_exit() {
  exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/FAILED"
    echo "QWEN_GATE2_FAILED: paste the final 100 log lines; do not change the recipe" >&2
  fi
}
trap on_exit EXIT

if [[ ! -x "${venv_python}" ]]; then
  echo "The Gate 2 environment is missing; run setup_qwen_gate2_l40s.sh first" >&2
  exit 1
fi

"${venv_python}" "${cloud_root}/research/training/qwen_gate2_probe.py" \
  --config "${cloud_root}/research/training/configs/qwen_gate2_probe.json" \
  --manifest "${cloud_root}/probe/manifest.jsonl" \
  --model "${model_dir}" \
  --output "${output_dir}"

cp "${cloud_root}/research/training/configs/qwen_gate2_probe.json" "${output_dir}/"
cp "${cloud_root}/probe/selection.json" "${output_dir}/"
date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/DONE"
echo "QWEN_GATE2_DONE: checkpoint reload and resume passed; copy ${output_dir} before stopping the Pod"
