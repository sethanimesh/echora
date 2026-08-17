#!/usr/bin/env bash
set -euo pipefail

cloud_root=/workspace/echora
venv_python=/workspace/venv-qwen-pilot-isolated/bin/python
model_dir=/workspace/models/Qwen3-ASR-1.7B-hf
run_id=qwen3-asr-1.7b-torgo-pilot-v1
output_dir="${cloud_root}/checkpoints/${run_id}"
status_dir="${cloud_root}/status/${run_id}"
latest="${output_dir}/checkpoint-latest.pt"
mode="${1:-fresh}"

if [[ "${mode}" != "fresh" && "${mode}" != "resume" ]]; then
  echo "Usage: run_qwen_pilot.sh fresh|resume" >&2
  exit 2
fi

mkdir -p "${output_dir}" "${status_dir}"
exec 9>"${cloud_root}/.qwen_pilot.lock"
if ! flock -n 9; then
  echo "QWEN_PILOT_FAILED: another pilot process is already running" >&2
  exit 1
fi

on_exit() {
  exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/FAILED"
    echo "QWEN_PILOT_FAILED: paste result.json if present, otherwise the final 150 log lines" >&2
  fi
}
trap on_exit EXIT

if [[ ! -x "${venv_python}" ]]; then
  echo "Pilot environment is missing; run setup_qwen_pilot_l40s.sh first" >&2
  exit 1
fi

resume_args=()
if [[ "${mode}" == "fresh" ]]; then
  if [[ -f "${latest}" || -f "${output_dir}/checkpoint-best.pt" || -f "${output_dir}/result.json" ]]; then
    echo "QWEN_PILOT_FAILED: fresh mode refuses existing pilot state in ${output_dir}" >&2
    echo "Use a clean persistent volume for a from-zero run; nothing was deleted." >&2
    exit 1
  fi
  echo "QWEN_PILOT_FRESH: loading the verified official foundation; no checkpoint will be restored"
else
  if [[ ! -f "${latest}" ]]; then
    echo "QWEN_PILOT_FAILED: resume requested but ${latest} does not exist" >&2
    exit 1
  fi
  resume_args=(--resume "${latest}")
  echo "QWEN_PILOT_RESUME: ${latest}"
fi

rm -f "${status_dir}/DONE" "${status_dir}/FAILED"
cp "${cloud_root}/research/training/configs/qwen_torgo_pilot_v1.json" "${output_dir}/"
cp "${cloud_root}/pilot_data/stage_report.json" "${output_dir}/"

"${venv_python}" "${cloud_root}/research/training/qwen_torgo_pilot.py" \
  --config "${cloud_root}/research/training/configs/qwen_torgo_pilot_v1.json" \
  --data "${cloud_root}/pilot_data" \
  --model "${model_dir}" \
  --output "${output_dir}" \
  "${resume_args[@]}"

date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/DONE"
echo "QWEN_PILOT_DONE: result, adapter and resumable checkpoints are in ${output_dir}"
