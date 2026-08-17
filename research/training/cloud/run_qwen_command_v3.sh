#!/usr/bin/env bash
set -euo pipefail

cloud_root=/workspace/echora
venv_python=/workspace/venv-qwen-command-v3/bin/python
model_dir=/workspace/models/Qwen3-ASR-1.7B-hf
run_id=qwen3-asr-1.7b-command-v3
output_dir="${cloud_root}/checkpoints/${run_id}"
status_dir="${cloud_root}/status/${run_id}"
v1_adapter="${cloud_root}/initial/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors"
latest="${output_dir}/checkpoint-latest.pt"
mode="${1:-fresh}"

if [[ "${mode}" != "fresh" && "${mode}" != "resume" ]]; then
  echo "Usage: run_qwen_command_v3.sh fresh|resume" >&2
  exit 2
fi
if [[ ! -x "${venv_python}" ]]; then
  echo "COMMAND_V3_FAILED: run setup_qwen_command_v3_l40s.sh first" >&2
  exit 1
fi
if [[ "${mode}" == "fresh" ]]; then
  if [[ -f "${latest}" || -f "${output_dir}/checkpoint-best-candidate.pt" || \
        -f "${output_dir}/result.json" || -f "${output_dir}/adapter.safetensors" ]]; then
    echo "COMMAND_V3_FAILED: fresh mode refuses existing training state in ${output_dir}" >&2
    echo "Nothing was deleted. Use resume only for this exact run." >&2
    exit 1
  fi
else
  if [[ ! -f "${latest}" ]]; then
    echo "COMMAND_V3_FAILED: resume requested but ${latest} is missing" >&2
    exit 1
  fi
fi

mkdir -p "${output_dir}" "${status_dir}"
exec 9>"${cloud_root}/.qwen_command_v3.lock"
if ! flock -n 9; then
  echo "COMMAND_V3_FAILED: another command-v3 process is running" >&2
  exit 1
fi
exec > >(tee -a "${output_dir}/command_v3.log") 2>&1

on_exit() {
  exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/FAILED"
    echo "COMMAND_V3_FAILED: paste the final 180 log lines; do not change the recipe"
  fi
}
trap on_exit EXIT
rm -f "${status_dir}/DONE" "${status_dir}/FAILED"

expected_v1_sha=4e211db6c6571a7a4efbe8e851b67d44f33feea41724d0ab7afcf7195e018e0c
actual_v1_sha="$(sha256sum "${v1_adapter}" | cut -d ' ' -f 1)"
if [[ "${actual_v1_sha}" != "${expected_v1_sha}" ]]; then
  echo "COMMAND_V3_FAILED: verified v1 baseline adapter identity changed" >&2
  exit 1
fi

cp "${cloud_root}/research/training/configs/qwen_command_v3.json" "${output_dir}/"
cp "${cloud_root}/pilot_data/stage_report.json" "${output_dir}/"
cp "${cloud_root}/pilot_data/literal_diagnostic_report.json" "${output_dir}/"
cp "${cloud_root}/pilot_data/command_v3_report.json" "${output_dir}/"

resume_args=()
if [[ "${mode}" == "resume" ]]; then
  resume_args=(--resume "${latest}")
fi
"${venv_python}" "${cloud_root}/research/training/qwen_command_v3.py" \
  --config "${cloud_root}/research/training/configs/qwen_command_v3.json" \
  --data "${cloud_root}/pilot_data" \
  --model "${model_dir}" \
  --v1-adapter "${v1_adapter}" \
  --output "${output_dir}" \
  "${resume_args[@]}"

date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/DONE"
echo "COMMAND_V3_DONE: copy ${output_dir} to the SSD and paste SHARE_THIS_WITH_CODEX.txt"
