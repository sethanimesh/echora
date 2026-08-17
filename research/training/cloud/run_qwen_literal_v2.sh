#!/usr/bin/env bash
set -euo pipefail

cloud_root=/workspace/echora
venv_python=/workspace/venv-qwen-pilot-isolated/bin/python
model_dir=/workspace/models/Qwen3-ASR-1.7B-hf
run_id=qwen3-asr-1.7b-literal-ctc-v2
output_dir="${cloud_root}/checkpoints/${run_id}"
diagnostic_dir="${output_dir}/diagnostics"
status_dir="${cloud_root}/status/${run_id}"
initial_adapter="${cloud_root}/initial/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors"
latest="${output_dir}/checkpoint-latest.pt"
mode="${1:-fresh}"

if [[ "${mode}" != "fresh" && "${mode}" != "resume" ]]; then
  echo "Usage: run_qwen_literal_v2.sh fresh|resume" >&2
  exit 2
fi
if [[ ! -x "${venv_python}" ]]; then
  echo "LITERAL_V2_FAILED: run setup_qwen_literal_v2_l40s.sh first" >&2
  exit 1
fi

if [[ "${mode}" == "fresh" ]]; then
  if [[ -f "${latest}" || -f "${output_dir}/checkpoint-best.pt" || \
        -f "${output_dir}/result.json" || -f "${output_dir}/adapter.safetensors" ]]; then
    echo "LITERAL_V2_FAILED: fresh mode refuses existing training state in ${output_dir}" >&2
    echo "Nothing was deleted. Use resume only if checkpoint-latest.pt belongs to this run." >&2
    exit 1
  fi
else
  if [[ ! -f "${latest}" ]]; then
    echo "LITERAL_V2_FAILED: resume requested but ${latest} is missing" >&2
    exit 1
  fi
fi

mkdir -p "${output_dir}" "${diagnostic_dir}" "${status_dir}"
exec 9>"${cloud_root}/.qwen_literal_v2.lock"
if ! flock -n 9; then
  echo "LITERAL_V2_FAILED: another literal-v2 process is running" >&2
  exit 1
fi
exec > >(tee -a "${output_dir}/literal_v2.log") 2>&1

on_exit() {
  exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/FAILED"
    echo "LITERAL_V2_FAILED: paste the final 150 log lines; do not change the recipe"
  fi
}
trap on_exit EXIT
rm -f "${status_dir}/DONE" "${status_dir}/FAILED"
expected_initial_sha=4e211db6c6571a7a4efbe8e851b67d44f33feea41724d0ab7afcf7195e018e0c
actual_initial_sha="$(sha256sum "${initial_adapter}" | cut -d ' ' -f 1)"
if [[ "${actual_initial_sha}" != "${expected_initial_sha}" ]]; then
  echo "LITERAL_V2_FAILED: verified v1 adapter identity changed" >&2
  exit 1
fi
cp "${cloud_root}/research/training/configs/qwen_literal_ctc_v2.json" "${output_dir}/"
cp "${cloud_root}/pilot_data/stage_report.json" "${output_dir}/"
cp "${cloud_root}/pilot_data/literal_diagnostic_report.json" "${output_dir}/"

if [[ ! -f "${diagnostic_dir}/v1_top5.summary.json" ]]; then
  "${venv_python}" "${cloud_root}/research/training/literal_diagnostic.py" \
    --config "${cloud_root}/research/training/configs/qwen_literal_ctc_v2.json" \
    --model "${model_dir}" \
    --adapter "${initial_adapter}" \
    --manifest "${cloud_root}/pilot_data/manifests/literal_diagnostic.jsonl" \
    --output "${diagnostic_dir}/v1_top5.tsv" \
    --beams 5
fi

resume_args=()
if [[ "${mode}" == "resume" ]]; then
  resume_args=(--resume "${latest}")
fi
"${venv_python}" "${cloud_root}/research/training/qwen_literal_ctc_v2.py" \
  --config "${cloud_root}/research/training/configs/qwen_literal_ctc_v2.json" \
  --data "${cloud_root}/pilot_data" \
  --model "${model_dir}" \
  --initial-adapter "${initial_adapter}" \
  --output "${output_dir}" \
  "${resume_args[@]}"

"${venv_python}" "${cloud_root}/research/training/literal_diagnostic.py" \
  --config "${cloud_root}/research/training/configs/qwen_literal_ctc_v2.json" \
  --model "${model_dir}" \
  --adapter "${output_dir}/adapter.safetensors" \
  --ctc-head "${output_dir}/ctc_head.safetensors" \
  --manifest "${cloud_root}/pilot_data/manifests/literal_diagnostic.jsonl" \
  --output "${diagnostic_dir}/v2_top5_and_ctc.tsv" \
  --beams 5

"${venv_python}" "${cloud_root}/research/training/make_literal_v2_decision.py" \
  --v1-summary "${diagnostic_dir}/v1_top5.summary.json" \
  --v1-tsv "${diagnostic_dir}/v1_top5.tsv" \
  --v2-summary "${diagnostic_dir}/v2_top5_and_ctc.summary.json" \
  --v2-tsv "${diagnostic_dir}/v2_top5_and_ctc.tsv" \
  --result "${output_dir}/result.json" \
  --output "${output_dir}/next_decision.json"

date -u +%Y-%m-%dT%H:%M:%SZ > "${status_dir}/DONE"
echo "LITERAL_V2_DONE: copy ${output_dir} back to the SSD and paste SHARE_THIS_WITH_CODEX.txt"
