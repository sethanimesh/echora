#!/usr/bin/env bash
set -euo pipefail

cloud_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_dir="$(cd "${cloud_script_dir}/.." && pwd)"
project_dir="$(cd "${training_dir}/../.." && pwd)"
archive_path="${1:-/private/tmp/echora-qwen-command-v3.tar.gz}"
v1_adapter="${2:-/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors}"
stage_dir="$(mktemp -d /private/tmp/echora-qwen-command-v3.XXXXXX)"
normal_dir="${project_dir}/data/derived/common_voice/pilot-v1"

cleanup_stage() {
  rm -rf "${stage_dir}"
}
trap cleanup_stage EXIT

if [[ ! -f "${v1_adapter}" ]]; then
  echo "Missing verified v1 adapter: ${v1_adapter}" >&2
  exit 1
fi
actual_v1_sha="$(shasum -a 256 "${v1_adapter}" | awk '{print $1}')"
expected_v1_sha=4e211db6c6571a7a4efbe8e851b67d44f33feea41724d0ab7afcf7195e018e0c
if [[ "${actual_v1_sha}" != "${expected_v1_sha}" ]]; then
  echo "Refusing unverified v1 adapter: ${actual_v1_sha}" >&2
  exit 1
fi
for manifest in train.jsonl dev.jsonl test.jsonl; do
  if [[ ! -f "${normal_dir}/${manifest}" ]]; then
    echo "Missing ${normal_dir}/${manifest}; normal retention data is mandatory" >&2
    exit 1
  fi
done

"${project_dir}/.venv/bin/python" "${training_dir}/stage_qwen_pilot.py" \
  --config "${training_dir}/configs/qwen_command_v3.json" \
  --torgo-dir "${project_dir}/data/derived/torgo/folds/M04" \
  --normal-dir "${normal_dir}" \
  --personal-dir "${project_dir}/research/benchmarks/clips" \
  --output "${stage_dir}/pilot_data" \
  --cloud-root /workspace/echora/pilot_data

"${project_dir}/.venv/bin/python" "${training_dir}/stage_literal_diagnostic.py" \
  --pilot-data "${stage_dir}/pilot_data" \
  --pause-root "${project_dir}/data/derived/pause_stress" \
  --cloud-root /workspace/echora/pilot_data

"${project_dir}/.venv/bin/python" "${training_dir}/stage_qwen_command_v3.py" \
  --config "${training_dir}/configs/qwen_command_v3.json" \
  --pilot-data "${stage_dir}/pilot_data" \
  --cloud-root /workspace/echora/pilot_data

mkdir -p \
  "${stage_dir}/research/training/cloud" \
  "${stage_dir}/research/training/configs" \
  "${stage_dir}/research/benchmarks" \
  "${stage_dir}/initial/qwen3-asr-1.7b-torgo-pilot-v1"
cp "${training_dir}/qwen_torgo_pilot.py" "${stage_dir}/research/training/"
cp "${training_dir}/qwen_command_model.py" "${stage_dir}/research/training/"
cp "${training_dir}/qwen_command_v3.py" "${stage_dir}/research/training/"
cp "${training_dir}/run_qwen_command_adapter.py" "${stage_dir}/research/training/"
cp "${training_dir}/augment.py" "${stage_dir}/research/training/"
cp "${training_dir}/COMMAND-V3-README.md" "${stage_dir}/research/training/"
cp "${training_dir}/configs/qwen_command_v3.json" "${stage_dir}/research/training/configs/"
cp "${project_dir}/research/benchmarks/metrics.py" "${stage_dir}/research/benchmarks/"
cp "${cloud_script_dir}/qwen_command_v3_requirements.txt" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/setup_qwen_command_v3_l40s.sh" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/run_qwen_command_v3.sh" "${stage_dir}/research/training/cloud/"
cp "${v1_adapter}" \
  "${stage_dir}/initial/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors"

COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --uid 0 \
  --gid 0 \
  --uname root \
  --gname root \
  -czf "${archive_path}" \
  -C "${stage_dir}" pilot_data research initial
echo "Created ${archive_path}"
du -h "${archive_path}"
shasum -a 256 "${archive_path}"
