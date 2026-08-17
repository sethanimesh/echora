#!/usr/bin/env bash
set -euo pipefail

cloud_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_dir="$(cd "${cloud_script_dir}/.." && pwd)"
project_dir="$(cd "${training_dir}/../.." && pwd)"
archive_path="${1:-/private/tmp/echora-qwen-literal-v2.tar.gz}"
initial_adapter="${2:-/Volumes/Extreme Pro/echora/checkpoints/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors}"
stage_dir="$(mktemp -d /private/tmp/echora-qwen-literal-v2.XXXXXX)"
normal_dir="${project_dir}/data/derived/common_voice/pilot-v1"

cleanup_stage() {
  rm -rf "${stage_dir}"
}
trap cleanup_stage EXIT

if [[ ! -f "${initial_adapter}" ]]; then
  echo "Missing verified v1 adapter: ${initial_adapter}" >&2
  exit 1
fi
actual_adapter_sha="$(shasum -a 256 "${initial_adapter}" | awk '{print $1}')"
expected_adapter_sha=4e211db6c6571a7a4efbe8e851b67d44f33feea41724d0ab7afcf7195e018e0c
if [[ "${actual_adapter_sha}" != "${expected_adapter_sha}" ]]; then
  echo "Refusing unverified v1 adapter: ${actual_adapter_sha}" >&2
  exit 1
fi

for manifest in train.jsonl dev.jsonl test.jsonl; do
  if [[ ! -f "${normal_dir}/${manifest}" ]]; then
    echo "Missing ${normal_dir}/${manifest}; normal-speech retention data is mandatory" >&2
    exit 1
  fi
done

"${project_dir}/.venv/bin/python" "${training_dir}/stage_qwen_pilot.py" \
  --config "${training_dir}/configs/qwen_literal_ctc_v2.json" \
  --torgo-dir "${project_dir}/data/derived/torgo/folds/M04" \
  --normal-dir "${normal_dir}" \
  --personal-dir "${project_dir}/research/benchmarks/clips" \
  --output "${stage_dir}/pilot_data" \
  --cloud-root /workspace/echora/pilot_data

"${project_dir}/.venv/bin/python" "${training_dir}/stage_literal_diagnostic.py" \
  --pilot-data "${stage_dir}/pilot_data" \
  --pause-root "${project_dir}/data/derived/pause_stress" \
  --cloud-root /workspace/echora/pilot_data

mkdir -p \
  "${stage_dir}/research/training/cloud" \
  "${stage_dir}/research/training/configs" \
  "${stage_dir}/research/benchmarks" \
  "${stage_dir}/initial/qwen3-asr-1.7b-torgo-pilot-v1"
cp "${training_dir}/qwen_torgo_pilot.py" "${stage_dir}/research/training/"
cp "${training_dir}/qwen_literal_ctc_v2.py" "${stage_dir}/research/training/"
cp "${training_dir}/literal_ctc.py" "${stage_dir}/research/training/"
cp "${training_dir}/literal_diagnostic.py" "${stage_dir}/research/training/"
cp "${training_dir}/make_literal_v2_decision.py" "${stage_dir}/research/training/"
cp "${training_dir}/augment.py" "${stage_dir}/research/training/"
cp "${training_dir}/configs/qwen_literal_ctc_v2.json" "${stage_dir}/research/training/configs/"
cp "${project_dir}/research/benchmarks/metrics.py" "${stage_dir}/research/benchmarks/"
cp "${cloud_script_dir}/qwen_literal_v2_requirements.txt" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/setup_qwen_literal_v2_l40s.sh" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/run_qwen_literal_v2.sh" "${stage_dir}/research/training/cloud/"
cp "${initial_adapter}" \
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
