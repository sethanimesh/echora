#!/usr/bin/env bash
set -euo pipefail

cloud_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_dir="$(cd "${cloud_script_dir}/.." && pwd)"
project_dir="$(cd "${training_dir}/../.." && pwd)"
archive_path="${1:-/private/tmp/echora-qwen-pilot-v1.tar.gz}"
stage_dir="$(mktemp -d /private/tmp/echora-qwen-pilot.XXXXXX)"
normal_dir="${project_dir}/data/derived/common_voice/pilot-v1"

cleanup_stage() {
  rm -rf "${stage_dir}"
}
trap cleanup_stage EXIT

for manifest in train.jsonl dev.jsonl test.jsonl; do
  if [[ ! -f "${normal_dir}/${manifest}" ]]; then
    echo "Missing ${normal_dir}/${manifest}" >&2
    echo "Prepare Common Voice first; the pilot refuses to omit retention data." >&2
    exit 1
  fi
done

"${project_dir}/.venv/bin/python" "${training_dir}/stage_qwen_pilot.py" \
  --config "${training_dir}/configs/qwen_torgo_pilot_v1.json" \
  --torgo-dir "${project_dir}/data/derived/torgo/folds/M04" \
  --normal-dir "${normal_dir}" \
  --personal-dir "${project_dir}/research/benchmarks/clips" \
  --output "${stage_dir}/pilot_data" \
  --cloud-root /workspace/echora/pilot_data

mkdir -p "${stage_dir}/research/training/cloud" "${stage_dir}/research/training/configs" "${stage_dir}/research/benchmarks"
cp "${training_dir}/qwen_torgo_pilot.py" "${stage_dir}/research/training/"
cp "${training_dir}/run_qwen_adapter.py" "${stage_dir}/research/training/"
cp "${training_dir}/augment.py" "${stage_dir}/research/training/"
cp "${training_dir}/configs/qwen_torgo_pilot_v1.json" "${stage_dir}/research/training/configs/"
cp "${project_dir}/research/benchmarks/metrics.py" "${stage_dir}/research/benchmarks/"
cp "${cloud_script_dir}/qwen_pilot_requirements.txt" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/setup_qwen_pilot_l40s.sh" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/run_qwen_pilot.sh" "${stage_dir}/research/training/cloud/"

COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --uid 0 \
  --gid 0 \
  --uname root \
  --gname root \
  -czf "${archive_path}" \
  -C "${stage_dir}" pilot_data research
echo "Created ${archive_path}"
du -h "${archive_path}"
