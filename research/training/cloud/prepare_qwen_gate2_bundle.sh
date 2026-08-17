#!/usr/bin/env bash
set -euo pipefail

cloud_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
training_dir="$(cd "${cloud_script_dir}/.." && pwd)"
project_dir="$(cd "${training_dir}/../.." && pwd)"
archive_path="${1:-/private/tmp/echora-qwen-gate2.tar.gz}"
stage_dir="$(mktemp -d /private/tmp/echora-qwen-gate2.XXXXXX)"

cleanup_stage() {
  rm -rf "${stage_dir}"
}
trap cleanup_stage EXIT

"${project_dir}/.venv/bin/python" "${training_dir}/stage_qwen_gate2.py" \
  --train-manifest "${project_dir}/data/derived/torgo/folds/M04/train.jsonl" \
  --config "${training_dir}/configs/qwen_gate2_probe.json" \
  --output "${stage_dir}/probe" \
  --cloud-root /workspace/echora/probe

mkdir -p "${stage_dir}/research/training/cloud" "${stage_dir}/research/training/configs"
cp "${training_dir}/qwen_gate2_probe.py" "${stage_dir}/research/training/"
cp "${training_dir}/configs/qwen_gate2_probe.json" "${stage_dir}/research/training/configs/"
cp "${cloud_script_dir}/qwen_gate2_requirements.txt" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/setup_qwen_gate2_l40s.sh" "${stage_dir}/research/training/cloud/"
cp "${cloud_script_dir}/run_qwen_gate2.sh" "${stage_dir}/research/training/cloud/"

# macOS otherwise stores AppleDouble/xattr records and the local UID in the
# archive. Some RunPod containers cannot restore that ownership even as root.
COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --uid 0 \
  --gid 0 \
  --uname root \
  --gname root \
  -czf "${archive_path}" \
  -C "${stage_dir}" probe research
echo "Created ${archive_path}"
du -h "${archive_path}"
