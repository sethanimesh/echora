#!/usr/bin/env bash
set -euo pipefail

cloud_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd "${cloud_script_dir}/.." && pwd)"
project_dir="$(cd "${bench_dir}/../.." && pwd)"
archive_path="${1:-/private/tmp/echora-cloud-gate.tar.gz}"
stage_dir="$(mktemp -d /private/tmp/echora-cloud-gate.XXXXXX)"

cleanup_stage() {
  rm -rf "${stage_dir}"
}
trap cleanup_stage EXIT

"${project_dir}/.venv/bin/python" "${bench_dir}/stage_cloud_gate.py" \
  --manifest "${project_dir}/data/derived/torgo/foundation_gate_400.jsonl" \
  --output "${stage_dir}/gate" \
  --cloud-root /workspace/echora/gate

mkdir -p "${stage_dir}/research/benchmarks/cloud"
cp "${bench_dir}/common.py" "${stage_dir}/research/benchmarks/"
cp "${bench_dir}/metrics.py" "${stage_dir}/research/benchmarks/"
cp "${bench_dir}/evaluate_manifest.py" "${stage_dir}/research/benchmarks/"
cp "${bench_dir}/run_parakeet_tdt_1_1b.py" "${stage_dir}/research/benchmarks/"
cp "${cloud_script_dir}/setup_l40s.sh" "${stage_dir}/research/benchmarks/cloud/"
cp "${cloud_script_dir}/run_parakeet_gate.sh" "${stage_dir}/research/benchmarks/cloud/"

tar -czf "${archive_path}" -C "${stage_dir}" gate research
echo "Created ${archive_path}"
du -h "${archive_path}"
