#!/usr/bin/env bash
set -euo pipefail

cloud_root=/workspace/echora
python_bin=/workspace/venv-nemo/bin/python
hf_bin=/workspace/venv-nemo/bin/hf
model_path="${cloud_root}/models/parakeet-tdt-1.1b.nemo"
manifest_path="${cloud_root}/gate/foundation_gate_400.jsonl"
result_path="${cloud_root}/research/benchmarks/results/foundation_gate/parakeet_tdt_1_1b.tsv"
expected_size=4283136000
expected_sha=9c563d52bdffeacbac0c5b894fdea9be82fea3a6bd8bb8018ff57888e2b5d988

# Prevent two terminals from appending the same utterances to the resumable TSV.
exec 9>"${cloud_root}/.parakeet_gate.lock"
if ! flock -n 9; then
  echo "Another Parakeet gate process is already running." >&2
  exit 1
fi

test -x "${python_bin}"
test -f "${manifest_path}"
mkdir -p "${cloud_root}/models"
mkdir -p "$(dirname "${result_path}")"

if [[ ! -f "${model_path}" ]]; then
  HF_HUB_DISABLE_XET=1 "${hf_bin}" download \
    nvidia/parakeet-tdt-1.1b parakeet-tdt-1.1b.nemo \
    --local-dir "${cloud_root}/models"
fi

actual_size="$(stat -c '%s' "${model_path}")"
actual_sha="$(sha256sum "${model_path}" | cut -d ' ' -f 1)"
if [[ "${actual_size}" != "${expected_size}" ]]; then
  echo "Wrong model size: ${actual_size}; expected ${expected_size}" >&2
  exit 1
fi
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  echo "Wrong model SHA-256: ${actual_sha}; expected ${expected_sha}" >&2
  exit 1
fi
echo "Verified official nvidia/parakeet-tdt-1.1b foundation checkpoint."

cd "${cloud_root}/research/benchmarks"
export ECHORA_PARAKEET_TDT_1_1B="${model_path}"

# One utterance first. The evaluator is resumable, so the full run keeps this row.
"${python_bin}" evaluate_manifest.py \
  --runner parakeet_tdt_1_1b \
  --device cuda \
  --manifest "${manifest_path}" \
  --output "${result_path}" \
  --batch-size 1 \
  --limit 1

"${python_bin}" evaluate_manifest.py \
  --runner parakeet_tdt_1_1b \
  --device cuda \
  --manifest "${manifest_path}" \
  --output "${result_path}" \
  --batch-size 4 \
  --report-every 50

row_count="$(($(wc -l < "${result_path}") - 1))"
if [[ "${row_count}" != "400" ]]; then
  echo "Incomplete result: ${row_count}/400 utterances" >&2
  exit 1
fi
echo "PARAKEET_GATE_DONE: ${result_path} has 400/400 utterances"
