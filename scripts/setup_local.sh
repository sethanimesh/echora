#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v npm >/dev/null || { echo "npm is required" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (brew install ffmpeg)" >&2; exit 1; }

if git rev-parse --git-dir >/dev/null 2>&1; then
  git config core.hooksPath .githooks
fi

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements-dev.txt
npm --prefix frontend ci
.venv/bin/python scripts/import_profiles.py --apply

cd models/echora-qwen3-asr-command-v3
shasum -a 256 -c CHECKSUMS.sha256

echo "ECHORA_SETUP_DONE"
