#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -x .venv/bin/uvicorn ] || { echo "Run ./scripts/setup_local.sh first" >&2; exit 1; }

cleanup() {
  if [ -n "${API_PID:-}" ]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONPATH="$ROOT/backend:$ROOT" .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "Echora is starting. The model normally becomes ready in about 10 seconds."
echo "UI:  http://localhost:3000"
echo "API: http://localhost:8000/docs"

npm --prefix frontend run dev
