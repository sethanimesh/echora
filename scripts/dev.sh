#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

[ -x .venv/bin/python ] || { echo "Run ./scripts/setup_local.sh first" >&2; exit 1; }

cleanup() {
  if [ -n "${WEB_PID:-}" ]; then
    kill "$WEB_PID" 2>/dev/null || true
    wait "$WEB_PID" 2>/dev/null || true
  fi
  if [ -n "${GAZE_PID:-}" ]; then
    kill "$GAZE_PID" 2>/dev/null || true
    wait "$GAZE_PID" 2>/dev/null || true
  fi
  if [ -n "${API_PID:-}" ]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONPATH="$ROOT/backend:$ROOT" .venv/bin/python -m uvicorn app.main:app --host "${ECHORA_BIND_HOST:-127.0.0.1}" --port 8000 &
API_PID=$!

if [ "${ECHORA_GAZE_ENABLED:-false}" = "true" ]; then
  [ -x .venv-gaze/bin/python ] || { echo "Install the optional gaze environment first; see communication/experiments/gazefollower/README.md" >&2; exit 1; }
  PYTHONPATH="$ROOT/backend:$ROOT" .venv-gaze/bin/python -m uvicorn communication.experiments.gazefollower.server:app --host 127.0.0.1 --port 8767 &
  GAZE_PID=$!
fi

echo "Echora is starting. The model normally becomes ready in about 10 seconds."
echo "UI:  http://localhost:3000"
echo "API: http://localhost:8000/docs"

npm --prefix frontend run dev &
WEB_PID=$!
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
