#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/echora
RUNTIME="$ROOT/runtime"
PID_FILE="$RUNTIME/pod-service.pid"
LOG_FILE="$RUNTIME/pod-service.log"
: "${ECHORA_WORKER_TOKEN:?Set ECHORA_WORKER_TOKEN before starting}"

mkdir -p "$RUNTIME"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Echora service is already running with PID $(cat "$PID_FILE")"
  exit 0
fi

export PYTHONPATH="$ROOT/backend:$ROOT"
export ECHORA_ASR_BACKEND=local
export ECHORA_DEVICE=cuda
export ECHORA_MODEL_ROOT="$ROOT/models/echora-qwen3-asr-command-v3"

nohup /workspace/venv-echora-inference/bin/uvicorn app.worker:app \
  --host 0.0.0.0 --port 8000 >"$LOG_FILE" 2>&1 &
echo "$!" >"$PID_FILE"

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "ECHORA_POD_SERVICE_READY"
    curl -fsS http://127.0.0.1:8000/health
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    tail -n 80 "$LOG_FILE" >&2
    exit 1
  fi
  sleep 2
done

echo "Service did not become ready; inspect $LOG_FILE" >&2
exit 1
