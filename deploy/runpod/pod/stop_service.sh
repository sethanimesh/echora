#!/usr/bin/env bash
set -euo pipefail

PID_FILE=/workspace/echora/runtime/pod-service.pid
if [ ! -f "$PID_FILE" ]; then
  echo "No Echora PID file is present"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "$PID_FILE"
echo "ECHORA_POD_SERVICE_STOPPED (the Pod itself is still running)"
