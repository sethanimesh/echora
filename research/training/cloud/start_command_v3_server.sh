#!/usr/bin/env bash
# Start (or restart) the persistent command-v3 literal-ASR diagnostic server.
# Run this on the Pod.  The foundation loads once and stays resident.
set -euo pipefail

ROOT=/workspace/echora
RUN=qwen3-asr-1.7b-command-v3
VENV=/workspace/venv-qwen-command-v3/bin/python
SERVE=$ROOT/serve
LOG=$SERVE/server.log

CONFIG=$ROOT/research/training/configs/qwen_command_v3.json
MODEL=/workspace/models/Qwen3-ASR-1.7B-hf
V3=$ROOT/checkpoints/$RUN/adapter.safetensors
V1=$ROOT/initial/qwen3-asr-1.7b-torgo-pilot-v1/adapter.safetensors

for path in "$VENV" "$CONFIG" "$V3"; do
  [ -e "$path" ] || { echo "missing required path: $path" >&2; exit 1; }
done
[ -d "$MODEL" ] || { echo "missing foundation: $MODEL" >&2; exit 1; }

if pgrep -f "serve_command_v3.py" >/dev/null 2>&1; then
  echo "stopping existing server"
  pkill -f "serve_command_v3.py" || true
  sleep 3
fi

mkdir -p "$SERVE/inbox" "$SERVE/outbox"
rm -f "$SERVE/outbox/SERVER_READY"

V1_ARGS=()
if [ -f "$V1" ]; then
  V1_ARGS=(--v1-adapter "$V1")
else
  echo "note: v1 baseline adapter not found, serving v3 only" >&2
fi

cd "$ROOT/research/training"
nohup "$VENV" serve_command_v3.py \
  --config "$CONFIG" \
  --model "$MODEL" \
  --adapter "$V3" \
  "${V1_ARGS[@]}" \
  --inbox "$SERVE/inbox" \
  --outbox "$SERVE/outbox" \
  --beams 5 \
  >"$LOG" 2>&1 &

echo "server starting, pid $!"
echo "log: $LOG"
echo -n "waiting for the foundation to load"
for _ in $(seq 1 180); do
  if [ -f "$SERVE/outbox/SERVER_READY" ]; then
    echo
    echo "COMMAND_V3_SERVER_READY"
    cat "$SERVE/outbox/SERVER_READY"
    exit 0
  fi
  if ! pgrep -f "serve_command_v3.py" >/dev/null 2>&1; then
    echo
    echo "server exited during startup:" >&2
    tail -n 40 "$LOG" >&2
    exit 1
  fi
  echo -n "."
  sleep 2
done
echo
echo "timed out waiting for the server; check $LOG" >&2
exit 1
