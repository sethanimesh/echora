#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${ECHORA_POD_HOST:?Set ECHORA_POD_HOST}"
: "${ECHORA_POD_PORT:?Set ECHORA_POD_PORT}"
: "${ECHORA_POD_USER:=root}"
: "${ECHORA_POD_KEY:?Set ECHORA_POD_KEY}"

SSH=(ssh -o IdentitiesOnly=yes -i "$ECHORA_POD_KEY" -p "$ECHORA_POD_PORT")
RSYNC_SHELL="ssh -o IdentitiesOnly=yes -i $ECHORA_POD_KEY -p $ECHORA_POD_PORT"

"${SSH[@]}" "$ECHORA_POD_USER@$ECHORA_POD_HOST" "mkdir -p /workspace/echora"
rsync -az --info=progress2 -e "$RSYNC_SHELL" \
  "$ROOT/backend" \
  "$ROOT/models" \
  "$ROOT/deploy" \
  "$ECHORA_POD_USER@$ECHORA_POD_HOST:/workspace/echora/"

echo "ECHORA_POD_TRANSFER_DONE"
