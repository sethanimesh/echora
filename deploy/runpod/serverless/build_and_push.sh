#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
: "${ECHORA_IMAGE:?Set ECHORA_IMAGE, for example docker.io/name/echora-asr:v1}"

docker buildx build \
  --platform linux/amd64 \
  --file "$ROOT/deploy/runpod/serverless/Dockerfile" \
  --tag "$ECHORA_IMAGE" \
  --push \
  "$ROOT"

echo "ECHORA_SERVERLESS_IMAGE_PUSHED=$ECHORA_IMAGE"
