#!/usr/bin/env bash
set -euo pipefail

: "${SCAIL2_MODEL_DIR:?Set SCAIL2_MODEL_DIR to the host model directory}"
: "${SCAIL2_INPUT_DIR:?Set SCAIL2_INPUT_DIR to the host input directory}"
: "${SCAIL2_OUTPUT_DIR:?Set SCAIL2_OUTPUT_DIR to the host output directory}"
: "${SCAIL2_GPU_0:?Set SCAIL2_GPU_0 to the first host GPU index}"
: "${SCAIL2_GPU_1:?Set SCAIL2_GPU_1 to the second host GPU index}"

SCAIL2_IMAGE=${SCAIL2_IMAGE:-localhost/scail2-inference:0.1.3}
SCAIL2_CONTAINER_NAME=${SCAIL2_CONTAINER_NAME:-scail2-worker-013}
SCAIL2_HTTP_BIND_ADDRESS=${SCAIL2_HTTP_BIND_ADDRESS:-0.0.0.0}
SCAIL2_HTTP_PORT=${SCAIL2_HTTP_PORT:-8000}

test -d "$SCAIL2_MODEL_DIR"
test -d "$SCAIL2_INPUT_DIR"
mkdir -p "$SCAIL2_OUTPUT_DIR"
test -f "$SCAIL2_MODEL_DIR/derived/SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors"

if podman container exists "$SCAIL2_CONTAINER_NAME"; then
  echo "Container already exists: $SCAIL2_CONTAINER_NAME" >&2
  echo "Stop/remove it explicitly or set SCAIL2_CONTAINER_NAME." >&2
  exit 1
fi

podman run --detach \
  --name "$SCAIL2_CONTAINER_NAME" \
  --restart unless-stopped \
  --ipc host \
  --publish "${SCAIL2_HTTP_BIND_ADDRESS}:${SCAIL2_HTTP_PORT}:8000" \
  --device "nvidia.com/gpu=$SCAIL2_GPU_0" \
  --device "nvidia.com/gpu=$SCAIL2_GPU_1" \
  --volume "$SCAIL2_MODEL_DIR":/models:ro \
  --volume "$SCAIL2_INPUT_DIR":/inputs:ro \
  --volume "$SCAIL2_OUTPUT_DIR":/outputs \
  --env SCAIL2_CHECKPOINT_DIR=/models \
  --env SCAIL2_DIT_CHECKPOINT=/models/derived/SCAIL-2-lightx2v-r128-dpo-alpha1-full-bf16.safetensors \
  --env SCAIL2_PROFILE=scail2-512p-bf16-v1 \
  --env SCAIL2_HTTP_HOST=0.0.0.0 \
  --env SCAIL2_OUTPUT_AUDIO_MODE=driving \
  "$SCAIL2_IMAGE"

echo "Started $SCAIL2_CONTAINER_NAME"
echo "Local health URL: http://127.0.0.1:${SCAIL2_HTTP_PORT}/v1/health"
if [[ "$SCAIL2_HTTP_BIND_ADDRESS" == "0.0.0.0" ]]; then
  echo "Remote URL: http://<deployment-host-IP>:${SCAIL2_HTTP_PORT}/v1/health"
fi
