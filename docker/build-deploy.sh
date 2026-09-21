#!/usr/bin/env bash
# Build the self-contained SAM-V image (default tag sam_v:deploy).
# Requires the deps image from docker/build.sh and initialised submodules.
# Needs no network: it only copies the source onto the deps image.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-sam_v:deploy}"
BASE_IMAGE="${BASE_IMAGE:-sam_v:latest}"

docker build -t "$IMAGE" -f "$HERE/Dockerfile.deploy" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" "$ROOT"
echo "Built $IMAGE"
