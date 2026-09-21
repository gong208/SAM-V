#!/usr/bin/env bash
# Run a container with the project dir bind-mounted and GPUs attached.
#
#   ./docker/run.sh                 -> interactive bash inside the container
#   ./docker/run.sh python foo.py   -> run a one-off command, then exit
#
# The project is mounted at /workspace, so edits on the host are seen instantly
# in the container and vice versa. Set DATA_DIR to also mount a dataset root:
#
#   DATA_DIR=/mnt/datasets ./docker/run.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-sam_v:latest}"
GPUS="${GPUS:-all}"
DATA_DIR="${DATA_DIR:-}"

mounts=( -v "$PROJECT_ROOT":/workspace )
[ -n "$DATA_DIR" ] && mounts+=( -v "$DATA_DIR":"$DATA_DIR" )

# -it only when attached to a terminal, so the script also works from CI / pipes.
tty_args=(); [ -t 0 ] && [ -t 1 ] && tty_args=(-it)

docker run --rm "${tty_args[@]}" \
  --gpus "$GPUS" \
  --shm-size=16g \
  "${mounts[@]}" \
  -w /workspace \
  -e PYTHONPATH=/workspace:/workspace/submodules/sam-hq:/workspace/submodules/vggt \
  "$IMAGE" "$@"
