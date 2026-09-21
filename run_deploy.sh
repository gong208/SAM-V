#!/usr/bin/env bash
# Run the self-contained sam_v:deploy image with GPUs + weights + datasets.
#
# Usage:
#   ./run_deploy.sh                       -> interactive bash inside the baked image
#   ./run_deploy.sh python some_script.py -> run a one-off command, then exit
#
# Weights are linked in by docker/entrypoint.sh from the mounted /weights dir
# (model.pt + sam_vit_h_4b8939.pth). The trained fusion head is passed per-run via
# --sam_v_ckpt, e.g.:
#   ./run_deploy.sh python benchmarks/sam_vggt_3dtracking_benchmark.py
#       --sam_v_ckpt /weights/sam_v.pth ...
#
# Modes:
#   default            -> code baked in the image (immutable, reproducible)
#   DEV=1 ./run_deploy.sh -> bind-mount this host dir over the baked code for live edits
#
# Env overrides:
#   IMAGE   (default sam_v:deploy)
#   GPUS    (default all)            e.g. GPUS='"device=0,1"'
#   WEIGHTS (default ./checkpoints)   host dir mounted read-write at /weights
#   DATA_DIR (unset)                  optional dataset root, mounted at the same path
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-sam_v:deploy}"
GPUS="${GPUS:-all}"
WEIGHTS="${WEIGHTS:-$HERE/checkpoints}"
DATA_DIR="${DATA_DIR:-}"

mounts=( -v "$WEIGHTS":/weights )
[ -n "$DATA_DIR" ] && mounts+=( -v "$DATA_DIR":"$DATA_DIR" )

# DEV=1 mounts the host source over the baked code so edits are live on both sides.
if [ "${DEV:-0}" = "1" ]; then
  mounts+=( -v "$HERE":/workspace )
fi

# -it only when attached to a terminal, so the script also works from CI / pipes.
tty_args=(); [ -t 0 ] && [ -t 1 ] && tty_args=(-it)

docker run --rm "${tty_args[@]}" \
  --gpus "$GPUS" \
  --shm-size=16g \
  "${mounts[@]}" \
  "$IMAGE" "$@"
