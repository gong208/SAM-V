#!/usr/bin/env bash
# Entry point for the self-contained SAM-V deploy image.
#
# Code + deps are baked into the image. Model weights are NOT (they are large and
# license-restricted), so they are supplied at runtime. This script links the base
# SAM / VGGT weights into the paths the code expects, from either a mounted
# /weights directory or explicit env vars, then runs the given command.
#
# Provide weights one of two ways:
#   1) Mount a dir containing model.pt and sam_vit_h_4b8939.pth at /weights:
#        -v /host/weights:/weights
#   2) Or point env vars at specific files:
#        -e VGGT_WEIGHT=/path/model.pt -e SAM_WEIGHT=/path/sam_vit_h_4b8939.pth
#
# The trained fusion checkpoint is passed to the script via --sam_v_ckpt.
set -euo pipefail

ROOT=/workspace
VGGT_DST="$ROOT/submodules/vggt/checkpoints/model.pt"
SAM_DST="$ROOT/submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth"
mkdir -p "$(dirname "$VGGT_DST")" "$(dirname "$SAM_DST")"

link() {  # src dst — no-op if src missing; always returns 0 (set -e safe)
  if [ -e "$1" ]; then
    ln -sf "$1" "$2"
    echo "[entrypoint] linked $2 -> $1"
  fi
}
# /weights convention (by conventional filenames)
link /weights/model.pt "$VGGT_DST"
link /weights/sam_vit_h_4b8939.pth "$SAM_DST"
# explicit env overrides win
if [ -n "${VGGT_WEIGHT:-}" ]; then link "$VGGT_WEIGHT" "$VGGT_DST"; fi
if [ -n "${SAM_WEIGHT:-}" ];  then link "$SAM_WEIGHT"  "$SAM_DST"; fi

if [ ! -e "$VGGT_DST" ]; then
  echo "[entrypoint] WARNING: VGGT weight not found at $VGGT_DST (mount /weights/model.pt or set VGGT_WEIGHT)" >&2
fi
if [ ! -e "$SAM_DST" ]; then
  echo "[entrypoint] WARNING: SAM weight not found at $SAM_DST (mount /weights/sam_vit_h_4b8939.pth or set SAM_WEIGHT)" >&2
fi

exec "$@"
