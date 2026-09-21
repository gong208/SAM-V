#!/usr/bin/env bash
# Initialise the git submodules SAM-V depends on, at their pinned commits:
#   submodules/vggt       facebookresearch/vggt
#   submodules/sam-hq     patched fork (plain MaskDecoder, no HQ-only args)
#   submodules/sam2       facebookresearch/sam2        (SAM2 baseline only)
#   submodules/panst3r    naver/panst3r                 (baseline only)
#   submodules/point-sam  zyc00/Point-SAM               (baseline only)
#   submodules/odin       ayushjain1144/odin            (baseline only)
#
# Safe to re-run. Run from anywhere.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "[setup] initializing submodules under $PROJECT_ROOT/submodules ..."
git -C "$PROJECT_ROOT" submodule update --init --recursive
echo "[setup] done."

# Model weights are NOT in git. Download them into the submodule checkpoint dirs:
#   submodules/vggt/checkpoints/model.pt                  (VGGT-1B)
#   submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth    (SAM vit_h)
# See README.md for the links.
