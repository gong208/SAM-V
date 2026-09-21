#!/usr/bin/env bash
# Fresh-environment install check for the SAM-V release.
#
# Run this on a machine WITH network access.
#
#   ./tools/check_clean_install.sh <repo-url-or-path> [scratch-dir]
#
# Set BRANCH= to test a branch other than the repo's default. This matters when
# testing a LOCAL path: `git clone /path/to/repo` checks out that repo's HEAD,
# which is whatever branch it happens to be on -- not necessarily the release
# branch. The script refuses to continue if the tree it cloned has no release
# content, rather than reporting a green run against the wrong branch.
#
#   BRANCH=release-prep ./tools/check_clean_install.sh /root/sam_vggt
#
# NOTE: this script makes its OWN clone into a scratch directory and tests that.
# Whatever checkout you launch it from is only the delivery mechanism for the
# script -- it is not what gets tested, and it is never modified. That is
# deliberate: the point is to see what a stranger gets, not what you have.
#
# Because it clones, a local <repo-url-or-path> is tested at its COMMITTED state.
# Uncommitted work in your checkout will not be exercised.
#
# It initialises the submodules, builds a fresh venv from the pinned
# requirements, and verifies that every entry point imports and every config
# loads. It never touches your existing environment.
#
# Weights and datasets are NOT downloaded; the script stops short of anything
# that needs them and says so.

set -uo pipefail

REPO="${1:?usage: [BRANCH=<branch>] check_clean_install.sh <repo-url-or-path> [scratch-dir]}"
WORK="${2:-$(mktemp -d)/samv-install-check}"
BRANCH="${BRANCH:-}"
# The pins target Python 3.10 (numpy 1.26.1 / torch 2.3.1). Prefer it, but fall
# back to whatever python3 exists rather than failing on a box that ships 3.12.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif command -v python3.10 >/dev/null; then PY=python3.10
else PY=python3
fi

pass=0; fail=0
step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  PASS  %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

step "0. environment"
command -v "$PY" >/dev/null || { echo "  $PY not found; set PYTHON=/path/to/python"; exit 1; }
echo "  python : $($PY --version 2>&1)  [$PY]"
case "$($PY --version 2>&1)" in
  *" 3.10."*) ;;
  *) echo "  NOTE   : the pins were built for Python 3.10; this is not 3.10, so a"
     echo "           resolution failure below may be the version, not the pins."
     echo "           Re-run with PYTHON=python3.10 if you have it." ;;
esac
echo "  git    : $(git --version)"
echo "  target : $WORK"

step "1. clone with submodules"
rm -rf "$WORK"; mkdir -p "$(dirname "$WORK")"
if [ -n "$BRANCH" ]; then
  echo "  branch : $BRANCH (explicit)"
  git clone --branch "$BRANCH" --recurse-submodules "$REPO" "$WORK" 2>&1 | tail -3
else
  echo "  branch : (repo default)"
  git clone --recurse-submodules "$REPO" "$WORK" 2>&1 | tail -3
fi
if [ -d "$WORK/.git" ]; then ok "clone --recurse-submodules"; else bad "clone"; exit 1; fi
cd "$WORK" || exit 1
echo "  got    : $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

step "1b. the cloned tree actually contains the release"
missing=""
for f in LICENSE NOTICE requirements.txt tools/export_release_checkpoint.py \
         configs/train/hypersim_pretrain.yaml configs/paper/scannetpp_finetune.yaml \
         utils/provenance.py utils/checkpoint.py; do
  [ -e "$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  bad "cloned tree is missing release files:$missing"
  echo
  echo "  You almost certainly cloned the wrong branch. A local path clones that"
  echo "  repo's CURRENT branch, which for /root/sam_vggt is 'main'. Re-run with:"
  echo "      BRANCH=release-prep $0 $REPO"
  exit 1
fi
ok "release files present"

step "2. every submodule populated at its pinned commit"
git submodule status | while read -r line; do
  case "$line" in
    -*) echo "  EMPTY   $line" ;;
    +*) echo "  WRONG   $line (not at the pinned commit)" ;;
    *)  echo "  ok      $line" ;;
  esac
done
if git submodule status | grep -q '^[-+]'; then bad "submodules"; else ok "all six submodules at pinned commits"; fi

step "3. fresh venv from the pinned requirements"
"$PY" -m venv .venv || { bad "venv creation"; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip >/dev/null
if pip install -r requirements.txt 2>&1 | tail -5; then
  ok "pip install -r requirements.txt"
else
  bad "pip install -r requirements.txt"
fi

step "4. torch sees the GPU (informational)"
python - <<'PY'
try:
    import torch
    print(f"  torch {torch.__version__}, cuda available: {torch.cuda.is_available()}"
          + (f", device: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
except Exception as e:
    print(f"  torch import failed: {e}")
PY

step "5. every entry point imports"
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
for m in \
  benchmarks/sam_vggt_3dtracking_benchmark.py \
  benchmarks/compare_baseline_sam2.py \
  benchmarks/timing/benchmark_inference_time.py \
  training/trainer.py \
  demos/infer_custom_images.py \
  masks/everything_mode_demo.py \
  tools/export_release_checkpoint.py
do
  if python "$m" --help >/dev/null 2>&1; then ok "$m --help"; else bad "$m --help"; fi
done

step "6. configs load, and name the env vars they need"
python - <<'PY'
import os, sys
from pathlib import Path
# `import training` resolves to the vggt submodule's own training package (it has
# an __init__.py, ours does not), so use the same idiom training/trainer.py uses:
# put training/ on the path and import config directly. See trainer.py:46.
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "training"))
from config import load_config
os.environ.setdefault("SCANNETPP_ROOT", "/tmp/x")
os.environ.setdefault("HYPERSIM_ROOT", "/tmp/x")
os.environ.setdefault("STAGE1_CHECKPOINT", "/tmp/x.pth")
bad = 0
for cfg in sorted(Path("configs").rglob("*.yaml")):
    try:
        c = load_config(str(cfg))
        roots = [d.get("root") for d in c.datasets]
        assert not any("${" in str(r) for r in roots), f"unexpanded: {roots}"
        print(f"  PASS  {cfg}")
    except Exception as e:
        print(f"  FAIL  {cfg}: {e}"); bad += 1
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] && ok "all configs load" || bad "config loading"

step "summary"
echo "  $pass passed, $fail failed"
echo
echo "  NOT covered here (they need weights and data):"
echo "    - downloading VGGT-1B and SAM ViT-H into submodules/*/checkpoints/"
echo "    - downloading the SAM-V release checkpoint"
echo "    - any actual inference or the benchmark itself"
echo "  Scratch dir left at: $WORK"
exit $(( fail > 0 ))
