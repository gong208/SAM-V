#!/usr/bin/env python3
"""
PanSt3R vocabulary resolution.

PanSt3R's classifier is open-vocabulary in a narrow sense: instead of a fixed
C-way head it embeds a runtime class-name list with SigLIP and scores each query
by cosine similarity.  The masks are vocabulary-independent (a fixed set of 200
learnable queries), but the **class scores gate post-processing** -- they decide
which queries survive `cls_threshold` and they weight the per-pixel argmax
(`postprocess.py:41-42,63`).  So the vocabulary decides *which* of the 200
queries survive, which is what drives recall and T-mIoU.

Candidates (see `notes/plans/baselines.md`):

| ID | `--vocab`       | content                                          | eligibility |
|----|-----------------|--------------------------------------------------|-------------|
| V0 | `demo_default`  | upstream demo's default: `CLASS_NAMES['scannet']` | **main row** |
| V1 | `train_union`   | union of the shipped training label spaces       | appendix    |
| V2 | `split_labels`  | ScanNet++100 (scannetpp) / ScanNet-20 (scannet)  | appendix    |
| V3 | `generic`       | `["object"]`                                     | diagnostic  |
| V4 | `scene_gt`      | the scene's own annotated categories             | **oracle**  |

**V0 `demo_default` is the only main-row-eligible value** (decision 2026-08-25).
The survivor gate is `sigmoid(logit_scale*cos).max > 0.1` with no softmax
denominator, so a longer class list monotonically raises how many of the 200
queries survive; since unmatched predictions are not penalised by the frozen
scorer, vocabulary size is a free recall knob.  SAM-V never sees a class name
and has no counterpart to tune, so the reported PanSt3R configuration is
upstream's own default rather than one selected on the evaluation split.  V1's
`train_union` is additionally a list *we* assembled, not one upstream
publishes.  V1/V2 stay so the appendix vocabulary-sensitivity numbers remain
re-derivable; they are not used in new runs.

V0, V1 and V2 are the same list for every scene and are fixed before seeing any
data, so none leaks.  **V4 does leak** -- it varies with the contents of the
scene being scored -- and is appendix-only, daggered.

A resolved vocabulary is always recorded as the explicit list *plus* its
sha256, never as the flag name alone: the underlying lists can be edited, and
CLAUDE.md §6 requires a result to be able to state what produced it.

Ordering is **sorted**, never `set()` iteration order.  The upstream demo builds
its list with `list(set(...))` (`tools/demo_panst3r.py:224-228`), which is not
reproducible across runs and would make `category_id` meaningless; copying that
idiom would silently break reproducibility.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPO_ROOT / "submodules" / "panst3r" / "tools" / "demo_panst3r.py"

VOCAB_CHOICES = ("demo_default", "train_union", "split_labels", "generic", "scene_gt")

# V0: what `tools/demo_panst3r.py` runs out of the box -- `run_model(...,
# class_set=['scannet'])` at :190, the Gradio control at :721 defaulting to
# `value=['scannet']` and labelled "ScanNet++ (100)".  Split-independent by
# construction: the same 100 names are used for scannet and scannetpp.
#
# The list is parsed out of a submodule file an upstream update could change
# underneath us, and a silent change there would invalidate every published row
# without touching a single command line -- so the resolution is pinned to this
# hash and `resolve_vocab` raises if it ever moves.
DEMO_DEFAULT_KEY = "scannet"
DEMO_DEFAULT_SHA256 = "e26d33824b28923f854aa42490181901cdadad26dc9a5f2a851e23a837c2c936"
DEMO_DEFAULT_SIZE = 100

# V3: there is no class-agnostic *mode* -- the classifier requires a non-empty
# class_names.  A single generic name is the closest approximation; its
# behaviour is itself diagnostic (see notes/plans/baselines.md).
GENERIC_VOCAB: List[str] = ["object"]

# V2 for the scannet split.  The ScanNet-20 benchmark label set is not shipped
# in the PanSt3R repo (only ScanNet++100 / COCO / ADE20k are), so it is written
# out explicitly here.  Note `otherfurniture` is a benchmark bookkeeping label
# rather than a natural language phrase; it is kept because dropping it would
# make this something other than ScanNet-20.
SCANNET20: List[str] = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]


def load_upstream_class_names(demo_path: Path = DEMO_PATH) -> Dict[str, List[str]]:
    """
    Parse the `CLASS_NAMES` dict literal out of `tools/demo_panst3r.py`.

    Parsed rather than imported: importing the demo module pulls in gradio,
    viser and matplotlib for a dict of strings.  Keys are `scannet`
    (ScanNet++ top-100, despite the name), `coco` (133), `ade20k` (150).
    """
    src = Path(demo_path).read_text()
    marker = "CLASS_NAMES = {"
    start = src.index(marker)
    end = src.index("\n}\n", start)
    literal = src[start + len("CLASS_NAMES = "):end + 2]
    names = ast.literal_eval(literal)
    if not isinstance(names, dict) or not names:
        raise ValueError(f"Could not parse CLASS_NAMES from {demo_path}")
    return {k: list(v) for k, v in names.items()}


def scene_gt_class_names(scene_dir: Path) -> List[str]:
    """
    V4: the exact category set of the scene's own per-frame ISAT annotations.

    Deliberately **not** the scene's `isat.yaml`: measured 2026-08-23, that file
    is the annotator's working palette, not a derived index -- it carries
    leftovers from an unrelated project (`giraffe`, `cake`, `sky` in indoor
    scenes) and on `scannet/scene0762_00` is *missing* four categories that are
    actually annotated.  Using it would understate the oracle on some scenes and
    inflate the vocabulary on others.
    """
    image_dir = Path(scene_dir) / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No 'images' directory in {scene_dir}")
    categories = set()
    n_files = 0
    for path in sorted(image_dir.glob("*.json")):
        with path.open() as f:
            payload = json.load(f)
        n_files += 1
        for obj in payload.get("objects", []):
            category = obj.get("category")
            if category and category != "__background__":
                categories.add(str(category))
    if n_files == 0:
        raise FileNotFoundError(f"No ISAT annotation json found in {image_dir}")
    if not categories:
        raise ValueError(f"No annotated categories found in {image_dir}")
    return sorted(categories)


def vocab_sha256(class_names: Sequence[str]) -> str:
    """Hash of the exact resolved list, order included."""
    payload = json.dumps(list(class_names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_vocab(
    vocab: Optional[str] = None,
    vocab_file: Optional[Path] = None,
    scene_dir: Optional[Path] = None,
    split: Optional[str] = None,
    demo_path: Path = DEMO_PATH,
) -> Tuple[List[str], Dict[str, object]]:
    """
    Resolve `--vocab` / `--vocab-file` to an explicit sorted list plus the
    metadata that goes into `tracks.json`'s `config` and `notes/results.md`.
    """
    if (vocab is None) == (vocab_file is None):
        raise ValueError("pass exactly one of vocab / vocab_file")

    source: str
    components: Dict[str, int] = {}

    if vocab_file is not None:
        names = [
            line.strip()
            for line in Path(vocab_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        source = f"file:{vocab_file}"
        vocab_id = "custom"
    elif vocab == "demo_default":
        # V0.  The upstream demo's own default class set, used unchanged for
        # both splits and both checkpoints.  Not split-dependent: `split` is
        # recorded in the run config but must not alter what this resolves to.
        names = sorted(load_upstream_class_names(demo_path)[DEMO_DEFAULT_KEY])
        source = (f"CLASS_NAMES['{DEMO_DEFAULT_KEY}'] (upstream demo default, "
                  f"ScanNet++ top-100) in {demo_path.name}")
        vocab_id = "demo_default"
    elif vocab == "train_union":
        # V1.  Only 3 of PanSt3R's 5 training label spaces ship in the repo
        # (ScanNet++100, COCO, ADE20k); ASE44 and Infinigen76 do not.  The
        # 3-way union is 316 unique names against the plan's predicted ~400.
        # Fixed and GT-independent either way, which is what main-row
        # eligibility requires.
        upstream = load_upstream_class_names(demo_path)
        union = set()
        for key, values in upstream.items():
            components[key] = len(values)
            union |= set(values)
        names = sorted(union)
        source = f"3-way union of CLASS_NAMES{sorted(upstream)} in {demo_path.name}"
        vocab_id = "train_union"
    elif vocab == "split_labels":
        # V2.  Per-split, but split-level -- the same list for every scene of a
        # split, known before seeing any scene.  Does not leak.
        if split not in ("scannetpp", "scannet"):
            raise ValueError(f"--vocab split_labels needs split in scannetpp|scannet, got {split!r}")
        if split == "scannetpp":
            names = sorted(load_upstream_class_names(demo_path)["scannet"])
            source = f"CLASS_NAMES['scannet'] (ScanNet++ top-100) in {demo_path.name}"
        else:
            names = sorted(SCANNET20)
            source = "ScanNet-20 benchmark label set (vocab.SCANNET20)"
        vocab_id = f"split_labels:{split}"
    elif vocab == "generic":
        names = list(GENERIC_VOCAB)
        source = "vocab.GENERIC_VOCAB"
        vocab_id = "generic"
    elif vocab == "scene_gt":
        # V4.  ORACLE -- appendix only, daggered.
        if scene_dir is None:
            raise ValueError("--vocab scene_gt needs a scene_dir")
        names = scene_gt_class_names(scene_dir)
        source = f"per-frame ISAT annotation categories of {Path(scene_dir).name}"
        vocab_id = f"scene_gt:{Path(scene_dir).name}"
    else:
        raise ValueError(f"unknown --vocab {vocab!r}; choose from {VOCAB_CHOICES}")

    # Deduplicate while staying sorted; the classifier indexes into this list.
    names = sorted(dict.fromkeys(names))
    if not names:
        raise ValueError(f"resolved vocabulary is empty (vocab={vocab!r}, file={vocab_file!r})")

    if vocab == "demo_default":
        # Pinned: see DEMO_DEFAULT_SHA256.  Fail loudly rather than publish a
        # row whose vocabulary silently changed under an upstream update.
        got = vocab_sha256(names)
        if len(names) != DEMO_DEFAULT_SIZE or got != DEMO_DEFAULT_SHA256:
            raise ValueError(
                "demo_default vocabulary does not match its pin: "
                f"C={len(names)} (expected {DEMO_DEFAULT_SIZE}), sha256={got} "
                f"(expected {DEMO_DEFAULT_SHA256}). "
                f"{demo_path} changed; do not publish rows until this is resolved."
            )

    meta: Dict[str, object] = {
        "vocab": vocab if vocab is not None else "custom",
        "vocab_id": vocab_id,
        "vocab_size": len(names),
        "vocab_sha256": vocab_sha256(names),
        "vocab_source": source,
        "vocab_gt_derived": vocab == "scene_gt",
        "vocab_class_names": names,
    }
    if components:
        meta["vocab_components"] = components
    return names, meta


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Resolve and print a PanSt3R vocabulary.")
    p.add_argument("--vocab", choices=VOCAB_CHOICES)
    p.add_argument("--vocab-file", type=Path)
    p.add_argument("--scene_dir", type=Path)
    p.add_argument("--split", choices=["scannetpp", "scannet"])
    p.add_argument("--show", action="store_true", help="print the full list")
    args = p.parse_args()

    names, meta = resolve_vocab(args.vocab, args.vocab_file, args.scene_dir, args.split)
    printable = {k: v for k, v in meta.items() if k != "vocab_class_names"}
    print(json.dumps(printable, indent=2))
    if args.show:
        print("\n".join(names))


if __name__ == "__main__":
    main()
