#!/usr/bin/env python3
"""
The ODIN closed-vocabulary ceiling on the IGGT 3D-tracking benchmark.

ODIN is a closed-vocabulary segmenter: every mask it emits is emitted under a
label drawn from a fixed class table.  A GT object whose category has no name in
that table can still be *proposed* (see "soft, not hard" below), but nothing in
ODIN is trained to name it, so the fraction of the benchmark's GT objects the
table can name is the natural bound to publish beside an ODIN row -- the same
role `pointsam_ceiling.py` plays for Point-SAM's voxel sampling.

What makes this number authoritative is the **denominator**.  It is the GT
object set the *frozen* scorer enumerates and matches, obtained by importing
`load_scene_inputs` and `evaluate_scene` from
`benchmarks/sam_vggt_3dtracking_benchmark.py` and reading back the `object_ids`
that `evaluate_scene` itself returns (called with zero predictions, so nothing
but the enumeration runs).  No re-implementation, no `isat.yaml`, no
hand-curated object list.  Concretely the frozen rule is:

  * frames  = `<scene>/images/<stem>.jpg` where `<stem>` is `frame_\\d+` or
    `\\d+`, each with a sibling `<stem>_label.npy` (benchmark:101-153);
  * GT maps = those `.npy` instance maps, nearest-resized to 1024x1024
    (benchmark:203-220);
  * objects = `sorted(set(np.unique(gt_maps)) - {<=0} - ignore_ids)`
    (benchmark:625-630).

So id 0 is background, and an object that survives in no evaluated frame is not
in the denominator at all -- it never appears in the union.  Measured this way
the benchmark has **58** GT objects on ScanNet++ and **66** on ScanNet.

The scorer never reads the ISAT `*.json` files, so *category names* have to come
from somewhere else.  They come from those same ISAT jsons: each annotated
object carries `group` (the instance id written into `_label.npy`, cf.
`visualization/visualize_3dtrackingbenchmark.py:136-155`) and `category`.  The
script asserts, per scene, that the json's non-`__background__` group set is
*exactly* the frozen object-id set, and fails loudly otherwise -- a silent
mismatch there would put names against the wrong objects.  `isat.yaml` is
deliberately not used: it is the annotator's working palette, not a derived
index (the same finding recorded in `benchmarks/baselines/vocab.py`).

Matching rule -- three tiers, all applied mechanically, all reported:

  T1 `strict`    casefold + collapse whitespace, then exact string equality.
  T2 `lexical`   T1 + mechanical morphology (open/close-compound spacing,
                 trailing annotator index digits, a small regular
                 plural->singular rule) + a short, published equivalence table
                 of same-referent names.  **This is the headline rule.**
  T3 `head_noun` T2 + head-noun match in either direction (a multi-word GT name
                 matches on its last token; a single-token GT name matches a
                 vocabulary phrase whose last token it is).  Upper bound only.

Normalisation is applied to *both* sides, so the vocabulary is normalised by the
same code as the GT names.  Catch-all bookkeeping labels are excluded as match
targets (`otherfurniture` in ScanNet-20, `object` in ScanNet200): they would
"cover" anything and mean nothing.  The equivalence table is small and each
entry is justified in `notes/plans/odin-vocab-ceiling.md`; it is written down
once and never tuned against a resulting number.

Class tables are read from upstream `odin/global_vars.py` by AST literal
evaluation (never executed) and pinned by sha256, so an upstream edit fails the
run instead of silently moving a published number -- the same pin discipline as
`vocab.py`'s `DEMO_DEFAULT_SHA256`.

Runs in the project env with the mandatory PYTHONPATH, no GPU::

    export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:\\
$PWD/submodules/vggt"
    python benchmarks/baselines/odin_vocab_ceiling.py \\
        --benchmark_root $BENCH \\
        --odin_root $PWD/submodules/odin \\
        --output_dir results/baselines/odin_vocab_ceiling
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.provenance import write_run_provenance  # noqa: E402
from benchmarks.sam_vggt_3dtracking_benchmark import (  # noqa: E402
    TARGET_SIZE,
    discover_scene_dirs,
    evaluate_scene,
    load_scene_inputs,
)

SPLITS = ("scannetpp", "scannet")
TIERS = ("strict", "lexical", "head_noun")

# --------------------------------------------------------------------------
# ODIN class tables
# --------------------------------------------------------------------------
# Pins over the *raw literals* as parsed out of odin/global_vars.py at upstream
# 25f5b85 ("merge").  A vocabulary that changed underneath a published ceiling
# would invalidate the row without touching a command line.
_PINS = {
    "NAME_MAP20": "4c88a44b9169a942eb4f0c4a7f30871d464e769b83442a8c70eaeaffd671429b",
    "SCANNET_CLASS_LABELS_200": "f8cf3fcb99840873357ebe6958eae29378b3cb79a2ad8c43a96412ecfcd73092",
}

# SKIP_CLASSES as shipped in upstream's own reference configs.  1-indexed into
# the class table; `odin_model.py:1268` subtracts 1.
SKIP_CLASSES = {
    "scannet20": [19, 20],       # scripts/scannet/scannet_swin.sh:26  -> wall, floor
    "scannet200": [119, 200],    # scripts/scannet200/scannet200_swin.sh:26
}

# Bookkeeping catch-alls: present in the table, but matching a real object name
# against them would be meaningless.
CATCH_ALL_TARGETS = {"otherfurniture", "object"}

# --------------------------------------------------------------------------
# The matching rule
# --------------------------------------------------------------------------
# T2 equivalence classes.  Each is a set of names treated as the same referent.
# Every entry is justified in notes/plans/odin-vocab-ceiling.md.  Kept
# deliberately small; names not listed here are matched lexically or not at all.
EQUIVALENCE_CLASSES: Tuple[FrozenSet[str], ...] = (
    frozenset({"sofa", "couch"}),                                   # dictionary synonyms
    frozenset({"trashcan", "trash can", "trash bin", "trashbin"}),  # compound spacing
    frozenset({"landlinephone", "landline phone", "telephone"}),    # spacing + synonym
    frozenset({"refridgerator", "refrigerator", "fridge"}),         # upstream misspelling
    frozenset({"tv", "television"}),                                # abbreviation
    frozenset({"whiteboard", "white board"}),                       # compound spacing
)

_WS_RE = re.compile(r"\s+")
_INDEX_SUFFIX_RE = re.compile(r"^(?P<stem>.*[^\W\d_])\d+$")


def normalise_strict(name: str) -> str:
    """T1: casefold, punctuation-to-space, collapse whitespace."""
    text = str(name).replace("_", " ").replace("-", " ").strip().casefold()
    return _WS_RE.sub(" ", text)


def _singularise(token: str) -> str:
    """A small regular plural rule.  Applied to both GT and vocabulary names."""
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("ves"):
        return token[:-3] + "f"
    if token.endswith(("sses", "ches", "shes", "xes", "zes")):
        return token[:-2]
    return token[:-1]


def normalise_lexical(name: str) -> str:
    """T2 normalisation: T1 + index-suffix strip + singularise the last token."""
    text = normalise_strict(name)
    if not text:
        return text
    tokens = text.split(" ")
    m = _INDEX_SUFFIX_RE.match(tokens[-1])
    if m is not None:
        tokens[-1] = m.group("stem")
    tokens[-1] = _singularise(tokens[-1])
    return " ".join(t for t in tokens if t)


def _equivalence_index() -> Dict[str, int]:
    index: Dict[str, int] = {}
    for i, klass in enumerate(EQUIVALENCE_CLASSES):
        for member in klass:
            index[normalise_lexical(member)] = i
    return index


_EQUIV_INDEX = _equivalence_index()


def lexical_keys(name: str) -> Set[str]:
    """Every T2 key a name resolves to (its own form + its equivalence class)."""
    base = normalise_lexical(name)
    keys = {base}
    klass = _EQUIV_INDEX.get(base)
    if klass is not None:
        keys |= {normalise_lexical(m) for m in EQUIVALENCE_CLASSES[klass]}
    return keys


def head_noun(name: str) -> str:
    text = normalise_lexical(name)
    return text.split(" ")[-1] if text else text


class Vocabulary:
    """One ODIN class table, pre-normalised at every tier."""

    def __init__(self, vocab_id: str, names: Sequence[str], skipped: Sequence[str]) -> None:
        self.vocab_id = vocab_id
        self.names = list(names)
        self.skipped = list(skipped)
        targets = [n for n in names if normalise_strict(n) not in CATCH_ALL_TARGETS]
        self.excluded_catch_alls = [n for n in names if normalise_strict(n) in CATCH_ALL_TARGETS]
        self.targets = targets
        self.strict: Set[str] = {normalise_strict(n) for n in targets}
        self.lexical: Set[str] = set()
        for n in targets:
            self.lexical |= lexical_keys(n)
        self.head_nouns: Set[str] = {head_noun(n) for n in targets}

    def match(self, gt_name: str, tier: str) -> Optional[str]:
        """Return the vocabulary entry `gt_name` is covered by at `tier`, or None."""
        if tier == "strict":
            key = normalise_strict(gt_name)
            for n in self.targets:
                if normalise_strict(n) == key:
                    return n
            return None
        keys = lexical_keys(gt_name)
        for n in self.targets:
            if lexical_keys(n) & keys:
                return n
        if tier == "lexical":
            return None
        if tier != "head_noun":
            raise ValueError(f"unknown tier {tier!r}")
        gt_head = head_noun(gt_name)
        for n in self.targets:
            if head_noun(n) == gt_head:
                return n
        return None


# --------------------------------------------------------------------------
# Reading the upstream tables
# --------------------------------------------------------------------------

def _literal_assignment(source: str, target_name: str, path: Path) -> Any:
    """AST-evaluate one module-level literal assignment.  Never executes the module."""
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == target_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"{target_name} not found as a module-level literal in {path}")


def sha256_of(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_odin_vocabularies(
    odin_root: Path,
    pins: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Vocabulary], Dict[str, str]]:
    """Both ODIN class tables, SKIP_CLASSES applied, pinned by sha256."""
    gv_path = Path(odin_root) / "odin" / "global_vars.py"
    if not gv_path.is_file():
        raise FileNotFoundError(
            f"{gv_path} not found.  Pass --odin_root (git worktrees have empty "
            f"submodule dirs; the ODIN clone lives in the main checkout)."
        )
    source = gv_path.read_text()
    name_map20 = _literal_assignment(source, "NAME_MAP20", gv_path)
    labels200 = _literal_assignment(source, "SCANNET_CLASS_LABELS_200", gv_path)

    digests = {
        "NAME_MAP20": sha256_of({str(k): v for k, v in name_map20.items()}),
        "SCANNET_CLASS_LABELS_200": sha256_of(list(labels200)),
    }
    if pins:
        for key, expected in pins.items():
            if digests[key] != expected:
                raise ValueError(
                    f"{key} in {gv_path} does not match its pin: got {digests[key]}, "
                    f"expected {expected}.  Upstream changed; do not publish a ceiling "
                    f"until this is resolved."
                )

    tables = {
        "scannet20": {int(k): v for k, v in name_map20.items()},
        "scannet200": {i + 1: v for i, v in enumerate(labels200)},
    }
    vocabs: Dict[str, Vocabulary] = {}
    for vid, table in tables.items():
        skip_ids = SKIP_CLASSES[vid]
        skipped = [table[i] for i in skip_ids if i in table]
        kept = [v for k, v in sorted(table.items()) if k not in set(skip_ids)]
        vocabs[vid] = Vocabulary(vid, kept, skipped)
    return vocabs, digests


# --------------------------------------------------------------------------
# GT objects, straight out of the frozen scorer
# --------------------------------------------------------------------------

class GTNameMismatch(RuntimeError):
    """The ISAT jsons do not name exactly the objects the frozen scorer enumerates."""


def frozen_object_ids(scene_dir: Path, ignore_instance_ids: Sequence[int]) -> List[int]:
    """
    The GT object ids `evaluate_scene` matches, read back from `evaluate_scene`
    itself with zero predictions.  Nothing is re-implemented here.
    """
    scene_inputs = load_scene_inputs(scene_dir, tuple(TARGET_SIZE))
    metrics = evaluate_scene([], scene_inputs, ignore_instance_ids)
    return [int(o) for o in metrics["object_ids"]]


def isat_categories(scene_dir: Path) -> Dict[int, Dict[str, int]]:
    """
    `group -> {category: frame_count}` from the scene's per-frame ISAT jsons.
    `__background__` (always group 0 in this benchmark) is excluded.
    """
    image_dir = Path(scene_dir) / "images"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No 'images' directory in {scene_dir}")
    out: Dict[int, Dict[str, int]] = {}
    n_files = 0
    for path in sorted(image_dir.glob("*.json")):
        with path.open() as f:
            payload = json.load(f)
        n_files += 1
        for obj in payload.get("objects", []):
            group = obj.get("group")
            category = obj.get("category")
            if group is None or category is None or category == "__background__":
                continue
            bucket = out.setdefault(int(group), {})
            bucket[str(category)] = bucket.get(str(category), 0) + 1
    if n_files == 0:
        raise FileNotFoundError(f"No ISAT annotation json found in {image_dir}")
    return out


def scene_objects(scene_dir: Path, ignore_instance_ids: Sequence[int]) -> List[Dict[str, Any]]:
    """One record per GT object the frozen scorer enumerates, with its name(s)."""
    object_ids = frozen_object_ids(scene_dir, ignore_instance_ids)
    cats = isat_categories(scene_dir)
    named = set(cats)
    enumerated = set(object_ids)
    if named != enumerated:
        raise GTNameMismatch(
            f"{scene_dir.name}: ISAT groups != frozen scorer object ids.\n"
            f"  scorer only : {sorted(enumerated - named)}\n"
            f"  ISAT only   : {sorted(named - enumerated)}"
        )
    records: List[Dict[str, Any]] = []
    for oid in object_ids:
        counts = cats[oid]
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        records.append({
            "scene_id": scene_dir.name,
            "object_id": oid,
            "categories": {k: v for k, v in ordered},
            "primary_category": ordered[0][0],
            "ambiguous": len(ordered) > 1,
        })
    return records


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def cover_object(record: Dict[str, Any], vocab: Vocabulary, tier: str) -> Dict[str, Any]:
    """
    An object is covered iff **every** category string ever attached to it is
    covered.  Two objects in this benchmark carry two names across frames; the
    conservative reading is the one that does not let a single lucky frame
    label rescue an object.  `--ambiguity primary` scores the most frequent
    label instead, for the sensitivity check.
    """
    per_name = {name: vocab.match(name, tier) for name in record["categories"]}
    return {
        "matches": per_name,
        "covered_all": all(v is not None for v in per_name.values()),
        "covered_primary": per_name[record["primary_category"]] is not None,
    }


def measure(records: Sequence[Dict[str, Any]], vocab: Vocabulary) -> Dict[str, Any]:
    out: Dict[str, Any] = {"num_gt_objects": len(records), "tiers": {}}
    for tier in TIERS:
        covered_all = 0
        covered_primary = 0
        uncovered_names: Dict[str, int] = {}
        per_object: List[Dict[str, Any]] = []
        for rec in records:
            res = cover_object(rec, vocab, tier)
            covered_all += int(res["covered_all"])
            covered_primary += int(res["covered_primary"])
            if not res["covered_all"]:
                for name, hit in res["matches"].items():
                    if hit is None:
                        uncovered_names[name] = uncovered_names.get(name, 0) + 1
            per_object.append({
                "scene_id": rec["scene_id"],
                "object_id": rec["object_id"],
                "categories": rec["categories"],
                "covered": res["covered_all"],
                "matched_vocab_entry": res["matches"].get(rec["primary_category"]),
            })
        n = len(records)
        out["tiers"][tier] = {
            "covered": covered_all,
            "coverage": covered_all / n if n else float("nan"),
            "covered_primary_label": covered_primary,
            "coverage_primary_label": covered_primary / n if n else float("nan"),
            "uncovered_names": dict(sorted(uncovered_names.items(), key=lambda kv: (-kv[1], kv[0]))),
            "per_object": per_object,
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark_root", type=Path,
                   default=Path(os.environ.get(
                       "IGGT_BENCHMARK_ROOT",
                       os.environ.get("BENCH", ""))),
                   help="directory holding the scannetpp/ and scannet/ splits")
    p.add_argument("--odin_root", type=Path,
                   default=Path(os.environ.get("ODIN_ROOT", REPO_ROOT / "submodules" / "odin")),
                   help="upstream ODIN clone (worktrees: point at the main checkout)")
    p.add_argument("--splits", nargs="*", default=list(SPLITS), choices=list(SPLITS))
    p.add_argument("--ignore_instance_ids", nargs="*", type=int, default=[],
                   help="GT instance ids to exclude, as passed to the scorer (runs used none)")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="write ceiling.json + provenance here (default: print only)")
    p.add_argument("--check_pins", action="store_true",
                   help="fail if the ODIN class tables do not match the recorded sha256 pins")
    p.add_argument("--print_pins", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    vocabs, digests = load_odin_vocabularies(
        args.odin_root, _PINS if args.check_pins else None)
    if args.print_pins:
        print(json.dumps(digests, indent=2))

    report: Dict[str, Any] = {
        "denominator_source": (
            "evaluate_scene(annotations=[], ...)['object_ids'] from "
            "benchmarks/sam_vggt_3dtracking_benchmark.py (frozen)"
        ),
        "name_source": "per-frame ISAT <stem>.json 'group' -> 'category' (isat.yaml NOT used)",
        "odin_root": str(args.odin_root),
        "odin_table_sha256": digests,
        "skip_classes": SKIP_CLASSES,
        "skipped_names": {k: v.skipped for k, v in vocabs.items()},
        "excluded_catch_all_targets": {k: v.excluded_catch_alls for k, v in vocabs.items()},
        "equivalence_classes": [sorted(c) for c in EQUIVALENCE_CLASSES],
        "tiers": list(TIERS),
        "splits": {},
    }

    for split in args.splits:
        records: List[Dict[str, Any]] = []
        for scene_dir in discover_scene_dirs(args.benchmark_root / split, None):
            records.extend(scene_objects(scene_dir, args.ignore_instance_ids))
        split_report: Dict[str, Any] = {
            "num_gt_objects": len(records),
            "num_scenes": len({r["scene_id"] for r in records}),
            "ambiguous_objects": [
                {"scene_id": r["scene_id"], "object_id": r["object_id"],
                 "categories": r["categories"]}
                for r in records if r["ambiguous"]
            ],
            "distinct_category_names": sorted({n for r in records for n in r["categories"]}),
            "vocabularies": {vid: measure(records, vocab) for vid, vocab in vocabs.items()},
        }
        report["splits"][split] = split_report

    # ---- console table -----------------------------------------------------
    print("\n=== ODIN vocabulary ceiling (denominator = frozen evaluate_scene object_ids) ===")
    header = f"{'vocabulary':<12}{'split':<12}" + "".join(f"{t:>18}" for t in TIERS)
    print(header)
    print("-" * len(header))
    for vid in vocabs:
        for split in args.splits:
            s = report["splits"][split]["vocabularies"][vid]
            cells = "".join(
                f"{s['tiers'][t]['covered']:>7}/{s['num_gt_objects']:<3} "
                f"{s['tiers'][t]['coverage']:.3f}".rjust(18)
                for t in TIERS
            )
            print(f"{vid:<12}{split:<12}{cells}")
    for vid in vocabs:
        for split in args.splits:
            s = report["splits"][split]["vocabularies"][vid]
            names = s["tiers"]["lexical"]["uncovered_names"]
            print(f"\nuncovered @ lexical -- {vid} / {split} "
                  f"({s['num_gt_objects'] - s['tiers']['lexical']['covered']} objects, "
                  f"{len(names)} distinct names):")
            print("  " + (", ".join(f"{n} x{c}" if c > 1 else n for n, c in names.items()) or "(none)"))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_run_provenance(args.output_dir, config={
            "benchmark_root": str(args.benchmark_root),
            "odin_root": str(args.odin_root),
            "splits": list(args.splits),
            "ignore_instance_ids": list(args.ignore_instance_ids),
            "odin_table_sha256": digests,
        })
        out = args.output_dir / "ceiling.json"
        with out.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
