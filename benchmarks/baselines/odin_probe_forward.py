#!/usr/bin/env python3
"""ODIN Phase 2, stage B: one forward pass on one scene, and the overlays to judge it.

`notes/plans/odin-baseline.md` §4 Phase 2 -- the real kill-switch. Consumes the
`geometry.npz` written by `odin_geometry_probe.py` (project env) and runs ODIN's
inference path on it, once per geometry source.

**`ODIN.forward` is deliberately NOT called.** Even on the eval branch it runs
`prepare_targets` and `adjust_masks_for_highres`, which need GT annotations we do not
have (plan §6 item 5). The eval path is reimplemented here instead:

    normalize -> ImageList -> backbone(images, multi_scale_xyz, ...)
      -> sem_seg_head(...) -> inference_2d_per_image(...)

Geometry is injected as `multi_scale_xyz` directly (plan §3 option (a)): VGGT's
`world_points` already ARE per-pixel world coordinates, so ODIN's own projection math is
never invoked and no intrinsics are needed.

Runs in the ODIN env, and installs nothing:

    PYTHONPATH=$PWD/submodules/odin $ODIN_ENV/bin/python \\
        benchmarks/baselines/odin_probe_forward.py --geometry vggt ...
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Baseline checkpoints are not redistributed with this repository. Stage them
# yourself and point $BASELINE_WEIGHTS at the directory holding them; the
# per-baseline subdirectory layout is documented in benchmarks/baselines/README.md.
_BASELINE_WEIGHTS = Path(os.environ.get(
    "BASELINE_WEIGHTS", str(REPO_ROOT / "checkpoints" / "baselines")))
from utils.provenance import write_run_provenance  # noqa: E402

ODIN_ROOT = str(REPO_ROOT / "submodules" / "odin")
CFG_FILE = f"{ODIN_ROOT}/configs/scannet_context/swin_3d.yaml"
CKPT = str(_BASELINE_WEIGHTS / "odin" / "scannet200_swin_31.5_76k_5.5k.pth")

# Mirrors `scripts/scannet200/scannet200_swin.sh` minus the training-only knobs, with the
# three deliberate inference deviations of plan §1. Identical to the Phase-1 load test.
OPTS: List[str] = [
    "INPUT.FRAME_LEFT", "7",
    "INPUT.FRAME_RIGHT", "7",
    "INPUT.SAMPLING_FRAME_NUM", "15",
    "INPUT.IMAGE_SIZE", "512",
    "INPUT.MIN_SIZE_TEST", "512",
    "INPUT.MAX_SIZE_TEST", "512",
    "INPUT.VOXELIZE", "True",
    "MODEL.DECODER_3D", "True",
    "MODEL.CROSS_VIEW_CONTEXTUALIZE", "True",
    "MODEL.CROSS_VIEW_BACKBONE", "True",
    "MODEL.PIXEL_DECODER_PANET", "True",
    "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "200",
    "MODEL.MASK_FORMER.TEST.SEMANTIC_ON", "True",
    "MODEL.MASK_FORMER.DICE_WEIGHT", "6.0",
    "MODEL.MASK_FORMER.MASK_WEIGHT", "15.0",
    "MODEL.MASK_FORMER.TRAIN_NUM_POINTS", "50000",
    "MODEL.FREEZE_BACKBONE", "False",
    "USE_MLP_POSITIONAL_ENCODING", "True",
    "HIGH_RES_SUBSAMPLE", "True",
    # PHASE-2 DISCOVERY, not in the plan: the non-ghost 3D path REQUIRES this.
    # With USE_GHOST_POINTS False + INPUT.VOXELIZE True, the pixel decoder returns
    # mask features on the res2 IMAGE grid, and `odin_transformer_decoder.py:305`
    # then asserts `mask_features.shape[-1] == 1` -- i.e. it demands voxelised mask
    # features. Only the `HIGH_RES_INPUT and not training and not USE_GHOST_POINTS`
    # branch at :296 produces those. It is also what makes `pred_masks` come out as
    # [Q, n_voxels] rather than [Q, V, h, w], which is what `upsample_pred_masks`'s
    # trilinear path (and therefore `inference_2d_per_image`) consumes.
    # This is upstream's own pairing for sensor-only 3D: every non-ghost 3D script
    # (`scripts/ai2thor/*.sh`, `scripts/alfred/alfred_resnet.sh`) sets it True.
    # It is a train/test config mismatch on top of the one plan §1 already flags,
    # because the ScanNet200 checkpoint trained with HIGH_RES_INPUT False.
    "HIGH_RES_INPUT", "True",
    "SKIP_CLASSES", "[119, 200]",
    "SAMPLING_STRATEGY", "consecutive",
    "MAX_FRAME_NUM", "-1",
    "USE_WANDB", "False",
    "USE_GHOST_POINTS", "False",
    "USE_SEGMENTS", "False",
    "EVAL_PER_IMAGE", "True",
    "MODEL.WEIGHTS", "",
    "DATASETS.TRAIN", "('scannet200_context_instance_train_200cls_single_highres_100k',)",
    "DATASETS.TEST", "('scannet200_context_instance_val_200cls_single_highres_100k',)",
]


# --------------------------------------------------------------------------------------
# network + stub instrumentation (plan §7b: a forward pass is the first real test)
# --------------------------------------------------------------------------------------
NETWORK_ATTEMPTS: List[str] = []
STUB_CALLS: List[str] = []


def block_network() -> None:
    """Make any outbound connection fail loudly and be recorded.

    CLAUDE.md §4: the compute node must be treated as having no outbound internet. The
    unproven claim from Phase 1 is `odin_model.py:44`'s tokenizer imports -- nothing calls
    `from_pretrained` during construction, but that was never established past model build.
    """
    real_connect = socket.socket.connect

    def guard(self, address, *a, **kw):  # noqa: ANN001
        NETWORK_ATTEMPTS.append(repr(address))
        raise RuntimeError(f"outbound network blocked by the probe: {address!r}")

    socket.socket.connect = guard  # type: ignore[method-assign]
    _ = real_connect


def instrument_stubs() -> Dict[str, str]:
    """Record (and still refuse) any call into the three stubbed modules."""
    import pytorch3d.ops
    import pyviz3d.visualizer
    import wandb

    def wrap(mod, name: str, label: str) -> None:
        orig = getattr(mod, name)

        def rec(*a, **kw):  # noqa: ANN001
            STUB_CALLS.append(label)
            return orig(*a, **kw)

        setattr(mod, name, rec)

    wrap(pytorch3d.ops, "knn_points", "pytorch3d.ops.knn_points")
    wrap(pyviz3d.visualizer, "Visualizer", "pyviz3d.visualizer.Visualizer")
    for fn in ("init", "log", "finish"):
        wrap(wandb, fn, f"wandb.{fn}")
    return {
        "pytorch3d": pytorch3d.__file__,
        "pyviz3d": pyviz3d.__file__,
        "wandb": wandb.__file__,
    }


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------
def resize_shortest_edge(h: int, w: int, short: int, longest: int) -> Tuple[int, int]:
    """detectron2's ResizeShortestEdge geometry, reimplemented for the xyz maps.

    The images themselves go through the real transform; this mirrors it so the geometry
    lands on the same pixel grid.
    """
    scale = short / min(h, w)
    nh, nw = h * scale, w * scale
    if max(nh, nw) > longest:
        scale = scale * longest / max(nh, nw)
        nh, nw = h * scale, w * scale
    return int(nh + 0.5), int(nw + 0.5)


def build_inputs(geom_npz: Path, scene_dir: Path, which: str, scale_multiplier: float,
                 cfg, device: str) -> Dict[str, object]:
    from detectron2.data import transforms as T
    from detectron2.data import detection_utils as utils

    g = np.load(geom_npz, allow_pickle=True)
    frames = [str(x) for x in g["frames"]]
    key = {"vggt": "xyz_vggt", "gt": "xyz_gt"}[which]
    if key not in g:
        raise KeyError(f"{geom_npz} carries no '{key}' (no GT geometry for this scene)")
    xyz_native = g[key].astype(np.float32)  # [V,H,W,3]
    n, H, W, _ = xyz_native.shape

    aug = T.ResizeShortestEdge(
        cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST_SAMPLING
    )
    images = []
    for stem in frames:
        img = utils.read_image(str(scene_dir / "images" / f"{stem}.jpg"), format=cfg.INPUT.FORMAT)
        img, _ = T.apply_transform_gens([aug], img)
        images.append(torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))))
    h, w = images[0].shape[-2:]
    assert (h, w) == resize_shortest_edge(H, W, cfg.INPUT.MIN_SIZE_TEST,
                                          cfg.INPUT.MAX_SIZE_TEST), \
        f"xyz resize target {(h, w)} does not match the image transform"

    # geometry onto the resized grid, nearest -- the same interpolation ODIN uses for its
    # own depth maps (`MODEL.INTERPOLATION_METHOD`, `interpolate_depth(..., 'nearest')`).
    xyz = torch.from_numpy(xyz_native).permute(0, 3, 1, 2)
    xyz = F.interpolate(xyz, size=(h, w), mode="nearest").permute(0, 2, 3, 1)
    if scale_multiplier != 1.0:
        xyz = xyz * scale_multiplier

    return {"frames": frames, "images": images, "xyz": xyz, "native_hw": (H, W),
            "resized_hw": (h, w), "scale_multiplier": scale_multiplier,
            "scale_m_per_unit": float(g["scale_m_per_unit"]) if "scale_m_per_unit" in g else None}


def make_multi_scale(xyz_padded: torch.Tensor, hw: Tuple[int, int], device: str
                     ) -> List[torch.Tensor]:
    """[res5, res4, res3, res2] xyz, the order ODIN's head expects (coarse -> fine).

    `dataset_mapper_scannet.get_multiview_xyz` builds res2/res3/res4/res5 at strides
    4/8/16/32 of the PADDED image and then reverses the list; this is the same thing with
    the depth-unprojection step replaced by the point map we were handed.
    """
    v, hp, wp, _ = xyz_padded.shape
    out = []
    for s in (4, 8, 16, 32):
        t = F.interpolate(xyz_padded.permute(0, 3, 1, 2), size=(hp // s, wp // s),
                          mode="nearest").permute(0, 2, 3, 1)
        out.append(t[None].to(device))          # [1, V, h_s, w_s, 3]
    return out[::-1]


# --------------------------------------------------------------------------------------
# the eval path
# --------------------------------------------------------------------------------------
@torch.no_grad()
def run_forward(model, inputs: Dict[str, object], cfg, device: str,
                with_instances: bool = True) -> Dict[str, object]:
    """One ODIN eval-path forward pass.

    ``with_instances`` controls whether upstream's own ``inference_2d_per_image`` head is
    additionally called. Phase 2 wants it (it is the reference the max-over-class rule is
    compared against); Phase 3's runner passes ``False`` because that head re-runs
    ``upsample_pred_masks`` and applies the literal ``topk`` rule the plan rejects
    (§6 item 6), so calling it would only add cost the row does not pay.
    """
    from detectron2.structures import ImageList

    images_raw = [im.to(device) for im in inputs["images"]]
    v = len(images_raw)
    images = [(x - model.pixel_mean) / model.pixel_std for x in images_raw]
    images = ImageList.from_tensors(images, model.size_divisibility)
    Hp, Wp = images.tensor.shape[-2:]
    h, w = inputs["resized_hw"]

    xyz = inputs["xyz"].to(device)                       # [V,h,w,3]
    if (Hp, Wp) != (h, w):
        xyz = F.pad(xyz.permute(0, 3, 1, 2), (0, Wp - w, 0, Hp - h),
                    mode="constant", value=0).permute(0, 2, 3, 1)
    original_xyz = xyz.contiguous()                      # [V,Hp,Wp,3]

    from odin.modeling.backproject.backproject import multiscsale_voxelize
    multi_scale_xyz = make_multi_scale(original_xyz, (Hp, Wp), device)
    voxel_size = cfg.INPUT.VOXEL_SIZE[::-1]              # coarse -> fine, as load_3d_data
    multiview_data = {
        "multi_scale_xyz": multi_scale_xyz,
        "multi_scale_p2v": multiscsale_voxelize(multi_scale_xyz, voxel_size),
    }
    n_vox = [int(p.max().item()) + 1 for p in multiview_data["multi_scale_p2v"]]

    shape = [1, v, Hp, Wp]
    t0 = time.perf_counter()
    features = model.backbone(images.tensor, multi_scale_xyz, shape=shape,
                              multiview_data=multiview_data, decoder_3d=True)
    outputs = model.sem_seg_head(
        features, shape=shape, multiview_data=multiview_data,
        scannet_pc=None, scannet_p2v=None, segments=None, decoder_3d=True,
        captions=None, positive_map_od=None, num_classes=None,
        scene_names=[inputs.get("scene_id", "probe")],
    )
    torch.cuda.synchronize()
    t_net = time.perf_counter() - t0

    mask_cls = outputs["pred_logits"][0]                 # [Q, C+1]
    mask_pred = outputs["pred_masks"][0]                 # [Q, N_voxels_res2]

    mv = {"multi_scale_xyz": multi_scale_xyz[-1][0],
          "multi_scale_p2v": multiview_data["multi_scale_p2v"][-1][0]}
    bi = {"original_xyz": original_xyz}

    t0 = time.perf_counter()
    per_query = model.upsample_pred_masks(mask_pred, bi, mv, shape,
                                          downsample=False, interp="trilinear")
    torch.cuda.synchronize()
    t_up = time.perf_counter() - t0                      # [Q, V, Hp, Wp]

    # upstream's own head, called unmodified -- kept as the reference the plan's rule is
    # argued against, NOT as the path Phase 3 ships (see the docstring).
    instances = (model.inference_2d_per_image(mask_cls, mask_pred, (h, w), shape, bi,
                                              multiview_data=mv)
                 if with_instances else None)

    # the plan's max-over-class-per-query rule (§6 item 6 / §7): every query survives
    # exactly once, so the vocabulary cannot decide which masks exist.
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    if cfg.SKIP_CLASSES is not None:
        keep = torch.ones(cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, device=scores.device)
        keep[torch.tensor(cfg.SKIP_CLASSES, device=scores.device) - 1] = 0
        scores = scores[:, keep.bool()]
    cls_score, cls_idx = scores.max(dim=-1)

    return {
        "per_query_logits": per_query[:, :, :h, :w],
        "mask_cls": mask_cls,
        "query_class_score": cls_score,
        "query_class_idx": cls_idx,
        "instances": instances,
        "n_voxels_per_scale": n_vox,
        "seconds": {"backbone_head": t_net, "upsample": t_up},
        "padded_hw": (int(Hp), int(Wp)),
    }


# --------------------------------------------------------------------------------------
# overlays
# --------------------------------------------------------------------------------------
PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
    [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
], dtype=np.uint8)


def load_rgb(scene_dir: Path, frames: Sequence[str], hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    out = []
    for stem in frames:
        im = Image.open(scene_dir / "images" / f"{stem}.jpg").convert("RGB")
        out.append(np.array(im.resize((w, h), Image.BILINEAR)))
    return np.stack(out)


def instance_overlay(rgb: np.ndarray, masks: np.ndarray, ids: Sequence[int],
                     alpha: float = 0.6) -> np.ndarray:
    """Per-pixel winner-takes-all overlay of a set of query masks."""
    out = rgb.astype(np.float32).copy()
    for k, qid in enumerate(ids):
        m = masks[k]
        if not m.any():
            continue
        col = PALETTE[qid % len(PALETTE)].astype(np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * col
    return out.astype(np.uint8)


def save_strip(path: Path, rows: Sequence[np.ndarray], labels: Sequence[str],
               pad: int = 4) -> None:
    h, w = rows[0].shape[:2]
    n = len(rows)
    canvas = np.full((h + 18, n * (w + pad) - pad, 3), 255, dtype=np.uint8)
    for i, r in enumerate(rows):
        canvas[18:18 + h, i * (w + pad): i * (w + pad) + w] = r
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for i, lab in enumerate(labels):
        d.text((i * (w + pad) + 2, 3), lab, fill=(0, 0, 0))
    im.save(path)


def render_overlays(out_dir: Path, tag: str, rgb: np.ndarray, logits: torch.Tensor,
                    cls_score: torch.Tensor, cls_idx: torch.Tensor,
                    class_names: Sequence[str], top_k: int = 12) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    q, v, h, w = logits.shape
    binary = (logits > 0).cpu().numpy()
    area = binary.reshape(q, -1).sum(1)
    score = cls_score.cpu().numpy()

    # (1) every query, winner-takes-all by class score where they overlap
    order = np.argsort(-score)
    frames_img = []
    for f in range(v):
        frames_img.append(instance_overlay(rgb[f], binary[order][:, f], list(order)))
    save_strip(out_dir / f"overlay_all_queries_{tag}.png", frames_img,
               [f"view {i}" for i in range(v)])

    # (2) the top-k queries by class score, one row each
    top = [int(i) for i in np.argsort(-score) if area[i] > 0][:top_k]
    tiles, labels = [], []
    for qi in top:
        row = [instance_overlay(rgb[f], binary[qi][f][None], [qi]) for f in range(v)]
        tiles.append(np.concatenate(row, axis=1))
        labels.append(f"q{qi} {class_names[int(cls_idx[qi])]} s={score[qi]:.2f} "
                      f"px={int(area[qi])}")
    if tiles:
        h2, w2 = tiles[0].shape[:2]
        canvas = np.full((len(tiles) * (h2 + 20), w2, 3), 255, dtype=np.uint8)
        for i, t in enumerate(tiles):
            canvas[i * (h2 + 20) + 20: i * (h2 + 20) + 20 + h2] = t
        im = Image.fromarray(canvas)
        d = ImageDraw.Draw(im)
        for i, lab in enumerate(labels):
            d.text((4, i * (h2 + 20) + 4), lab, fill=(0, 0, 0))
        im.save(out_dir / f"overlay_top{top_k}_{tag}.png")

    return {
        "n_queries": int(q),
        "n_queries_nonempty": int((area > 0).sum()),
        "query_area_px": {int(i): int(area[i]) for i in top},
        "top_queries": [{"query": int(i), "class": class_names[int(cls_idx[i])],
                         "score": float(score[i]), "area_px": int(area[i]),
                         "views_present": int((binary[i].reshape(v, -1).sum(1) > 0).sum())}
                        for i in top],
    }


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry_npz", required=True)
    ap.add_argument("--scene_dir", required=True, help="3DTrackingBenchmark/<split>/<scene>")
    ap.add_argument("--geometry", choices=["vggt", "gt"], default="vggt")
    ap.add_argument("--scale_multiplier", type=float, default=1.0,
                    help="DIAGNOSTIC ONLY: multiply the metric geometry by this factor. "
                         "Used for the scale sensitivity sweep; the convention is NOT "
                         "chosen from its results.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--top_k", type=int, default=12)
    args = ap.parse_args()

    block_network()

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.modeling import build_model
    from detectron2.data import MetadataCatalog
    from odin import add_maskformer2_config, add_maskformer2_video_config

    stub_files = instrument_stubs()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(CFG_FILE)
    cfg.merge_from_list(OPTS)
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    model.to(args.device)

    class_names = list(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)
    if cfg.SKIP_CLASSES is not None:
        skip = {c - 1 for c in cfg.SKIP_CLASSES}
        class_names = [n for i, n in enumerate(class_names) if i not in skip]

    scene_dir = Path(args.scene_dir)
    inputs = build_inputs(Path(args.geometry_npz), scene_dir, args.geometry,
                          args.scale_multiplier, cfg, args.device)
    inputs["scene_id"] = scene_dir.name

    res = run_forward(model, inputs, cfg, args.device)

    tag = args.geometry if args.scale_multiplier == 1.0 else \
        f"{args.geometry}_x{args.scale_multiplier:g}"
    out = Path(args.output_dir)
    rgb = load_rgb(scene_dir, inputs["frames"], inputs["resized_hw"])
    viz = render_overlays(out, tag, rgb, res["per_query_logits"],
                          res["query_class_score"], res["query_class_idx"],
                          class_names, top_k=args.top_k)

    report = {
        "scene_id": scene_dir.name,
        "geometry": args.geometry,
        "scale_multiplier": args.scale_multiplier,
        "scale_m_per_unit": inputs["scale_m_per_unit"],
        "frames": inputs["frames"],
        "native_hw": list(inputs["native_hw"]),
        "resized_hw": list(inputs["resized_hw"]),
        "padded_hw": list(res["padded_hw"]),
        "n_voxels_per_scale_res5_to_res2": res["n_voxels_per_scale"],
        "seconds": res["seconds"],
        "instances_per_view": [len(i) for i in res["instances"]] if res["instances"] else None,
        "viz": viz,
        "stub_modules": stub_files,
        "stub_calls_reached": STUB_CALLS,
        "network_attempts": NETWORK_ATTEMPTS,
        "argv": list(sys.argv),
        "command": " ".join(sys.argv),
        "python": sys.executable,
        "opts": OPTS,
    }
    out.mkdir(parents=True, exist_ok=True)
    # CLAUDE.md §6. One provenance.json per output directory: it reflects the LAST run
    # written there, while every forward_report_<tag>.json carries its own argv.
    write_run_provenance(out, {"geometry": args.geometry, "tag": tag,
                               "scale_multiplier": args.scale_multiplier,
                               "ckpt": args.ckpt, "cfg_file": CFG_FILE, "opts": OPTS})
    with (out / f"forward_report_{tag}.json").open("w") as f:
        json.dump(report, f, indent=2)

    # per-query binary masks, bit-packed -- what the scale sensitivity comparison reads
    # (and the shape Phase 3's tracks.json will be built from).
    binary = (res["per_query_logits"] > 0).cpu().numpy()
    np.savez_compressed(
        out / f"query_masks_{tag}.npz",
        packed=np.packbits(binary, axis=None),
        shape=np.array(binary.shape),
        class_score=res["query_class_score"].cpu().numpy(),
        class_idx=res["query_class_idx"].cpu().numpy(),
        frames=np.array(inputs["frames"]),
    )

    print(json.dumps({k: v for k, v in report.items() if k != "viz"}, indent=2))
    print("nonempty queries:", viz["n_queries_nonempty"], "/", viz["n_queries"])
    for t in viz["top_queries"]:
        print(f"  q{t['query']:<3} {t['class']:<20} score={t['score']:.3f} "
              f"area={t['area_px']:<7} views={t['views_present']}")
    print("stub calls reached:", STUB_CALLS or "NONE")
    print("network attempts  :", NETWORK_ATTEMPTS or "NONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
