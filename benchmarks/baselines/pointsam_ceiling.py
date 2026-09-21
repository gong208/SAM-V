"""Phase 5.2: the Point-SAM sampling ceiling.

Pushes **GT masks** through the identical route predictions take —
VGGT world points -> 1024x1024 grid -> stride-N subsample -> voxel merge (one voxel carries
one label) -> scatter back to every pixel of each voxel -> nearest-upsample to 1024x1024 —
and measures IoU against the original GT mask.

Whatever this returns is an upper bound on any Point-SAM number: the representation cannot
express a better mask than this, no matter how good the segmentation is. A Point-SAM row of
0.5 on objects whose ceiling is 0.6 is a *sampling* result, not a segmentation result, and the
two support very different sentences in the paper.

Measured **per GT object and binned by GT mask area**, never as a scene aggregate: a mean over
objects hides exactly the small objects the voxel merge destroys, which is the failure mode
worth knowing about.

The sampling path is imported from `pointsam_sampling.py`, the same module Stage A uses, so
"identical" is structural rather than a promise. Runs in the project env with the mandatory
PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmarks.baselines import pointsam_sampling as ps
from utils.provenance import write_run_provenance
from benchmarks.sam_vggt_3dtracking_benchmark import TARGET_SIZE, load_scene_inputs

# GT-area bins in pixels on the 1024x1024 grid, summed over frames.
AREA_BINS = [(0, 2_000), (2_000, 10_000), (10_000, 50_000), (50_000, 200_000),
             (200_000, np.inf)]
AREA_LABELS = ["<2k", "2k-10k", "10k-50k", "50k-200k", ">200k"]


def bin_of(area: float) -> int:
    for i, (lo, hi) in enumerate(AREA_BINS):
        if lo <= area < hi:
            return i
    return len(AREA_BINS) - 1


def objects_ceiling(xyz: np.ndarray, pix: np.ndarray, gt: np.ndarray, n_frames: int,
                    stride: int, voxel_rel: float, crop_radius: float, voxel_scope: str,
                    voxel_id: np.ndarray = None, voxel_xyz: np.ndarray = None) -> List[Dict]:
    """Per-GT-object ceiling, given a scene's stride cloud and its pixel map.

    Split out of `ceiling_for_scene` so `prepare_pointsam_inputs.py --verify_ceiling` can run
    the *same* code over the cloud it actually wrote to `inputs.npz`. If Stage A's cloud and
    the ceiling harness's cloud disagree, the ceiling row is decoration and the shared-path
    claim is false -- so the two must be checked against each other, not assumed equal.
    """
    gt_at_points = gt[pix[:, 0], pix[:, 1], pix[:, 2]]
    n_vox = len(voxel_xyz) if voxel_xyz is not None else 0
    objects: List[Dict] = []
    for oid in sorted({int(i) for i in np.unique(gt) if i > 0}):
        gt_mask = gt == oid                                        # [F,1024,1024]
        area = int(gt_mask.sum())
        if area == 0:
            continue
        labels = (gt_at_points == oid).astype(np.int8)

        # Anchor the crop the way Stage B does -- at a prompt inside the object. The GT
        # object's own medoid is the most favourable anchor available, which is what a
        # ceiling wants; a real prompt group can only do worse.
        sel = np.flatnonzero(labels)
        if len(sel) == 0:
            objects.append(dict(
                object_id=oid, gt_area=area, area_bin=AREA_LABELS[bin_of(area)],
                ceiling_iou=0.0, recovered_area=0, gt_points_sampled=0,
                crop_voxels=0, crop_covers_gt=0.0))
            continue
        obj_xyz = xyz[sel]
        anchor = obj_xyz[int(np.argmin(np.linalg.norm(obj_xyz - obj_xyz.mean(0), axis=1)))]

        # THE shared definition of what one Point-SAM call sees. Stage B calls this exact
        # function per prompt group; here it is called per GT object. Anything the crop or the
        # voxel merge cannot express is unreachable, and that is what the ceiling reports.
        call = ps.prompt_group_cloud(xyz, anchor, crop_radius, voxel_rel, voxel_scope,
                                     global_voxel_id=voxel_id, global_voxel_xyz=voxel_xyz)
        n_crop_vox = int(len(call["voxel_xyz"]))
        # The best mask this call could possibly return: each voxel takes its points' majority
        # GT label. A real Point-SAM prediction is one of these, never better.
        vox_label = ps.majority_label_per_voxel(call["voxel_of_point"],
                                                labels[call["point_index"]], n_crop_vox)
        point_values = np.zeros(len(xyz), dtype=np.int8)
        point_values[call["point_index"]] = vox_label[call["voxel_of_point"]]

        recovered = ps.scatter_to_full_masks(point_values, pix, n_frames, stride, TARGET_SIZE)
        inter = int((recovered & gt_mask).sum())
        union = int((recovered | gt_mask).sum())
        # How much of the object the crop can even reach, separating "the crop is too small"
        # from "the voxel merge is too coarse".
        in_r = np.linalg.norm(obj_xyz - anchor[None, :], axis=1) <= crop_radius
        objects.append(dict(
            object_id=oid, gt_area=area, area_bin=AREA_LABELS[bin_of(area)],
            ceiling_iou=(inter / union) if union else 0.0,
            recovered_area=int(recovered.sum()),
            gt_points_sampled=int(labels.sum()),
            crop_voxels=n_crop_vox,
            crop_covers_gt=float(in_r.mean()) if crop_radius > 0 else 1.0,
        ))

    return objects


def ceiling_for_scene(scene_dir: Path, model, device: str, stride: int, voxel_size: float,
                      conf_thresh: float, voxel_rel: float = 0.0, crop_rel: float = 0.0,
                      voxel_scope: str = "global") -> Dict:
    """One scene's per-object ceiling.

    `crop_rel > 0` reproduces what Stage B actually does: Point-SAM is only ever shown the
    crop around a prompt group, so any pixel whose voxel lies outside that crop is
    **unreachable** and the ceiling must say so. The Phase-5.2 ceilings were measured with no
    crop and are therefore an upper bound on this one.

    `voxel_scope`:
      - `global` -- voxelise the whole scene once, then crop the voxel cloud. This is the
        Stage A / Stage B split as designed: Stage A pre-voxelises, Stage B crops.
      - `crop_local` -- crop the *point* cloud first, then voxelise inside the crop at
        `voxel_rel` x the crop's own diagonal. A crop holds far fewer points than a scene, so
        this buys finer resolution at a similar per-call point count -- but it forces Stage A
        to hand Stage B the un-voxelised cloud and Stage B to voxelise once per prompt group.
    """
    scene = load_scene_inputs(scene_dir, TARGET_SIZE)
    gt = scene.gt_instance_maps                                   # [F,1024,1024] int
    n_frames = gt.shape[0]

    imgs = ps.load_native_rgb([Path(p) for p in scene.image_paths])
    t0 = time.perf_counter()
    wp, conf, coords = ps.vggt_forward(model, imgs, device)
    xyz_grid, conf_grid = ps.world_points_to_grid(wp, conf, coords, TARGET_SIZE)
    t_vggt = time.perf_counter() - t0

    sub = ps.stride_subsample(xyz_grid, conf_grid, stride, conf_thresh)
    n_before = n_frames * len(range(0, TARGET_SIZE[0], stride)) ** 2

    # VGGT's world scale is ARBITRARY and differs per scene (measured: 1 VGGT unit ~ 2.07 m on
    # 0a76e06478). An absolute --voxel_size is therefore a different physical voxel in every
    # scene, which makes ceilings incomparable across scenes. --voxel_rel expresses the voxel as
    # a fraction of this scene's own cloud diagonal, which is scale-invariant.
    diag = ps.cloud_diagonal(sub["xyz"])
    effective_voxel = voxel_rel * diag if voxel_rel > 0 else voxel_size
    crop_radius = crop_rel * diag
    voxel_id, voxel_xyz = ps.voxel_merge(sub["xyz"], effective_voxel)
    n_vox = len(voxel_xyz)

    pix = sub["pixel"]
    objects = objects_ceiling(sub["xyz"], pix, gt, n_frames, stride, voxel_rel, crop_radius,
                              voxel_scope, voxel_id, voxel_xyz)

    return dict(
        scene_id=scene.scene_id, frames=n_frames, num_objects=len(objects),
        points_before_voxel=int(len(sub["xyz"])), points_on_grid=int(n_before),
        voxels=n_vox, stride=stride, voxel_size=voxel_size, voxel_rel=voxel_rel,
        effective_voxel=effective_voxel, cloud_diagonal=diag, conf_thresh=conf_thresh,
        crop_rel=crop_rel, crop_radius=crop_radius, voxel_scope=voxel_scope,
        conf_mean=float(conf_grid.mean()),
        conf_percentiles={str(p): float(np.percentile(conf_grid, p)) for p in (1, 10, 50, 90, 99)},
        dropped_by_conf=float(1.0 - len(sub["xyz"]) / n_before),
        vggt_seconds=t_vggt, objects=objects,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark_root", required=True)
    ap.add_argument("--scene_ids", default="", help="comma-separated; default all")
    ap.add_argument("--vggt_ckpt", default="submodules/vggt/checkpoints/model.pt")
    ap.add_argument("--output_dir", required=True)
    # Defaults are THE FROZEN CONFIG (pointsam_sampling.py). They are exposed as flags only so
    # the sweep that produced the published table stays re-runnable; the published ceiling is
    # the one measured at the defaults.
    ap.add_argument("--stride", type=int, default=ps.STRIDE)
    ap.add_argument("--voxel_size", type=float, default=0.02,
                    help="absolute, in VGGT's arbitrary units -- prefer --voxel_rel")
    ap.add_argument("--voxel_rel", type=float, default=ps.VOXEL_REL,
                    help="voxel as a fraction of the scene cloud diagonal (scale-invariant); "
                         "0.0023 ~ 2 cm in a room of ~8.7 m diagonal")
    ap.add_argument("--conf_thresh", type=float, default=ps.CONF_THRESH)
    ap.add_argument("--crop_rel", type=float, default=ps.CROP_REL,
                    help="crop radius as a fraction of the scene cloud diagonal; 0 disables "
                         "the crop, which is what the Phase-5.2 ceilings were measured at "
                         "(and is NOT what Stage B does)")
    ap.add_argument("--voxel_scope", choices=["global", "crop_local"], default=ps.VOXEL_SCOPE,
                    help="global: voxelise the scene once, then crop (the Stage A/B split). "
                         "crop_local: crop first, then voxelise inside the crop.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.benchmark_root)
    scenes = ([root / s for s in args.scene_ids.split(",")] if args.scene_ids
              else sorted(p for p in root.iterdir() if p.is_dir()))

    model = ps.load_vggt(args.vggt_ckpt, args.device)
    results = []
    for sd in scenes:
        r = ceiling_for_scene(sd, model, args.device, args.stride, args.voxel_size,
                              args.conf_thresh, args.voxel_rel, args.crop_rel,
                              args.voxel_scope)
        results.append(r)
        ious = [o["ceiling_iou"] for o in r["objects"]]
        cv = [o["crop_voxels"] for o in r["objects"]]
        print(f"[{r['scene_id']}] {r['num_objects']} obj  pts {r['points_on_grid']}->"
              f"{r['points_before_voxel']}->{r['voxels']} vox  "
              f"crop {min(cv)}-{max(cv)} vox/call (med {int(np.median(cv))})  "
              f"ceiling median={np.median(ious):.4f} min={min(ious):.4f} "
              f"({r['vggt_seconds']:.1f}s VGGT)")

    # per-area-bin distribution -- the thing the gate is actually about
    all_objs = [o for r in results for o in r["objects"]]
    vox_desc = (f"voxel_rel={args.voxel_rel} (~{results[0]['effective_voxel']:.4f} VGGT units)"
                if args.voxel_rel > 0 else f"voxel={args.voxel_size}")
    print(f"\n=== ceiling by GT area, stride={args.stride} {vox_desc} "
          f"crop_rel={args.crop_rel} scope={args.voxel_scope} "
          f"conf_thresh={args.conf_thresh} ({len(all_objs)} objects) ===")
    print(f"{'bin':10s} {'n':>4s} {'median':>8s} {'p10':>8s} {'min':>8s} "
          f"{'crop_vox':>9s} {'cover':>6s}")
    summary = {}
    for lab in AREA_LABELS:
        sel = [o for o in all_objs if o["area_bin"] == lab]
        v = np.array([o["ceiling_iou"] for o in sel])
        if not len(v):
            continue
        med, p10, mn = float(np.median(v)), float(np.percentile(v, 10)), float(v.min())
        cv = float(np.median([o["crop_voxels"] for o in sel]))
        cov = float(np.median([o["crop_covers_gt"] for o in sel]))
        summary[lab] = dict(n=len(v), median=med, p10=p10, min=mn,
                            median_crop_voxels=cv, median_crop_covers_gt=cov)
        print(f"{lab:10s} {len(v):4d} {med:8.4f} {p10:8.4f} {mn:8.4f} {cv:9.0f} {cov:6.3f}")
    overall = np.array([o["ceiling_iou"] for o in all_objs])
    print(f"{'ALL':10s} {len(overall):4d} {np.median(overall):8.4f} "
          f"{np.percentile(overall,10):8.4f} {overall.min():8.4f}")

    payload = dict(config=vars(args), frozen_config=ps.frozen_config(),
                   by_area_bin=summary,
                   overall=dict(n=len(overall), median=float(np.median(overall)),
                                p10=float(np.percentile(overall, 10)),
                                min=float(overall.min())),
                   scenes=results)
    (out / "ceiling.json").write_text(json.dumps(payload, indent=2))
    write_run_provenance(out, config=vars(args))
    print(f"\nwrote {out / 'ceiling.json'}")


if __name__ == "__main__":
    main()
