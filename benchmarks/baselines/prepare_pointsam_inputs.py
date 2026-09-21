"""Stage A: everything Point-SAM needs, built in the project env.

Point-SAM lives in `$POINTSAM_ENV` and cannot import sam-hq, VGGT or this repo's model
package. So the two things it needs -- a point cloud with colours, and prompt groups in 3D --
are produced here and handed over as one `.npz` per scene:

    RGB frames -> VGGT world points -> 1024x1024 grid -> stride-N cloud (+ colours)
    RGB frames -> sam-hq AMG proposals -> pole_plus_diverse prompts -> 3D prompt coordinates

Two things this file deliberately does NOT do.

**It does not re-implement the sampling geometry.** `pointsam_sampling.py` owns it, and
`pointsam_ceiling.py` imports the same module. That is what makes the published ceiling a
statement about *this* run rather than a separate measurement standing next to it.

**It does not re-implement prompt generation.** `build_dense_prompt_groups` is imported from
the frozen `sam_vggt_3dtracking_benchmark` and called unmodified, so Point-SAM is prompted
from the same proposals, the same sampling method and the same seed SAM-V got. Prompts are
regenerated here rather than copied out of SAM-V's run directories, so `prepare_seconds`
carries the AMG cost the timing row is supposed to charge Point-SAM for. `--prompt_cache_from`
checks the result against a SAM-V run's cache: on `0a76e06478` that is the same 381 groups on
the same frames, with 2 groups landing a different sampled point and 48 differing in
`proposal_area` by a few boundary pixels -- GPU non-determinism in the AMG forward, not a
code-path difference. See `compare_prompt_cache`.

**It does not voxelise.** Under the frozen `VOXEL_SCOPE = "crop_local"` the voxel merge happens
inside each prompt group's crop, in Stage B, at `VOXEL_REL` x the *crop's* diagonal -- a finer
voxel than a scene-wide merge at the same per-call point count. So Stage A hands over the
un-voxelised stride grid and Stage B calls `pointsam_sampling.prompt_group_cloud` per group.

Project env (`python`) with the mandatory PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.baselines import pointsam_sampling as ps  # noqa: E402
from benchmarks.baselines.pointsam_ceiling import objects_ceiling  # noqa: E402
from utils.provenance import write_run_provenance  # noqa: E402
from benchmarks.sam_vggt_3dtracking_benchmark import (  # noqa: E402  -- FROZEN, unmodified
    PROPOSAL_BOX_NMS_THRESH,
    PROPOSAL_MAX_MASKS_PER_FRAME,
    PROPOSAL_MIN_MASK_REGION_AREA,
    PROPOSAL_POINTS_PER_BATCH,
    PROPOSAL_POINTS_PER_SIDE,
    PROPOSAL_PRED_IOU_THRESH,
    PROPOSAL_SAM_MODEL_TYPE,
    PROPOSAL_STABILITY_SCORE_THRESH,
    TARGET_SIZE,
    build_dense_prompt_groups,
    discover_scene_dirs,
    load_scene_inputs,
    sam_model_registry,
)


def proposal_constants() -> Dict[str, Any]:
    """The frozen proposal settings, recorded so a run can prove it used them."""
    return dict(
        proposal_sam_model_type=PROPOSAL_SAM_MODEL_TYPE,
        proposal_points_per_side=PROPOSAL_POINTS_PER_SIDE,
        proposal_points_per_batch=PROPOSAL_POINTS_PER_BATCH,
        proposal_pred_iou_thresh=PROPOSAL_PRED_IOU_THRESH,
        proposal_stability_score_thresh=PROPOSAL_STABILITY_SCORE_THRESH,
        proposal_box_nms_thresh=PROPOSAL_BOX_NMS_THRESH,
        proposal_min_mask_region_area=PROPOSAL_MIN_MASK_REGION_AREA,
        proposal_max_masks_per_frame=PROPOSAL_MAX_MASKS_PER_FRAME,
    )


def sample_colors(image_tensor: torch.Tensor, pixel: np.ndarray) -> np.ndarray:
    """Point colours, read at each point's own pixel on the benchmark's 1024x1024 grid.

    Taken from `SceneInputs.image_tensor`, i.e. the *scorer's* view of the frames, so a point's
    colour and its mask live on the same grid by construction.
    """
    img = image_tensor.permute(0, 2, 3, 1).cpu().numpy()          # [N,H,W,3] float
    return img[pixel[:, 0], pixel[:, 1], pixel[:, 2]].clip(0, 255).astype(np.uint8)


def compare_prompt_cache(ours: Path, theirs: Path) -> Dict[str, Any]:
    """Check our freshly generated prompt cache against a SAM-V run's, group for group.

    The proposal + sampling code is the frozen module's, called unmodified, so this should
    match. It is checked rather than asserted in prose because "same code path" and "same
    prompts" are different claims, and only the second makes the Point-SAM row a controlled
    comparison.

    It reports magnitudes, not a boolean, because **bit-equality is not achievable**: the SAM
    AMG forward is not run-to-run deterministic on GPU, so mask logits move by ~1e-5, a few
    boundary pixels flip, and `proposal_area` / the RLE shift slightly. Measured on
    `0a76e06478`: same group count, same frames, 48/381 areas differing by boundary pixels
    and **2/381 groups landing a different sampled point**. Group count and frame assignment
    are the parts that must match exactly; those are flagged as hard mismatches.
    """
    a = json.loads(ours.read_text())["prompt_groups"]
    b = json.loads(theirs.read_text())["prompt_groups"]
    if len(a) != len(b):
        return dict(match=False, reason=f"group count {len(a)} vs {len(b)}")
    bad_frame = [ga["prompt_id"] for ga, gb in zip(a, b)
                 if ga["frame_index"] != gb["frame_index"]]
    if bad_frame:
        return dict(match=False, reason=f"frame_index differs in {len(bad_frame)} group(s)")
    pts = sum(1 for ga, gb in zip(a, b)
              if not np.array_equal(np.asarray(ga["points"]), np.asarray(gb["points"])))
    area = sum(1 for ga, gb in zip(a, b) if ga["proposal_area"] != gb["proposal_area"])
    return dict(match=True, num_groups=len(a), groups_with_different_points=pts,
                groups_with_different_area=area,
                note="AMG is not bit-deterministic on GPU; count and frames must match, "
                     "per-group boundary jitter is expected")


def verify_against_ceiling(xyz: np.ndarray, pixel: np.ndarray, gt: np.ndarray,
                           n_frames: int, crop_radius: float, scene_id: str,
                           ceiling_json: Path) -> Dict[str, Any]:
    """THE STAGE-A GATE: does Stage A's cloud reproduce the published ceiling?

    Pushes the scene's GT masks through the cloud this run just wrote to `inputs.npz`, using
    `pointsam_ceiling.objects_ceiling` -- the same function that produced the published
    numbers -- and compares per object. If they disagree, the ceiling row is decoration: it
    would be describing a different point cloud from the one Point-SAM is given.

    Exact equality is not expected. Stage A and the ceiling harness each run their own VGGT
    forward, and that forward is not bit-deterministic on GPU, so world points move slightly
    and a few points land in different voxels. The tolerance is on the *magnitude* of the
    disagreement, reported here rather than hidden behind a boolean.
    """
    objs = objects_ceiling(xyz, pixel, gt, n_frames, ps.STRIDE, ps.VOXEL_REL, crop_radius,
                           ps.VOXEL_SCOPE)
    published = json.loads(Path(ceiling_json).read_text())
    ref = next((sc for sc in published["scenes"] if sc["scene_id"] == scene_id), None)
    if ref is None:
        return dict(ok=False, reason=f"{scene_id} not in {ceiling_json}")
    refmap = {o["object_id"]: o for o in ref["objects"]}
    diffs, rows = [], []
    for o in objs:
        r = refmap.get(o["object_id"])
        if r is None:
            return dict(ok=False, reason=f"object {o['object_id']} missing from the ceiling")
        d = o["ceiling_iou"] - r["ceiling_iou"]
        diffs.append(d)
        rows.append(dict(object_id=o["object_id"], stage_a=o["ceiling_iou"],
                         published=r["ceiling_iou"], delta=d))
    a = np.abs(diffs)
    return dict(ok=bool(a.max() < 0.01), num_objects=len(objs),
                max_abs_delta=float(a.max()), mean_abs_delta=float(a.mean()),
                num_exact=int((a == 0).sum()), tolerance=0.01,
                ceiling_json=str(ceiling_json), per_object=rows)


def prepare_scene(scene_dir: Path, vggt_model, proposal_sam_model, device: str,
                  scene_output_dir: Path, prompt_cache_from: Path = None,
                  verify_ceiling: Path = None) -> Dict[str, Any]:
    """Build one scene's Stage-A payload. Returns the summary; writes `inputs.npz`."""
    t_start = time.perf_counter()
    scene = load_scene_inputs(scene_dir, TARGET_SIZE)
    n_frames = int(scene.image_tensor.shape[0])

    # ---- geometry: VGGT -> 1024 grid -> stride cloud -------------------------------------
    imgs = ps.load_native_rgb([Path(p) for p in scene.image_paths])
    t0 = time.perf_counter()
    wp, conf, coords = ps.vggt_forward(vggt_model, imgs, device)
    xyz_grid, conf_grid = ps.world_points_to_grid(wp, conf, coords, TARGET_SIZE)
    t_vggt = time.perf_counter() - t0

    sub = ps.stride_subsample(xyz_grid, conf_grid, ps.STRIDE, ps.CONF_THRESH)
    xyz, pixel = sub["xyz"], sub["pixel"]
    n_on_grid = n_frames * len(range(0, TARGET_SIZE[0], ps.STRIDE)) ** 2
    rgb = sample_colors(scene.image_tensor, pixel)

    diag = ps.cloud_diagonal(xyz)
    crop_radius = ps.CROP_REL * diag

    # ---- prompts: the FROZEN proposal + sampling path, called unmodified ------------------
    t0 = time.perf_counter()
    prompt_args = Namespace(
        prompt_source="sam_dense_masks",
        prompt_sampling_method=ps.PROMPT_SAMPLING_METHOD,
        prompt_points_per_mask=ps.PROMPT_POINTS_PER_MASK,
        ls_level="large",                      # unused for sam_dense_masks; kept for the API
    )
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    prompt_groups, prompt_summary = build_dense_prompt_groups(
        scene, proposal_sam_model, scene_output_dir, prompt_args
    )
    t_prompts = time.perf_counter() - t0
    if not prompt_groups:
        raise RuntimeError(f"{scene_dir.name}: no prompt groups produced")

    prompt_check: Dict[str, Any] = {}
    if prompt_cache_from is not None:
        ours = Path(prompt_summary["cache_path"])
        theirs = sorted((Path(prompt_cache_from) / scene_dir.name).glob("prompt_groups_*"))
        prompt_check = (compare_prompt_cache(ours, theirs[0]) if theirs
                        else dict(match=False, reason="no cache found to compare against"))
        prompt_check["compared_against"] = str(theirs[0]) if theirs else str(prompt_cache_from)

    # ---- prompts 2D -> 3D ----------------------------------------------------------------
    # A prompt is a pixel on one frame, so its 3D coordinate is that pixel's world point --
    # read off the FULL 1024 grid, not the strided one, so the prompt lands exactly where the
    # proposal put it rather than at the nearest sampled point.
    g = len(prompt_groups)
    k = int(ps.PROMPT_POINTS_PER_MASK)
    prompt_xy = np.zeros((g, k, 2), dtype=np.float32)
    prompt_xyz = np.zeros((g, k, 3), dtype=np.float32)
    prompt_frame = np.zeros(g, dtype=np.int32)
    prompt_id = np.zeros(g, dtype=np.int32)
    prompt_area = np.zeros(g, dtype=np.int64)
    prompt_piou = np.zeros(g, dtype=np.float32)
    prompt_stab = np.zeros(g, dtype=np.float32)
    anchor = np.zeros((g, 3), dtype=np.float32)

    for i, grp in enumerate(prompt_groups):
        f = int(grp["frame_index"])
        pts = np.asarray(grp["points"], dtype=np.float32)[:k]          # (x, y)
        cols = np.clip(np.round(pts[:, 0]).astype(int), 0, TARGET_SIZE[1] - 1)
        rows = np.clip(np.round(pts[:, 1]).astype(int), 0, TARGET_SIZE[0] - 1)
        prompt_xy[i, :len(pts)] = pts
        prompt_xyz[i, :len(pts)] = xyz_grid[f, rows, cols]
        prompt_frame[i] = f
        prompt_id[i] = int(grp["prompt_id"])
        prompt_area[i] = int(grp.get("proposal_area", 0))
        prompt_piou[i] = float(grp.get("proposal_predicted_iou", 0.0))
        prompt_stab[i] = float(grp.get("proposal_stability_score", 0.0))
        # The crop is anchored at the pole point -- prompt 0, the deepest interior point of
        # the proposal (masks/prompt_sampling.py `_pole_point`) -- which is the prompt least
        # likely to sit on a boundary and pull the crop off the object.
        anchor[i] = prompt_xyz[i, 0]

    prepare_seconds = time.perf_counter() - t_start

    ceiling_check: Dict[str, Any] = {}
    if verify_ceiling is not None:
        ceiling_check = verify_against_ceiling(
            xyz, pixel, scene.gt_instance_maps, n_frames, crop_radius,
            scene.scene_id, verify_ceiling)
    conf_flat = sub["conf"]
    summary = dict(
        scene_id=scene.scene_id,
        frames=n_frames,
        frame_names=list(scene.frame_names),
        target_size=list(TARGET_SIZE),
        num_points=int(len(xyz)),
        points_on_grid=int(n_on_grid),
        cloud_diagonal=diag,
        crop_radius=crop_radius,
        num_prompt_groups=int(g),
        world_points_conf=dict(
            mean=float(conf_grid.mean()),
            percentiles={str(p): float(np.percentile(conf_grid, p))
                         for p in (1, 5, 10, 25, 50, 75, 90, 99)},
            kept_mean=float(conf_flat.mean()),
            fraction_dropped=float(1.0 - len(xyz) / n_on_grid),
        ),
        vggt_seconds=t_vggt,
        prompt_seconds=t_prompts,
        prepare_seconds=prepare_seconds,
        prompts_cached=bool(prompt_summary.get("cached", False)),
        prompt_check=prompt_check,
        ceiling_check={k: v for k, v in ceiling_check.items() if k != "per_object"},
        ceiling_check_per_object=ceiling_check.get("per_object", []),
        prompt_summary={kk: vv for kk, vv in prompt_summary.items() if kk != "frames"},
        frozen_config=ps.frozen_config(),
        proposal_constants=proposal_constants(),
    )

    np.savez_compressed(
        scene_output_dir / "inputs.npz",
        xyz=xyz, rgb=rgb, conf=conf_flat, pixel=pixel,
        prompt_xyz=prompt_xyz, prompt_xy=prompt_xy, prompt_frame=prompt_frame,
        prompt_id=prompt_id, prompt_area=prompt_area,
        prompt_proposal_iou=prompt_piou, prompt_proposal_stability=prompt_stab,
        anchor=anchor,
        cloud_diagonal=np.float32(diag), crop_radius=np.float32(crop_radius),
        stride=np.int32(ps.STRIDE), target_size=np.array(TARGET_SIZE, dtype=np.int32),
        frame_names=np.array(scene.frame_names, dtype=object),
    )
    (scene_output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark_root", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--scene_ids", nargs="*", default=None)
    ap.add_argument("--vggt_ckpt", default="submodules/vggt/checkpoints/model.pt")
    ap.add_argument("--proposal_sam_checkpoint",
                    default=str(REPO_ROOT / "submodules" / "sam-hq" / "checkpoints"
                               / "sam_vit_h_4b8939.pth"))
    ap.add_argument("--prompt_cache_from", type=Path, default=None,
                    help="a SAM-V run directory (e.g. .../3dtracking_benchmark_paper/"
                         "scannetpp_samprompt) whose prompt cache the freshly generated "
                         "prompts are checked against, group for group. Read-only.")
    ap.add_argument("--verify_ceiling", type=Path, default=None,
                    help="path to a published ceiling.json; runs the Stage-A gate -- push this "
                         "scene's GT masks through the cloud just written and check the "
                         "per-object ceilings against it")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    scene_dirs = discover_scene_dirs(args.benchmark_root, args.scene_ids)
    if not scene_dirs:
        raise SystemExit(f"no scenes discovered under {args.benchmark_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_provenance(args.output_dir, config=vars(args) | {
        "benchmark_root": str(args.benchmark_root), "output_dir": str(args.output_dir),
        "frozen_config": ps.frozen_config(), "proposal_constants": proposal_constants(),
    })

    print(f"Building proposal SAM model ({PROPOSAL_SAM_MODEL_TYPE}) from "
          f"{args.proposal_sam_checkpoint} ...")
    proposal_sam_model = sam_model_registry[PROPOSAL_SAM_MODEL_TYPE](
        checkpoint=args.proposal_sam_checkpoint)
    proposal_sam_model.to(device=args.device).eval()
    vggt_model = ps.load_vggt(args.vggt_ckpt, args.device)

    summaries: List[Dict[str, Any]] = []
    for sd in scene_dirs:
        s = prepare_scene(sd, vggt_model, proposal_sam_model, args.device,
                          args.output_dir / sd.name, args.prompt_cache_from,
                          args.verify_ceiling)
        summaries.append(s)
        print(f"[{s['scene_id']}] {s['frames']} frames  {s['num_points']} pts  "
              f"diag={s['cloud_diagonal']:.3f}  crop_r={s['crop_radius']:.3f}  "
              f"M={s['num_prompt_groups']} prompt groups  "
              f"conf mean={s['world_points_conf']['mean']:.2f} "
              f"dropped={s['world_points_conf']['fraction_dropped']:.3f}  "
              f"({s['prepare_seconds']:.1f}s: vggt {s['vggt_seconds']:.1f}s, "
              f"prompts {s['prompt_seconds']:.1f}s"
              f"{' [cached]' if s['prompts_cached'] else ''})")
        if s["prompt_check"]:
            c = s["prompt_check"]
            if c.get("match"):
                print(f"    prompt-cache check vs SAM-V: MATCH ({c['num_groups']} groups; "
                      f"{c['groups_with_different_points']} with different points, "
                      f"{c['groups_with_different_area']} with different area -- AMG jitter)")
            else:
                print(f"    prompt-cache check vs SAM-V: MISMATCH -- {c.get('reason')}")
        if s["ceiling_check"]:
            c = s["ceiling_check"]
            if "reason" in c:
                print(f"    STAGE-A GATE: FAILED -- {c['reason']}")
            else:
                print(f"    STAGE-A GATE: {'PASS' if c['ok'] else '*** FAIL ***'}  "
                      f"{c['num_objects']} objects vs published ceiling, "
                      f"{c['num_exact']} exact, max|delta|={c['max_abs_delta']:.6f}, "
                      f"mean|delta|={c['mean_abs_delta']:.6f} (tol {c['tolerance']})")

    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summaries, indent=2))
    tot_m = sum(s["num_prompt_groups"] for s in summaries)
    print(f"\n{len(summaries)} scene(s), {tot_m} prompt groups total "
          f"({tot_m / len(summaries):.0f}/scene) -> {args.output_dir}")


if __name__ == "__main__":
    main()
