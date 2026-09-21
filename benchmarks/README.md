# benchmarks/ — evaluation entry points

Every evaluation script lives here.

| Tier | Directory | Meaning |
|---|---|---|
| live | `benchmarks/*.py` | use these for new work |
| baselines | `benchmarks/baselines/` | PanSt3R, Point-SAM and ODIN adapters — see its own README |
| timing | `benchmarks/timing/` | wall-clock measurement, not accuracy |

**Metric and matching semantics are frozen.** Every reported table depends on
them. If you believe a metric is wrong, report it with evidence and stop — do not
fix it in place, because a silent change invalidates every comparison.

Each entry point writes `provenance.json` + `git_diff.patch` into its output
directory (git sha, branch, dirty flag, exact command, resolved config,
interpreter) before it does any work, via `utils/provenance.py`. A run that
cannot state its commit does not go in a paper.

## `sam_vggt_3dtracking_benchmark.py` — the main eval entry point

SAM-V every-object inference on the IGGT 3D-tracking splits (ScanNet /
ScanNet++); the paper's Table 2. It owns the canonical metric definitions, the
Hungarian matching and the precision/recall counting — they live inside this
file, not in a shared module, so that the semantics behind the published numbers
cannot drift.

The only script that emits the full metric set: T-mIoU, T-SR, T-SR@0.5, pooled
IoU (the paper's **O-IoU**) and P/R at IoU 0.1–0.9. Supports `grid` and
`sam_dense_masks` prompt sources, threshold-grid sweeps, and the `--nms_score` /
`--nms_iou_type` flags.

`--nms_iou_type box` is required to reproduce the published numbers; `mask` is
the default and the better choice for new work. See the root README's caveats.

The optional `sam_dense_masks_ls` prompt source needs a LangSplat-modified SAM
fork that is not a submodule here; it never beat the sam-hq generator, and the
import is lazy, so the default path needs only `vggt` and `sam-hq`.

## `compare_baseline_sam2.py` — SAM-V vs SAM2, prompt-conditioned

A different track from the 3D-tracking benchmark and not comparable to it:
**Hypersim** scenes, a single target instance, GT point prompts on one frame, and
`pose_near` / `pose_diverse` frame sampling (the paper's *continuous* /
*diverse*). This is the paper's Table 1.

Needs the `sam2` submodule and a Hypersim `test` split.

## `timing/benchmark_inference_time.py`

Per-scene core inference time (excluding image I/O). Each model runs in its own
environment and writes a partial JSON; `--merge` combines them. Pass checkpoint
and dataset paths explicitly.

Timings are only comparable when measured on the same device class as the
published numbers (a single NVIDIA L40S) with nothing else on the node.
