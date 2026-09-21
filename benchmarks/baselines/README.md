# Baselines: PanSt3R, Point-SAM and ODIN on the IGGT 3D-tracking benchmark

Design decisions, protocol-fairness arguments and measured findings
are summarised in the root README's caveats. This file is the
*operational* reference: the stage contract, and the install commands exactly as they ran.

---

## The stage contract

Inference must run in each baseline's own environment, scoring must run in the project env
(importing `benchmarks.sam_vggt_3dtracking_benchmark` pulls in sam-hq, vggt and
segment_anything). Hence three stages:

```
stage A  (project env)   prepare inputs        -> Point-SAM and ODIN only
stage B  (baseline env)  run their inference   -> preds/<scene>/tracks.json
stage C  (project env)   score, frozen code    -> dataset_summary.json + precision_recall_summary.csv
```

`tracks.json` is the contract between B and C, one per scene:

```json
{"scene_id": "0a76e06478",
 "frame_names": ["frame_000000.jpg", "..."],
 "target_size": [1024, 1024],
 "method": "panst3r_v2",
 "config": {"vocab": "train_union", "vocab_sha256": "...", "...": "..."},
 "infer_seconds": {"prepare": 0.0, "infer": 12.3, "assemble": 0.4},
 "tracks": [{"score": 0.91, "per_view_masks": {"0": {"size": [1024,1024], "counts": [...]}}}]}
```

- `per_view_masks` maps **frame index** (as a string) to an RLE mask. The RLE is exactly what the
  frozen `encode_mask_to_rle` emits, so the frozen `decode_mask` consumes it unchanged.
  Frames on which a track is absent are **omitted**, matching SAM-V's own convention.
- `frame_names` must equal the frozen module's own listing for that scene, element-wise.
  `score_baseline.py` asserts this and **exits non-zero** on mismatch — a silent misalignment
  would produce plausible, wrong numbers.
- `infer_seconds` is a three-key breakdown, never a scalar; the number reported in the table is
  their **sum** (`track_io.total_infer_seconds`). Single-stage methods write `0.0` for the keys
  they do not use, so the sum is well-defined everywhere.
- `method` (`panst3r_v1` / `panst3r_v2` / `point_sam` / `odin_scannet200_swin`) is what keeps the two PanSt3R checkpoints
  in separate output trees rather than overwriting each other.

`track_io.py` is import-safe in **either** env: it uses the frozen RLE helpers when the heavy
deps are importable and byte-identical local copies otherwise.

Every runner writes `provenance.json` + `git_diff.patch` into its output directory (git sha,
branch, dirty flag, exact command, resolved config, interpreter), via `utils/provenance.py`.
The two SAM-V eval entry points write the same files. A run that cannot state its commit does
not go in a paper.

## Files

| File | Env | Role |
|---|---|---|
| `track_io.py` | either | `tracks.json` read/write, RLE, frame discovery, `total_infer_seconds()` |
| `score_baseline.py` | project | **the single scoring path** for all baseline configs |
| `samv_to_tracks.py` | project | verification utility: an existing SAM-V run → `tracks.json` |
| `vocab.py` | panst3r | resolves `--vocab` to an explicit list + sha256 |
| `panst3r_infer.py` | panst3r | RGB frames → per-view panoptic → `tracks.json` |
| `pointsam_probe.py` | pointsam | Phase-5.1 sanity probe: GT-depth cloud -> Point-SAM -> overlays |
| `pointsam_sampling.py` | either | **THE FROZEN CONFIG** + the sampling path every stage shares |
| `pointsam_ceiling.py` | project | the published sampling ceiling, per GT object and area bin |
| `prepare_pointsam_inputs.py` | project | stage A: VGGT cloud + AMG prompt groups -> `inputs.npz` |
| `pointsam_infer.py` | pointsam | stage B: one `predict_masks` per prompt group -> `masks.npz` |
| `pointsam_assemble.py` | project | stage C: scatter -> panoramic NMS -> `tracks.json` |
| `odin_geometry_probe.py` | project | ODIN stage A (`--mode dump`) + the frozen metric-scale convention (`--mode calibrate`) |
| `odin_probe_forward.py` | odin | Phase-2 kill-switch: one forward pass + overlays; **owns `OPTS` and the reimplemented eval path** |
| `odin_infer.py` | odin | stage B: geometry -> Q x V masks -> `tracks.json` |

---

## Environments

### Project env (scoring, stage A/C)

Unchanged: `python`, invoked by absolute path, with

```bash
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
```

No dependency was added to it.

### PanSt3R env — `$PANST3R_ENV`

Upstream: `naver/panst3r` @ `0253909812ce570d35f2325a20a591e4785f7d7d`
("Ckpt defaults", 2026-03-20), cloned to `submodules/panst3r`.

**Commands as they actually ran** (not the README's, see the deviation below):

```bash
git clone https://github.com/naver/panst3r submodules/panst3r

/usr/bin/python3.11 -m venv $PANST3R_ENV          # Python 3.11.0rc1
$PANST3R_ENV/bin/pip install -U pip setuptools wheel
$PANST3R_ENV/bin/pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
$PANST3R_ENV/bin/pip install xformers==0.0.30
cd submodules/panst3r && $PANST3R_ENV/bin/pip install -e .   # pulls must3r + pyrender fork
$PANST3R_ENV/bin/pip install matplotlib                      # for the geometry figures
```

**Deviation from the upstream README — read this before re-running.** The README prescribes

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
```

`download.pytorch.org/whl/cu126` redirects the NVIDIA CUDA runtime wheels to
`pypi.nvidia.com`. If that host is unreachable from your network (some proxies reset the TLS
connection mid-transfer), the install fails even though `download.pytorch.org` and `pypi.org`
work.

PyPI's own default `torch==2.7.0` linux/x86_64 wheel **is** the cu126 build, so installing from
plain PyPI yields the same CUDA version the README asks for. The build script asserts
`torch.version.cuda == "12.6"` rather than trusting this.

**Two more things to know before a paper run.**

1. **PanSt3R downloads two backbones from the HF Hub at model-construction time** —
   `facebook/dinov2-large` and `google/siglip-base-patch16-224` (~1.2 GB, cached under
   `~/.cache/huggingface`). Weights must be staged ahead of time on an offline node, so
   the cache must be warm (or `HF_HOME` pointed at a pre-staged copy) before any benchmark run.
   It is warm on this node now; it is not part of the checkpoint staging above.
2. **The `pip install -e .` is editable and points at the worktree**
   (`.../worktrees/baselines-phase0/submodules/panst3r/src/panst3r`). If that worktree is
   removed, `import panst3r` breaks. Re-point it (`pip install -e <new path>`) or install
   non-editable before treating the env as durable.

Not installed: the optional cuRoPE extension
(`pip install --no-build-isolation "git+https://github.com/naver/croco.git@croco_module#egg=curope&subdirectory=curope"`).
It is a speed optimisation, not a correctness requirement, but MUSt3R says so out loud at import:
`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`.
If PanSt3R timings are reported in the paper's timing table, they must be labelled as measured
**without** cuRoPE — otherwise the baseline is charged for an optimisation we declined to build.

#### PanSt3R checkpoints

Staged to `$BASELINE_WEIGHTS/panst3r/`, both MD5s verified against the
upstream README table:

| Variant | File | Upscaler | MD5 | Verified |
|---|---|---|---|---|
| v1 | `panst3r_v1_512_5ds.pth` | PixelShuffle | `901ac0e17153f9b15decaa796511b1be` | **OK** |
| v2 | `panst3r_v2_512_5ds.pth` | LoftUp | `6a7a47a2ea635de05af45ded8dcabd57` | **OK** |

```bash
DEST=$BASELINE_WEIGHTS/panst3r
BASE=https://download.europe.naverlabs.com/ComputerVision/PanSt3R
mkdir -p "$DEST" && cd "$DEST"
curl -fL --retry 3 -O "$BASE/panst3r_v1_512_5ds.pth"
curl -fL --retry 3 -O "$BASE/panst3r_v2_512_5ds.pth"
md5sum panst3r_v1_512_5ds.pth panst3r_v2_512_5ds.pth
```

Weights are staged ahead of time and never downloaded inside a benchmark run.

### Point-SAM env — `$POINTSAM_ENV`

Upstream: `zyc00/Point-SAM` @ `25f4fd9675479f3c951e830771595855ff7ed466`
("Revise citation for Point-SAM paper", 2025-10-04), cloned to `submodules/point-sam`.
`third_party/torkit3d` submodule @ `235ecf60497271136f5552cb45bb7cf75ab1cb09`.

**Commands as they actually ran:**

```bash
git clone https://github.com/zyc00/Point-SAM submodules/point-sam
cd submodules/point-sam && git submodule update --init third_party/torkit3d

/usr/bin/python3.10 -m venv $POINTSAM_ENV                     # Python 3.10.12
$POINTSAM_ENV/bin/pip install -U pip setuptools wheel
$POINTSAM_ENV/bin/pip install "numpy<2" torch==2.4.1 torchvision==0.19.1
$POINTSAM_ENV/bin/pip install "timm>=1.0,<1.1" hydra-core omegaconf accelerate \
    safetensors scipy h5py trimesh huggingface_hub matplotlib
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.9" \
    $POINTSAM_ENV/bin/pip install --no-build-isolation third_party/torkit3d
```

Verified after install: `torch 2.4.1+cu121`, `torch.cuda.is_available()`, device `NVIDIA L40S`,
capability `(8, 9)`, and the three torkit3d CUDA ops actually exercised
(`sample_farthest_points`, `batch_index_select`, `knn_points`).

**Three deviations from the upstream README — read before re-running.**

1. **torch 2.4.1, not the latest.** This node exports `LD_LIBRARY_PATH=/usr/local/cuda/lib64`,
   whose CUDA 12.2 `libnvJitLink.so.12` exports symbol versions only up to `__nvJitLink*_12_2`.
   torch **2.5.1**'s bundled `libcusparse` requires `__nvJitLinkComplete_12_4` and therefore
   **fails at `import torch`** on this node:
   `ImportError: .../libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4`.
   torch 2.4.1 (cu121) needs only `_12_1`, which 12.2 provides. Pinning 2.4.1 keeps the env
   working under the ambient environment instead of requiring every caller to prepend the
   wheel's `nvidia/nvjitlink/lib` to `LD_LIBRARY_PATH`. The build asserts this by running an
   actual cuSPARSE op, not by inspecting version strings. (This is a *different* failure from
   the `pypi.nvidia.com` proxy reset that affects PanSt3R; both bite the same install step.)
2. **`--no-build-isolation` is required for torkit3d.** Its `setup.py` imports `torch` at
   module scope, and pip's isolated build env has no torch:
   `ModuleNotFoundError: No module named 'torch'` during *Getting requirements to build wheel*.
3. **Point-SAM is NOT pip-installed.** Runners put the clone on `sys.path` (which is what
   upstream's own `evaluation/inference.py` does with `sys.path.append(".")`). This deliberately
   avoids the editable-install fragility that `$PANST3R_ENV` has, where deleting the worktree
   breaks `import panst3r`.

**apex is not needed.** It is used only by `replace_with_fused_layernorm`, which swaps every
`nn.LayerNorm` for `apex.normalization.FusedLayerNorm` — same `normalized_shape`, `eps`,
`elementwise_affine`, and same state-dict keys. It is a speed optimisation with no effect on
outputs or on checkpoint loading. Not installed; runners default to stock `nn.LayerNorm` and
record `apex: false` in their provenance.

#### Point-SAM checkpoint

Staged to `$BASELINE_WEIGHTS/point_sam/`:

| File | Bytes | MD5 | SHA256 |
|---|---|---|---|
| `model.safetensors` | 1244245632 | `8480240b01d13e93e01d0002cd800946` | `bfe0aa4fee2d3c08251271e597954e6a2d26209d003c724d0ff049f578410ab2` |

```bash
DEST=$BASELINE_WEIGHTS/point_sam
HFSHA=fb2cd5cd8047e3681491d82fc41b880a9a914fc8
mkdir -p "$DEST" && cd "$DEST"
curl -fL --retry 3 -o model.safetensors \
  "https://huggingface.co/yuchen0187/Point-SAM/resolve/$HFSHA/model.safetensors"
md5sum model.safetensors && sha256sum model.safetensors
```

Unlike PanSt3R, upstream publishes **no checksum** to verify against, so the download is pinned
to the HF repo revision `fb2cd5cd8047e3681491d82fc41b880a9a914fc8` (content-addressed) and our
own MD5/SHA256 are recorded above as the reference for future re-staging. `PROVENANCE.txt` next
to the file records the repo, revision, size and fetch date. `load_model` reports **no missing
and no unexpected keys** against `configs/model/default.yaml`, so the ViT-L config is the right
one.

The model constructs `timm.create_model("eva02_large_patch14_448", pretrained=False)`, so
**nothing is downloaded at model-construction time** — unlike PanSt3R, this baseline satisfies
the root README's no-runtime-download rule as-is.

---

## Running

### Stage C — score any baseline (project env)

```bash
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
BENCH=$BENCH

python benchmarks/baselines/score_baseline.py \
    --preds_dir      results/baselines/panst3r_v2_scannetpp/preds \
    --benchmark_root "$BENCH/scannetpp" \
    --output_dir     results/baselines/panst3r_v2_scannetpp
```

### Verifying the scorer (the round-trip gate)

Re-scores an existing SAM-V run through the baseline path; must reproduce the published numbers
exactly, or no baseline number is trustworthy.

`--run_dir` is an existing SAM-V benchmark output directory (the one holding
`dataset_summary.json`). Point `$SAMV_RUN` at it.

```bash
python benchmarks/baselines/samv_to_tracks.py \
    --run_dir "$SAMV_RUN" \
    --out_dir /tmp/rt/scannetpp/preds
python benchmarks/baselines/score_baseline.py \
    --preds_dir /tmp/rt/scannetpp/preds \
    --benchmark_root "$BENCH/scannetpp" \
    --output_dir /tmp/rt/scannetpp/scored
```

Must reproduce, to full printed precision:

| | T-mIoU | O-IoU | P@50 | R@50 |
|---|---|---|---|---|
| ScanNet++ | 0.7842516812664859 | 0.793869328145483 | 0.7941176470588235 | 0.8915094339622641 |

Bitwise identical on both splits on 2026-08-23, and re-verified at each session since. See

### Point-SAM — the three-stage pipeline

The config lives in **one** place: the frozen constants at the top of `pointsam_sampling.py`.
Stage A, Stage B, Stage C and the ceiling all import them, and none of them accepts a loose
CLI override for them. That is deliberate: a ceiling measured at settings the run does not use
is not that run's ceiling, and publishing the per-area-bin ceiling beside the row is the
condition under which the Point-SAM row was approved.

```bash
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
BENCH=$BENCH
SPLIT=scannetpp
RUN=results/baselines/pointsam_m2

# Stage A (project env): VGGT cloud + sam-hq AMG prompt groups -> inputs.npz
#   --verify_ceiling runs THE STAGE-A GATE; --prompt_cache_from checks the prompts against
#   SAM-V's own run. Both are cheap; run them.
python benchmarks/baselines/prepare_pointsam_inputs.py \
    --benchmark_root "$BENCH/$SPLIT" --output_dir "$RUN/stage_a" \
    --prompt_cache_from "$SAMV_RUN" \
    --verify_ceiling "results/baselines/pointsam_ceiling/frozen_${SPLIT}/ceiling.json"

# Stage B (pointsam env): one predict_masks call per prompt group -> masks.npz
$POINTSAM_ENV/bin/python benchmarks/baselines/pointsam_infer.py \
    --stage_a_dir "$RUN/stage_a" --output_dir "$RUN/infer"

# Stage C (project env): scatter -> panoramic NMS -> tracks.json, then score
python benchmarks/baselines/pointsam_assemble.py \
    --stage_a_dir "$RUN/stage_a" --infer_dir "$RUN/infer" --output_dir "$RUN/preds"
python benchmarks/baselines/score_baseline.py \
    --preds_dir "$RUN/preds" --benchmark_root "$BENCH/$SPLIT" --output_dir "$RUN/scored"
```

Cost, measured on `0a76e06478` (one L40S): 126 s/scene — 21.5 s Stage A (19.7 s of it the AMG),
101 s Stage B (M = 381 prompt groups at 0.264 s each), 3.7 s Stage C. About 40 min for all 19
benchmark scenes.

**`--max_groups N` on Stage B caps the prompt groups per scene.** It exists for cost
extrapolation only; a capped run is not a scoreable row and must never be scored as one.

**Stage B's cost scales with M, the prompt-group count**, not just with point count: upstream
has no `set_pointcloud()` and the point encoder re-runs on every call. `M` is recorded in
`infer_summary.json` and carried into `tracks.json`'s `config`, and it belongs next to any
Point-SAM timing that gets published.

### ODIN — the three-stage pipeline

`method` string: **`odin_scannet200_swin`**.

The row (19/19 scenes, frozen `evaluate_scene`, commit `c07ec06`; full precision and caveats in
**0.5809 / 0.6305 / 0.6367 / 0.6964** (T-mIoU / O-IoU / P@50 / R@50). Both splits ran with
`--scene_ids` omitted and identical flags; the run tree is
`results/baselines/odin_scannet200_swin/{scannetpp,scannet}/{stage_a,preds,scored}`.

Like Point-SAM, ODIN needs geometry it cannot produce itself, so it is a three-stage
pipeline rather than PanSt3R's two. Unlike Point-SAM, its stage A is the *geometry probe*
in `--mode dump` -- there is no separate `prepare_*` script:

```bash
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
BENCH=$BENCH
SPLIT=scannetpp
RUN=results/baselines/odin_phase3

# Stage A (project env): RGB -> VGGT -> metric, gravity-aligned geometry.npz
CUDA_VISIBLE_DEVICES=0 python \
    benchmarks/baselines/odin_geometry_probe.py \
    --benchmark_root "$BENCH/$SPLIT" --scene_ids 0a76e06478 --mode dump \
    --device cuda:0 --output_dir "$RUN/stage_a"

# Stage B (odin env): geometry + RGB -> 100 queries -> tracks.json
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/submodules/odin \
    $ODIN_ENV/bin/python benchmarks/baselines/odin_infer.py \
    --stage_a_dir "$RUN/stage_a" --benchmark_root "$BENCH/$SPLIT" \
    --scene_ids 0a76e06478 --output_dir "$RUN/$SPLIT"

# Stage C (project env), frozen scorer
python benchmarks/baselines/score_baseline.py \
    --preds_dir "$RUN/$SPLIT/preds" --benchmark_root "$BENCH/$SPLIT" \
    --scene_ids 0a76e06478 --output_dir "$RUN/$SPLIT/scored"
```

`--scene_ids` is omitted for a full split. `--geometry gt` on stage B selects the
ScanNet++-only GT-depth geometry (the plan's Phase-6 appendix probe); **the row is
`vggt`, on both splits, and geometry sources are never mixed across splits.**

**Four things about stage B that belong next to any ODIN number.**

1. **Query -> track is max-over-class per query**, not upstream's literal `topk(100)` over
   the flattened `Q x C`. Upstream's rule lets one query occupy several slots under
   different labels (measured on `0a76e06478`: its 100 slots hold only **81 distinct
   queries**) and makes *which* queries survive depend on the class posterior, which would
   turn the closed-vocabulary ceiling from soft into partly hard. Each scene's
   `probe.json` records `upstream_topk_distinct_queries` so the two rules stay comparable.
2. **No NMS.** ODIN is a set predictor solved jointly over all views, as PanSt3R's panoptic
   argmax was. Recorded as `nms: "none"` in `tracks.json`'s `config`, not left implicit.
3. **Masks reach 1024x1024 by one bilinear on the logits, then `> 0`** -- ODIN runs at
   `IMAGE_SIZE 512` and its `ResizeShortestEdge` has no crop, so the two resizes compose
   into one. `> 0` is upstream's own threshold (`odin_model.py:1286`).
4. **`SKIP_CLASSES [119, 200]` is upstream's typo and is run under, not corrected**: in
   `SCANNET200_NAME_MAP` 119 = *mini fridge*, 199 = *wall*, 200 = *floor*, so upstream drops
   mini fridge + floor and keeps wall while its comment says "floor and wall". It is
   recorded verbatim in the config with a `skip_classes_note`; `submodules/odin` is
   unpatched.

Queries whose mask is empty everywhere at 1024x1024 are not emitted (95/100 on
`0a76e06478`). That is metric-neutral -- `hungarian_match` drops zero-IoU assignments and
the P/R counters only visit GT objects and their matched predictions -- and both counts
are in `probe.json`.

`infer_seconds["prepare"]` carries stage A's VGGT+convention wall clock through
ScanNet++, 4.37 s/scene ScanNet** -- strictly serial, single idle L40S, `perf_counter` +
`cuda.synchronize()` and 2 discarded whole-scene warmups, which is what `--timing_warmup 2`
on both stages turns on. Without that flag the clock is the Phase-3/4 bare `perf_counter`,
`tracks.json` marks it `timing_is_phase3_wiring_only: true`, and **it is not reportable**.

Two things belong beside that number. Only ~10 % of it is ODIN's own forward (0.47 s/scene
on both splits): stage A's VGGT geometry is ~51 % and the resize-to-1024x1024 plus RLE
assembly ~38 %. And ODIN, like PanSt3R, never pays the sam-hq AMG. Reading stage A's
`geometry.npz` back off disk plus the JPEG read/resize is measured but **not** charged
(`build_inputs_excluded`, 0.18 / 0.09 s) -- Point-SAM's stage B excludes its own
`inputs.npz` load for the same reason.

#### ODIN env — `$ODIN_ENV`

cloned to `submodules/odin` (untracked, unpatched). Checkpoint
`scannet200_swin_31.5_76k_5.5k.pth` staged to
`$BASELINE_WEIGHTS/odin/` with a `PROVENANCE.txt`
(md5 `9520f7fdc02727b52f3f13394c6d6d6f`); it loads `strict=True` with 0 missing and 0
unexpected keys. Three module-scope imports on the `import odin` path -- `pytorch3d`,
`pyviz3d`, `wandb` -- are stubbed in `$ODIN_ENV/odin_stubs/` and are never entered on
the eval path (instrumented and confirmed empty in every run's `probe.json`). Never
`import functions` or `import modules` in adapter code running in this env: MSDeformAttn's
`setup.py` installs two generically-named top-level packages that would shadow them.

### The Point-SAM sampling ceiling

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/baselines/pointsam_ceiling.py \
    --benchmark_root "$BENCH/scannetpp" \
    --output_dir results/baselines/pointsam_ceiling/frozen_scannetpp
```

Every flag defaults to the frozen config, so the bare command above is the published ceiling.
The flags exist so the Phase-5.2 sweep stays re-runnable (`--crop_rel 0 --voxel_scope global`
reproduces the pre-crop numbers bitwise), not so the run can be retuned.
