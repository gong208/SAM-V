# SAM-V: Geometry-Aware Segment Anything for Multi-View Instance Segmentation

Jiangshan Gong, Yuqun Wu, Qiqian Fu, Yao Xiao, Chuhang Zou, Shenlong Wang, Derek Hoiem


SAM-V takes `N` RGB frames of one scene and user-provided point prompts, and predicts per-frame consistent masks of the *same* instance.


Two usage modes:

- **Prompt-conditioned single instance segmentation** — as in SAM, the user provides point prompts for an arbitrary target instance in the scene.
- **Every-object segmentation** — SAM proposals per frame → point prompts inside each proposal → SAM-V per prompt group → mask-overlap NMS.

## Repo layout

```
model/            SamVGGT (sam_vggt_model.py) — fusion modules + SAM decoder call
masks/            automatic_mask_generator.py (every-object mode + panoramic NMS)
                  prompt_sampling.py (centroid / pole_plus_diverse / ...)
training/         trainer.py, config.py (TrainConfig + YAML loader), viz.py
utils/            dataloader.py, loss_mask.py, misc.py, provenance.py
benchmarks/       every evaluation entry point — see benchmarks/README.md
                    sam_vggt_3dtracking_benchmark.py   every-object (Table 2)
                    compare_baseline_sam2.py           prompt-conditioned (Table 1)
                    baselines/                         PanSt3R, Point-SAM, ODIN adapters
                    timing/                            wall-clock, not accuracy
preprocessing/    dataset preparation + offline SAM embeddings
configs/          train/ (new work), paper/ (as submitted), smoke/ (fast paths)
demos/            infer_custom_images.py, web/ (Flask demo)
tools/            export_release_checkpoint.py
docker/           Dockerfile + build/run scripts
submodules/       vggt, sam-hq, sam2, panst3r, point-sam, odin
results/          (gitignored) run outputs
```

## Setup

**1. Clone with submodules.**

```bash
git clone --recurse-submodules https://github.com/gong208/SAM-V.git
cd sam-v
```

- `sam-hq` is a **patched fork** — it uses the plain `MaskDecoder`, with the HQ-only
`hq_token_only` / `interm_embeddings` / `vit_dim` arguments removed. No manual
patching needed.
- The `sam2`, `panst3r`, `point-sam` and `odin` submodules are only
needed to reproduce the baseline rows.
- SAM-V itself needs `vggt` and `sam-hq`.

**2. Environment** (Python 3.10):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The pins target **Linux with CUDA 12.1** (`torch==2.3.1`, which pulls its own CUDA
runtime). `timm`, `matplotlib`, `scikit-image` and `scikit-learn` are already in
there — do not install them separately, or you will upgrade off the pins.

**3. Base weights**:

- VGGT-1B → `submodules/vggt/checkpoints/model.pt`
  ([huggingface.co/facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt))
- SAM ViT-H → `submodules/sam-hq/checkpoints/sam_vit_h_4b8939.pth` ([sam-hq](https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view))

**4. The SAM-V checkpoints.** A released checkpoint carries only the three
trained modules; the frozen SAM and VGGT weights are loaded from step 3, so they
are not redistributed here. Pass the SAM-V checkpoint as `--sam_v_ckpt`.


| file | what it is |
|---|---|
| `sam_v_stage2.pth` | **Stage 2.** The checkpoint for every object segmentation evaluation and inference. |
| `sam_v_stage1.pth` | **Stage 1.** Hypersim pre-training. The initialisation stage 2 started from; use it as `${STAGE1_CHECKPOINT}` to re-run the fine-tune. |

Both live at **[huggingface.co/Frank-Gong123/SAM-V](https://huggingface.co/Frank-Gong123/SAM-V)**:

```bash
pip install huggingface_hub   # already in requirements.txt
huggingface-cli download Frank-Gong123/SAM-V sam_v_stage2.pth --local-dir checkpoints
```

**5. `PYTHONPATH` is mandatory and non-obvious.** The repo root alone is not
enough: the `vggt` submodule ships its own `training` package which shadows this
one. Always export all three, and invoke scripts by path.

```bash
export PYTHONPATH="$PWD:$PWD/submodules/sam-hq:$PWD/submodules/vggt"
```

## Quick start demo with your own images

```bash
python demos/infer_custom_images.py \
  --image_dir /path/to/frames \
  --sam_v_ckpt /path/to/sam_v.pth \
  --output_dir results/demo
```

There is also a Flask demo under `demos/web/` for clicking prompts in a browser.

## Datasets

Nothing here is redistributed. Download each dataset from its maintainer and comply with its
terms. The three variables below are what the commands in the rest of this README refer to.

```bash
export HYPERSIM_ROOT=/path/to/hypersim                     # stage 1, Table 1
export SCANNETPP_ROOT=/path/to/processed_scannetpp_v2      # stage 2
export BENCH=/path/to/IGGT_Benchmark/3DTrackingBenchmark   # Table 2
```

### Hypersim — stage-1 pre-training and Table 1

Source: **[apple-aiml-research/ml-hypersim](https://github.com/apple-aiml-research/ml-hypersim)**.
The full release is far larger than SAM-V needs, so use `contrib/99991/download.py`, which
fetches individual files and takes `--contains` filters.

One SAM-V scene is one upstream *(scene, camera trajectory)* pair: `ai_001_001_00` is scene
`ai_001_001`, camera `cam_00`. SAM-V reads exactly three things per scene:

| SAM-V path | Upstream file | Used for |
|---|---|---|
| `color/NNNNN.jpg` | `<scene>/images/scene_cam_XX_final_preview/frame.NNNN.color.jpg` | RGB input |
| `instance/NNNNN.png` (uint16) | `<scene>/images/scene_cam_XX_geometry_hdf5/frame.NNNN.semantic_instance.hdf5` | GT instance masks. The ids are scene-global, so the same id is the same physical object in every frame — that is what makes the multi-view supervision well defined. |
| `pose/NNNNN.txt` (4×4 camera-to-world, metres) | `<scene>/_detail/cam_XX/camera_keyframe_positions.hdf5` and `camera_keyframe_orientations.hdf5`, with the positions scaled by `meters_per_asset_unit` from `<scene>/_detail/metadata_scene.csv` | the `pose_near` / `pose_diverse` frame samplers |

So, per camera trajectory:

```bash
./download.py --contains scene_cam_00_final_preview --contains .color.jpg --silent
./download.py --contains scene_cam_00_geometry_hdf5 --contains .semantic_instance.hdf5 --silent
./download.py --contains _detail --silent
```

Nothing else is required. In particular skip `final_hdf5/` (the raw radiance images),
`position`, `normal_*`, `tex_coord`, `render_entity_id`, and the `*_preview` renders of
anything but colour. `depth_meters.hdf5` and `semantic.hdf5` are fine to keep if you want
them for your own work, but no SAM-V training or evaluation code opens them.

Camera intrinsics are **not** downloaded. Hypersim renders every scene with the same 60°
horizontal field of view, so for the 1024×768 images `fx = fy = 512 / tan(30°) = 886.81`,
`cx = 512`, `cy = 384` — constant across the dataset.

Convert the above into `$HYPERSIM_ROOT/<split>/<scene_id>/` with zero-padded 5-digit frame
stems. Stage 1 reads `train/` and `val/`; the Table 1 evaluation reads whatever directory
`--split` names, in the same layout. Then generate the two derived directories (run them once
per split you created):

```bash
python preprocessing/valid_instance_id.py \
  --hypersim_roots "$HYPERSIM_ROOT/train" "$HYPERSIM_ROOT/val"
python preprocessing/precompute_instance_presence.py \
  --roots "$HYPERSIM_ROOT/train" "$HYPERSIM_ROOT/val"
```

The finished layout:

```
$HYPERSIM_ROOT/<split>/<scene_id>/
  color/              00000.jpg  ...   downloaded
  instance/           00000.png  ...   downloaded
  pose/               00000.txt  ...   downloaded
  valid_ids/          00000.json ...   generated by valid_instance_id.py
  instance_presence/  00000.json ...   generated by precompute_instance_presence.py
```

Those last two are generated locally, not downloaded. If you get a Hypersim tree from
somewhere else and it carries extra directories — `masks/`, `dino_feat/`, `pos_encoding/`,
`region-instance/`, `region-labels/`, `transform.json` — they are artefacts of other
pipelines, not part of the Hypersim release, and SAM-V never reads them.

### ScanNet++ v2 — stage-2 fine-tuning

We fine-tune on the **processed** ScanNet++ v2 tree published with IGGT:
**[lifuguan/InsScene-15K](https://huggingface.co/datasets/lifuguan/InsScene-15K/tree/main/processed_scannetpp_v2)**.
It is one zip split byte-wise across 53 parts (~227 GB), so `cat` the parts back together
before unzipping:

```bash
huggingface-cli download lifuguan/InsScene-15K --repo-type dataset \
  --include "processed_scannetpp_v2/*" --local-dir /path/to/download
cat /path/to/download/processed_scannetpp_v2/processed_scannetpp_v2.zip.* \
  > processed_scannetpp_v2.zip
unzip processed_scannetpp_v2.zip
```

**The train/val split is the official ScanNet++ one, not IGGT's.** We do not use the
partition that ships with InsScene-15K. Scenes are assigned by the official ScanNet++ v2
`nvs_sem_train.txt` (856 scenes) and `nvs_sem_val.txt` (50 scenes) lists, obtained from
[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) under its terms of use. Put the split
files at `$SCANNETPP_ROOT/splits/` and move each scene directory into `$SCANNETPP_ROOT/train`
or `$SCANNETPP_ROOT/val` to match.

Then generate the derived directories. `pose/` is unpacked from the downloaded
`scene_iphone_metadata.npz` here — it is not a separate download:

```bash
python preprocessing/valid_instance_id.py \
  --scannetpp_root "$SCANNETPP_ROOT/train" \
  --scannetpp_scene_lists "$SCANNETPP_ROOT/splits/nvs_sem_train.txt" --num_workers 8
python preprocessing/precompute_instance_presence.py \
  --scannetpp_root "$SCANNETPP_ROOT/train" \
  --scannetpp_scene_lists "$SCANNETPP_ROOT/splits/nvs_sem_train.txt" --num_workers 8
python preprocessing/precompute_sam_embeddings_scannetpp.py \
  --dataset_root "$SCANNETPP_ROOT" --splits train val
```

(repeat the first two with `val` / `nvs_sem_val.txt`). The finished layout:

```
$SCANNETPP_ROOT/
  splits/           nvs_sem_train.txt, nvs_sem_val.txt   from ScanNet++
  <split>/<scene_id>/
    images/                    frame_000010.jpg  ...  downloaded
    refined_ins_ids/           frame_000010.png  ...  downloaded (GT instance masks, uint16)
    scene_iphone_metadata.npz                         downloaded (camera trajectories)
    pose/                      frame_000010.txt  ...  generated by valid_instance_id.py
    valid_ids/                 frame_000010.json ...  generated by valid_instance_id.py
    object_union.json                                 generated by valid_instance_id.py
    instance_presence/         frame_000010.json ...  generated by precompute_instance_presence.py
    sam_embeddings/            frame_000010.pt   ...  generated by precompute_sam_embeddings_scannetpp.py
```

`sam_embeddings/` is large but optional in principle: it caches the frozen SAM image encoder
so stage 2 can skip that forward pass. `configs/paper/scannetpp_finetune.yaml` sets
`use_offline_sam_embeddings: true` and expects it.

### IGGT 3D-tracking benchmark — Table 2

Source: **[lifuguan/IGGT_Benchmark](https://huggingface.co/datasets/lifuguan/IGGT_Benchmark)**.

```bash
huggingface-cli download lifuguan/IGGT_Benchmark --repo-type dataset \
  --include "3D Tracking Benchmark/*" --local-dir /path/to/IGGT_Benchmark
```

Upstream the directory is named `3D Tracking Benchmark`, with spaces; rename it to
`3DTrackingBenchmark` and point `$BENCH` at it. It contains the two splits, each scene being
`<scene_id>/images/` with `frame_*.jpg` and the matching GT `frame_*_label.npy`:

```
$BENCH/{scannet,scannetpp}/<scene_id>/images/frame_XXXXXX.jpg
                                             frame_XXXXXX_label.npy
```

Raw ScanNet and raw ScanNet++ are **not** needed for Table 2 — the benchmark ships the frames
it scores.

## Reproducing the paper

With `$HYPERSIM_ROOT`, `$SCANNETPP_ROOT` and `$BENCH` exported as in **Datasets** above:

### Table 2 — every-object segmentation

```bash
python benchmarks/sam_vggt_3dtracking_benchmark.py \
  --sam_v_ckpt /path/to/sam_v.pth \
  --benchmark_root "$BENCH/scannetpp" \
  --prompt_source sam_dense_masks \
  --prompt_sampling_method pole_plus_diverse \
  --prompt_points_per_mask 5 \
  --nms_score iou_preds \
  --nms_iou_type box \
  --device cuda:0 \
  --output_dir results/3dtracking_scannetpp
```

Swap `scannetpp` for `scannet` for the other split. We used **`--nms_iou_type box`** for our evaluation.

The run writes `dataset_summary.json`, `precision_recall_summary.csv`, per-scene
`summary.json`, and `provenance.json` + `git_diff.patch` recording the exact
commit and command.

### Table 1 — prompt-conditioned single instance segmentation

```bash
python benchmarks/compare_baseline_sam2.py \
  --sam_v_ckpt /path/to/sam_v.pth \
  --dataset_root "$HYPERSIM_ROOT" --split test \
  --strategies pose_near,pose_diverse --num_points 10 --seed 0 \
  --frame_count 16 --scene_limit 0 \
  --output_dir results/sam2_baseline
```

The repo's strategy names are `pose_near` / `pose_diverse`; the paper calls the
same two settings *continuous* / *diverse*.

We evaluate this on checkpoint `sam_v_stage1.pth`

### Baselines

`benchmarks/baselines/` holds the PanSt3R, Point-SAM and ODIN adapters. Every
baseline is scored by the *same* frozen `evaluate_scene` that scores SAM-V, on the
same frame lists. See `benchmarks/baselines/README.md` for the per-stage
commands and the environments each one needs.

## Training

Two stages. Both take one YAML config; unknown keys are rejected, so a typo fails
loudly.

### Stage 1 — Hypersim pre-training

```bash
torchrun --nproc_per_node=8 training/trainer.py \
  --config configs/train/hypersim_pretrain.yaml
```

> `configs/train/hypersim_pretrain.yaml` is the recipe of the released stage-1
> checkpoint.

### Stage 2 — ScanNet++ fine-tuning

```bash
export STAGE1_CHECKPOINT=/path/to/stage1.pth
torchrun --nproc_per_node=8 training/trainer.py \
  --config configs/paper/scannetpp_finetune.yaml
```

Stage 2 needs offline SAM image embeddings; build them first with
`preprocessing/precompute_sam_embeddings_scannetpp.py`.

## Licence

Apache-2.0 — see `LICENSE`. `NOTICE` records the third-party code this builds on
and, importantly, what is *not* redistributed here: VGGT ships a Meta
research-materials agreement rather than an OSI licence and is required to run
SAM-V at all, and the PanSt3R baseline is under a NAVER non-commercial licence.
Read `NOTICE` before any commercial use.

The released checkpoints are licensed separately under **CC BY-NC 4.0**. See the
[model card](https://huggingface.co/Frank-Gong123/SAM-V) for their licence and
provenance.

## Citation

```bibtex
@article{gong2026samv,
  title   = {SAM-V: Geometry-Aware Segment Anything for Multi-View Instance Segmentation},
  author  = {Gong, Jiangshan and Wu, Yuqun and Fu, Qiqian and Xiao, Yao and
             Zou, Chuhang and Wang, Shenlong and Hoiem, Derek},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

<!-- TODO: replace arXiv:XXXX.XXXXX with the real identifier once the preprint is
     posted. Do not cite a venue until acceptance is decided. -->
