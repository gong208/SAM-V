# Docker for SAM-V

Two images, both built from the repository root:

| Image | Built by | Code | Use |
|---|---|---|---|
| `sam_v:latest` (deps) | `docker/build.sh` → `Dockerfile` | bind-mounted from the host | development |
| `sam_v:deploy` (self-contained) | `docker/build-deploy.sh` → `Dockerfile.deploy` | baked in | running anywhere with only weights and data mounted |

Both install exactly `requirements.txt` — the same pins as a native install —
on Python 3.10 (torch 2.3.1 with its CUDA 12.1 runtime; the host driver must
support CUDA 12.1). Weights are never baked in.

Docker covers SAM-V itself. The PanSt3R, Point-SAM and ODIN baselines each need
their own environment; see `benchmarks/baselines/README.md`.

## Build

```bash
git submodule update --init --recursive   # the deploy image bakes these in
bash docker/build.sh                      # -> sam_v:latest (needs network)
bash docker/build-deploy.sh               # -> sam_v:deploy (no network)
```

Behind a proxy, export `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` before
`build.sh`; they are forwarded to the build only when set, and cleared from the
image.

## Run the deploy image

Put these in one host directory and mount it at `/weights`:

- `model.pt` — VGGT-1B ([facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B))
- `sam_vit_h_4b8939.pth` — SAM ViT-H
- `sam_v_stage2.pth` — SAM-V ([Frank-Gong123/SAM-V](https://huggingface.co/Frank-Gong123/SAM-V))

`docker/entrypoint.sh` links the two base weights into place (or set
`VGGT_WEIGHT` / `SAM_WEIGHT` to their paths instead). Reproducing the ScanNet++
row of Table 2:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v /host/weights:/weights \
  -v /host/3DTrackingBenchmark:/data \
  -v /host/out:/out \
  sam_v:deploy \
  python benchmarks/sam_vggt_3dtracking_benchmark.py \
    --sam_v_ckpt /weights/sam_v_stage2.pth \
    --benchmark_root /data/scannetpp \
    --prompt_source sam_dense_masks \
    --prompt_sampling_method pole_plus_diverse \
    --prompt_points_per_mask 5 \
    --nms_score iou_preds \
    --nms_iou_type box \
    --output_dir /out/scannetpp
```

`--nms_iou_type box` is required for the published numbers; see the main README.
`./run_deploy.sh <command>` wraps the same `docker run`, mounting `WEIGHTS`
(default `./checkpoints`) at `/weights` and optionally `DATA_DIR`.

To share the image without a registry:

```bash
docker save sam_v:deploy | gzip > sam_v_deploy.tar.gz
docker load < sam_v_deploy.tar.gz          # on the target machine
```

## Develop with the deps image

```bash
bash docker/run.sh                                   # shell at /workspace
bash docker/run.sh python benchmarks/sam_vggt_3dtracking_benchmark.py --help
```

`run.sh` bind-mounts the repository at `/workspace`, attaches all GPUs (`GPUS=` to
override) and uses `--shm-size=16g`. `DATA_DIR=/path` also mounts a dataset root at
the same path. Base weights are read from `submodules/*/checkpoints/` in your
checkout.

For multi-GPU training use a larger shared-memory segment (`--shm-size=64g`);
16 GB deadlocks the DataLoader workers on the full ScanNet++ set.
