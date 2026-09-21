# SAM-V Web Demo

This is an optional research demo. It is intentionally separate from the main
project requirements used for training and benchmark reproduction.

## Install

```bash
pip install -r requirements-web.txt
```

Use the same Python environment that already runs SamVGGT inference.
If `pip` fails with an SSL EOF while your shell has proxy variables set, bypass
the proxy for this install:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  pip install -r requirements-web.txt
```

## Run

```bash
PYTHONPATH=$PWD uvicorn demos.web.app:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860`, upload related images, click object points on one
or more frames, and run inference.
If your shell exports proxy variables, use `localhost` in the browser or set
`NO_PROXY=localhost,127.0.0.1` before command-line checks against the local
server.

## Configuration

- `SAM_V_CKPT`: SamVGGT checkpoint. Defaults to
  `checkpoints/sam_vggt_epoch0020_scannetpp_v2_finetune.pth`.
- `DEVICE`: default `cuda:0`.
- `AMP_DTYPE`: `fp16`, `bf16`, or `none`; default `fp16`.
- `MAX_FRAMES`: default `16`.
- `MAX_UPLOAD_MB`: default `128`.
- `OUTPUT_TTL_SECONDS`: delete generated result folders older than this many
  seconds; default `3600`, set to `0` to disable cleanup.
- `INFER_RATE_LIMIT_PER_MINUTE`: per-client limit for `/api/infer`; default `10`,
  set to `0` to disable rate limiting.
- `SAM_VGGT_LAZY=1`: load the model on first inference instead of startup.
- `SAM_VGGT_MOCK=1`: run a deterministic fake mask path for UI/API testing.

For a public no-auth deployment, keep cleanup on and set explicit limits:

```bash
PUBLIC_DEMO=1 \
INFER_RATE_LIMIT_PER_MINUTE=10 \
OUTPUT_TTL_SECONDS=3600 \
MAX_FRAMES=16 \
MAX_UPLOAD_MB=128 \
PYTHONPATH=$PWD uvicorn demos.web.app:app --host 127.0.0.1 --port 7860 --workers 1
```

## Tests

```bash
python -m unittest demos/web/tests/test_web_demo_inference.py
```
