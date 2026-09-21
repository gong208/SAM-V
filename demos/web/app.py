from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from demos.web.inference import (
    UploadedImage,
    build_uploaded_sample,
    cleanup_output_dirs,
    decode_image_bytes,
    mock_predict,
    render_outputs,
    timed_call,
)


WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
RUNTIME_DIR = WEB_DIR / "runtime"
REPO_ROOT = WEB_DIR.parents[1]


class Settings:
    def __init__(self) -> None:
        self.sam_v_ckpt = os.getenv(
            "SAM_V_CKPT",
            str(REPO_ROOT / "checkpoints" / "sam_v_stage2.pth"),
        )
        self.device = os.getenv("DEVICE", "cuda:0")
        self.amp_dtype = os.getenv("AMP_DTYPE", "fp16")
        self.max_frames = int(os.getenv("MAX_FRAMES", "16"))
        self.max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "128"))
        self.output_ttl_seconds = int(os.getenv("OUTPUT_TTL_SECONDS", "3600"))
        self.infer_rate_limit_per_minute = int(os.getenv("INFER_RATE_LIMIT_PER_MINUTE", "10"))
        self.sam_model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
        self.sam_encode_chunk = int(os.getenv("SAM_ENCODE_CHUNK", "8"))
        self.mock = os.getenv("SAM_VGGT_MOCK", "0") == "1"
        self.lazy = os.getenv("SAM_VGGT_LAZY", "0") == "1"


class InferenceBusyError(RuntimeError):
    """Raised when the single GPU inference slot is already occupied."""


class InferenceRateLimiter:
    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0) -> None:
        self.limit_per_minute = int(limit_per_minute)
        self.window_seconds = float(window_seconds)
        self.lock = threading.Lock()
        self.hits_by_client: dict[str, deque[float]] = {}

    def allow(self, client_id: str, now: float | None = None) -> bool:
        if self.limit_per_minute <= 0:
            return True

        ts = time.time() if now is None else now
        cutoff = ts - self.window_seconds
        with self.lock:
            hits = self.hits_by_client.setdefault(client_id, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit_per_minute:
                return False
            hits.append(ts)
            return True


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: torch.nn.Module | None = None
        self.device = torch.device(settings.device)
        self.load_lock = threading.Lock()
        self.infer_lock = threading.Lock()

    def load(self) -> None:
        if self.settings.mock or self.model is not None:
            return
        with self.load_lock:
            if self.model is not None:
                return
            if self.device.type == "cuda" and not torch.cuda.is_available():
                self.device = torch.device("cpu")

            from benchmarks.compare_baseline_sam2 import load_samvggt_checkpoint
            from model.sam_vggt_model import build_sam_vggt

            model = build_sam_vggt(
                device=str(self.device),
                sam_model_type=self.settings.sam_model_type,
                sam_encode_chunk=self.settings.sam_encode_chunk,
            )
            load_samvggt_checkpoint(model, self.settings.sam_v_ckpt, self.device)
            model.eval()
            self.model = model

    def predict(self, sample: dict[str, Any], amp_dtype: str) -> dict[str, Any]:
        if not self.infer_lock.acquire(blocking=False):
            raise InferenceBusyError("SAM-V is already running inference. Try again shortly.")
        try:
            if self.settings.mock:
                return mock_predict(sample)
            self.load()
            assert self.model is not None
            from benchmarks.compare_baseline_sam2 import infer_samvggt

            return infer_samvggt(self.model, sample, device=self.device, amp_dtype=amp_dtype)
        finally:
            self.infer_lock.release()


settings = Settings()
model_service = ModelService(settings)
infer_rate_limiter = InferenceRateLimiter(settings.infer_rate_limit_per_minute)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_output_dirs(RUNTIME_DIR, settings.output_ttl_seconds)
    if not settings.lazy:
        model_service.load()
    yield


app = FastAPI(title="SAM-V", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=RUNTIME_DIR), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "mock": settings.mock,
        "device": str(model_service.device),
        "max_frames": settings.max_frames,
        "max_upload_mb": settings.max_upload_mb,
        "output_ttl_seconds": settings.output_ttl_seconds,
        "infer_rate_limit_per_minute": settings.infer_rate_limit_per_minute,
        "model_loaded": model_service.model is not None,
    }


def _parse_prompts(payload: str) -> list[dict[str, Any]]:
    try:
        prompts = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="prompts must be valid JSON.") from exc
    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(status_code=400, detail="At least one prompt is required.")
    return prompts


async def _read_images(files: list[UploadFile]) -> list[UploadedImage]:
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required.")
    if len(files) > settings.max_frames:
        raise HTTPException(status_code=400, detail=f"At most {settings.max_frames} images are allowed.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    total = 0
    images: list[UploadedImage] = []
    for file in files:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not an image upload.")
        data = await file.read()
        total += len(data)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"Uploads exceed {settings.max_upload_mb} MB.")
        try:
            images.append(decode_image_bytes(file.filename or "upload", data))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return images


def _client_id(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        first = forwarded_for.split(",", 1)[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


@app.post("/api/infer")
async def infer(
    request: Request,
    images: Annotated[list[UploadFile], File()],
    prompts: Annotated[str, Form()],
    amp_dtype: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    client_id = _client_id(request)
    if not infer_rate_limiter.allow(client_id):
        raise HTTPException(
            status_code=429,
            detail=f"Too many inference requests. Limit is {settings.infer_rate_limit_per_minute} per minute.",
        )

    prompt_list = _parse_prompts(prompts)
    uploaded_images = await _read_images(images)
    try:
        sample = build_uploaded_sample(uploaded_images, prompt_list)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request_id = uuid.uuid4().hex
    output_dir = RUNTIME_DIR / request_id
    cleanup_output_dirs(RUNTIME_DIR, settings.output_ttl_seconds)
    dtype = amp_dtype or settings.amp_dtype
    if dtype not in {"fp16", "bf16", "none"}:
        raise HTTPException(status_code=400, detail="amp_dtype must be fp16, bf16, or none.")

    try:
        result, runtime_seconds = timed_call(model_service.predict, sample, dtype)
        frames = render_outputs(sample, result["pred_binary"], output_dir, f"/outputs/{request_id}")
    except InferenceBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    warnings = []
    if model_service.device.type == "cpu":
        warnings.append("Running on CPU; inference will be slow.")
    return {
        "frames": frames,
        "runtime_seconds": runtime_seconds,
        "warnings": warnings,
    }
