from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from demos.web.inference import build_sample_from_image_arrays
from demos.web.inference import (
    UploadedImage,
    build_uploaded_sample,
    cleanup_output_dirs,
    render_outputs,
)


class FakeUploadFile:
    def __init__(self, filename: str, content_type: str, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data

    async def read(self) -> bytes:
        return self._data


class WebDemoInferenceTests(unittest.TestCase):
    def test_prompt_scaling_uses_prompt_frame_size(self) -> None:
        arrays = [
            np.zeros((50, 100, 3), dtype=np.float32),
            np.zeros((100, 200, 3), dtype=np.float32),
        ]
        sample = build_sample_from_image_arrays(
            stems=["a", "b"],
            arrays=arrays,
            prompts=[{"frame_index": 1, "x": 100, "y": 50, "label": 1}],
        )

        coord = sample["point_coords"][0].tolist()
        self.assertAlmostEqual(coord[0], 512.0, places=4)
        self.assertAlmostEqual(coord[1], 512.0, places=4)
        self.assertEqual(sample["images"].shape[-2:], (1024, 1024))

    def test_build_uploaded_sample_rejects_out_of_bounds_prompt(self) -> None:
        image = UploadedImage(
            filename="a.png",
            stem="a",
            array=np.zeros((20, 30, 3), dtype=np.float32),
            width=30,
            height=20,
        )
        with self.assertRaises(ValueError):
            build_uploaded_sample(
                [image],
                [{"frame_index": 0, "x": 30, "y": 10, "label": 1}],
            )

    def test_render_outputs_writes_overlay_and_mask(self) -> None:
        arrays = [np.zeros((20, 30, 3), dtype=np.float32)]
        sample = build_sample_from_image_arrays(
            stems=["a"],
            arrays=arrays,
            prompts=[{"frame_index": 0, "x": 10, "y": 10, "label": 1}],
        )
        pred = np.zeros((1, 1024, 1024), dtype=bool)
        pred[0, 100:120, 100:120] = True

        with tempfile.TemporaryDirectory() as td:
            frames = render_outputs(sample, pred, output_dir=Path(td), url_prefix="/outputs/test")
            overlay_path = os.path.join(td, "frame_000_overlay.png")
            mask_path = os.path.join(td, "frame_000_mask.png")
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["foreground_pixels"], 400)
            self.assertTrue(os.path.isfile(overlay_path))
            self.assertTrue(os.path.isfile(mask_path))

            with Image.open(overlay_path) as overlay:
                self.assertEqual(overlay.size, (1024, 1024))
                self.assertEqual(overlay.mode, "RGBA")
            with Image.open(mask_path) as mask:
                self.assertEqual(mask.size, (1024, 1024))
                self.assertEqual(mask.mode, "L")

    def test_cleanup_output_dirs_removes_only_expired_request_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_dir = root / "old_request"
            fresh_dir = root / "fresh_request"
            old_dir.mkdir()
            fresh_dir.mkdir()
            (old_dir / "frame_000_overlay.png").write_text("old", encoding="utf-8")
            (fresh_dir / "frame_000_overlay.png").write_text("fresh", encoding="utf-8")
            (root / ".gitignore").write_text("*", encoding="utf-8")

            os.utime(old_dir, (800.0, 800.0))
            os.utime(fresh_dir, (950.0, 950.0))

            removed = cleanup_output_dirs(root, ttl_seconds=100, now=1000.0)

            self.assertEqual(removed, 1)
            self.assertFalse(old_dir.exists())
            self.assertTrue(fresh_dir.is_dir())
            self.assertTrue((root / ".gitignore").is_file())

    def test_cleanup_output_dirs_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_dir = root / "old_request"
            old_dir.mkdir()
            os.utime(old_dir, (800.0, 800.0))

            removed = cleanup_output_dirs(root, ttl_seconds=0, now=1000.0)

            self.assertEqual(removed, 0)
            self.assertTrue(old_dir.is_dir())


@unittest.skipIf(
    importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("multipart") is None,
    "FastAPI upload support is an optional web dependency.",
)
class WebDemoAppTests(unittest.TestCase):
    def assert_http_error(self, coro, status_code: int) -> None:
        with self.assertRaises(Exception) as ctx:
            asyncio.run(coro)
        self.assertEqual(ctx.exception.status_code, status_code)

    def test_app_imports_in_mock_lazy_mode(self) -> None:
        os.environ["SAM_VGGT_MOCK"] = "1"
        os.environ["SAM_VGGT_LAZY"] = "1"

        from demos.web.app import model_service, settings

        self.assertTrue(settings.mock)
        self.assertTrue(settings.lazy)
        self.assertEqual(settings.max_frames, 16)
        self.assertEqual(settings.max_upload_mb, 128)
        self.assertEqual(settings.output_ttl_seconds, 3600)
        self.assertEqual(settings.infer_rate_limit_per_minute, 10)
        self.assertIsNone(model_service.model)

    def test_app_title_is_sam_v(self) -> None:
        from demos.web.app import app

        self.assertEqual(app.title, "SAM-V")

    def test_infer_rate_limiter_limits_per_client(self) -> None:
        from demos.web.app import InferenceRateLimiter

        limiter = InferenceRateLimiter(limit_per_minute=2, window_seconds=60)
        self.assertTrue(limiter.allow("1.2.3.4", now=1000.0))
        self.assertTrue(limiter.allow("1.2.3.4", now=1001.0))
        self.assertFalse(limiter.allow("1.2.3.4", now=1002.0))
        self.assertTrue(limiter.allow("5.6.7.8", now=1002.0))
        self.assertTrue(limiter.allow("1.2.3.4", now=1061.0))

    def test_model_service_reports_busy_without_queueing(self) -> None:
        from demos.web.app import InferenceBusyError, model_service

        acquired = model_service.infer_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            with self.assertRaises(InferenceBusyError):
                model_service.predict({}, "none")
        finally:
            model_service.infer_lock.release()

    def test_read_images_rejects_empty_upload_list(self) -> None:
        from demos.web.app import _read_images

        self.assert_http_error(_read_images([]), 400)

    def test_read_images_rejects_non_image_file_type(self) -> None:
        from demos.web.app import _read_images

        files = [FakeUploadFile("notes.txt", "text/plain", b"not an image")]
        self.assert_http_error(_read_images(files), 400)

    def test_read_images_rejects_empty_file(self) -> None:
        from demos.web.app import _read_images

        files = [FakeUploadFile("empty.png", "image/png", b"")]
        self.assert_http_error(_read_images(files), 400)

    def test_read_images_rejects_frame_limit(self) -> None:
        from demos.web.app import _read_images, settings

        files = [
            FakeUploadFile(f"{i}.png", "image/png", b"unused")
            for i in range(settings.max_frames + 1)
        ]
        self.assert_http_error(_read_images(files), 400)


if __name__ == "__main__":
    unittest.main()
