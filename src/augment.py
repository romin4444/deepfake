
"""
augment.py — compression / social-media re-encoding augmentations.

The training-time goal is to make the detector survive the same laundering
pipeline that hurts real deployments: resize, JPEG, noise, and (optionally)
H.264 round-tripping when ffmpeg is available.
"""
from __future__ import annotations

import io
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def _h264_roundtrip(img: Image.Image, crf: int) -> Image.Image:
    """Best-effort H.264 re-encode of a single frame via ffmpeg.

    This is intentionally optional. If ffmpeg is unavailable, the function
    simply returns the input image.
    """
    if shutil.which("ffmpeg") is None:
        return img

    img = img.convert("RGB")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_png = td / "in.png"
        out_mp4 = td / "out.mp4"
        out_png = td / "out.png"
        img.save(in_png)

        cmd1 = [
            "ffmpeg", "-y", "-loglevel", "quiet",
            "-loop", "1", "-i", str(in_png),
            "-t", "0.2",
            "-vf", "scale=iw:ih",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", str(int(crf)),
            str(out_mp4),
        ]
        cmd2 = [
            "ffmpeg", "-y", "-loglevel", "quiet",
            "-i", str(out_mp4),
            "-frames:v", "1",
            str(out_png),
        ]
        try:
            subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if out_png.exists():
                return Image.open(out_png).convert("RGB")
        except Exception:
            pass
    return img


class CompressionAugment:
    def __init__(self, cfg):
        self.enabled = cfg.get("enabled", True)
        self.jpeg_quality = cfg.get("jpeg_quality", [40, 75, 95])
        self.resize_scales = cfg.get("resize_scales", [0.5, 0.75, 1.0])
        self.noise_sigma = cfg.get("gaussian_noise_sigma", [0.0, 2.0, 4.0])
        self.h264_crf = cfg.get("h264_crf", [])
        self.hflip_prob = cfg.get("hflip_prob", 0.5)
        self.h264_prob = cfg.get("h264_prob", 0.1)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img
        img = img.convert("RGB")

        if random.random() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        scale = random.choice(self.resize_scales)
        if scale != 1.0:
            w, h = img.size
            small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))),
                               Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)

        q = random.choice(self.jpeg_quality)
        if q < 100:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(q), optimize=True)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        if self.h264_crf and random.random() < self.h264_prob:
            img = _h264_roundtrip(img, random.choice(self.h264_crf))

        sigma = random.choice(self.noise_sigma)
        if sigma > 0:
            arr = np.asarray(img, np.float32)
            arr += np.random.normal(0, sigma, arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        return img


class FixedDegrade:
    """Deterministic single-op degradation for the robustness battery."""
    def __init__(self, op: str, level):
        self.op, self.level = op, level

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if self.op == "jpeg":
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(self.level), optimize=True)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        if self.op == "resize":
            w, h = img.size
            s = float(self.level)
            small = img.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        if self.op == "noise":
            arr = np.asarray(img, np.float32) + np.random.normal(0, float(self.level), (img.size[1], img.size[0], 3))
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if self.op == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=float(self.level)))
        return img


ROBUSTNESS_BATTERY = [
    ("clean",        "none",   0),
    ("jpeg_q40",     "jpeg",   40),
    ("jpeg_q30",     "jpeg",   30),
    ("resize_0.5",   "resize", 0.5),
    ("resize_0.35",  "resize", 0.35),
    ("noise_5",      "noise",  5),
    ("blur_1.5",     "blur",   1.5),
]
