
"""
datasets.py — unified data layer for face/video deepfake detection.

Supports:
  (A) Frame-folder layout:
        data/frames/<dataset>/<split>/{real,fake}/<video_id>/frame_xxxx.jpg

  (B) CSV manifest:
        columns: path,label,video_id,dataset,split

The dataset groups frames by video_id and returns a clip tensor of shape
[T, C, H, W]. Compression augmentation is applied per-frame.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import CompressionAugment, FixedDegrade


class ResizeToTensorNormalize:
    """Minimal replacement for torchvision transforms."""
    def __init__(self, image_size: int):
        self.image_size = int(image_size)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return (ten - self.mean) / self.std


def _base_transform(image_size):
    return ResizeToTensorNormalize(image_size)


class VideoClipDataset(Dataset):
    def __init__(self, cfg, datasets, split, train=False, degrade=None):
        self.cfg = cfg
        self.split = split
        self.T = cfg.data.frames_per_clip
        self.image_size = cfg.data.image_size
        self.tf = _base_transform(self.image_size)
        self.aug = CompressionAugment(cfg.data.augment) if train else None
        self.degrade = degrade

        self.clips = []
        if cfg.data.get("manifest"):
            self._load_manifest(cfg.data.manifest, datasets, split)
        else:
            self._scan_folders(cfg.data.root, datasets, split)

        if not self.clips:
            print(f"[WARN] no clips found for split={split} datasets={datasets}")

    def _add_grouped(self, frames_by_video):
        for (vid, label, ds), frames in frames_by_video.items():
            frames = sorted(set(frames))
            if frames:
                self.clips.append((frames, label, vid, ds))

    def _scan_folders(self, root, datasets, split):
        root = Path(root)
        grouped = defaultdict(list)
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for ds in datasets:
            for label_name, label in (("real", 0), ("fake", 1)):
                base = root / ds / split / label_name
                if not base.exists():
                    continue
                for vid_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                    for img in vid_dir.iterdir():
                        if img.is_file() and img.suffix.lower() in exts:
                            grouped[(vid_dir.name, label, ds)].append(str(img))
        self._add_grouped(grouped)

    def _load_manifest(self, manifest, datasets, split):
        grouped = defaultdict(list)
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row.get("split") and row["split"] != split:
                    continue
                if datasets and row.get("dataset") not in datasets:
                    continue
                key = (row["video_id"], int(row["label"]), row.get("dataset", "?"))
                grouped[key].append(row["path"])
        self._add_grouped(grouped)

    def _sample_frames(self, frames):
        n = len(frames)
        if n >= self.T:
            if self.split == "train":
                start = random.randint(0, n - self.T)
                idx = list(range(start, start + self.T))
            else:
                idx = [min(n - 1, int(i * n / self.T)) for i in range(self.T)]
            return [frames[i] for i in idx]
        if n == 0:
            return []
        return frames + [frames[-1]] * (self.T - n)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        frames, label, vid, ds = self.clips[i]
        chosen = self._sample_frames(frames)
        tensors = []
        for fp in chosen:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                img = Image.new("RGB", (self.image_size, self.image_size))
            if self.degrade is not None:
                img = self.degrade(img)
            if self.aug is not None:
                img = self.aug(img)
            tensors.append(self.tf(img))
        if not tensors:
            tensors = [self.tf(Image.new("RGB", (self.image_size, self.image_size))) for _ in range(self.T)]
        clip = torch.stack(tensors, 0)
        return clip, label, vid


def make_loaders(cfg):
    from torch.utils.data import DataLoader
    pin = torch.cuda.is_available() and getattr(cfg, "device", "auto") != "cpu"
    train_ds = VideoClipDataset(cfg, cfg.data.train_datasets, "train", train=True)
    val_ds = VideoClipDataset(cfg, cfg.data.val_datasets, "val", train=False)
    common = dict(batch_size=cfg.train.batch_size,
                  num_workers=cfg.data.num_workers,
                  pin_memory=pin)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader
