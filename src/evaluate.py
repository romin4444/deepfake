"""
evaluate.py — full evaluation of a trained checkpoint.

  python -m src.evaluate --config configs/default.yaml --ckpt outputs/run1/best.pt

Produces, per test dataset:
  - clean cross-dataset metrics (AUC/ACC/EER/AP/P/R), video-level
  - robustness battery (JPEG/resize/noise/blur sweep)
  - calibrated probabilities (temperature scaling fit on val)
Writes outputs/<run>/eval_report.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .datasets import VideoClipDataset
from .model import DeepfakeVideoDetector
from .metrics import (compute_metrics, aggregate_video, pretty,
                      TemperatureScaler)
from .augment import FixedDegrade, ROBUSTNESS_BATTERY
from .train import pick_device


@torch.no_grad()
def _infer(model, loader, device):
    model.eval()
    s, y, v, logits = [], [], [], []
    for clip, label, vid in loader:
        out = model(clip.to(device))
        lg = out[:, 1] - out[:, 0]            # binary logit
        probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
        s.extend(probs.tolist())
        y.extend(label.numpy().tolist())
        v.extend(list(vid))
        logits.extend(lg.cpu().numpy().tolist())
    return np.array(s), np.array(y), v, np.array(logits)


def _loader(cfg, datasets, split, degrade=None):
    ds = VideoClipDataset(cfg, datasets, split, train=False, degrade=degrade)
    return DataLoader(ds, batch_size=cfg.train.batch_size,
                      num_workers=cfg.data.num_workers,
                      pin_memory=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    device = pick_device(cfg.device)

    model = DeepfakeVideoDetector(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"[eval] loaded {args.ckpt} (val_auc={state.get('val_auc')})")

    report = {"checkpoint": args.ckpt, "datasets": {}}

    # Optional calibration on val split
    scaler = None
    if cfg.eval.calibrate:
        val_s, val_y, _, val_logits = _infer(
            model, _loader(cfg, cfg.data.val_datasets, "val"), device
        )
        if len(np.unique(val_y)) > 1:
            scaler = TemperatureScaler().fit(val_logits, val_y)
            print(f"[eval] calibrated temperature T={scaler.T:.3f}")

    for ds in cfg.data.test_datasets:
        print(f"\n=== test dataset: {ds} ===")
        entry = {"clean": None, "robustness": {}}

        # clean
        s, y, v, lg = _infer(model, _loader(cfg, [ds], "test"), device)
        if len(y) == 0:
            print(f"  [skip] no data for {ds}")
            continue
        vy, vs = aggregate_video(v, s, y)
        m = compute_metrics(vy, vs, cfg.eval.threshold)
        entry["clean"] = m
        print("  " + pretty("clean", m))
        if scaler is not None:
            _, cvs = aggregate_video(v, scaler.transform(lg), y)
            entry["clean_calibrated"] = compute_metrics(vy, cvs,
                                                        cfg.eval.threshold)

        # robustness battery
        if cfg.eval.robustness_battery:
            for name, op, level in ROBUSTNESS_BATTERY:
                deg = None if op == "none" else FixedDegrade(op, level)
                s, y, v, _ = _infer(model, _loader(cfg, [ds], "test", deg),
                                    device)
                vy, vs = aggregate_video(v, s, y)
                rm = compute_metrics(vy, vs, cfg.eval.threshold)
                entry["robustness"][name] = rm
                print(f"    {name:14s} AUC={rm['auc']:.4f} ACC={rm['accuracy']:.4f}")

        report["datasets"][ds] = entry

    out = Path(cfg.output_dir) / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
