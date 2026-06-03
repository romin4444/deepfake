"""
train.py — training loop for the video deepfake detector.

  python -m src.train --config configs/default.yaml
  python -m src.train --config configs/default.yaml model.backbone=efficientnet_b4 train.epochs=10

Features: mixed precision (GPU), cosine LR with warmup, label smoothing,
gradient clipping, early stopping on val AUC, best-checkpoint saving.
"""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import load_config
from .datasets import make_loaders
from .model import DeepfakeVideoDetector
from .metrics import compute_metrics, aggregate_video, pretty


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pick_device(want):
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, video_level=True):
    model.eval()
    all_s, all_y, all_v = [], [], []
    for clip, label, vid in loader:
        clip = clip.to(device)
        logits = model(clip)
        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        all_s.extend(probs.tolist())
        all_y.extend(label.numpy().tolist())
        all_v.extend(list(vid))
    if video_level:
        y, s = aggregate_video(all_v, all_s, all_y)
    else:
        y, s = np.array(all_y), np.array(all_s)
    return compute_metrics(y, s, threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"[train] device={device}  output={out}")

    train_loader, val_loader = make_loaders(cfg)
    print(f"[train] train clips={len(train_loader.dataset)} "
          f"val clips={len(val_loader.dataset)}")

    model = DeepfakeVideoDetector(cfg).to(device)
    total, trn = model.count_params()
    print(f"[train] params total={total/1e6:.1f}M  trainable={trn/1e6:.3f}M "
          f"({100*trn/max(1,total):.2f}%)")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    use_amp = cfg.train.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup_steps = steps_per_epoch * cfg.train.warmup_epochs

    best_auc, best_epoch, patience = -1.0, -1, 0
    history = []
    gstep = 0

    # Wall-clock budget: stop training before this many hours elapse so the
    # job always finishes (checkpoint saved) within the Kaggle/time limit.
    budget_h = cfg.train.get("time_budget_hours", 0)
    deadline = time.time() + budget_h * 3600 if budget_h else None
    if deadline:
        print(f"[train] wall-clock budget = {budget_h}h "
              f"(will stop early to leave time for eval)")

    stop_for_time = False
    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for it, (clip, label, _) in enumerate(train_loader):
            clip, label = clip.to(device), label.to(device)
            lr = cosine_lr(gstep, total_steps, warmup_steps, cfg.train.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(clip)
                loss = crit(logits, label)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                         cfg.train.grad_clip)
            scaler.step(opt); scaler.update()
            running += loss.item(); gstep += 1
            if it % cfg.train.log_every == 0:
                print(f"  e{epoch} it{it}/{steps_per_epoch} "
                      f"loss={loss.item():.4f} lr={lr:.2e}")
            if deadline and time.time() > deadline:
                print(f"  [time] budget reached mid-epoch {epoch}; "
                      f"stopping training.")
                stop_for_time = True
                break

        val_m = evaluate(model, val_loader, device, cfg.eval.threshold,
                         cfg.eval.video_level)
        print(f"[epoch {epoch}] train_loss={running/steps_per_epoch:.4f} "
              f"({time.time()-t0:.1f}s) | " + pretty("val", val_m))
        history.append({"epoch": epoch, "train_loss": running/steps_per_epoch,
                        "val": val_m})

        if val_m["auc"] > best_auc:
            best_auc, best_epoch, patience = val_m["auc"], epoch, 0
            torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                        "epoch": epoch, "val_auc": best_auc},
                       out / "best.pt")
            print(f"  -> saved best (AUC={best_auc:.4f})")
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[train] early stop at epoch {epoch} "
                      f"(best AUC={best_auc:.4f} @ epoch {best_epoch})")
                break

        if stop_for_time:
            # always keep a usable checkpoint even if val didn't improve
            if best_epoch < 0:
                torch.save({"model": model.state_dict(), "cfg": dict(cfg),
                            "epoch": epoch, "val_auc": val_m["auc"]},
                           out / "best.pt")
                print("  -> saved checkpoint (time budget, first epoch)")
            print("[train] stopped on wall-clock budget.")
            break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done. best val AUC={best_auc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
