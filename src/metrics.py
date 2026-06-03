"""
metrics.py — the metrics IEEE venues expect for deepfake detection:
AUC, accuracy, EER, AP, precision/recall, plus video-level aggregation
and temperature-scaling calibration.
"""
from __future__ import annotations
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import (roc_auc_score, accuracy_score, roc_curve,
                             average_precision_score, precision_score,
                             recall_score, confusion_matrix)


def equal_error_rate(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def aggregate_video(video_ids, scores, labels):
    """Average frame/clip scores per video_id -> one score & label per video."""
    by_vid_s, by_vid_y = defaultdict(list), {}
    for v, s, y in zip(video_ids, scores, labels):
        by_vid_s[v].append(s)
        by_vid_y[v] = y
    vids = list(by_vid_s.keys())
    vs = np.array([np.mean(by_vid_s[v]) for v in vids])
    vy = np.array([by_vid_y[v] for v in vids])
    return vy, vs


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    out = {}
    # AUC/AP need both classes present
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, scores))
        out["ap"] = float(average_precision_score(y_true, scores))
        out["eer"], out["eer_thr"] = equal_error_rate(y_true, scores)
    else:
        out["auc"] = out["ap"] = out["eer"] = float("nan")
        out["eer_thr"] = 0.5
    out["accuracy"] = float(accuracy_score(y_true, preds))
    out["precision"] = float(precision_score(y_true, preds, zero_division=0))
    out["recall"] = float(recall_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(
        y_true, preds, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return out


class TemperatureScaler:
    """Single-parameter calibration: p = sigmoid(logit / T). Fit on val."""
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, labels, max_iter=200, lr=0.01):
        logits = torch.tensor(np.asarray(logits), dtype=torch.float32)
        labels = torch.tensor(np.asarray(labels), dtype=torch.float32)
        T = torch.nn.Parameter(torch.ones(1))
        opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)
        bce = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = bce(logits / T.clamp(min=1e-3), labels)
            loss.backward()
            return loss
        opt.step(closure)
        self.T = float(T.detach().clamp(min=1e-3))
        return self

    def transform(self, logits):
        return 1.0 / (1.0 + np.exp(-np.asarray(logits) / self.T))


def pretty(name, m):
    return (f"{name:16s} AUC={m['auc']:.4f} ACC={m['accuracy']:.4f} "
            f"EER={m['eer']:.4f} AP={m['ap']:.4f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f}")
