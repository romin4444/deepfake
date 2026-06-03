"""
lora.py — minimal LoRA (Low-Rank Adaptation) for linear layers.

We wrap selected nn.Linear modules (attention qkv / proj) of a frozen
backbone with a low-rank residual:  y = W0 x + (B A) x * (alpha / r).
Only A and B are trainable, giving the parameter-efficient fine-tuning that
the research identified as the cross-dataset SOTA recipe (Forensics Adapter /
LNCLIP-DF lineage).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(rank, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero -> adapter starts as identity (no change to pretrained fn)

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(module: nn.Module, target_names, rank=8, alpha=16) -> int:
    """Recursively replace nn.Linear children whose attribute name contains
    any string in target_names. Returns number of layers adapted."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, rank, alpha))
            count += 1
        else:
            count += inject_lora(child, target_names, rank, alpha)
    return count
