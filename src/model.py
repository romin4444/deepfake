
"""
model.py — backbone + temporal head + classifier for video deepfake detection.

The design intentionally keeps the backbone frozen and uses parameter-efficient
adaptation (LoRA or LayerNorm tuning) so the detector can be trained on limited
GPU budgets while preserving cross-dataset transfer.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .lora import inject_lora


class TinyBackbone(nn.Module):
    """Pure-PyTorch fallback that runs in a sandbox without timm/open_clip.

    This is not the production backbone. It exists so the codebase can be
    imported and smoke-tested anywhere.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)
        self.num_features = out_dim

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.proj(x)


def _build_backbone(name, pretrained):
    """Return (module, feature_dim, forward_fn)."""
    if name == "tiny_cnn":
        m = TinyBackbone(out_dim=256)
        return m, m.num_features, lambda mod, x: mod(x)

    if name == "efficientnet_b4":
        try:
            import timm
            m = timm.create_model("efficientnet_b4", pretrained=pretrained,
                                  num_classes=0)
            return m, m.num_features, lambda mod, x: mod(x)
        except Exception as e:
            print(f"[model] timm unavailable, falling back to tiny_cnn: {e}")
            m = TinyBackbone(out_dim=256)
            return m, m.num_features, lambda mod, x: mod(x)

    if name == "dinov2_vits14":
        try:
            import timm
            m = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                  pretrained=pretrained, num_classes=0,
                                  img_size=224)
            return m, m.num_features, lambda mod, x: mod(x)
        except Exception as e:
            print(f"[model] DINOv2 unavailable, falling back to tiny_cnn: {e}")
            m = TinyBackbone(out_dim=256)
            return m, m.num_features, lambda mod, x: mod(x)

    if name == "clip_vit_l14":
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai" if pretrained else None
            )
            visual = model.visual
            dim = getattr(visual, "output_dim", 768)
            return visual, dim, lambda mod, x: mod(x)
        except Exception as e:
            print(f"[model] open_clip unavailable, falling back to tiny_cnn: {e}")
            m = TinyBackbone(out_dim=256)
            return m, m.num_features, lambda mod, x: mod(x)

    raise ValueError(f"unknown backbone {name!r}")


class AttentionPool(nn.Module):
    def __init__(self, dim, layers=2, heads=8, dropout=0.1):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], 1)
        x = self.encoder(x)
        return x[:, 0]


class TemporalHead(nn.Module):
    def __init__(self, kind, dim, layers, dropout):
        super().__init__()
        self.kind = kind
        if kind == "attention":
            # IMPORTANT: pass dropout by keyword; positional use would fill `heads`
            # and break the encoder construction.
            self.pool = AttentionPool(dim, layers=layers, dropout=dropout)
        elif kind == "gru":
            self.pool = nn.GRU(dim, dim, layers, batch_first=True)
        elif kind == "mean":
            self.pool = None
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "mean":
            return x.mean(1)
        if self.kind == "gru":
            out, _ = self.pool(x)
            return out[:, -1]
        return self.pool(x)


class DeepfakeVideoDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mc = cfg.model
        self.backbone, dim, self._bb_forward = _build_backbone(
            mc.backbone, mc.pretrained
        )

        if mc.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if mc.peft == "lora":
            n = inject_lora(self.backbone, mc.lora_targets, mc.lora_rank,
                            mc.lora_alpha)
            print(f"[model] injected LoRA into {n} linear layers "
                  f"(rank={mc.lora_rank})")
        elif mc.peft == "layernorm":
            tuned = 0
            for m in self.backbone.modules():
                if isinstance(m, (nn.LayerNorm,)):
                    for p in m.parameters():
                        p.requires_grad = True
                        tuned += p.numel()
            print(f"[model] LayerNorm-only fine-tuning enabled (~{tuned} params)")
        elif mc.peft not in ("none", None):
            raise ValueError(f"unknown peft mode {mc.peft!r}")

        self.temporal = TemporalHead(mc.temporal, dim, mc.temporal_layers,
                                     mc.dropout)
        self.dropout = nn.Dropout(mc.dropout)
        self.classifier = nn.Linear(dim, mc.num_classes)
        self.feat_dim = dim

    def forward(self, clip):
        b, t, c, h, w = clip.shape
        frames = clip.reshape(b * t, c, h, w)
        feats = self._bb_forward(self.backbone, frames)
        if feats.dim() == 4:
            feats = feats.flatten(1)
        feats = feats.reshape(b, t, -1)
        clip_feat = self.temporal(feats)
        return self.classifier(self.dropout(clip_feat))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, train
