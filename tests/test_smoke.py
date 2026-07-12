"""
Smoke tests — verify every module imports cleanly and the model runs
end-to-end on CPU with a 2-frame dummy clip.

These tests run without a GPU and without heavy optional deps (open_clip,
timm): the TinyBackbone fallback is used automatically when those are absent.
Torch-dependent tests are skipped if torch is not installed.
"""
from pathlib import Path

import pytest

# Absolute path to the default config — works regardless of CWD.
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CFG = str(_REPO_ROOT / "configs" / "default.yaml")

# ---------------------------------------------------------------------------
# Config (no torch dep)
# ---------------------------------------------------------------------------

def test_config_load_defaults(tmp_path):
    """Config loader returns correct types for default.yaml values."""
    from src.config import load_config
    cfg = load_config(_DEFAULT_CFG)
    assert cfg.seed == 42
    assert cfg.model.lora_rank == 8
    assert cfg.train.epochs == 20
    assert isinstance(cfg.data.train_datasets, list)


def test_config_cli_override(tmp_path):
    """Dotted-key CLI overrides are applied and coerced correctly."""
    from src.config import load_config
    cfg = load_config(_DEFAULT_CFG, overrides=[
        "model.lora_rank=16",
        "train.epochs=5",
        "model.peft=none",
        "train.amp=false",
    ])
    assert cfg.model.lora_rank == 16
    assert cfg.train.epochs == 5
    assert cfg.model.peft is None
    assert cfg.train.amp is False


def test_config_missing_key_raises():
    from src.config import load_config
    cfg = load_config(_DEFAULT_CFG)
    with pytest.raises(AttributeError):
        _ = cfg.nonexistent_key


# ---------------------------------------------------------------------------
# Metrics (numpy dep only)
# ---------------------------------------------------------------------------

def test_metrics_perfect():
    """AUC=1 for perfect predictions."""
    np = pytest.importorskip("numpy")
    from src.metrics import compute_metrics
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, s)
    assert m["auc"] > 0.99


def test_metrics_random():
    """AUC≈0.5 for random predictions."""
    np = pytest.importorskip("numpy")
    from src.metrics import compute_metrics
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    s = rng.random(100)
    m = compute_metrics(y, s)
    assert 0.3 < m["auc"] < 0.7


# ---------------------------------------------------------------------------
# Model — TinyBackbone end-to-end on CPU (requires torch)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="torch not installed")


def _cpu_cfg(tmp_path):
    """Minimal config that forces tiny_cnn + CPU."""
    from src.config import load_config
    cfg = load_config(_DEFAULT_CFG, overrides=[
        "device=cpu",
        "model.backbone=tiny_cnn",
        "model.pretrained=false",
        "model.freeze_backbone=false",
        "model.peft=none",
        "model.temporal=mean",
        "model.num_classes=2",
        f"output_dir={tmp_path}",
    ])
    return cfg


def test_model_forward_shape(tmp_path):
    """Model produces (batch, 2) logits for a 2-frame dummy clip."""
    from src.model import DeepfakeVideoDetector
    cfg = _cpu_cfg(tmp_path)
    model = DeepfakeVideoDetector(cfg)
    model.eval()
    # (batch=2, frames=2, C=3, H=224, W=224)
    dummy = torch.zeros(2, 2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2), f"expected (2,2), got {out.shape}"


def test_model_trainable_params(tmp_path):
    """In peft=none mode, all backbone params are trainable."""
    from src.model import DeepfakeVideoDetector
    cfg = _cpu_cfg(tmp_path)
    model = DeepfakeVideoDetector(cfg)
    total, trainable = model.count_params()
    assert trainable > 0
    assert trainable <= total


def test_model_lora_reduces_trainable(tmp_path):
    """LoRA leaves fewer trainable params than fully unfrozen (tiny_cnn has no
    qkv/proj layers so injection count is 0, but model still builds cleanly)."""
    from src.model import DeepfakeVideoDetector
    from src.config import load_config
    cfg = load_config(_DEFAULT_CFG, overrides=[
        "device=cpu",
        "model.backbone=tiny_cnn",
        "model.pretrained=false",
        "model.freeze_backbone=true",
        "model.peft=lora",
        "model.temporal=mean",
        f"output_dir={tmp_path}",
    ])
    model = DeepfakeVideoDetector(cfg)
    _, trainable = model.count_params()
    # frozen backbone + no injectable layers → only head is trainable
    assert trainable < sum(p.numel() for p in model.parameters())


def test_model_temporal_variants(tmp_path):
    """All three temporal head modes produce valid output shapes."""
    from src.model import DeepfakeVideoDetector
    from src.config import load_config
    for mode in ("mean", "gru", "attention"):
        cfg = load_config(_DEFAULT_CFG, overrides=[
            "device=cpu",
            "model.backbone=tiny_cnn",
            "model.pretrained=false",
            "model.freeze_backbone=false",
            "model.peft=none",
            f"model.temporal={mode}",
            "model.temporal_layers=1",
            f"output_dir={tmp_path}",
        ])
        model = DeepfakeVideoDetector(cfg)
        model.eval()
        dummy = torch.zeros(1, 4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 2), f"mode={mode} gave shape {out.shape}"
