"""
config.py — load YAML config and apply dotted-key CLI overrides.

Usage:
    cfg = load_config("configs/default.yaml",
                      overrides=["model.lora_rank=16", "train.epochs=30"])
"""
from __future__ import annotations
import copy
import yaml


def _set_dotted(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _coerce(s: str):
    """Turn a CLI string into bool/int/float/list/None where possible."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x.strip()) for x in inner.split(",")]
    return s


class Cfg(dict):
    """Dict with attribute access (cfg.model.lora_rank)."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def load_config(path: str, overrides: list[str] | None = None) -> Cfg:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override (need key=value): {ov}")
        key, val = ov.split("=", 1)
        _set_dotted(cfg, key.strip(), _coerce(val.strip()))
    return Cfg(cfg)


if __name__ == "__main__":
    c = load_config("configs/default.yaml",
                    overrides=["model.lora_rank=32", "train.epochs=5"])
    print("lora_rank:", c.model.lora_rank, "| epochs:", c.train.epochs)
    print("train_datasets:", c.data.train_datasets)
