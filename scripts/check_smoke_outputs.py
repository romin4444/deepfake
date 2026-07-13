"""
check_smoke_outputs.py — assert the zero-data mock smoke path produced the
expected artifacts.

Run after:
    python scripts/create_mock_data.py
    python -m src.train    --config configs/mock.yaml
    python -m src.evaluate --config configs/mock.yaml --ckpt outputs/mock_run/best.pt

This verifies the smoke path actually ran end-to-end (checkpoint, per-epoch
history, and a well-formed evaluation report) rather than just exiting 0. It
checks structure and value ranges, not exact metrics, so it stays deterministic
and non-flaky across environments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path("outputs/mock_run")


def _fail(msg: str) -> None:
    print(f"[check_smoke_outputs] FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    ckpt = RUN_DIR / "best.pt"
    history = RUN_DIR / "history.json"
    report = RUN_DIR / "eval_report.json"

    for p in (ckpt, history, report):
        if not p.exists():
            _fail(f"expected artifact missing: {p}")

    hist = json.loads(history.read_text())
    if not isinstance(hist, list) or len(hist) != 2:
        _fail(f"history.json should record 2 epochs, got {hist!r}")

    rep = json.loads(report.read_text())
    datasets = rep.get("datasets", {})
    if "mock_dataset" not in datasets:
        _fail(f"eval report missing mock_dataset entry: keys={list(datasets)}")

    clean = datasets["mock_dataset"].get("clean", {})
    auc = clean.get("auc")
    if not isinstance(auc, (int, float)) or not (0.0 <= auc <= 1.0):
        _fail(f"clean AUC not a valid metric in [0,1]: {auc!r}")

    battery = datasets["mock_dataset"].get("robustness", {})
    if not battery:
        _fail("robustness battery is empty; evaluation did not run fully")

    print(
        f"[check_smoke_outputs] OK: 2 epochs, clean AUC={auc:.4f}, "
        f"{len(battery)} robustness conditions"
    )


if __name__ == "__main__":
    main()
