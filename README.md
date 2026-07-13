# Parameter-Efficient Video Deepfake Detector

[![CI](https://github.com/romin4444/deepfake/actions/workflows/ci.yml/badge.svg)](https://github.com/romin4444/deepfake/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

A small, **runnable** training/evaluation codebase for a face/video **deepfake
detector**. It implements a modern cross-dataset recipe — a frozen vision
foundation backbone (CLIP ViT-L/14 or DINOv2) adapted with **LoRA**, a
**temporal aggregation head**, and **compression augmentation** for robustness
to social-media re-encoding — and wires it to a clean data → train → evaluate
pipeline with cross-dataset and robustness metrics.

> **Status: research / educational.** The code is CPU smoke-tested end-to-end in
> CI, but real training needs a GPU and large, access-gated datasets. **No
> trained checkpoint and no benchmark numbers are bundled** — the metrics you
> get come from *your* run. Performance figures in this repo are literature-
> derived expectations (cited below), not measurements of this code.

## Contents

- [Quick start (no GPU, ~1 min)](#quick-start-no-gpu-1-min)
- [How it works](#how-it-works)
- [Full workflow (GPU)](#full-workflow-gpu)
- [Run on Kaggle](#run-on-kaggle)
- [Expected results](#expected-results-literature-not-measured-here)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Responsible use](#responsible-use)
- [Citations & references](#citations--references)
- [Contributing](#contributing) · [License](#license)

## Quick start (no GPU, ~1 min)

Verify the whole pipeline on a synthetic mock dataset — no downloads, no GPU.
This is exactly what CI exercises.

```bash
pip install -r requirements.txt        # torch, numpy, scikit-learn, pillow, pyyaml, tqdm
python scripts/create_mock_data.py     # writes data/frames/mock_dataset/...
python -m src.train    --config configs/mock.yaml
python -m src.evaluate --config configs/mock.yaml --ckpt outputs/mock_run/best.pt
```

You'll see a 2-epoch training run and an evaluation report (clean metrics +
robustness battery) written to `outputs/mock_run/eval_report.json`. The mock
config uses a tiny CNN backbone so it runs anywhere; swap in `configs/default.yaml`
and a real dataset for actual training.

Run the tests:

```bash
pip install pytest ruff
pytest tests/ -v
ruff check src/ scripts/ tests/
```

## How it works

1. **Parameter-efficient backbone.** A frozen CLIP/DINOv2 (or an EfficientNet
   fallback) is adapted with LoRA on its attention projections, so only a small
   fraction of parameters train. If `open_clip`/`timm` aren't installed, a small
   built-in CNN backbone is used automatically so the code always runs.
2. **Temporal head.** Per-frame features are aggregated per clip with a
   transformer attention pool, a GRU, or mean pooling (`model.temporal`).
3. **Compression-robust training.** Every training frame is randomly degraded
   (JPEG, resize, noise, optional H.264 via ffmpeg) to emulate the laundering
   that hurts real-world detection. Evaluation includes a fixed robustness
   battery so you can see the degradation curve.

Metrics are video-level by default (frame scores averaged per `video_id`) and
include AUC/ACC/EER/AP/precision/recall plus temperature-scaling calibration.

## Full workflow (GPU)

```bash
# 1) Discover data sources (Kaggle / HuggingFace / form-gated academic sets)
python -m src.fetch_data --list
python -m src.fetch_data --kaggle xhlulu/140k-real-and-fake-faces --out data/raw

# 2) Videos -> face-cropped frame folders
python scripts/extract_frames.py --videos data/raw/dfdc/train --label fake \
  --dataset dfdc --split train --out data/frames --frames-per-video 16 --face-crop

# 3) Train (any config field is overridable as key.subkey=value)
python -m src.train --config configs/default.yaml \
  model.backbone=clip_vit_l14 model.peft=lora \
  'data.train_datasets=[faceforensics]' 'data.val_datasets=[faceforensics]'

# 4) Cross-dataset + robustness evaluation
python -m src.evaluate --config configs/default.yaml --ckpt outputs/run1/best.pt \
  'data.test_datasets=[celebdf,dfdc]'
```

Frame layout expected by the loader:
`data/frames/<dataset>/<split>/{real,fake}/<video_id>/frame_xxxx.jpg`
(a CSV manifest with `path,label,video_id,dataset,split` also works).

> Note: quote CLI list overrides (`'data.test_datasets=[celebdf,dfdc]'`) so your
> shell doesn't interpret the brackets.

Form-gated academic sets (FaceForensics++, Celeb-DF v2, DF40) print their
request URLs; DFDC is on Kaggle. This repo bundles **no** dataset.

## Run on Kaggle

The project runs directly in a Kaggle GPU notebook. See **[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)**
for the full walkthrough, and:

- **[KAGGLE_NOTEBOOK.py](KAGGLE_NOTEBOOK.py)** — cell-by-cell notebook for a
  DFDC/140k-faces run (extract frames → train → evaluate → save to
  `/kaggle/working/final`).
- **[KAGGLE_NOTEBOOK_8H.py](KAGGLE_NOTEBOOK_8H.py)** — a time-budgeted first run
  (`configs/fast8h.yaml`) that downloads small subsets and auto-stops training
  to finish inside Kaggle's session limit.

If you just want to confirm the code runs on Kaggle before wiring up a large
dataset, run the [quick start](#quick-start-no-gpu-1-min) mock commands in a
notebook cell — they need no input data.

## Expected results (literature, not measured here)

These ranges come from published cross-dataset deepfake-detection work and are
provided as sanity-check targets for your own runs — **they are not results of
this code.** No checkpoint or benchmark output is shipped in this repo.

| Setting | Reported AUC (literature) |
|---|---|
| In-distribution (FF++) | ~0.97–0.99 |
| Cross-dataset (Celeb-DF v2) | ~0.92–0.96 |
| Cross-dataset (DFDC) | ~0.78–0.87 |
| In-the-wild (Deepfake-Eval-2024) | ~0.70–0.82 |

The in-the-wild row is the one that matters for deployment: the
Deepfake-Eval-2024 benchmark reports commercial detectors around ~0.78 on real
social-media content, far below in-distribution numbers.

## Project layout

```
configs/            default / kaggle / fast8h / mock YAML configs
src/config.py       YAML + dotted-key CLI-override loader
src/fetch_data.py   dataset source registry + Kaggle/HF download helpers
src/download_latest.py  direct-download helpers for a time-boxed first run
scripts/extract_frames.py  video -> face-cropped frame folders (OpenCV)
scripts/create_mock_data.py  synthetic dataset for the quick start / tests
src/augment.py      compression / social-media augmentation + robustness battery
src/datasets.py     video-clip dataset (frame folders or CSV manifest)
src/lora.py         minimal LoRA for linear layers
src/model.py        backbone + temporal head + classifier
src/metrics.py      AUC/ACC/EER/AP + video aggregation + temperature scaling
src/train.py        training loop (AMP, cosine LR, early stop, time budget)
src/evaluate.py     cross-dataset + robustness evaluation
tests/              CPU smoke tests (run in CI)
experiments/        exploratory scratch work — see experiments/README.md
```

## Limitations

- **Cross-generator generalization is unsolved.** Expect large drops on
  generators never seen in training; evaluate on an in-the-wild set (e.g.
  Deepfake-Eval-2024) before making any deployment claim.
- **Probabilities are calibrated estimates, not proof.** A high score is not
  forensic evidence.
- **The default face crop is a Haar cascade** (`extract_frames.py`) chosen for
  zero extra dependencies; swap in RetinaFace/MTCNN for production-quality crops.
- **No pretrained weights or benchmark results are provided.** You must train on
  your own data/GPU.

## Responsible use

This is a **detection** tool for identifying manipulated media, intended for
research, education, media forensics, and platform-integrity work. Please:

- **Do not** treat model outputs as definitive proof of manipulation. Deepfake
  detection is probabilistic and brittle; a qualified human should review before
  any consequential decision.
- **Do not** use this project to build, improve, or evade detection of
  non-consensual, abusive, or deceptive synthetic media.
- Respect the **licenses and consent terms** of every dataset you use. Several
  referenced datasets are access-gated for good reason.
- Be aware of the real-world harms this technology enables — see
  [`reports/Deepfake_Numbers_Factcheck_Report.md`](reports/Deepfake_Numbers_Factcheck_Report.md)
  for a sourced overview.

## Citations & references

The recipe follows a well-documented line of cross-dataset detection work.
Please cite the primary sources rather than this repository:

- **CLIP** — Radford et al., *Learning Transferable Visual Models From Natural
  Language Supervision*, 2021.
- **DINOv2** — Oquab et al., *DINOv2: Learning Robust Visual Features without
  Supervision*, 2023.
- **LoRA** — Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- **PEFT-for-detection lineage** — e.g. *Forensics Adapter*, *Effort*, and
  CLIP/LayerNorm-tuning approaches (LNCLIP-DF) for cross-dataset deepfake
  detection (2024–2025).
- **Benchmarks** — FaceForensics++ (Rössler et al., 2019), Celeb-DF v2 (Li et
  al., 2020), DFDC (Dolhansky et al., 2020), Deepfake-Eval-2024 (Chandra et al.,
  HuggingFace `nuriachandra/Deepfake-Eval-2024`).

## Contributing

Issues and PRs welcome. Before opening a PR:

```bash
ruff check src/ scripts/ tests/
pytest tests/ -v
```

Keep changes focused, add/adjust tests for behavior changes, and avoid
committing datasets, checkpoints, or generated outputs (see `.gitignore`).

## License

MIT — see [LICENSE](LICENSE).
