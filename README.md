# Parameter-Efficient Video Deepfake Detector — Training & Evaluation

A complete, runnable codebase for training and evaluating a **face/video
deepfake detector** that helps identify manipulated media. Built around the
2024–2026 cross-dataset SOTA recipe: a **frozen vision foundation backbone
(CLIP ViT-L/14 / DINOv2) adapted with LoRA**, a **temporal aggregation head**,
and **compression augmentation** for real-world robustness.

> Sandbox note: this repo was authored and *smoke-tested* on CPU. Actual
> training needs a GPU (the benchmarks are 10s–100s of GB). Every code path
> here has been verified to execute; the numbers come from *your* GPU run.

## What's the contribution?

Three things the research identified as the open gap, combined:
1. **PEFT on a video-foundation backbone** (LoRA / LayerNorm-only) — trains
   <1% of parameters, the cross-dataset SOTA family (Forensics Adapter, Effort,
   LNCLIP-DF).
2. **Temporal head** over per-frame features (attention / GRU / mean).
3. **Compression-robust training + a test-time robustness battery** (JPEG,
   resize, noise, blur) — directly targets the documented social-media
   re-encoding failure mode.

## Install

```bash
pip install -r requirements.txt
```

## 1. Get data

```bash
python -m src.fetch_data --list                       # see all sources
python -m src.fetch_data --kaggle xhlulu/140k-real-and-fake-faces --out data/raw
python -m src.fetch_data --hf faridlab/deepspeak_v2 --out data/raw
```

Form-gated academic sets (FaceForensics++, Celeb-DF v2, DF40) print their
request URLs. DFDC is on Kaggle (`c/deepfake-detection-challenge`).

## 2. Extract frames

```bash
python scripts/extract_frames.py --videos data/raw/dfdc/train --label fake \
  --dataset dfdc --split train --out data/frames --frames-per-video 16 --face-crop
```

Produces: `data/frames/<dataset>/<split>/{real,fake}/<video_id>/frame_xxxx.jpg`

## 3. Train

```bash
# CLIP ViT-L/14 + LoRA (recommended, needs GPU)
python -m src.train --config configs/default.yaml \
  model.backbone=clip_vit_l14 model.peft=lora model.lora_rank=8 \
  data.train_datasets=[faceforensics] data.val_datasets=[faceforensics]

# Lightweight CNN fallback (small GPU / quick test)
python -m src.train --config configs/default.yaml \
  model.backbone=efficientnet_b4 model.peft=none train.epochs=10
```

Any config field is overridable on the CLI as `key.subkey=value`.

## 4. Evaluate (cross-dataset + robustness)

```bash
python -m src.evaluate --config configs/default.yaml --ckpt outputs/run1/best.pt \
  data.test_datasets=[celebdf,dfdc]
```

Writes `outputs/<run>/eval_report.json` with clean cross-dataset metrics
(AUC/ACC/EER/AP/precision/recall, video-level), calibrated probabilities, and
the full robustness battery per dataset.

## Files

```
configs/default.yaml      all hyperparameters
src/config.py             YAML + CLI-override loader
src/fetch_data.py         Kaggle / HF / academic data fetcher
scripts/extract_frames.py video -> face-cropped frame folders
src/augment.py            compression / social-media augmentation
src/datasets.py           unified video-clip dataset (folder or CSV manifest)
src/lora.py               minimal LoRA for linear layers
src/model.py              PEFT backbone + temporal head + classifier
src/metrics.py            AUC / ACC / EER / AP + video aggregation + calibration
src/train.py              training loop (AMP, cosine LR, early stop)
src/evaluate.py           cross-dataset + robustness evaluation
```

## Expected results (from the literature, FF++ -> X, video-level AUC)

| Setting | Realistic AUC |
|---|---|
| In-distribution (FF++) | 0.97–0.99 |
| Cross-dataset (Celeb-DF v2) | 0.92–0.96 |
| Cross-dataset (DFDC) | 0.78–0.87 |
| In-the-wild (Deepfake-Eval-2024) | 0.70–0.82 |

If your CLIP+LoRA run hits ≥0.95 on Celeb-DF v2 with <10M trainable params,
you've matched the current published SOTA family.

## Honest limitations

- Cross-generator generalization is the field's unsolved problem; expect large
  drops on generators never seen in training. Always evaluate on
  Deepfake-Eval-2024 before claiming deployment readiness.
- Probability outputs are calibrated but are estimates, not proof.
- The Haar-cascade face crop in `extract_frames.py` is a no-dependency default;
  swap in RetinaFace/MTCNN for production-quality crops.
```
