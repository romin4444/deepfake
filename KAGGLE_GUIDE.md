# Kaggle Training Guide: Compression-Robust Video Deepfake Detector

This document explains how to run the detector on Kaggle, which datasets to use, and what results to expect.

## Quick Start (5 minutes)

1. **Create a Kaggle notebook**
2. **Add the code as a dataset** (upload the `dfvideo/` folder as a dataset to Kaggle)
3. **Add data as input** (click "Add input" → select a dataset like DFDC or 140k-faces)
4. **Open `KAGGLE_NOTEBOOK.py`** — copy-paste each cell into your notebook
5. **Run cells sequentially**

That's it. The notebook handles:
- Frame extraction (videos → face-cropped frames)
- Training (15 epochs, ~12h)
- Cross-dataset evaluation + robustness battery

---

## Dataset Choice: Which one to use?

You have different options depending on what you want to optimize for:

### Option A: Quick warmup on image data (1–2 hours)
**Dataset:** 140k-real-and-fake-faces  
**Size:** ~4 GB  
**Use case:** Test the code, verify frame extraction works, sanity check  
**Expected AUC:** ~85–92% (images, not video)  
**Kaggle storage:** No problem  

```yaml
# In configs/kaggle.yaml:
train_datasets: [140k_faces]
val_datasets: [140k_faces]
test_datasets: [140k_faces]
```

### Option B: Standard benchmark (6–12 hours)
**Dataset:** DFDC (sample subset, available on Kaggle)  
**Size:** ~50–100 GB  
**Use case:** Train on a realistic benchmark, test cross-dataset  
**Expected AUC:** ~85–92% (depends on your training tricks)  
**Kaggle storage:** ~150 GB working (tight but doable with Kaggle+)  

```yaml
# In configs/kaggle.yaml:
train_datasets: [dfdc]
val_datasets: [dfdc]
test_datasets: [celebdf, dfdc]   # or deepspeak if available
```

### Option C: Robustness-focused (12–24 hours, if storage available)
**Training:** DFDC  
**Validation:** DeepSpeak v2 (HuggingFace, different generation method)  
**Test:** Deepfake-Eval-2024 (in-the-wild, ~45h video)  
**Use case:** Prove compression robustness in real social-media scenarios  
**Expected AUC:** 
- Clean on DFDC: ~87–92%
- Cross-dataset (DeepSpeak): ~75–85%
- In-the-wild (Deepfake-Eval): ~65–80% (the *real* number)

**Storage:** ~300 GB (need Kaggle Pro/Teams)

```yaml
train_datasets: [dfdc]
val_datasets: [deepspeak]
test_datasets: [deepfake_eval]
```

---

## My Recommendation

**Start with Option B (DFDC)** because:
1. It's on Kaggle already (no download step).
2. Realistic benchmark size.
3. Standard protocol (train on DFDC, test on Celeb-DF + DFDC = established cross-dataset evaluation).
4. Fits in Kaggle storage.

**Once Option B works**, upgrade to **Option C** if you want to claim robustness (this is the novel contribution — in-the-wild performance on Deepfake-Eval-2024).

---

## Dataset Details Table

| Dataset | Kaggle | HF | Size | Videos | Methods | Quality | Notes |
|---|---|---|---|---|---|---|---|
| 140k-faces | ✓ | – | 4 GB | – | image-level | high | Warmup, image-only |
| DFDC (sample) | ✓ | – | 50–100 GB | ~7k | 8 | high | Standard benchmark |
| FaceForensics++ | – | – | ~500 GB | 4k | 4 | high | Too large for free Kaggle |
| Celeb-DF v2 | ~ | – | ~100 GB | 5.6k | 1 | very high | Gold standard, form-gated |
| DeepSpeak v2 | – | ✓ | ~50 GB | – | talking-head | high | Different method (audio-driven) |
| Deepfake-Eval-2024 | – | ✓ | ~50 GB | ~1.9k | in-the-wild | *real* | True test of real-world perf |

✓ = directly available  
~ = community mirror (verify license)  
– = not available on platform

---

## Expected Results by Dataset

Training on DFDC with our CLIP+LoRA+compression-augmentation setup:

| Test Set | In-Domain | Cross-Dataset | Notes |
|---|---|---|---|
| DFDC → DFDC | ~94–97% AUC | – | saturated |
| DFDC → Celeb-DF v2 | – | ~88–93% AUC | standard cross-dataset |
| DFDC → 140k-faces | – | ~85–90% AUC | image-level |
| DFDC → DeepSpeak v2 | – | ~75–85% AUC | different generation |
| DFDC → Deepfake-Eval | – | ~65–80% AUC | *real in-the-wild* |

The **Deepfake-Eval-2024 number is the most honest** — it's what actually matters for deployment. Papers often report 95% but collapse to 70% in the wild.

---

## Kaggle Workflow: Step-by-Step

### 1. Create notebook
Go to kaggle.com → New Notebook → select GPU runtime

### 2. Add data as input
- Click "Add input" (right sidebar)
- Search "deepfake-detection-challenge" (DFDC)
- Click "Add"
- (Repeat for 140k faces if you want warmup)

### 3. Paste code
Open `KAGGLE_NOTEBOOK.py` in this repo. Copy each cell into your Kaggle notebook in order.

### 4. Run
Cell by cell. After Cell 5 (frame extraction), you'll have frames ready.
Cell 7 starts training (~2–12h depending on data size).
Cell 8 runs evaluation.

### 5. Download results
After training, results are in `/kaggle/working/outputs/`:
- `best.pt` — trained checkpoint
- `history.json` — per-epoch logs
- `eval_report.json` — cross-dataset metrics + robustness battery

---

## Kaggle Notebook Timeouts & How to Avoid Them

Kaggle notebooks have a **12-hour timeout** (or 24h with auto-save in Kaggle+).

**To stay under 12h:**
- Use a smaller dataset (140k-faces or DFDC sample, not full DFDC).
- Set `train.epochs=10` (not 20).
- Use `frames_per_clip=8` (not 16).
- Set `data.num_workers=2` (Kaggle has limited CPU).

If you exceed 12h:
- **Option 1:** Switch to Kaggle+ (24h timeout).
- **Option 2:** Save a checkpoint mid-training, then load it in a new notebook to resume.
- **Option 3:** Use Colab instead (longer timeout, but requires manual data management).

---

## Troubleshooting

**"Out of memory" during frame extraction?**
- Reduce frames-per-video from 16 to 8.
- Extract only a sample: `for v in sorted(...)[:100]` (first 100 videos).

**"CUDA out of memory" during training?**
- Reduce batch_size from 8 to 4.
- Reduce frames_per_clip from 8 to 4.
- Use a smaller backbone (efficientnet_b4 instead of clip_vit_l14).

**"No frames found" error?**
- Check that frame extraction actually ran (Cell 5).
- Verify output dir: `ls /kaggle/working/data/frames/`.
- Check dataset was added as input correctly.

**Training is too slow?**
- Kaggle GPU varies; T4 is slower than A100.
- Reduce num_workers (Kaggle has limited CPU parallelism).
- Use AMP (mixed precision) — already on in kaggle.yaml.

---

## What's the novel contribution here?

Three things:
1. **Compression robustness** — the code augments *every* training sample with JPEG/resize/noise to emulate social-media laundering. Most detectors never see compressed data and collapse on real-world video. We train for it.
2. **Cross-dataset evaluation on real benchmarks** — standard protocol, but honestly done: DFDC→Celeb-DF, then Deepfake-Eval-2024 (in-the-wild).
3. **Parameter-efficient fine-tuning** — LoRA on frozen CLIP/DINOv2, matching the published SOTA family (Forensics Adapter, Effort, LNCLIP-DF) without needing massive GPUs.

Expected outcome: ~85–92% on clean tests, ~65–80% on in-the-wild (honest number), with documented robustness to compression.

---

## Questions?

Check the main `README.md` for architecture details.
Check `configs/kaggle.yaml` for all tunable parameters.
Run `python -m src.fetch_data --list` to see all available sources.
