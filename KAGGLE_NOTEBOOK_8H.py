# ============================================================================
# KAGGLE NOTEBOOK — 8-HOUR FIRST RUN
# Direct-download LATEST datasets -> train (time-budgeted) -> multi-dataset eval
# ============================================================================
# Budget: download ~1h + extract ~1.5h + train ~4h + eval ~1h  =  ~7.5h
# Everything auto-stops at 6h training so eval always completes inside 8h.
#
# Copy each CELL into a Kaggle notebook (GPU runtime). Run top to bottom.
# ============================================================================

# ----------------------------------------------------------------------------
# CELL 1 — install
# ----------------------------------------------------------------------------
!pip install -q open_clip_torch timm scikit-learn pyyaml tqdm \
    opencv-python-headless huggingface_hub datasets modelscope kaggle

# ----------------------------------------------------------------------------
# CELL 2 — put the code on the path
#   (upload dfvideo_detector_kaggle.zip as a Kaggle *dataset*, add it as input)
# ----------------------------------------------------------------------------
import sys, os, zipfile, shutil
SRC_ZIP = None
for root, _, files in os.walk('/kaggle/input'):
    for f in files:
        if f.endswith('.zip') and 'dfvideo' in f.lower():
            SRC_ZIP = os.path.join(root, f)
if SRC_ZIP:
    with zipfile.ZipFile(SRC_ZIP) as z:
        z.extractall('/kaggle/working')
    print('extracted code from', SRC_ZIP)
os.chdir('/kaggle/working/dfvideo')
sys.path.insert(0, '/kaggle/working/dfvideo')
print('cwd =', os.getcwd())

# ----------------------------------------------------------------------------
# CELL 3 — DIRECT DOWNLOAD of the latest datasets into Kaggle storage
#   (subset-limited so we stay inside the budget; scale up on later runs)
# ----------------------------------------------------------------------------
# 3a. Image warmup (reliable, fast) — 140k real/fake faces
#     Needs Kaggle API token (Add-ons -> Secrets, or ~/.kaggle/kaggle.json)
!python -m src.download_latest --dataset faces140k --out /kaggle/working/data/raw

# 3b. LATEST in-the-wild benchmark — Deepfake-Eval-2024 (images split = fast)
!python -m src.download_latest --dataset deepfake_eval --split image \
    --limit 600 --out /kaggle/working/data/raw

# 3c. (optional, large) AI-generated video subset — comment out if tight on time
# !python -m src.download_latest --dataset genvideo100k --limit 400 \
#     --out /kaggle/working/data/raw

print('downloads done; contents:')
!ls -R /kaggle/working/data/raw | head -40

# ----------------------------------------------------------------------------
# CELL 4 — organize into the frame-folder layout the trainer expects
#   data/frames/<dataset>/<split>/{real,fake}/<video_id>/frame_xxxx.jpg
#   For IMAGE datasets each image becomes a 1-frame "clip".
# ----------------------------------------------------------------------------
import glob, random
from pathlib import Path
from PIL import Image

def place_images(src_glob, dataset, split, label):
    """Copy images into per-item frame folders (1 frame per image)."""
    files = sorted(glob.glob(src_glob, recursive=True))
    random.Random(0).shuffle(files)
    base = Path(f'/kaggle/working/data/frames/{dataset}/{split}/{label}')
    n = 0
    for i, fp in enumerate(files):
        d = base / f'{dataset}_{split}_{label}_{i:05d}'
        d.mkdir(parents=True, exist_ok=True)
        try:
            Image.open(fp).convert('RGB').save(d / 'frame_0000.jpg')
            n += 1
        except Exception:
            pass
    print(f'  {dataset}/{split}/{label}: {n} items')
    return n

# --- 140k faces (image warmup): has 'real'/'fake' subfolders after unzip ---
RF = '/kaggle/working/data/raw/faces140k'
# the kaggle set has train/valid/test dirs with real/fake; adjust globs to match
place_images(f'{RF}/**/real*/**/*.jpg', 'faces140k', 'train', 'real')
place_images(f'{RF}/**/fake*/**/*.jpg', 'faces140k', 'train', 'fake')
place_images(f'{RF}/**/real*/**/*.jpg', 'faces140k', 'val', 'real')
place_images(f'{RF}/**/fake*/**/*.jpg', 'faces140k', 'val', 'fake')

# --- Deepfake-Eval-2024 images (test only): label from its metadata CSV ---
# The HF repo ships a CSV with columns like (filename,label). Map real=0/fake=1.
import csv
DE = '/kaggle/working/data/raw/deepfake_eval'
csvs = glob.glob(f'{DE}/**/*.csv', recursive=True)
if csvs:
    print('deepfake_eval metadata:', csvs[0])
    with open(csvs[0]) as f:
        rows = list(csv.DictReader(f))
    # heuristics for label/filename column names
    def col(row, *names):
        for n in names:
            for k in row:
                if k.lower() == n: return row[k]
        return None
    placed = {'real': 0, 'fake': 0}
    for i, r in enumerate(rows[:600]):
        fn = col(r, 'filename', 'file', 'path', 'name')
        lab = (col(r, 'label', 'ground_truth', 'class') or '').lower()
        if not fn: continue
        label = 'fake' if lab in ('fake', '1', 'ai', 'manipulated', 'true') else 'real'
        matches = glob.glob(f'{DE}/**/{os.path.basename(fn)}', recursive=True)
        if not matches: continue
        d = Path(f'/kaggle/working/data/frames/deepfake_eval/test/{label}/de_{i:05d}')
        d.mkdir(parents=True, exist_ok=True)
        try:
            Image.open(matches[0]).convert('RGB').save(d / 'frame_0000.jpg')
            placed[label] += 1
        except Exception:
            pass
    print('  deepfake_eval/test:', placed)
else:
    print('  [warn] no deepfake_eval CSV found; check the download in CELL 3')

!echo "frame folders:"; find /kaggle/working/data/frames -maxdepth 3 -type d | head -30

# ----------------------------------------------------------------------------
# CELL 5 — TRAIN (time-budgeted: auto-stops at 6h)
# ----------------------------------------------------------------------------
!python -m src.train --config configs/fast8h.yaml \
    data.root=/kaggle/working/data/frames \
    data.train_datasets=[faces140k] \
    data.val_datasets=[faces140k] \
    train.time_budget_hours=6.0 \
    output_dir=/kaggle/working/outputs

# ----------------------------------------------------------------------------
# CELL 6 — MULTI-DATASET EVALUATION (clean + robustness battery)
# ----------------------------------------------------------------------------
!python -m src.evaluate --config configs/fast8h.yaml \
    --ckpt /kaggle/working/outputs/best.pt \
    data.root=/kaggle/working/data/frames \
    data.test_datasets=[deepfake_eval,faces140k] \
    output_dir=/kaggle/working/outputs

# ----------------------------------------------------------------------------
# CELL 7 — show results
# ----------------------------------------------------------------------------
import json
rep = json.load(open('/kaggle/working/outputs/eval_report.json'))
for ds, m in rep['datasets'].items():
    print(f'\n=== {ds} ===')
    if m.get('clean'):
        c = m['clean']; print(f"  clean: AUC={c['auc']:.4f} ACC={c['accuracy']:.4f} EER={c['eer']:.4f}")
    for name, rm in (m.get('robustness') or {}).items():
        print(f"    {name:14s} AUC={rm['auc']:.4f}")

# ----------------------------------------------------------------------------
# CELL 8 — persist results (Kaggle auto-saves /kaggle/working, but be explicit)
# ----------------------------------------------------------------------------
import shutil
os.makedirs('/kaggle/working/final', exist_ok=True)
for f in ['best.pt', 'history.json', 'eval_report.json']:
    p = f'/kaggle/working/outputs/{f}'
    if os.path.exists(p): shutil.copy(p, f'/kaggle/working/final/{f}')
print('saved to /kaggle/working/final/ — download from the notebook Output tab')
