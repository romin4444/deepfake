# ============================================================================
# KAGGLE NOTEBOOK: Compression-Robust Video Deepfake Detector
# ============================================================================
# Copy each cell below into a Kaggle notebook. Run sequentially.
# Expected runtime: 12–24h (depends on dataset size + GPU speed)

# ============================================================================
# CELL 1: Setup & Install
# ============================================================================
!pip install -q -r requirements.txt open_clip_torch
!pip install -q opencv-python-headless

# ============================================================================
# CELL 2: Clone/Upload Code
# ============================================================================
import os
os.chdir('/kaggle/working')

# Option A: Clone from GitHub (if you pushed it)
# !git clone https://github.com/your-username/dfvideo.git
# %cd dfvideo

# Option B: Code is already available (uploaded as dataset)
# Assume it's in /kaggle/input/dfvideo-detector/

# For this tutorial, we assume the code is in /kaggle/working/dfvideo/
import sys
sys.path.insert(0, '/kaggle/working/dfvideo')

# ============================================================================
# CELL 3: Check Available Data on Kaggle
# ============================================================================
import os
print("Available datasets in /kaggle/input/:")
for d in os.listdir('/kaggle/input/'):
    print(f"  - {d}")

# Look for: deepfake-detection-challenge (DFDC), 140k-real-and-fake-faces, etc.

# ============================================================================
# CELL 4: Fetch Data (if not already on Kaggle)
# ============================================================================
# If DFDC is not in /kaggle/input/, download from Kaggle:

# !kaggle datasets download -d deepfake-detection-challenge -p /kaggle/input/
# (or manually: go to Kaggle dataset page, click 'Add to' notebook, then 'Input')

# List what we have:
!ls -lah /kaggle/input/ | head -10

# ============================================================================
# CELL 5: Extract Frames from Videos
# ============================================================================
# This turns raw videos → face-cropped frame folders
# Takes ~1-2 hours depending on dataset size

import subprocess
os.makedirs('/kaggle/working/data/frames', exist_ok=True)

# Example: extract DFDC train fakes
# Assumes DFDC is in /kaggle/input/deepfake-detection-challenge or similar
dfdc_path = '/kaggle/input/deepfake-detection-challenge/dfdc/train'

if os.path.exists(dfdc_path):
    print(f"[frame extraction] found DFDC at {dfdc_path}")
    # Extract a SAMPLE (first 100 videos) to fit in Kaggle memory
    videos = []
    for label_dir in ['fake', 'real']:
        label_path = os.path.join(dfdc_path, label_dir)
        if os.path.exists(label_path):
            videos.extend([(os.path.join(label_path, v), label_dir) 
                          for v in sorted(os.listdir(label_path))[:100]])
    
    print(f"[frame extraction] extracting {len(videos)} videos (sample)...")
    import cv2
    from PIL import Image
    import numpy as np
    
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for i, (video_file, label) in enumerate(videos):
        if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        
        video_id = os.path.splitext(os.path.basename(video_file))[0]
        out_dir = f'/kaggle/working/data/frames/dfdc/train/{label}/{video_id}'
        os.makedirs(out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        indices = [int(j * total_frames / 16) for j in range(16)]  # 16 frames
        
        saved = 0
        for frame_idx, fi in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            
            # Face crop
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                margin = int(0.3 * h)
                x0 = max(0, x - margin)
                y0 = max(0, y - margin)
                x1 = min(frame.shape[1], x + w + margin)
                y1 = min(frame.shape[0], y + h + margin)
                frame = frame[y0:y1, x0:x1]
            
            cv2.imwrite(f'{out_dir}/frame_{frame_idx:04d}.jpg', frame)
            saved += 1
        cap.release()
        
        if i % 20 == 0:
            print(f"  [{i}/{len(videos)}] extracted {saved} frames from {video_id}")

print("[frame extraction] done!")

# ============================================================================
# CELL 6: Verify Frame Dataset
# ============================================================================
import subprocess
result = subprocess.run(['find', '/kaggle/working/data/frames', '-name', '*.jpg'],
                       capture_output=True, text=True)
n_frames = len(result.stdout.strip().split('\n'))
print(f"Total frames extracted: {n_frames}")
print("Sample structure:")
subprocess.run(['ls', '-la', '/kaggle/working/data/frames/dfdc/train/fake/'], 
               capture_output=False)

# ============================================================================
# CELL 7: Train
# ============================================================================
os.chdir('/kaggle/working/dfvideo')

import subprocess
result = subprocess.run([
    'python', '-m', 'src.train',
    '--config', 'configs/kaggle.yaml',
    'train.epochs=10',                              # shorter for Kaggle timeout
    'model.backbone=clip_vit_l14',
    'model.peft=lora',
    'output_dir=/kaggle/working/outputs'
], capture_output=False)

print(f"Training exit code: {result.returncode}")

# ============================================================================
# CELL 8: Evaluate (Cross-Dataset + Robustness)
# ============================================================================
result = subprocess.run([
    'python', '-m', 'src.evaluate',
    '--config', 'configs/kaggle.yaml',
    '--ckpt', '/kaggle/working/outputs/best.pt',
    'data.test_datasets=[celebdf]',                 # or deepfake_eval if available
    'output_dir=/kaggle/working/outputs'
], capture_output=False)

print(f"Evaluation exit code: {result.returncode}")

# ============================================================================
# CELL 9: Inspect Results
# ============================================================================
import json

# Training history
history_path = '/kaggle/working/outputs/history.json'
if os.path.exists(history_path):
    with open(history_path) as f:
        hist = json.load(f)
    print("Training history (last 3 epochs):")
    for entry in hist[-3:]:
        print(f"  Epoch {entry['epoch']}: "
              f"loss={entry['train_loss']:.4f}, "
              f"val_auc={entry['val'].get('auc', 'N/A')}")

# Evaluation report
eval_path = '/kaggle/working/outputs/eval_report.json'
if os.path.exists(eval_path):
    with open(eval_path) as f:
        report = json.load(f)
    print("\nEvaluation results (cross-dataset):")
    for ds, metrics in report.get('datasets', {}).items():
        print(f"  {ds}:")
        if 'clean' in metrics:
            clean = metrics['clean']
            print(f"    clean:    AUC={clean.get('auc', 0):.4f} "
                  f"ACC={clean.get('accuracy', 0):.4f}")
        if 'robustness' in metrics:
            print(f"    robustness battery: {len(metrics['robustness'])} conditions")

# ============================================================================
# CELL 10: Save Results to Output
# ============================================================================
# Kaggle auto-saves /kaggle/working/outputs, but you can also:
import shutil
shutil.copy('/kaggle/working/outputs/best.pt', 
            '/kaggle/output/best_checkpoint.pt')
shutil.copy('/kaggle/working/outputs/eval_report.json', 
            '/kaggle/output/eval_report.json')
print("Results saved to /kaggle/output/")
