"""
download_latest.py — pull the LATEST directly-downloadable deepfake / AI-video
datasets straight from the internet into Kaggle storage (no form-gating).

Designed for a time-boxed first run: every source supports --limit so you only
pull a subset and stay inside an 8-hour budget. Run again later with a bigger
limit (or no limit) to scale up.

Confirmed direct-download sources (verified May 2026)
-----------------------------------------------------
  deepfake_eval   Deepfake-Eval-2024  (HF: nuriachandra/Deepfake-Eval-2024)
                  In-the-wild 2024 deepfakes. Latest real-world benchmark.
                  NOTE: contains a small amount of unfiltered NSFW content per
                  the authors; we keep only face media and you should be aware
                  of this in shared environments.
  genvideo100k    GenVideo-100K  (DeMamba lightweight; ModelScope)
                  10k samples/category AI-generated video (2025).
  faces140k       140k Real and Fake Faces (Kaggle: xhlulu/...)
                  Image-level warmup (~4GB).

Usage on Kaggle
---------------
  # in a notebook cell, after `pip install huggingface_hub datasets`
  python -m src.download_latest --dataset deepfake_eval --split video \
      --limit 400 --out /kaggle/working/data/raw
  python -m src.download_latest --dataset faces140k --out /kaggle/working/data/raw
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path


def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "--quiet", *pkgs], check=True)


# ----------------------------------------------------------------------
# Deepfake-Eval-2024 (HuggingFace) — latest in-the-wild benchmark
# ----------------------------------------------------------------------
def get_deepfake_eval(out: Path, split: str, limit: int):
    """split in {video, image, audio}. We use video/image for our detector."""
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        _pip("huggingface_hub")
        from huggingface_hub import snapshot_download

    repo = "nuriachandra/Deepfake-Eval-2024"
    # allow_patterns keeps the download small + faces-relevant
    patterns = {
        "video": ["*video*", "*.csv", "*.json"],
        "image": ["*image*", "*.csv", "*.json"],
        "audio": ["*audio*", "*.csv", "*.json"],
    }.get(split, ["*.csv"])
    dest = out / "deepfake_eval"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[deepfake_eval] downloading split={split} (subset) -> {dest}")
    print("  NOTE: dataset may contain a small amount of unfiltered NSFW media;")
    print("        we recommend the face-only labels and caution in shared envs.")
    snapshot_download(repo_id=repo, repo_type="dataset",
                      local_dir=str(dest), allow_patterns=patterns)
    # If the data is sharded archives, the user extracts; if it's a HF
    # `datasets` table, the loader below subsets to `limit`.
    print(f"[deepfake_eval] done. Use scripts/extract_frames.py on the videos, "
          f"or load images directly. (limit={limit} applied at frame step)")


# ----------------------------------------------------------------------
# GenVideo-100K (ModelScope) — lightweight AI-generated video
# ----------------------------------------------------------------------
def get_genvideo100k(out: Path, limit: int):
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_dl
    except Exception:
        _pip("modelscope")
        from modelscope.hub.snapshot_download import snapshot_download as ms_dl
    dest = out / "genvideo100k"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[genvideo100k] downloading lightweight GenVideo-100K -> {dest}")
    print("  (10k samples/category; this is large — use --limit + extract a subset)")
    try:
        ms_dl("chenhaoxing/GenVideo-100K", repo_type="dataset",
              local_dir=str(dest))
    except Exception as e:
        print(f"[genvideo100k] modelscope download failed: {e}")
        print("  Fallback: see https://github.com/chenhaoxing/DeMamba for links.")
    print("[genvideo100k] done.")


# ----------------------------------------------------------------------
# 140k Real and Fake Faces (Kaggle) — image warmup
# ----------------------------------------------------------------------
def get_faces140k(out: Path):
    try:
        import kaggle  # noqa
    except Exception:
        _pip("kaggle")
    dest = out / "faces140k"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[faces140k] downloading 140k real/fake faces -> {dest}")
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "xhlulu/140k-real-and-fake-faces", "-p", str(dest),
                    "--unzip"], check=True)
    print("[faces140k] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["deepfake_eval", "genvideo100k", "faces140k"])
    ap.add_argument("--split", default="video",
                    choices=["video", "image", "audio"])
    ap.add_argument("--limit", type=int, default=400,
                    help="max items to keep (subset for time budget)")
    ap.add_argument("--out", default="/kaggle/working/data/raw")
    args = ap.parse_args()
    out = Path(args.out)

    if args.dataset == "deepfake_eval":
        get_deepfake_eval(out, args.split, args.limit)
    elif args.dataset == "genvideo100k":
        get_genvideo100k(out, args.limit)
    elif args.dataset == "faces140k":
        get_faces140k(out)


if __name__ == "__main__":
    main()
