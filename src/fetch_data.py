"""
fetch_data.py — one-stop helper to obtain face/video deepfake-detection
benchmarks from Kaggle, Hugging Face, and academic request forms.

This script does NOT bundle any data (datasets are large + access-gated).
It automates the *download* step once you have credentials, and prints the
manual steps for form-gated datasets.

Sources covered
----------------
Kaggle:
  - deepfake-detection-challenge (DFDC, ~470 GB full / sample subset available)
  - celeb-df-v2 (community mirrors exist; prefer official request form)
  - faceforensics  (community mirrors; prefer official form)
  - 140k-real-and-fake-faces (image-level warmup set, ~4 GB)
Hugging Face:
  - faridlab/deepspeak_v2 (talking-head deepfakes, ~50h)
  - GenVideo / DeMamba mirrors (AI-generated video)
Form-gated (manual):
  - FaceForensics++  (https://github.com/ondyari/FaceForensics)
  - Celeb-DF v2      (https://github.com/yuezunli/celeb-deepfakeforensics)
  - DF40             (https://github.com/YZY-stack/DF40)

Examples
--------
  # list everything and how to get it
  python -m src.fetch_data --list

  # download a small Kaggle warmup set (needs ~/.kaggle/kaggle.json)
  python -m src.fetch_data --kaggle xhlulu/140k-real-and-fake-faces --out data/raw

  # pull a HF dataset
  python -m src.fetch_data --hf faridlab/deepspeak_v2 --out data/raw
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

REGISTRY = {
    "kaggle": {
        "dfdc": ("deepfake-detection-challenge",
                 "Full DFDC competition data (~470GB). Use --kaggle "
                 "'c/deepfake-detection-challenge' after `kaggle competitions "
                 "download`. A sample subset is in the competition 'Data' tab."),
        "140k-faces": ("xhlulu/140k-real-and-fake-faces",
                       "Image-level real/fake faces (~4GB). Good CPU warmup."),
        "celeb-df-v2": ("reubenschmidt/celeb-df-v2",
                        "Community mirror of Celeb-DF v2 (verify license; "
                        "official form preferred)."),
    },
    "hf": {
        "deepspeak_v2": ("faridlab/deepspeak_v2",
                         "Talking-head deepfakes, UC Berkeley (Farid lab)."),
        "genvideo": ("Andyrasika/GenVideo-sample",
                     "Sample of AI-generated video (DeMamba/GenVideo lineage)."),
    },
    "manual": {
        "faceforensics++": "https://github.com/ondyari/FaceForensics  "
                           "(fill the EULA form; you get a download script).",
        "celeb-df-v2": "https://github.com/yuezunli/celeb-deepfakeforensics  "
                       "(request form).",
        "df40": "https://github.com/YZY-stack/DF40  (40-method benchmark; "
                "request via form).",
        "deepfake-eval-2024": "https://huggingface.co/datasets/nuriachandra/"
                              "Deepfake-Eval-2024  (in-the-wild test set).",
    },
}


def _run(cmd: list[str]):
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def list_sources():
    print("=" * 68)
    print("DEEPFAKE-DETECTION DATASET SOURCES")
    print("=" * 68)
    print("\n[KAGGLE]  (needs ~/.kaggle/kaggle.json; `pip install kaggle`)")
    for k, (slug, desc) in REGISTRY["kaggle"].items():
        print(f"  - {k:14s} slug={slug}\n      {desc}")
    print("\n[HUGGING FACE]  (`pip install datasets huggingface_hub`)")
    for k, (repo, desc) in REGISTRY["hf"].items():
        print(f"  - {k:14s} repo={repo}\n      {desc}")
    print("\n[FORM-GATED — manual download]")
    for k, url in REGISTRY["manual"].items():
        print(f"  - {k:18s} {url}")
    print("\nAfter download, run scripts/extract_frames.py to turn videos into")
    print("frame folders, then point configs/default.yaml:data.root at them.")


def kaggle_download(slug: str, out: str):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle  # noqa: F401
    except Exception:
        print("Installing kaggle..."); _run([sys.executable, "-m", "pip",
              "install", "--break-system-packages", "kaggle"])
    if slug.startswith("c/"):
        _run(["kaggle", "competitions", "download", "-c", slug[2:],
              "-p", str(out)])
    else:
        _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out),
              "--unzip"])
    print(f"Done -> {out}")


def hf_download(repo: str, out: str):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("Installing huggingface_hub..."); _run([sys.executable, "-m",
              "pip", "install", "--break-system-packages", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id=repo, repo_type="dataset",
                          local_dir=str(out / repo.split("/")[-1]))
    print(f"Done -> {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--kaggle", help="kaggle dataset slug or c/<competition>")
    ap.add_argument("--hf", help="hugging face dataset repo id")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    if args.list or (not args.kaggle and not args.hf):
        list_sources(); return
    if args.kaggle:
        kaggle_download(args.kaggle, args.out)
    if args.hf:
        hf_download(args.hf, args.out)


if __name__ == "__main__":
    main()
