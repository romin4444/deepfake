"""
extract_frames.py — turn raw videos into the frame-folder layout the
training pipeline expects.

  data/frames/<dataset>/<split>/{real,fake}/<video_id>/frame_xxxx.jpg

Usage:
  python scripts/extract_frames.py \
      --videos data/raw/dfdc/train --label fake \
      --dataset dfdc --split train --out data/frames \
      --frames-per-video 16 --face-crop

Requires: opencv-python (decoding). Face cropping uses OpenCV's built-in
Haar cascade by default (no extra weights); pass --no-face-crop to keep full
frames. For production, swap in RetinaFace/MTCNN where noted.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def get_face_detector():
    import cv2
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(path)


def extract_one(video_path, out_dir, n_frames, face_crop, detector, margin=0.3):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = [int(i * total / n_frames) for i in range(n_frames)]
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        if face_crop and detector is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
            if len(faces):
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                mx, my = int(w * margin), int(h * margin)
                x0, y0 = max(0, x - mx), max(0, y - my)
                x1, y1 = min(frame.shape[1], x + w + mx), min(frame.shape[0], y + h + my)
                frame = frame[y0:y1, x0:x1]
        cv2.imwrite(str(out_dir / f"frame_{j:04d}.jpg"), frame)
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir of video files")
    ap.add_argument("--label", required=True, choices=["real", "fake"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--frames-per-video", type=int, default=16)
    ap.add_argument("--face-crop", dest="face_crop", action="store_true", default=True)
    ap.add_argument("--no-face-crop", dest="face_crop", action="store_false")
    ap.add_argument("--exts", default=".mp4,.avi,.mov,.mkv,.webm")
    args = ap.parse_args()

    try:
        import cv2  # noqa
    except Exception:
        raise SystemExit("pip install opencv-python first")

    detector = get_face_detector() if args.face_crop else None
    exts = set(args.exts.split(","))
    vids = [p for p in Path(args.videos).rglob("*") if p.suffix.lower() in exts]
    print(f"found {len(vids)} videos")
    base = Path(args.out) / args.dataset / args.split / args.label
    tot = 0
    for i, v in enumerate(vids):
        vid_id = f"{args.dataset}_{args.split}_{args.label}_{v.stem}"
        n = extract_one(v, base / vid_id, args.frames_per_video,
                        args.face_crop, detector)
        tot += n
        if i % 50 == 0:
            print(f"  [{i}/{len(vids)}] {v.name} -> {n} frames")
    print(f"done: {tot} frames from {len(vids)} videos -> {base}")


if __name__ == "__main__":
    main()
