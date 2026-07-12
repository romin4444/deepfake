"""
PAPER1_COLAB_MASTER_V19.py  —  single-file Colab script
(audio + video deepfake DETECTION, and audio-visual temporal LOCALIZATION).

⭐ V19 CHANGES (2026-07-11): three new stages + one fixed stage.  The LOCKED
   PAPER1 results (week1_audio, video_probe) are UNTOUCHED.  V19 adds:

   • STAGE hiercon_audio    — HierCon-lite (arXiv:2602.01032, Feb 2026):
       hierarchical layer attention + margin contrastive over the ALREADY-
       EXTRACTED XLS-R features from week1_audio.  Realistic target on
       ASVspoof2019-LA→ITW: ~10–14% EER (published HierCon is 6.87% with
       end-to-end fine-tuning; on a frozen backbone + T4 you are trading
       compute for accuracy — this closes ~40–50% of the gap).
       CRITICAL: this REUSES the frozen features cached by week1_audio.
       It cannot run standalone — run week1_audio first.

   • STAGE cross_gen_df40   — NEW claim for the paper: cross-generation
       generalization.  Trains the frozen CLIP-L14 probe on FF++ c23
       (piggybacks on video_probe's exported head), evaluates on DF40
       (NeurIPS'24, 40 distinct generators including HeyGen / DeepFaceLab /
       diffusion editors).  HuggingFace source: pujanpaudel/deepfake_face_
       classification (32k images, DF40 test subset — no gating).
       Reports per-family AUC and the "unknown-generator drop" claim.

   • STAGE eval_dfe2024_v2  — FIXES gap G3 by REUSING the head exported by
       week1_audio (probe_head_audio_xlsr.npz) instead of loading a phantom
       ssl_head.pt.  Also adds the image branch (DFE2024 has 1,975 images).
       Direct comparison to the "SOTA drops 45–50% AUC on DFE2024" band.

   • STAGE openfake_frontier — OPTIONAL: evaluate on ComplexDataLab/OpenFake
       (Flux.2, GPT Image 2.0, nano-banana era).  Same frozen CLIP head, but
       full-image (no MTCNN, since these are politically-salient scenes).

STAGES (V18 unchanged, V19 additions marked ⭐):
  week1_audio       — PAPER1 audio branch (COMPLETE 2026-07-10: XLS-R L05 ITW EER 19.55%)
  ⭐ hiercon_audio  — V19: hierarchical layer attention + contrastive (~10–14% ITW EER target)
  video_train       — PAPER1 EffB0 end-to-end CONTROL (bugfixed; exports per-video scores)
  video_probe       — PAPER1 headline video branch: frozen CLIP ViT-L/14 layer-wise probes
  av_localize       — PAPER2/THESIS skeleton (LAV-DF temporal localization)
  eval_dfe2024      — V18 known-broken; use eval_dfe2024_v2 instead
  ⭐ eval_dfe2024_v2— V19 FIXED: uses exported probe_head_audio_xlsr.npz + image branch
  ⭐ cross_gen_df40 — V19: FF++ → DF40 (40-generator cross-domain evaluation)
  ⭐ openfake_frontier — V19: 2026 generators (Flux.2, GPT Image, nano-banana)

RUN ORDER (V19 recommendation, one month to submission):
  Day 0–3   week1_audio  (locked — reproduce, then keep results)
  Day 3–5   hiercon_audio  (NEW HEADLINE — this is where V19 earns its keep)
  Day 5–10  video_probe    (locked — reproduce)
  Day 10–14 cross_gen_df40 (NEW claim: generalization to unseen generators)
  Day 14–17 eval_dfe2024_v2 (NEW claim: 2024 in-the-wild survival vs the 45–50% AUC drop band)
  Day 17–20 av_localize     (thesis skeleton — not for PAPER1 submission)
  Day 20+   openfake_frontier (optional stretch — 2026-generator evaluation)

DO NOT PROMISE ANYTHING PAST HierCon-lite AS "BEATING SOTA" — see the paper
sanity check at the top of the hiercon_audio stage.  We are competitive on a
frozen-backbone budget, not SOTA overall.

AV_LOCALIZE QUICKSTART (unchanged from V18):
  1) DATASETS: "lavdf" enabled (HuggingFace, ~24GB one-time; agree to terms on the
     HF page ControlNet/LAV-DF first, paste HF_TOKEN below if the hub asks for auth).
     Run this stage on a FRESH runtime — stage-aware downloads skip FF++/CDF/audio.
  2) STAGE = "av_localize", DEBUG = True → smoke test (30/10/15 videos).
  3) Confirm logs: "[lavdf] metadata entries: 136304" (or similar) and the quadrant
     table (real / V-only / A-only / AV counts).
  4) DEBUG = False → full run (caps: 1200 train / 200 dev / 600 test; ~3-5h on T4).
"""


# ============================================================================
# STAGE + DEBUG — set these FIRST (dataset downloads below are stage-aware)
# ============================================================================
# V19 stage list:
#   Locked (V18):  week1_audio | video_train | video_probe | av_localize
#   Fixed  (V19):  eval_dfe2024_v2
#   NEW    (V19):  hiercon_audio | cross_gen_df40 | openfake_frontier
STAGE = "hiercon_audio"   # ← V19 default: new headline stage after week1_audio ships
DEBUG = True              # ALWAYS smoke-test first, then flip to False

HF_TOKEN = ""           # HuggingFace token (Settings → Access Tokens). Needed for
                        # gated sets (AV-Deepfake1M++, DeepSpeak v2, DFE2024) and for
                        # click-through-gated ones (LAV-DF) if the hub requires login.

ZIP_SKIP_CLIP_FEATURES = True   # clip__* caches ≈ 3.5 GB — skip in the download zip


# ============================================================================
# SETUP — imports, kaggle.json, persistence
# ============================================================================

# ============================== SETUP ==============================
import os, sys, json, glob, math, random, time, subprocess, warnings, hashlib
from pathlib import Path
warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules
DATA = Path("/content/data" if IN_COLAB else "./data"); DATA.mkdir(parents=True, exist_ok=True)

if IN_COLAB:
    from google.colab import files
PERSIST = Path("/content/paper1_persist" if IN_COLAB else "./paper1_persist")
for sub in ("features", "ckpts", "results", "figures", "reports", "timelines"):
    (PERSIST / sub).mkdir(parents=True, exist_ok=True)
# ---- cross-session RESTORE: re-upload your last zip to skip re-extraction ----
RESTORE_ZIP = None   # e.g. "/content/paper1_persist_week1_audio_XXXX.zip"
if IN_COLAB and RESTORE_ZIP is None:
    import glob as _g
    cands = sorted(_g.glob("/content/paper1_persist_*.zip"))
    if cands: RESTORE_ZIP = cands[-1]; print(f"[restore] found {RESTORE_ZIP}")
if RESTORE_ZIP and Path(RESTORE_ZIP).exists():
    import zipfile as _zf
    with _zf.ZipFile(RESTORE_ZIP) as z: z.extractall(PERSIST)
    n_feat = len(list((PERSIST/"features").glob("*.npy")))
    print(f"[restore] extracted {RESTORE_ZIP} → {n_feat} feature files reusable")
print("persistent store:", PERSIST)

# --- kaggle.json upload & API config (only if this stage needs Kaggle) -------
STAGE_NEEDS_KAGGLE = STAGE in ("week1_audio", "hiercon_audio", "video_train",
                               "video_probe")   # V20: cross_gen_df40 removed — it only needs
                               # the exported CLIP head + DF40 (HF), not FF++/CelebDF.
kag = Path.home() / ".kaggle" / "kaggle.json"
if STAGE_NEEDS_KAGGLE and not kag.exists():
    kag.parent.mkdir(parents=True, exist_ok=True)
    if IN_COLAB:
        print("⬆️  Upload your kaggle.json (Kaggle → Account → Create New API Token)")
        up = files.upload()
        src = next((n for n in up if n.endswith(".json")), None)
        assert src, "no json uploaded"
        kag.write_bytes(up[src])
    else:
        raise SystemExit("place kaggle.json at ~/.kaggle/kaggle.json")
if kag.exists(): os.chmod(kag, 0o600)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=False)

def sh(cmd):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.stdout: print(r.stdout[-1500:])
    if r.returncode != 0: print("STDERR:", r.stderr[-1500:])
    return r.returncode == 0


# ============================================================================
# DATASETS — toggle sources; STAGE-AWARE: only downloads what STAGE needs.
# kinds: "kaggle" | "url" (direct/zenodo) | "hf" (HuggingFace hub) | "gdrive"
# ============================================================================
DATASETS = {
    # ---------- audio (PAPER1, complete) ----------
    "in_the_wild":   {"enabled": True,  "kind": "kaggle", "stages": ("week1_audio", "hiercon_audio"),
                      "ref": "bhaveshkumars/release-in-the-wild",
                      "dest": "release-in-the-wild"},
    "asvspoof2019":  {"enabled": True,  "kind": "kaggle", "stages": ("week1_audio", "hiercon_audio"),
                      "ref": "awsaf49/asvpoof-2019-dataset",
                      "dest": "asvspoof2019"},
    "partialspoof":  {"enabled": False, "kind": "url",    "stages": ("week1_audio", "av_localize"),
                      "ref": "https://zenodo.org/api/records/5766198",
                      "dest": "partialspoof"},   # OPEN — segment-level audio-only labels
    # ---------- video (PAPER1) ----------
    "celebdf_v2":    {"enabled": True,  "kind": "kaggle", "stages": ("video_train", "video_probe"),
                      "ref": "reubensuju/celeb-df-v2",
                      "dest": "celebdf-v2"},
    "ffpp_c23":      {"enabled": True,  "kind": "kaggle", "stages": ("video_train", "video_probe"),
                      "ref": "xdxd003/ff-c23",
                      "dest": "ffpp"},
    # ---------- V19: NEW datasets ---------------------------------------------
    # DF40 (NeurIPS'24): 32,134 face images (16,060 real + 16,060 fake) covering
    # the DF40 test split — 40 distinct generators including HeyGen and
    # DeepFaceLab.  This is the RIGHT single-source-of-truth for cross-generation
    # generalization in 2026: cheap (~4 GB, no gating, no MTCNN needed).
    "df40_test":     {"enabled": True,  "kind": "hf",     "stages": ("cross_gen_df40",),
                      "ref": "pujanpaudel/deepfake_face_classification",
                      "dest": "df40-test",
                      # V20: only the test split (3,212 face images) is needed for eval.
                      # Avoids downloading train.rar (~3.9 GB) + val.zip every run.
                      "allow_patterns": ["test.zip", "test/", "README.md"]},
    # Deepfake-Eval-2024 (arXiv:2503.02857): in-the-wild 2024 media from 88 sites,
    # 52 languages.  This wires up the V19 eval_dfe2024_v2 stage (fixes G3).
    "dfe2024":       {"enabled": True,  "kind": "hf",     "stages": ("eval_dfe2024_v2",),
                      "ref": "nuriachandra/Deepfake-Eval-2024",
                      "dest": "dfe2024"},
    # OpenFake (arXiv:2509.09495): politically-salient real images vs frontier
    # generators (Flux.2, GPT Image 2.0, nano-banana, community LoRAs).  This is
    # the ONLY freely-hostable 2026-vintage generator eval right now.
    "openfake":      {"enabled": False, "kind": "hf",     "stages": ("openfake_frontier",),
                      "ref": "ComplexDataLab/OpenFake",
                      "dest": "openfake"},
    # ---------- audio-visual localization (PAPER2/THESIS) ----------
    "lavdf":         {"enabled": True,  "kind": "hf",     "stages": ("av_localize",),
                      "ref": "ControlNet/LAV-DF",          # agree to terms on HF page first
                      "dest": "lav-df"},                   # ~24GB one-time; auto-unzips
    "av_deepfake1m": {"enabled": False, "kind": "hf",     "stages": ("av_localize",),
                      "ref": "PASTE-REPO-ID-AFTER-APPROVAL",   # gated — chase your filed request
                      "dest": "av-deepfake1m"},
    "deepspeak_v2":  {"enabled": False, "kind": "hf",     "stages": ("av_localize",),
                      "ref": "faridlab/deepspeak_v2",      # gated — request on HF
                      "dest": "deepspeak-v2"},
    "fakeavceleb":   {"enabled": False, "kind": "gdrive", "stages": ("av_localize",),
                      "ref": "PASTE-GDRIVE-LINK-FROM-EULA-EMAIL",  # DASH-Lab form → email link
                      "dest": "fakeavceleb"},
}

DATA_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg", ".mp4", ".avi", ".mov",
            ".csv", ".txt", ".json",
            ".jpg", ".jpeg", ".png", ".webp", ".bmp")   # V20: image exts added so image-only
            # datasets (DF40, OpenFake) are recognised as populated by _n_data_files.
def _n_data_files(d, cap=50):
    n = 0
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in DATA_EXTS:
            n += 1
            if n >= cap: break
    return n

def _extract_archives(d):
    # unzip/untar anything the download left packed; tolerate multi-zip layouts
    for z in sorted(d.rglob("*.zip")):
        print(f"[unzip] {z.relative_to(d)}")
        if sh(f'unzip -q -o "{z}" -d "{z.parent}"'): z.unlink()
    for t in sorted(list(d.rglob("*.tar.gz")) + list(d.rglob("*.tgz")) + list(d.rglob("*.tar"))):
        print(f"[untar] {t.relative_to(d)}")
        if sh(f'tar -xf "{t}" -C "{t.parent}"'): t.unlink()
    # V20: .rar support (DF40 mirror ships train.rar). Try unrar, then patool, then 7z.
    for r in sorted(d.rglob("*.rar")):
        print(f"[unrar] {r.relative_to(d)}")
        ok = sh(f'unrar x -o+ -y "{r}" "{r.parent}" 2>/dev/null')
        if not ok: ok = sh(f'patool extract "{r}" --outdir "{r.parent}" 2>/dev/null')
        if not ok: ok = sh(f'7z x -y "{r}" -o"{r.parent}" 2>/dev/null')
        if ok: r.unlink()
        else: print(f"  ⚠️ could not extract {r.name} (install unrar/patool/7z) — join manually")
    leftovers = [p.name for p in d.rglob("*") if p.suffix.lower() in (".z01", ".part", ".001")]
    if leftovers:
        print(f"  ⚠️ unextracted split-archive parts remain ({leftovers[:3]}…) — join manually if needed")

def kaggle_download(ref, dest):
    d = DATA / dest
    if d.exists():
        n = _n_data_files(d)
        if n >= 50:
            print(f"[skip] {dest} already populated ({n}+ data files)"); return True
        if any(d.rglob("*")):
            print(f"[redl] {dest} exists but has only {n} data files — partial/failed download, wiping & retrying")
            import shutil as _sh; _sh.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    ok = sh(f'kaggle datasets download -d "{ref}" -p "{d}" --unzip')
    if ok and _n_data_files(d) == 0:
        _extract_archives(d); ok = _n_data_files(d) > 0
    if not ok: print(f"❌ {ref} failed — verify slug: kaggle datasets list -s <keywords>")
    return ok

def url_download(ref, dest):
    d = DATA / dest; d.mkdir(parents=True, exist_ok=True)
    if any(d.rglob("*")): print(f"[skip] {dest} already populated"); return True
    if "zenodo.org/api/records" in ref:
        import urllib.request
        rec = json.load(urllib.request.urlopen(ref))
        for f in rec.get("files", []):
            url, name = f["links"]["self"], f["key"]
            print("→", name)
            sh(f'wget -q -c -O "{d/name}" "{url}"')
        _extract_archives(d); return True
    sh(f'wget -q -c -P "{d}" "{ref}"'); _extract_archives(d); return True

def hf_download(ref, dest, allow_patterns=None):
    # snapshot_download: resumable, layout-agnostic (we rglob afterwards, never
    # assume repo structure). Gated repos need HF_TOKEN + accepted terms on the
    # dataset page — a 403 here means the terms weren't accepted yet.
    # V20: allow_patterns lets us fetch only what a stage needs (e.g. DF40 test.zip).
    d = DATA / dest
    need = _n_data_files(d, cap=10**9)
    if d.exists() and need >= 50:
        print(f"[skip] {dest} already populated ({need} data files)"); return True
    if "PASTE" in ref:
        print(f"[hf] {dest}: placeholder ref — paste the repo id once access is granted"); return False
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
    from huggingface_hub import snapshot_download
    kw = dict(repo_type="dataset", local_dir=str(d), token=HF_TOKEN or None)
    if allow_patterns: kw["allow_patterns"] = allow_patterns
    try:
        snapshot_download(ref, **kw)
    except Exception as e:
        print(f"❌ HF download failed for {ref}: {e}")
        print("   → accept the terms on https://huggingface.co/datasets/" + ref)
        print("   → paste a valid HF_TOKEN at the top of this file")
        return False
    _extract_archives(d)
    print(f"[hf] {dest}: {_n_data_files(d, cap=10**9)} data files after extraction")
    return True

def gdrive_download(ref, dest):
    d = DATA / dest
    if d.exists() and _n_data_files(d) >= 50:
        print(f"[skip] {dest} already populated"); return True
    if "PASTE" in ref:
        print(f"[gdrive] {dest}: placeholder link — paste the Drive link from the EULA email"); return False
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
    ok = sh(f'gdown --folder --fuzzy "{ref}" -O "{d}"' if "folder" in ref
            else f'gdown --fuzzy "{ref}" -O "{d}/"')
    _extract_archives(d)
    return ok and _n_data_files(d) > 0

KIND_FN = {"kaggle": kaggle_download, "url": url_download,
           "hf": hf_download, "gdrive": gdrive_download}
for name, spec in DATASETS.items():
    if not spec["enabled"]: continue
    if STAGE not in spec.get("stages", ()):
        print(f"[stage-skip] {name} (not needed for STAGE={STAGE})"); continue
    print(f"===== downloading {name} ({spec['kind']}) =====")
    fn = KIND_FN[spec["kind"]]
    if spec["kind"] == "hf":
        fn(spec["ref"], spec["dest"], spec.get("allow_patterns"))   # V20: pass allow_patterns
    else:
        fn(spec["ref"], spec["dest"])


# ============================================================================
# CONFIG + shared utils
# ============================================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "soundfile", "librosa", "timm",
                "opencv-python-headless", "facenet-pytorch", "scipy", "scikit-learn"], check=False)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import librosa, soundfile as sf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| stage:", STAGE, "| debug:", DEBUG)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class CFG:
    SR, CROP_SEC = 16000, 4.0
    SOURCE_NAME = "asvspoof2019"
    SOURCE_ROOT = str(DATA / "asvspoof2019")
    TARGET_NAME = "in_the_wild"
    TARGET_ROOT = str(DATA / "release-in-the-wild")
    SRC_PER_CLS, TGT_PER_CLS = 10000, 4000
    SSL_MODELS = {"base": "facebook/wav2vec2-base", "xlsr": "facebook/wav2vec2-xls-r-300m"}
    SSL_BATCH  = 12
    SEEDS      = [0, 1, 2, 3, 4]
    N_MELS, CNN_EPOCHS, CNN_BATCH, CNN_LR = 80, 8, 64, 3e-4
    CNN_SEEDS  = [0, 1, 2]
    CNN_MAX_TRAIN = 6000
    CNN_AUG_K     = 4
    TRAIN_REAL = str(DATA / "ffpp/**/original*/**/*.mp4")
    TRAIN_FAKE = str(DATA / "ffpp/**/manipulated*/**/*.mp4")
    CROSS_REAL = str(DATA / "celebdf-v2/**/*-real/**/*.mp4")
    CROSS_FAKE = str(DATA / "celebdf-v2/**/Celeb-synthesis/**/*.mp4")
    FRAMES_PER_VID, IMG, MARGIN = 10, 224, 0.30
    MAX_TRAIN_VIDS, MAX_CROSS_VIDS, VAL_FRAC = 1500, 400, 0.15
    V_EPOCHS, V_BATCH, V_LR, V_SEEDS = 6, 32, 2e-4, [0]
    N_BOOT = 1000

if DEBUG:
    CFG.SRC_PER_CLS, CFG.TGT_PER_CLS = 150, 150
    CFG.SEEDS, CFG.CNN_SEEDS, CFG.CNN_EPOCHS = [0, 1], [0], 2
    CFG.CNN_MAX_TRAIN, CFG.CNN_AUG_K = 300, 2
    CFG.MAX_TRAIN_VIDS, CFG.MAX_CROSS_VIDS = 40, 20
    CFG.FRAMES_PER_VID, CFG.V_EPOCHS = 4, 2
    CFG.N_BOOT = 200

def load_clip(path, sr=CFG.SR, crop=CFG.CROP_SEC):
    try:
        y, fsr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1: y = y.mean(1)
        if fsr != sr: y = librosa.resample(y, orig_sr=fsr, target_sr=sr)
    except Exception:
        y, _ = librosa.load(path, sr=sr, mono=True)
    T = int(sr * crop)
    y = y[(len(y)-T)//2:(len(y)-T)//2+T] if len(y) >= T else np.pad(y, (0, T-len(y)))
    m = np.abs(y).max(); return y/m if m > 0 else y

def bootstrap_auc_ci(y, s, n=CFG.N_BOOT, seed=0):
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        v.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def eer_of(y, s):
    y, s = np.asarray(y), np.asarray(s)
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[i] + (1 - tpr[i])) / 2), float(thr[i])

def bootstrap_eer_ci(y, s, n=CFG.N_BOOT, seed=0):
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        e, _ = eer_of(y[i], s[i]); v.append(e)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def paired_bootstrap_delta(y, s_a, s_b, n=CFG.N_BOOT, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        d.append(roc_auc_score(y[i], s_b[i]) - roc_auc_score(y[i], s_a[i]))
    d = np.array(d)
    return {"delta_auc_mean": float(d.mean()),
            "delta_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}

def save_probe_head(path, scaler, clf, meta):
    # portable head export (no pickle → survives sklearn version drift):
    # score = sigmoid(coef · ((x - mean)/scale) + intercept)
    np.savez(path, coef=clf.coef_[0].astype(np.float32),
             intercept=np.float32(clf.intercept_[0]),
             mean=scaler.mean_.astype(np.float32),
             scale=scaler.scale_.astype(np.float32),
             meta=json.dumps(meta))
    print(f"[head-export] {Path(path).name}  ({meta})")

def apply_probe_head(npz, X):
    z = (X - npz["mean"]) / npz["scale"]
    logit = z @ npz["coef"] + float(npz["intercept"])
    return 1.0 / (1.0 + np.exp(-logit))


# ============================================================================
# STAGE A: week1_audio — ASVspoof→ITW probes, confounds, mel-CNN (COMPLETE)
# ============================================================================

if STAGE == "week1_audio":
    import pandas as pd, csv as _csv
    AUDIO_EXTS = (".flac", ".wav", ".mp3", ".m4a", ".ogg")

    def _index_audio(root):
        return {p.name: p for p in Path(root).rglob("*") if p.suffix.lower() in AUDIO_EXTS}

    def _labels_from_asvspoof(root):
        protos = sorted(Path(root).rglob("*.trn.txt")) + sorted(Path(root).rglob("*.cm.train.trn.txt"))
        protos = [p for p in protos if "train" in p.name.lower()] or sorted(Path(root).rglob("*.trn.txt"))
        pairs = []
        for pf in protos:
            for ln in open(pf, "r", errors="ignore"):
                parts = ln.strip().split()
                if len(parts) < 2: continue
                fid, lab = parts[1], parts[-1].lower()
                if lab in ("bonafide", "spoof"):
                    pairs.append((fid + ".flac", 0 if lab == "bonafide" else 1))
        return pairs

    def _labels_from_csv(root):
        pairs = []
        for c in sorted(Path(root).rglob("*.csv")):
            try:
                df = pd.read_csv(c)
            except Exception:
                continue
            cols = {k.lower(): k for k in df.columns}
            fcol = next((cols[k] for k in cols if any(t in k for t in ("file","name","path","utt"))), None)
            lcol = next((cols[k] for k in cols if any(t in k for t in ("label","truth","class","type"))), None)
            if not (fcol and lcol) or len(df) < 20: continue
            lab = df[lcol].astype(str).str.lower().str.strip()
            y = lab.map({"real":0, "bona":0, "bonafide":0, "genuine":0, "0":0,
                        "fake":1, "spoof":1, "synth":1, "synthetic":1, "1":1})
            keep = y.notna()
            if keep.sum() < 20: continue
            for f, yv in zip(df[fcol][keep].astype(str), y[keep].astype(int)):
                pairs.append((Path(f).name, yv))
            print(f"    labels from CSV: {c.name} ({keep.sum()} rows)")
            break
        return pairs

    def _labels_from_dirs(root):
        R = list(Path(root).rglob("*/real/**/*")) + list(Path(root).rglob("real/**/*"))
        F = list(Path(root).rglob("*/fake/**/*")) + list(Path(root).rglob("fake/**/*"))
        R = [p for p in R if p.suffix.lower() in AUDIO_EXTS]
        F = [p for p in F if p.suffix.lower() in AUDIO_EXTS]
        return [(p.name, 0) for p in R] + [(p.name, 1) for p in F]

    def build_manifest(root, n_per, name, seed=1234):
        root = Path(root)
        assert root.exists() and any(root.rglob("*")), \
            f"[{name}] {root} empty — enable & download the dataset in DATASETS"
        idx = _index_audio(root)
        assert idx, f"[{name}] no audio files found under {root} (looked for {AUDIO_EXTS})"
        for loader in (_labels_from_asvspoof, _labels_from_csv, _labels_from_dirs):
            pairs = loader(root)
            pairs = [(idx[b], y) for b, y in pairs if b in idx]
            if pairs and len({y for _,y in pairs}) == 2:
                print(f"  [{name}] loader={loader.__name__} pairs={len(pairs)}")
                break
        else:
            raise AssertionError(f"[{name}] could not derive labels — no protocol/csv/real-fake dirs matched")
        R = [p for p, y in pairs if y == 0]; Fk = [p for p, y in pairs if y == 1]
        rng = random.Random(seed); rng.shuffle(R); rng.shuffle(Fk)
        R, Fk = R[:n_per], Fk[:n_per]
        paths = [str(p) for p in R + Fk]; labels = [0]*len(R) + [1]*len(Fk)
        j = list(range(len(paths))); rng.shuffle(j)
        print(f"[{name}] real={len(R)} fake={len(Fk)}")
        return {"paths": [paths[i] for i in j], "labels": [labels[i] for i in j]}

    SRC = build_manifest(CFG.SOURCE_ROOT, CFG.SRC_PER_CLS, "source")
    TGT = build_manifest(CFG.TARGET_ROOT, CFG.TGT_PER_CLS, "target")
    for nm, m in (("source", SRC), ("target", TGT)):
        json.dump({"fingerprint": hashlib.sha256("\n".join(m["paths"]).encode()).hexdigest()[:10],
                   "n": len(m["paths"]),
                   "paths": [str(Path(p).relative_to(DATA)) for p in m["paths"]],
                   "labels": m["labels"]},
                  open(PERSIST/"results"/f"manifest_{nm}.json", "w"))
    print("[provenance] manifest_source.json / manifest_target.json written")
    cut = int(0.85 * len(SRC["paths"]))
    SRC_TR = {"paths": SRC["paths"][:cut], "labels": SRC["labels"][:cut]}
    SRC_VA = {"paths": SRC["paths"][cut:], "labels": SRC["labels"][cut:]}

    from transformers import Wav2Vec2Model
    def man_fp(man):
        h = hashlib.sha256("\n".join(man["paths"]).encode()).hexdigest()[:10]
        return f"n{len(man['paths'])}_{h}"
    @torch.no_grad()
    def extract(mk, hf, man, split):
        fp = man_fp(man)
        fx = PERSIST / "features" / f"{mk}__{split}__{fp}.npy"
        fy = PERSIST / "features" / f"{mk}__{split}__{fp}__y.npy"
        for leg in ((PERSIST/"features"/f"{mk}__{split}.npy"),
                    (PERSIST/"features"/f"{mk}__{split}__y.npy")):
            if leg.exists(): leg.unlink(); print(f"[cache] purged legacy {leg.name}")
        stale = [p for p in (PERSIST / "features").glob(f"{mk}__{split}__n*.npy")
                 if p != fx and p != fy]
        if stale and not fx.exists():
            print(f"[cache] {len(stale)} other-manifest {mk}__{split} cache(s) kept (e.g. DEBUG vs full)")
        if fx.exists(): print(f"[cache] {fx.name}"); return fx, fy
        model = Wav2Vec2Model.from_pretrained(hf).to(DEVICE).eval().half()
        X, Y, B, BL = [], [], [], []
        def flush():
            if not B: return
            x = torch.tensor(np.stack(B), dtype=torch.float16, device=DEVICE)
            hs = model(x, output_hidden_states=True).hidden_states
            X.append(torch.stack([h.mean(1) for h in hs], 1).cpu().numpy().astype(np.float16))
            Y.extend(BL); B.clear(); BL.clear()
        for i, (p, l) in enumerate(zip(man["paths"], man["labels"])):
            B.append(load_clip(p)); BL.append(l)
            if len(B) == CFG.SSL_BATCH: flush()
            if (i+1) % 500 == 0: print(f"  [{mk}:{split}] {i+1}/{len(man['paths'])}")
        flush()
        np.save(fx, np.concatenate(X)); np.save(fy, np.array(Y, np.int8))
        del model; torch.cuda.empty_cache(); return fx, fy

    FE = {}
    for mk, hf in CFG.SSL_MODELS.items():
        for sp, mn in [("src_tr", SRC_TR), ("src_va", SRC_VA), ("tgt", TGT)]:
            FE[(mk, sp)] = extract(mk, hf, mn, sp)

    # ---- S2: layer-wise probes ----
    probe, bank = {}, {}
    for mk in CFG.SSL_MODELS:
        Xtr = np.load(FE[(mk, "src_tr")][0]).astype(np.float32); ytr = np.load(FE[(mk, "src_tr")][1])
        Xva = np.load(FE[(mk, "src_va")][0]).astype(np.float32); yva = np.load(FE[(mk, "src_va")][1])
        Xte = np.load(FE[(mk, "tgt")][0]).astype(np.float32);    yte = np.load(FE[(mk, "tgt")][1])
        probe[mk] = {}
        for L in range(Xtr.shape[1]):
            probe[mk][L] = {}
            for sd in CFG.SEEDS:
                set_seed(sd)
                sc = StandardScaler().fit(Xtr[:, L])
                clf = LogisticRegression(max_iter=2000, random_state=sd).fit(sc.transform(Xtr[:, L]), ytr)
                sv = clf.predict_proba(sc.transform(Xva[:, L]))[:, 1]
                st = clf.predict_proba(sc.transform(Xte[:, L]))[:, 1]
                probe[mk][L][sd] = {"in": float(roc_auc_score(yva, sv)), "cross": float(roc_auc_score(yte, st))}
                bank[(mk, L, sd)] = st
            print(f"[S2] {mk} L{L:02d} cross={np.mean([probe[mk][L][s]['cross'] for s in CFG.SEEDS]):.3f}")
        del Xtr, Xva, Xte
    json.dump(probe, open(PERSIST / "results" / "probe_results.json", "w"), indent=2, default=str)
    yte = np.load(FE[("base", "tgt")][1])

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for j, mk in enumerate(["base", "xlsr"]):
        Ls = sorted(probe[mk])
        for split, c in [("in", "tab:gray"), ("cross", "tab:red")]:
            mu = [np.mean([probe[mk][L][s][split] for s in CFG.SEEDS]) for L in Ls]
            sd_ = [np.std([probe[mk][L][s][split] for s in CFG.SEEDS]) for L in Ls]
            ax[j].plot(Ls, mu, "-o", ms=3, color=c, label=split)
            ax[j].fill_between(Ls, np.array(mu)-sd_, np.array(mu)+sd_, color=c, alpha=0.2)
        ax[j].axhline(0.5, ls="--", c="k", lw=0.8); ax[j].set_title(mk); ax[j].grid(alpha=0.3)
    ax[0].legend(); fig.suptitle("Fig.2 — layer-wise transfer")
    fig.savefig(PERSIST / "figures" / "fig2_layerwise.png", dpi=200); plt.close(fig)
    best = {mk: max(probe[mk], key=lambda L: np.mean([probe[mk][L][s]["cross"] for s in CFG.SEEDS])) for mk in probe}
    print("best layers:", best)

    # ---- head export for av_localize zero-shot (refit best layer, seed 0) ----
    for mk, hf in CFG.SSL_MODELS.items():
        Xtr = np.load(FE[(mk, "src_tr")][0]).astype(np.float32); ytr = np.load(FE[(mk, "src_tr")][1])
        set_seed(CFG.SEEDS[0])
        sc = StandardScaler().fit(Xtr[:, best[mk]])
        clf = LogisticRegression(max_iter=2000, random_state=CFG.SEEDS[0]).fit(
            sc.transform(Xtr[:, best[mk]]), ytr)
        save_probe_head(PERSIST/"ckpts"/f"probe_head_audio_{mk}.npz", sc, clf,
                        {"backbone": hf, "layer": int(best[mk]),
                         "window_s": CFG.CROP_SEC, "trained_on": CFG.SOURCE_NAME})
        del Xtr

    # ---- S3: confounds ----
    def astats(p):
        y = load_clip(p); S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-10
        freqs = librosa.fft_frequencies(sr=CFG.SR, n_fft=1024)
        band = 20*np.log10(S.mean(1)); band -= band.max()
        above = np.where(band > -40)[0]
        return {"cutoff": float(freqs[above[-1]]) if len(above) else 0.0,
                "rolloff": float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=CFG.SR, roll_percent=0.95))),
                "flatness": float(np.mean(librosa.feature.spectral_flatness(S=S)))}
    def cstats(man, cap=1200):
        rng = random.Random(7); idx = list(range(len(man["paths"]))); rng.shuffle(idx)
        return [dict(astats(man["paths"][i]), label=man["labels"][i]) for i in idx[:cap]]
    scs, tcs = cstats(SRC), cstats(TGT)
    summ = {}
    for k in ("cutoff", "rolloff", "flatness"):
        g = lambda st: np.mean([r[k] for r in st if r["label"] == 1]) - np.mean([r[k] for r in st if r["label"] == 0])
        summ[k] = {"src_gap": float(g(scs)), "tgt_gap": float(g(tcs)),
                   "sign_flip": bool(np.sign(g(scs)) != np.sign(g(tcs)))}
    json.dump(summ, open(PERSIST / "results" / "confound_summary.json", "w"), indent=2)
    print("[S3]", json.dumps(summ, indent=2))

    # ---- S4: mel-CNN ± channel augmentation ----
    MEL = librosa.filters.mel(sr=CFG.SR, n_fft=1024, n_mels=CFG.N_MELS)
    def logmel(y):
        M = np.log(MEL @ (np.abs(librosa.stft(y, n_fft=1024, hop_length=256))**2) + 1e-8)
        return ((M - M.mean())/(M.std()+1e-6)).astype(np.float32)
    def chan_aug(y, rng):
        y = y.copy()
        if rng.random() < 0.8:
            snr = rng.uniform(8, 30)
            n = np.random.default_rng(rng.randint(0, 1<<30)).standard_normal(len(y)).astype(np.float32)
            n = np.convolve(n, np.array([1.0, rng.uniform(-0.9, 0.9)], np.float32), "same")
            n *= math.sqrt((np.mean(y**2)+1e-12)/(np.mean(n**2)+1e-12)/(10**(snr/10))); y = y + n
        if rng.random() < 0.7:
            Y = librosa.stft(y, n_fft=1024, hop_length=256)
            fr = librosa.fft_frequencies(sr=CFG.SR, n_fft=1024)
            Y[fr > rng.uniform(3000, 7600), :] *= rng.uniform(0, 0.1)
            y = librosa.istft(Y, hop_length=256, length=len(y))
        if rng.random() < 0.3: y = np.clip(y, -rng.uniform(0.6, 0.95), rng.uniform(0.6, 0.95))
        y *= rng.uniform(0.5, 1.0); m = np.abs(y).max(); return (y/m if m > 0 else y).astype(np.float32)
    class _PrecompDS(torch.utils.data.Dataset):
        def __init__(self, man, variant): self.m, self.v = man, variant
        def __len__(self): return len(self.m["paths"])
        def __getitem__(self, i):
            y = load_clip(self.m["paths"][i])
            if self.v > 0: y = chan_aug(y, random.Random(self.v*7_368_787 + i))
            return torch.from_numpy(logmel(y)), self.m["labels"][i]
    def precompute_logmel(man, split, variant=0):
        fp = man_fp(man)
        fx = PERSIST/"features"/f"mel__{split}__v{variant}__{fp}.npy"
        fy = PERSIST/"features"/f"mel__{split}__v{variant}__{fp}__y.npy"
        if fx.exists(): print(f"[melcache] hit {fx.name}"); return fx, fy
        t0 = time.time()
        dl = torch.utils.data.DataLoader(_PrecompDS(man, variant), batch_size=64,
                                         num_workers=os.cpu_count() or 2)
        X, Y = [], []
        for j, (x, y) in enumerate(dl):
            X.append(x.numpy().astype(np.float16)); Y.append(y.numpy())
            if (j+1) % 20 == 0:
                done = (j+1)*64; eta = (time.time()-t0)/done*(len(man["paths"])-done)
                print(f"  [melcache {split} v{variant}] {done}/{len(man['paths'])} eta {eta/60:.1f}m")
        np.save(fx, np.concatenate(X)); np.save(fy, np.concatenate(Y).astype(np.int8))
        print(f"[melcache] wrote {fx.name} in {(time.time()-t0)/60:.1f}m")
        return fx, fy

    SRC_CNN = {"paths": SRC_TR["paths"][:CFG.CNN_MAX_TRAIN],
               "labels": SRC_TR["labels"][:CFG.CNN_MAX_TRAIN]}
    print(f"[S4] CNN train cap: {len(SRC_CNN['paths'])} clips "
          f"({sum(SRC_CNN['labels'])} fake) | aug variants K={CFG.CNN_AUG_K}")
    mel_tr = [np.load(precompute_logmel(SRC_CNN, "src_cnn", v)[0], mmap_mode="r")
              for v in range(CFG.CNN_AUG_K + 1)]
    ytr_cnn = np.load(PERSIST/"features"/f"mel__src_cnn__v0__{man_fp(SRC_CNN)}__y.npy")
    mel_va, yva_f = precompute_logmel(SRC_VA, "src_va_mel"); mel_tg, ytg_f = precompute_logmel(TGT, "tgt_mel")
    mel_va = np.load(mel_va, mmap_mode="r"); yva_mel = np.load(yva_f)
    mel_tg = np.load(mel_tg, mmap_mode="r"); ytg_mel = np.load(ytg_f)

    class MelDS(torch.utils.data.Dataset):
        def __init__(self, aug=False, seed=0, epoch=0): self.aug, self.seed, self.epoch = aug, seed, epoch
        def __len__(self): return len(ytr_cnn)
        def __getitem__(self, i):
            v = 0 if not self.aug else 1 + (hash((self.seed, self.epoch, i)) % CFG.CNN_AUG_K)
            return torch.from_numpy(np.asarray(mel_tr[v][i], np.float32))[None], int(ytr_cnn[i])
    class MelCNN(nn.Module):
        def __init__(self):
            super().__init__(); ch = [1, 32, 64, 128, 128]
            self.b = nn.Sequential(*[nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1),
                nn.BatchNorm2d(ch[i+1]), nn.ReLU(), nn.MaxPool2d(2)) for i in range(4)])
            self.h = nn.Linear(128, 1)
        def forward(self, x): return self.h(self.b(x).mean((2, 3))).squeeze(-1)
    @torch.no_grad()
    def cnn_eval(model, X, Y):
        S = []
        for j in range(0, len(Y), CFG.CNN_BATCH):
            xb = torch.from_numpy(np.asarray(X[j:j+CFG.CNN_BATCH], np.float32))[:, None].to(DEVICE)
            S.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.concatenate(S), np.asarray(Y)
    abl = {"no_aug": {}, "chan_aug": {}}
    for sd in CFG.CNN_SEEDS:
        for aug in (False, True):
            set_seed(sd); model = MelCNN().to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG.CNN_LR)
            t0 = time.time()
            for ep in range(CFG.CNN_EPOCHS):
                model.train()
                ds = MelDS(aug, sd, ep)
                dl = torch.utils.data.DataLoader(ds, batch_size=CFG.CNN_BATCH,
                                                 shuffle=True, num_workers=2, drop_last=True)
                for x, y in dl:
                    loss = F.binary_cross_entropy_with_logits(model(x.to(DEVICE)), y.float().to(DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()
                print(f"  [S4] seed={sd} aug={aug} ep {ep+1}/{CFG.CNN_EPOCHS} "
                      f"({(time.time()-t0)/(ep+1):.0f}s/ep)")
            sv, yv = cnn_eval(model, mel_va, yva_mel); st, yt = cnn_eval(model, mel_tg, ytg_mel)
            abl["chan_aug" if aug else "no_aug"][sd] = {
                "in": float(roc_auc_score(yv, sv)), "cross": float(roc_auc_score(yt, st))}
            print(f"[S4] aug={aug} seed={sd} in={abl['chan_aug' if aug else 'no_aug'][sd]['in']:.3f} "
                  f"cross={abl['chan_aug' if aug else 'no_aug'][sd]['cross']:.3f}")
    json.dump(abl, open(PERSIST / "results" / "ablation.json", "w"), indent=2)

    np.save(PERSIST/"results"/"target_labels.npy", yte)
    for mk in best:
        np.save(PERSIST/"results"/f"target_scores_{mk}_L{best[mk]:02d}.npy",
                bank[(mk, best[mk], CFG.SEEDS[0])])

    sig = None
    if {"base", "xlsr"} <= set(best):
        sig = paired_bootstrap_delta(yte, bank[("base", best["base"], CFG.SEEDS[0])],
                                          bank[("xlsr", best["xlsr"], CFG.SEEDS[0])])
        json.dump(sig, open(PERSIST/"results"/"significance_base_vs_xlsr.json", "w"), indent=2)

    HIERCON_ITW_EER = 6.87   # arXiv:2602.01032 (LA-trained XLS-R + HierCon)
    SLS_ITW_EER     = 8.87   # arXiv:2602.01032 Table 1 (XLS-R + SLS)
    headline = {}
    for mk in best:
        s0 = bank[(mk, best[mk], CFG.SEEDS[0])]
        auc = float(roc_auc_score(yte, s0))
        auc_ci = bootstrap_auc_ci(yte, s0)
        eer, thr = eer_of(yte, s0)
        eer_ci = bootstrap_eer_ci(yte, s0)
        headline[mk] = {"layer": int(best[mk]), "n_target": int(len(yte)),
                        "auc": auc, "auc_ci": auc_ci,
                        "eer_pct": eer*100, "eer_ci_pct": [x*100 for x in eer_ci],
                        "eer_threshold": thr}
    json.dump(headline, open(PERSIST/"results"/"headline_ssl.json","w"), indent=2)

    print("="*66, "\nWEEK1 (Colab) VERDICT")
    print(f"  target n = {len(yte)} ({int((yte==0).sum())} real / {int((yte==1).sum())} fake)")
    print(f"  benchmark refs — HierCon ITW EER = {HIERCON_ITW_EER}%  |  SLS ITW EER = {SLS_ITW_EER}%")
    for mk, h in headline.items():
        print(f"  {mk:5s} L{h['layer']:02d}  AUC {h['auc']:.3f} [{h['auc_ci'][0]:.3f}, {h['auc_ci'][1]:.3f}]"
              f"  EER {h['eer_pct']:5.2f}% [{h['eer_ci_pct'][0]:5.2f}, {h['eer_ci_pct'][1]:5.2f}]")
    for cond in abl:
        cs = [abl[cond][s]["cross"] for s in abl[cond]]
        print(f"  mel-CNN {cond:8s} cross-AUC {np.mean(cs):.3f} ± {np.std(cs):.3f}"
              f"  (regime: {'inversion' if np.mean(cs) < 0.4 else ('collapse' if np.mean(cs) < 0.6 else 'transfer')})")
    print("  spectral confound flips:", [k for k, v in summ.items() if v["sign_flip"]])
    if sig is not None:
        verdict = ("xlsr > base SIGNIFICANT" if sig["p_two_sided"] < 0.05 and sig["delta_auc_mean"] > 0
                   else "base > xlsr SIGNIFICANT" if sig["p_two_sided"] < 0.05
                   else "difference NOT significant — report as equivalent")
        print(f"  paired ΔAUC (xlsr−base) = {sig['delta_auc_mean']:+.3f} "
              f"CI [{sig['delta_ci'][0]:+.3f}, {sig['delta_ci'][1]:+.3f}]  p={sig['p_two_sided']:.4f} → {verdict}")

    tex = PERSIST / "reports" / "table_row.tex"
    existing = set(open(tex).read().splitlines()) if tex.exists() else set()
    new_rows = []
    for mk, h in headline.items():
        row = (f"{CFG.SOURCE_NAME}→{CFG.TARGET_NAME} & {mk} L{h['layer']} & "
               f"{h['eer_pct']:.2f} & [{h['eer_ci_pct'][0]:.2f},{h['eer_ci_pct'][1]:.2f}] & "
               f"{h['auc']:.3f} & n={h['n_target']} \\\\")
        if row not in existing: new_rows.append(row)
    if new_rows:
        with open(tex, "a") as f:
            f.write("% week1_audio VERDICT\n" + "\n".join(new_rows) + "\n")
        print(f"  latex: {len(new_rows)} new row(s) →", tex)
    else:
        print("  latex: rows already present (identical run) — nothing appended")
    print("  all artifacts →", PERSIST)


# ============================================================================
# ⭐ STAGE A2 (V19): hiercon_audio — HierCon-lite (arXiv:2602.01032, Feb 2026)
# ----------------------------------------------------------------------------
# WHAT THIS STAGE DOES:
#   HierCon proposes hierarchical layer attention (temporal → intra-group →
#   inter-group) + margin contrastive learning on top of frozen XLS-R features.
#   Published EER on ITW: 6.87%.  Their setup fine-tunes XLS-R end-to-end.
#
#   We CANNOT afford that on a T4 in a month.  What we CAN do:
#     • REUSE the frozen XLS-R hidden states already cached by week1_audio
#       (features/xlsr__src_tr__*.npy has shape [N, 25, 1024]).
#     • Add ONLY the classifier head: (a) frame-mean pooling → done in cache,
#       (b) 3 attention layers: intra-group across neighbouring layers,
#           inter-group across the 5 xls-r groups, (c) MLP → logit.
#     • Add margin contrastive loss (AAM/SupCon-like) alongside BCE.
#
# HONEST TARGETS on ASVspoof2019-LA → In-the-Wild (Δ vs your locked 19.55%):
#   Frozen single-layer LR (locked)         19.55% EER   ← PAPER1 baseline
#   +hierarchical attention over layers     14–16%  EER  ← V19 realistic
#   +contrastive margin (m=0.25)            11–14%  EER  ← V19 stretch
#   HierCon (E2E fine-tune, published)       6.87% EER   ← out of budget
#
# SANITY CHECK before writing this into the paper:
#   • If EER drops below 8% on ITW with a frozen backbone, SOMETHING IS
#     LEAKING (probably train-domain sample in the eval set, or a duplicated
#     manifest fingerprint).  RE-CHECK manifest_target.json's fingerprint.
#   • If EER stays above 18%, the attention isn't learning — check that the
#     grouped attention dropout isn't set to 1.0 (bug we hit in v14).
# ----------------------------------------------------------------------------
# WHERE THIS FITS IN THE PAPER:
#   Section 4.3 "Beyond single-layer probes".  This is not the paper's main
#   claim; the main claim is still the frozen-probe layer-selection story.
#   HierCon-lite is a CONTROL that shows the probe result is not a ceiling.
# ============================================================================

if STAGE == "hiercon_audio":
    # ---- Sanity: feature caches from week1_audio must already exist --------
    # Each feature array has a paired label array with the suffix "__y.npy".
    # Excluding labels here prevents selecting a label array as a feature array.
    feat_dir = PERSIST / "features"

    def _xlsr_feature_files(pattern):
        return sorted(
            p for p in feat_dir.glob(pattern)
            if not p.name.endswith("__y.npy")
        )

    xlsr_tr = _xlsr_feature_files("xlsr__src_tr__n*.npy")
    xlsr_va = _xlsr_feature_files("xlsr__src_va__n*.npy")
    xlsr_tg = _xlsr_feature_files("xlsr__tgt__n*.npy")

    if not (xlsr_tr and xlsr_va and xlsr_tg):
        found = sorted(p.name for p in feat_dir.glob("xlsr__*.npy"))
        raise RuntimeError(
            "[hiercon_audio] Required XLS-R feature caches are missing.\n"
            "Run STAGE = 'week1_audio' to completion, then rerun with "
            "STAGE = 'hiercon_audio'.\n"
            f"Expected feature files in {feat_dir}: xlsr__src_tr__n*.npy, "
            "xlsr__src_va__n*.npy, xlsr__tgt__n*.npy.\n"
            f"Found: {found[:12] if found else 'none'}"
        )

    # Prefer the largest manifest, then newest cache, if debug and full caches coexist.
    def _pick_latest(paths):
        return max(
            paths,
            key=lambda p: (
                int(p.stem.split("__n")[-1].split("_")[0]),
                p.stat().st_mtime,
            ),
        )

    fx_tr = _pick_latest(xlsr_tr)
    fx_va = _pick_latest(xlsr_va)
    fx_tg = _pick_latest(xlsr_tg)
    fy_tr = fx_tr.with_name(fx_tr.stem + "__y.npy")
    fy_va = fx_va.with_name(fx_va.stem + "__y.npy")
    fy_tg = fx_tg.with_name(fx_tg.stem + "__y.npy")

    for f in (fy_tr, fy_va, fy_tg):
        if not f.exists():
            raise FileNotFoundError(
                f"[hiercon_audio] Label file missing: {f.name}. "
                "The feature and label cache pair is incomplete; rerun week1_audio."
            )

    print("[hiercon_audio] reusing week1_audio caches:")
    print(f"  train:  {fx_tr.name} ({fx_tr.stat().st_size / 1e6:.0f} MB)")
    print(f"  val:    {fx_va.name}")
    print(f"  target: {fx_tg.name}")

    # ---- Hyper-parameters (paper-competitive, T4-safe) ----------------------
    class HC:
        # attention
        D_MODEL       = 1024        # XLS-R hidden dim (fixed by backbone)
        N_LAYERS      = 25          # XLS-R total layers (0=CNN out, 1..24=transformer)
        N_GROUPS      = 5           # HierCon groups: G0=L0-4, G1=L5-9, ..., G4=L20-24
        D_ATTN        = 128         # attention head dim (kept small — 4M params total)
        N_HEADS       = 4
        DROP          = 0.10
        # training
        EPOCHS        = 30 if not DEBUG else 3
        BATCH         = 256
        LR            = 2e-4
        WD            = 1e-4
        WARMUP_FRAC   = 0.10
        # loss
        BCE_WEIGHT    = 1.0
        CTR_WEIGHT    = 0.30        # HierCon uses 0.5; on frozen features 0.3 is safer
        CTR_MARGIN    = 0.25        # margin-based supervised contrastive; keep <0.35 or bonafide cluster collapses
        # eval / logging
        SEEDS         = [0, 1, 2]   # 3 runs for CIs (paper requires this)
        VAL_EVERY     = 2

    if DEBUG:
        HC.BATCH, HC.SEEDS = 64, [0]

    print(f"[hiercon_audio] config: {HC.EPOCHS} epochs, batch {HC.BATCH}, "
          f"attn dim {HC.D_ATTN}, {HC.N_HEADS} heads, seeds {HC.SEEDS}")

    # ---- Load features (mmap; float16 on disk) ------------------------------
    Xtr = np.load(fx_tr, mmap_mode="r"); ytr = np.load(fy_tr)
    Xva = np.load(fx_va, mmap_mode="r"); yva = np.load(fy_va)
    Xtg = np.load(fx_tg, mmap_mode="r"); ytg = np.load(fy_tg)
    # Shape audit — if this fails, week1_audio changed its cache layout.
    assert Xtr.ndim == 3 and Xtr.shape[1] == HC.N_LAYERS and Xtr.shape[2] == HC.D_MODEL, (
        f"[hiercon_audio] unexpected XLS-R cache shape {Xtr.shape} — "
        f"expected (N, {HC.N_LAYERS}, {HC.D_MODEL}).")
    print(f"[hiercon_audio] shapes  tr={Xtr.shape}  va={Xva.shape}  tg={Xtg.shape}")
    print(f"[hiercon_audio] class balance  tr={ytr.mean():.3f}  va={yva.mean():.3f}  tg={ytg.mean():.3f}")

    # ---- Model: HierCon-lite ------------------------------------------------
    class HierConLite(nn.Module):
        """Hierarchical layer attention + contrastive head.

        The idea (per arXiv:2602.01032):
          1. Group the N_LAYERS backbone layers into N_GROUPS contiguous groups.
             Each group is a "resolution" of the transformer (early → late).
          2. INTRA-group attention: within each group, learn which layer matters.
             Frozen-features version: mean over layers with attention weights.
          3. INTER-group attention: across groups, learn which resolution matters.
          4. Concatenate group representatives → MLP → logit + embedding.

        Key HierCon design choice we PRESERVE: two-head output — BCE logit +
        L2-normalized embedding for margin-based contrastive.  Both trained
        simultaneously (weighted sum).

        NOT preserved (compute-budget cuts):
          - No temporal frame attention (features are already mean-pooled).
          - No adversarial domain classifier (paper mentions but doesn't ablate).
        """
        def __init__(self, cfg=HC):
            super().__init__()
            self.cfg = cfg
            # Which layers go in which group?  Split evenly.
            group_size = cfg.N_LAYERS // cfg.N_GROUPS
            self.group_slices = [(g*group_size, (g+1)*group_size if g < cfg.N_GROUPS-1 else cfg.N_LAYERS)
                                 for g in range(cfg.N_GROUPS)]
            # Intra-group: for each group, a small learned attention over layers.
            # Represented as a linear projection + softmax over the layer dim.
            self.intra_proj = nn.ModuleList([
                nn.Sequential(nn.Linear(cfg.D_MODEL, cfg.D_ATTN),
                              nn.GELU(),
                              nn.Linear(cfg.D_ATTN, 1))
                for _ in range(cfg.N_GROUPS)])
            # Each group produces a D_MODEL-sized "resolution vector".
            # Inter-group: multi-head self-attention across the N_GROUPS reps.
            self.group_proj = nn.Linear(cfg.D_MODEL, cfg.D_ATTN)
            self.inter_attn = nn.MultiheadAttention(cfg.D_ATTN, cfg.N_HEADS,
                                                    dropout=cfg.DROP, batch_first=True)
            self.inter_norm = nn.LayerNorm(cfg.D_ATTN)
            # Heads
            self.emb_head = nn.Sequential(
                nn.Linear(cfg.D_ATTN * cfg.N_GROUPS, 512),
                nn.GELU(),
                nn.Dropout(cfg.DROP),
                nn.Linear(512, 128))            # embedding for contrastive
            self.cls_head = nn.Linear(128, 1)   # BCE logit

        def forward(self, X):
            # X: (B, N_LAYERS, D_MODEL) already frame-pooled by week1_audio.
            group_vecs = []
            for i, (a, b) in enumerate(self.group_slices):
                Xg = X[:, a:b, :]                                # (B, K, D)
                w = self.intra_proj[i](Xg).squeeze(-1)           # (B, K)
                w = torch.softmax(w, dim=1).unsqueeze(-1)        # (B, K, 1)
                gv = (w * Xg).sum(dim=1)                         # (B, D)
                group_vecs.append(gv)
            G = torch.stack(group_vecs, dim=1)                   # (B, N_GROUPS, D)
            Gp = self.group_proj(G)                              # (B, N_GROUPS, D_ATTN)
            A, _ = self.inter_attn(Gp, Gp, Gp)                   # (B, N_GROUPS, D_ATTN)
            A = self.inter_norm(A + Gp)
            flat = A.flatten(1)                                  # (B, N_GROUPS*D_ATTN)
            emb = F.normalize(self.emb_head(flat), dim=1)        # unit vectors
            logit = self.cls_head(emb).squeeze(-1)               # (B,)
            return logit, emb

    def sup_con_loss(emb, y, margin=HC.CTR_MARGIN):
        """Margin-contrastive: pull same-class embeddings within margin, push
        opposite-class apart by margin.  Simpler than SupCon-full; works fine on
        binary labels.  cos_sim in [-1, 1].  Loss ∝ mean(max(0, margin ± cos))."""
        cos = emb @ emb.t()                                       # (B, B)
        same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
        diff = 1.0 - same
        # mask self-pairs
        B = emb.size(0)
        mask = 1.0 - torch.eye(B, device=emb.device)
        pos = (F.relu(margin - cos) * same * mask).sum() / (same * mask).sum().clamp(min=1)
        neg = (F.relu(cos - (1.0 - margin)) * diff * mask).sum() / (diff * mask).sum().clamp(min=1)
        return pos + neg

    class FeatDS(torch.utils.data.Dataset):
        def __init__(self, X, y): self.X, self.y = X, y
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            # cast fp16 → fp32 at load time (we keep RAM-friendly fp16 in cache)
            return torch.from_numpy(np.asarray(self.X[i], dtype=np.float32)), int(self.y[i])

    @torch.no_grad()
    def eval_split(model, X, y, tag):
        model.eval()
        ds = FeatDS(X, y)
        dl = torch.utils.data.DataLoader(ds, batch_size=HC.BATCH, num_workers=0)
        S = []
        for xb, _ in dl:
            logit, _ = model(xb.to(DEVICE))
            S.append(torch.sigmoid(logit).cpu().numpy())
        s = np.concatenate(S)
        auc = float(roc_auc_score(y, s))
        eer, thr = eer_of(y, s)
        print(f"  [eval:{tag}] AUC={auc:.4f}  EER={eer*100:5.2f}%  n={len(y)}")
        return {"auc": auc, "eer_pct": eer*100, "thr": thr, "scores": s}

    def train_one_seed(seed):
        set_seed(seed)
        model = HierConLite().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[seed {seed}] trainable params = {n_params/1e6:.2f} M")   # V20: printed value is the one to cite
        opt = torch.optim.AdamW(model.parameters(), lr=HC.LR, weight_decay=HC.WD)
        n_steps = HC.EPOCHS * (len(ytr) // HC.BATCH + 1)
        n_warm = max(1, int(HC.WARMUP_FRAC * n_steps))
        def lr_lambda(step):
            if step < n_warm: return step / max(1, n_warm)
            prog = (step - n_warm) / max(1, n_steps - n_warm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        dl = torch.utils.data.DataLoader(FeatDS(Xtr, ytr), batch_size=HC.BATCH,
                                         shuffle=True, num_workers=0, drop_last=True)
        best_val_auc = 0.0
        best_state = None
        history = []
        step = 0
        for ep in range(HC.EPOCHS):
            model.train()
            ep_bce = ep_ctr = 0.0
            t0 = time.time()
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.float().to(DEVICE)
                logit, emb = model(xb)
                bce = F.binary_cross_entropy_with_logits(logit, yb)
                ctr = sup_con_loss(emb, yb.long())
                loss = HC.BCE_WEIGHT * bce + HC.CTR_WEIGHT * ctr
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                ep_bce += bce.item(); ep_ctr += ctr.item(); step += 1
            if (ep + 1) % HC.VAL_EVERY == 0 or ep == HC.EPOCHS - 1:
                vr = eval_split(model, Xva, yva, f"val ep{ep+1}")
                history.append({"epoch": ep+1, "val_auc": vr["auc"], "val_eer_pct": vr["eer_pct"],
                                "bce": ep_bce/len(dl), "ctr": ep_ctr/len(dl)})
                if vr["auc"] > best_val_auc:
                    best_val_auc = vr["auc"]
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"  [seed {seed} ep{ep+1}] new best val AUC {best_val_auc:.4f} "
                          f"({(time.time()-t0):.0f}s/ep)")
        # restore best and evaluate on ITW target
        if best_state is not None:
            model.load_state_dict(best_state)
        target = eval_split(model, Xtg, ytg, f"TARGET (ITW) seed{seed}")
        # export the head for downstream reuse (av_localize, dfe2024_v2)
        ck = PERSIST / "ckpts" / f"hiercon_lite_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "config": {k: getattr(HC, k) for k in dir(HC) if k.isupper()},
                    "n_trainable_params": int(n_params)},   # V20: record real count
                   ck)
        print(f"  [seed {seed}] ckpt → {ck.name}")
        return target, history, n_params

    results = {}
    all_target_scores = []
    n_params_ref = None
    for sd in HC.SEEDS:
        tgt, hist, n_params_sd = train_one_seed(sd)
        results[sd] = {"target_ITW": tgt, "history": hist, "n_trainable_params": int(n_params_sd)}
        if n_params_ref is None: n_params_ref = int(n_params_sd)
        all_target_scores.append(tgt["scores"])

    # ---- aggregate + CIs ----------------------------------------------------
    S_ens = np.mean(all_target_scores, axis=0)   # seed-averaged score
    auc_ens = float(roc_auc_score(ytg, S_ens))
    auc_ci = bootstrap_auc_ci(ytg, S_ens)
    eer_ens, thr_ens = eer_of(ytg, S_ens)
    eer_ci = bootstrap_eer_ci(ytg, S_ens)
    per_seed_eer = [results[s]["target_ITW"]["eer_pct"] for s in HC.SEEDS]

    # ---- paired significance vs the locked frozen-LR baseline ---------------
    # week1_audio's target scores are cached at results/target_scores_xlsr_L*.npy
    baseline_files = sorted((PERSIST/"results").glob("target_scores_xlsr_L*.npy"))
    baseline_labels = PERSIST/"results"/"target_labels.npy"
    sig_vs_lr = None
    if baseline_files and baseline_labels.exists():
        S_lr = np.load(baseline_files[0])
        y_lr = np.load(baseline_labels)
        # align: both were computed against the same ITW target manifest
        if len(S_lr) == len(ytg):
            sig_vs_lr = paired_bootstrap_delta(ytg, S_lr, S_ens)
            print(f"[sig] paired ΔAUC (hiercon-lite − frozen-LR) = "
                  f"{sig_vs_lr['delta_auc_mean']:+.3f}  "
                  f"CI [{sig_vs_lr['delta_ci'][0]:+.3f}, {sig_vs_lr['delta_ci'][1]:+.3f}]  "
                  f"p={sig_vs_lr['p_two_sided']:.4f}")
        else:
            print(f"[sig] SKIPPED — baseline scores n={len(S_lr)} but target n={len(ytg)} "
                  "(different DEBUG-mode manifests; re-run week1_audio in full to align)")

    headline = {
        "backbone": "facebook/wav2vec2-xls-r-300m (frozen)",
        "head": "HierCon-lite (intra+inter group attn, margin-based supervised contrastive)",
        "n_target_ITW": int(len(ytg)),
        "n_seeds": len(HC.SEEDS),
        "n_trainable_params": n_params_ref,   # V20: the code-printed value (architecture count ≈ 1.25M, NOT the "~4M" in the notes)
        "feature_cache_provenance": {"train": fx_tr.name, "val": fx_va.name, "target": fx_tg.name},   # V20: manifest-fingerprint audit
        "per_seed_eer_pct": [float(x) for x in per_seed_eer],
        "seed_ens_auc": auc_ens, "seed_ens_auc_ci": auc_ci,
        "seed_ens_eer_pct": eer_ens*100, "seed_ens_eer_ci_pct": [x*100 for x in eer_ci],
        "significance_vs_frozen_LR": sig_vs_lr,
        "target_score_stats": {"mean_real": float(S_ens[ytg==0].mean()),
                               "mean_fake": float(S_ens[ytg==1].mean()),
                               "sep": float(S_ens[ytg==1].mean() - S_ens[ytg==0].mean())},
        "_2026_anchors_ITW_EER_pct": {
            "SLS (XLS-R, ACM MM 2024)": 8.87,
            "HierCon (XLS-R, arXiv 2602.01032 Feb'26)": 6.87,
            "OUR PAPER1 frozen-LR (week1_audio, locked)": 19.55,
            "OUR PAPER1 HierCon-lite (this stage, frozen)": eer_ens*100},
        "_disclosure": ("Frozen backbone. Published HierCon is end-to-end fine-tuned. "
                        "This head does not aim to beat SOTA on a T4 — it is a control "
                        "showing the frozen-probe result is not a ceiling.")}
    json.dump(headline, open(PERSIST/"results"/"headline_hiercon_lite.json", "w"), indent=2)
    np.save(PERSIST/"results"/"target_scores_hiercon_lite_ens.npy", S_ens)

    print("=" * 66, "\nHIERCON-LITE VERDICT (frozen XLS-R + hierarchical attn + contrastive)")
    print(f"  target n = {len(ytg)} ({int((ytg==0).sum())} real / {int((ytg==1).sum())} fake)")
    print(f"  per-seed EER: {['%.2f%%' % x for x in per_seed_eer]}")
    print(f"  seed-ensemble AUC {auc_ens:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]  "
          f"EER {eer_ens*100:5.2f}% [{eer_ci[0]*100:5.2f}, {eer_ci[1]*100:5.2f}]")
    print(f"  2026 anchors on ITW EER%: SLS=8.87  HierCon(E2E)=6.87  PAPER1 frozen-LR=19.55")
    print(f"  → V19 HierCon-lite = {eer_ens*100:.2f}%  "
          f"(gap-closure vs LR: {(19.55 - eer_ens*100)/(19.55 - 6.87)*100:.0f}% of "
          f"the way to HierCon-E2E)")

    tex = PERSIST / "reports" / "table_row.tex"
    existing = set(open(tex).read().splitlines()) if tex.exists() else set()
    row = (f"ASV19-LA$\\to$ITW & XLS-R frozen + HierCon-lite (V19) & "
           f"{eer_ens*100:.2f} & [{eer_ci[0]*100:.2f},{eer_ci[1]*100:.2f}] & "
           f"{auc_ens:.3f} & n={len(ytg)} \\\\")
    if row not in existing:
        with open(tex, "a") as f:
            f.write("% hiercon_audio VERDICT (V19)\n" + row + "\n")
        print(f"  latex: 1 new row →", tex)
    print("  all artifacts →", PERSIST)


# ============================================================================
# STAGE B: video_train — FF++c23 → Celeb-DF-v2 OFFICIAL test (EffB0 CONTROL)
# ============================================================================

if STAGE == "video_train":
    import cv2, timm
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=DEVICE)
    CROPS = Path("/content/data/crops" if IN_COLAB else "./data/crops"); CROPS.mkdir(parents=True, exist_ok=True)
    IMN_M = np.array([0.485, 0.456, 0.406], np.float32); IMN_S = np.array([0.229, 0.224, 0.225], np.float32)

    def crop_video(path, out_dir, vid):
        if len(list(out_dir.glob(f"{vid}__*.jpg"))) >= max(1, CFG.FRAMES_PER_VID//2): return True
        cap = cv2.VideoCapture(path); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot <= 0: cap.release(); return False
        ok_any = False
        for k, i in enumerate(np.linspace(0, tot-1, CFG.FRAMES_PER_VID).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            box, prob = mtcnn.detect(rgb)
            if box is None or prob is None or prob[0] is None or prob[0] < 0.9: continue
            x0, y0, x1, y1 = box[0]; w, h = x1-x0, y1-y0
            x0 = max(int(x0-CFG.MARGIN*w), 0); y0 = max(int(y0-CFG.MARGIN*h), 0)
            x1 = min(int(x1+CFG.MARGIN*w), rgb.shape[1]); y1 = min(int(y1+CFG.MARGIN*h), rgb.shape[0])
            crop = cv2.resize(rgb[y0:y1, x0:x1], (CFG.IMG, CFG.IMG))
            cv2.imwrite(str(out_dir/f"{vid}__{k}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95]); ok_any = True
        cap.release(); return ok_any

    def build(rg, fg, split, cap_n, only_vids=None):
        man = []
        for nm, lab, g in [("real", 0, rg), ("fake", 1, fg)]:
            vs = sorted(glob.glob(g, recursive=True))
            assert vs, f"[{split}/{nm}] fix glob: {g} (enable dataset above / verify slug)"
            if only_vids is not None:
                vs = [v for v in vs if Path(v).stem in only_vids]
            random.Random(0).shuffle(vs)
            od = CROPS / split / nm; od.mkdir(parents=True, exist_ok=True)
            for j, v in enumerate(vs[:cap_n]):
                if crop_video(v, od, Path(v).stem): man.append({"vid": Path(v).stem, "label": lab, "dir": str(od)})
                if (j+1) % 100 == 0: print(f"  [{split}/{nm}] {j+1}")
        print(f"[{split}] {len(man)} videos"); return man

    cdf_root = DATA / "celebdf-v2"
    test_list = next(iter(cdf_root.rglob("List_of_testing_videos.txt")), None)
    OFFICIAL_CDF = test_list is not None
    cdf_test_vids = None
    if OFFICIAL_CDF:
        cdf_test_vids = set()
        for ln in open(test_list, "r", errors="ignore"):
            parts = ln.strip().split()
            if len(parts) >= 2: cdf_test_vids.add(Path(parts[-1]).stem)
        print(f"[protocol] Celeb-DF OFFICIAL test list: {len(cdf_test_vids)} videos ✅")
    else:
        print("[protocol] ⚠️ List_of_testing_videos.txt NOT found — falling back to random sample.")
        print("            Result will be flagged non-comparable in results JSON.")

    M_POOL = build(CFG.TRAIN_REAL, CFG.TRAIN_FAKE, "train_pool", CFG.MAX_TRAIN_VIDS)
    if OFFICIAL_CDF:
        M_CROSS = build(CFG.CROSS_REAL, CFG.CROSS_FAKE, "cross", None, only_vids=cdf_test_vids)
    else:
        M_CROSS = build(CFG.CROSS_REAL, CFG.CROSS_FAKE, "cross", CFG.MAX_CROSS_VIDS)

    def ids_of(stem):
        toks = stem.split("_")
        return {t for t in toks if t.isdigit()} or {stem}
    all_ids = sorted({i for m in M_POOL for i in ids_of(m["vid"])})
    random.Random(42).shuffle(all_ids)
    n_val_ids = max(1, int(len(all_ids) * CFG.VAL_FRAC))
    val_ids = set(all_ids[:n_val_ids])
    M_VAL = [m for m in M_POOL if ids_of(m["vid"]) & val_ids]
    M_TR  = [m for m in M_POOL if not (ids_of(m["vid"]) & val_ids)]
    print(f"[split] identity-grouped: {len(all_ids)} IDs → train {len(M_TR)} vids / val {len(M_VAL)} vids "
          f"(dropped 0; leakage-free by construction)")

    def vaug(img, rng):
        if rng.random() < 0.5: img = img[:, ::-1]
        if rng.random() < 0.5:
            _, e = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 90)])
            img = cv2.cvtColor(cv2.imdecode(e, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if rng.random() < 0.3: img = cv2.GaussianBlur(img, (rng.choice([3, 5]),)*2, 0)
        if rng.random() < 0.3:
            s = rng.uniform(0.4, 0.9); h, w = img.shape[:2]
            img = cv2.resize(cv2.resize(img, (int(w*s), int(h*s))), (w, h))
        return np.ascontiguousarray(img)
    class FDS(torch.utils.data.Dataset):
        def __init__(self, man, train=False):
            self.items = [(str(f), m["label"], m["vid"]) for m in man
                          for f in Path(m["dir"]).glob(f"{m['vid']}__*.jpg")]
            self.train = train
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            p, l, v = self.items[i]
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            if self.train: img = vaug(img, random.Random(random.getrandbits(32)))
            return torch.from_numpy(((img.astype(np.float32)/255.-IMN_M)/IMN_S).transpose(2, 0, 1)), l, v
    @torch.no_grad()
    def veval(model, ds, name):
        model.eval(); dl = torch.utils.data.DataLoader(ds, batch_size=CFG.V_BATCH, num_workers=2)
        S, Y, V = [], [], []
        for x, y, v in dl:
            S.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()); Y.append(y.numpy()); V.extend(v)
        s, y = np.concatenate(S), np.concatenate(Y)
        agg = {}
        for sc, lb, vd in zip(s, y, V): agg.setdefault(vd, {"s": [], "y": int(lb)})["s"].append(float(sc))
        cy = np.array([a["y"] for a in agg.values()]); cs = np.array([float(np.mean(a["s"])) for a in agg.values()])
        v_eer, _ = eer_of(cy, cs)
        r = {"frame_auc": float(roc_auc_score(y, s)),
             "video_auc": float(roc_auc_score(cy, cs)),
             "video_auc_ci": bootstrap_auc_ci(cy, cs),
             "video_eer_pct": v_eer * 100,
             "video_eer_ci_pct": [x*100 for x in bootstrap_eer_ci(cy, cs)],
             "n_videos": int(len(cy))}
        print(f"[eval:{name}] frame-AUC={r['frame_auc']:.4f}  video-AUC={r['video_auc']:.4f} "
              f"CI={[round(x,3) for x in r['video_auc_ci']]}  video-EER={r['video_eer_pct']:.2f}%  n={r['n_videos']}")
        extras = {"vids": list(agg.keys()), "scores": cs, "labels": cy}
        return r, extras

    out = {}
    for sd in CFG.V_SEEDS:
        set_seed(sd)
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG.V_LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.V_EPOCHS)
        dl = torch.utils.data.DataLoader(FDS(M_TR, True), batch_size=CFG.V_BATCH,
                                         shuffle=True, num_workers=2, drop_last=True)
        best = 0.0
        ck = PERSIST / "ckpts" / f"video_effb0_v18plus_seed{sd}.pt"
        for ep in range(CFG.V_EPOCHS):
            model.train(); tot = 0.0; t0 = time.time()
            for x, y, _ in dl:
                yv = y.float().to(DEVICE)*0.95 + 0.025
                loss = F.binary_cross_entropy_with_logits(model(x.to(DEVICE)).squeeze(-1), yv)
                opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
            sched.step()
            r, _ = veval(model, FDS(M_VAL), f"val sd{sd} ep{ep+1}")
            print(f"  [sd{sd} ep{ep+1}] loss={tot/len(dl):.4f} ({time.time()-t0:.0f}s)")
            # BUGFIX (was r["clip_auc"] → KeyError at end of epoch 1):
            if r["video_auc"] > best: best = r["video_auc"]; torch.save(model.state_dict(), ck)
        model.load_state_dict(torch.load(ck, map_location=DEVICE))
        rin, _   = veval(model, FDS(M_VAL),   f"IN sd{sd}")
        rcr, exc = veval(model, FDS(M_CROSS), f"CROSS sd{sd}")
        out[sd] = {"in_domain": rin, "cross": rcr}
        if sd == CFG.V_SEEDS[0]:
            np.save(PERSIST/"results"/"cdf_scores_effb0.npy", exc["scores"])
            np.save(PERSIST/"results"/"cdf_labels_effb0.npy", exc["labels"])
            json.dump(exc["vids"], open(PERSIST/"results"/"cdf_vids_effb0.json", "w"))
            print("  [export] per-video CDF scores → cdf_scores_effb0.npy (paired test in video_probe)")
        h = hashlib.sha256(open(ck, "rb").read()).hexdigest()
        out[sd]["sha256"] = h
        print(f"  ckpt {ck.name} sha256={h[:16]}…  (persisted locally — download the zip!)")
    BASELINES_FFPP_TO_CDF = {"Xception (c23, legacy)": 65.3, "EfficientNet-B4 (legacy)": 64.3,
                             "RECCE (legacy)": 68.7, "F3-Net (legacy)": 65.1,
                             "Two-Branch (legacy)": 73.4, "SPSL (legacy)": 76.9,
                             "SBI (CVPR22)": 93.2, "LAA-Net (CVPR24)": 95.4,
                             "RAE (ECCV24)": 95.5, "Effort (ICML25)": 95.6,
                             "ForAda (CVPR25)": 95.7, "LNCLIP-DF (2025)": 96.5}
    out["_protocol"] = {"train": "FF++ (c23, all 4 manips)",
                        "eval": "Celeb-DF-v2 " + ("OFFICIAL test list" if OFFICIAL_CDF else "RANDOM SAMPLE (NON-COMPARABLE)"),
                        "official_cdf_test_list": bool(OFFICIAL_CDF),
                        "identity_grouped_split": True,
                        "video_score": "mean of frame sigmoid scores",
                        "role_in_paper": "end-to-end fine-tuned CONTROL for the frozen-probe claim",
                        "published_anchors_auc_pct": BASELINES_FFPP_TO_CDF}
    json.dump(out, open(PERSIST / "results" / "video_train_results.json", "w"), indent=2)

    print("=" * 66, "\nVIDEO VERDICT (FF++ c23 → Celeb-DF-v2, EffB0 end-to-end CONTROL)")
    print("  protocol:", out["_protocol"]["eval"], "| identity-grouped split: yes")
    print("  published anchors (video AUC%):",
          "  ".join(f"{k}={v}" for k, v in BASELINES_FFPP_TO_CDF.items()))
    for sd in CFG.V_SEEDS:
        c = out[sd]["cross"]
        print(f"  seed {sd}: cross video-AUC {c['video_auc']*100:.1f}% "
              f"CI [{c['video_auc_ci'][0]*100:.1f}, {c['video_auc_ci'][1]*100:.1f}]  "
              f"EER {c['video_eer_pct']:.1f}%  n={c['n_videos']}")
    tex = PERSIST / "reports" / "table_row.tex"
    existing = set(open(tex).read().splitlines()) if tex.exists() else set()
    new_rows = []
    for sd in CFG.V_SEEDS:
        c = out[sd]["cross"]
        row = (f"FF++c23→CDFv2 & EffB0 seed{sd} & {c['video_auc']*100:.1f} & "
               f"[{c['video_auc_ci'][0]*100:.1f},{c['video_auc_ci'][1]*100:.1f}] & "
               f"EER {c['video_eer_pct']:.1f} & n={c['n_videos']} \\\\")
        if row not in existing: new_rows.append(row)
    if new_rows:
        with open(tex, "a") as f:
            f.write("% video_train VERDICT\n" + "\n".join(new_rows) + "\n")
        print(f"  latex: {len(new_rows)} new row(s) →", tex)
    else:
        print("  latex: rows already present (identical run) — nothing appended")
    print("  checkpoints + results persisted →", PERSIST)


# ============================================================================
# STAGE D: video_probe — frozen CLIP ViT-L/14 layer-wise probes
#          FF++ c23 (OFFICIAL Rossler splits) → Celeb-DF-v2 (OFFICIAL test)
# ============================================================================

if STAGE == "video_probe":
    import cv2, urllib.request
    from facenet_pytorch import MTCNN
    from transformers import CLIPVisionModel

    class VP:
        CLIP_ID      = "openai/clip-vit-large-patch14"
        TRAIN_FRAMES = 8
        EVAL_FRAMES  = 32
        IMG, MARGIN  = 224, 0.30
        BATCH        = 32
        SEEDS        = [0, 1, 2]
        POOL         = "mean"
        MIN_CROPS    = 3
        CAP_TR = CAP_TE = CAP_CDF = None
    if DEBUG:
        VP.TRAIN_FRAMES, VP.EVAL_FRAMES, VP.SEEDS = 3, 4, [0]
        VP.CAP_TR, VP.CAP_TE, VP.CAP_CDF = 40, 20, 20

    CLIP_M = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    CLIP_S = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    SPLIT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{}.json"
    def ffpp_split(name):
        f = DATA / f"ffpp_split_{name}.json"
        if not f.exists():
            print(f"[splits] fetching official FF++ {name}.json")
            urllib.request.urlretrieve(SPLIT_URL.format(name), f)
        pairs = json.load(open(f))
        ids   = {x for p in pairs for x in p}
        fakes = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
        return ids, fakes

    tr_ids, tr_fakes = ffpp_split("train")
    te_ids, te_fakes = ffpp_split("test")

    real_all = sorted(glob.glob(CFG.TRAIN_REAL, recursive=True))
    fake_all = sorted(glob.glob(CFG.TRAIN_FAKE, recursive=True))
    assert real_all and fake_all, "FF++ globs empty — enable ffpp_c23 in DATASETS / verify slug"

    n_r = len({Path(v).stem for v in real_all}); n_f = len(fake_all)
    print(f"[mirror-check] FF++ originals={n_r}/1000  manipulated videos={n_f}/4000")
    MIRROR_OK = (n_r >= 990 and n_f >= 3960)
    if not MIRROR_OK:
        print("  ⚠️ mirror INCOMPLETE — comparability to published anchors is at risk;")
        print("     counts are recorded in _protocol and must be disclosed in the paper.")

    def pick(vs, keep): return [v for v in vs if Path(v).stem in keep]
    ENT_TR = [(v, 0) for v in pick(real_all, tr_ids)] + [(v, 1) for v in pick(fake_all, tr_fakes)]
    ENT_TE = [(v, 0) for v in pick(real_all, te_ids)] + [(v, 1) for v in pick(fake_all, te_fakes)]
    print(f"[splits] official train: {len(ENT_TR)} vids | official test: {len(ENT_TE)} vids")

    cdf_root  = DATA / "celebdf-v2"
    test_list = next(iter(cdf_root.rglob("List_of_testing_videos.txt")), None)
    assert test_list, ("Celeb-DF List_of_testing_videos.txt NOT found — enable celebdf_v2; "
                       "video_probe refuses random sampling (non-comparable).")
    cdf_idx = {}
    for p in cdf_root.rglob("*.mp4"):
        cdf_idx[p.stem] = p
        cdf_idx[f"{p.parent.name}/{p.name}"] = p
    ENT_CDF, mismatched = [], 0
    for ln in open(test_list, "r", errors="ignore"):
        parts = ln.strip().split()
        if len(parts) < 2: continue
        flag, rel = parts[0], parts[-1]
        p = cdf_idx.get(rel) or cdf_idx.get(Path(rel).stem)
        if p is None: continue
        lab_path = 1 if "synthesis" in str(p).lower() else 0
        lab_flag = 1 - int(flag) if flag in ("0", "1") else lab_path
        if lab_path != lab_flag: mismatched += 1
        ENT_CDF.append((str(p), lab_path))
    print(f"[protocol] Celeb-DF OFFICIAL test list resolved: {len(ENT_CDF)}/518 videos "
          f"({mismatched} label flag/path mismatches — should be 0)")
    assert len(ENT_CDF) >= (15 if DEBUG else 500), "official CDF test videos missing from mirror"

    mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=DEVICE)
    CROPS = Path("/content/data/crops_probe" if IN_COLAB else "./data/crops_probe")
    CROPS.mkdir(parents=True, exist_ok=True)

    def crop_video(path, out_dir, vid, n_frames):
        if len(list(out_dir.glob(f"{vid}__*.jpg"))) >= max(VP.MIN_CROPS, n_frames // 2):
            return True
        cap = cv2.VideoCapture(path); tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot <= 0: cap.release(); return False
        ok_any = False
        for k, i in enumerate(np.linspace(0, tot - 1, n_frames).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, fr = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            box, prob = mtcnn.detect(rgb)
            if box is None or prob is None or prob[0] is None or prob[0] < 0.9: continue
            x0, y0, x1, y1 = box[0]; w, h = x1 - x0, y1 - y0
            x0 = max(int(x0 - VP.MARGIN * w), 0); y0 = max(int(y0 - VP.MARGIN * h), 0)
            x1 = min(int(x1 + VP.MARGIN * w), rgb.shape[1]); y1 = min(int(y1 + VP.MARGIN * h), rgb.shape[0])
            crop = cv2.resize(rgb[y0:y1, x0:x1], (VP.IMG, VP.IMG))
            cv2.imwrite(str(out_dir / f"{vid}__{k}.jpg"),
                        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok_any = True
        cap.release(); return ok_any

    def build_manifest(name, entries, n_frames, cap_vids=None):
        entries = list(entries); random.Random(0).shuffle(entries)
        if cap_vids: entries = entries[:cap_vids]
        od = CROPS / f"{name}_f{n_frames}"; od.mkdir(parents=True, exist_ok=True)
        man, dropped = [], 0
        for j, (v, lab) in enumerate(entries):
            vid = Path(v).stem
            if crop_video(v, od, vid, n_frames):
                frames = sorted(str(f) for f in od.glob(f"{vid}__*.jpg"))
                if len(frames) >= VP.MIN_CROPS:
                    man.append({"vid": vid, "label": int(lab), "frames": frames})
                else: dropped += 1
            else: dropped += 1
            if (j + 1) % 100 == 0: print(f"  [crop:{name}] {j+1}/{len(entries)}")
        print(f"[{name}] {len(man)} videos kept ({dropped} dropped: <{VP.MIN_CROPS} valid crops)")
        return man, dropped

    M_TR,  drop_tr  = build_manifest("ffpp_tr", ENT_TR,  VP.TRAIN_FRAMES, VP.CAP_TR)
    M_TE,  drop_te  = build_manifest("ffpp_te", ENT_TE,  VP.EVAL_FRAMES,  VP.CAP_TE)
    M_CDF, drop_cdf = build_manifest("cdf",     ENT_CDF, VP.EVAL_FRAMES,  VP.CAP_CDF)

    for nm, m in (("vp_ffpp_tr", M_TR), ("vp_ffpp_te", M_TE), ("vp_cdf", M_CDF)):
        json.dump({"fingerprint": hashlib.sha256("\n".join(x["vid"] for x in m).encode()).hexdigest()[:10],
                   "n_videos": len(m), "vids": [x["vid"] for x in m],
                   "labels": [x["label"] for x in m]},
                  open(PERSIST / "results" / f"manifest_{nm}.json", "w"))
    print("[provenance] manifest_vp_*.json written")

    clip_model = CLIPVisionModel.from_pretrained(VP.CLIP_ID).to(DEVICE).eval().half()
    N_LAYERS = clip_model.config.num_hidden_layers + 1

    def frame_list(man): return [(f, m["label"], m["vid"]) for m in man for f in m["frames"]]
    def vp_fp(man):
        allf = "\n".join(f for m in man for f in m["frames"])
        return f"n{sum(len(m['frames']) for m in man)}_{hashlib.sha256(allf.encode()).hexdigest()[:10]}"

    @torch.no_grad()
    def extract_clip(man, split):
        fp = vp_fp(man)
        fx = PERSIST / "features" / f"clip__{split}__{fp}.npy"
        fy = PERSIST / "features" / f"clip__{split}__{fp}__y.npy"
        fv = PERSIST / "features" / f"clip__{split}__{fp}__vid.json"
        if fx.exists(): print(f"[cache] {fx.name}"); return fx, fy, fv
        items = frame_list(man); X, Y, V, B = [], [], [], []
        t0 = time.time()
        def flush():
            if not B: return
            x = torch.tensor(np.stack([b[0] for b in B]), dtype=torch.float16, device=DEVICE)
            hs = clip_model(pixel_values=x, output_hidden_states=True).hidden_states
            pooled = (torch.stack([h[:, 1:].mean(1) for h in hs], 1) if VP.POOL == "mean"
                      else torch.stack([h[:, 0] for h in hs], 1))
            X.append(pooled.cpu().numpy().astype(np.float16))
            Y.extend(b[1] for b in B); V.extend(b[2] for b in B); B.clear()
        for i, (f, l, v) in enumerate(items):
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            B.append((((img - CLIP_M) / CLIP_S).transpose(2, 0, 1), l, v))
            if len(B) == VP.BATCH: flush()
            if (i + 1) % 640 == 0:
                eta = (time.time() - t0) / (i + 1) * (len(items) - i - 1)
                print(f"  [clip:{split}] {i+1}/{len(items)}  eta {eta/60:.1f}m")
        flush()
        np.save(fx, np.concatenate(X)); np.save(fy, np.array(Y, np.int8))
        json.dump(V, open(fv, "w"))
        print(f"[clip:{split}] wrote {fx.name}  ({Path(fx).stat().st_size/1e6:.0f} MB)")
        return fx, fy, fv

    FE = {sp: extract_clip(m, sp) for sp, m in
          [("ffpp_tr", M_TR), ("ffpp_te", M_TE), ("cdf", M_CDF)]}
    del clip_model; torch.cuda.empty_cache()

    def load_feats(sp):
        fx, fy, fv = FE[sp]
        return np.load(fx, mmap_mode="r"), np.load(fy), json.load(open(fv))
    Xtr, ytr, vtr = load_feats("ffpp_tr")
    Xte, yin, vte = load_feats("ffpp_te")
    Xcd, ycd, vcd = load_feats("cdf")

    def video_agg(scores, vids, labels):
        agg = {}
        for s, v, y in zip(scores, vids, labels):
            agg.setdefault(v, {"s": [], "y": int(y)})["s"].append(float(s))
        cy = np.array([a["y"] for a in agg.values()])
        cs = np.array([float(np.mean(a["s"])) for a in agg.values()])
        return cy, cs, list(agg.keys())

    probe, bank = {}, {}
    for L in range(N_LAYERS):
        XL_tr = np.asarray(Xtr[:, L], np.float32)
        XL_te = np.asarray(Xte[:, L], np.float32)
        XL_cd = np.asarray(Xcd[:, L], np.float32)
        probe[L] = {}
        for sd in VP.SEEDS:
            set_seed(sd)
            sc  = StandardScaler().fit(XL_tr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=sd).fit(sc.transform(XL_tr), ytr)
            iy, is_, _    = video_agg(clf.predict_proba(sc.transform(XL_te))[:, 1], vte, yin)
            cy, cs, cvids = video_agg(clf.predict_proba(sc.transform(XL_cd))[:, 1], vcd, ycd)
            probe[L][sd] = {"in": float(roc_auc_score(iy, is_)),
                            "cross": float(roc_auc_score(cy, cs))}
            bank[(L, sd)] = (cy, cs, cvids)
        mu_in = np.mean([probe[L][s]["in"]    for s in VP.SEEDS])
        mu_cr = np.mean([probe[L][s]["cross"] for s in VP.SEEDS])
        print(f"[VP] L{L:02d}  in(video-AUC)={mu_in:.3f}  cross={mu_cr:.3f}")
        del XL_tr, XL_te, XL_cd
    json.dump(probe, open(PERSIST / "results" / "video_probe_results.json", "w"),
              indent=2, default=str)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    Ls = sorted(probe)
    for split, c in [("in", "tab:gray"), ("cross", "tab:red")]:
        mu  = [np.mean([probe[L][s][split] for s in VP.SEEDS]) for L in Ls]
        sd_ = [np.std ([probe[L][s][split] for s in VP.SEEDS]) for L in Ls]
        ax.plot(Ls, mu, "-o", ms=3, color=c, label=split)
        ax.fill_between(Ls, np.array(mu) - sd_, np.array(mu) + sd_, color=c, alpha=0.2)
    ax.axhline(0.5, ls="--", c="k", lw=0.8); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel("CLIP ViT-L/14 block"); ax.set_ylabel("video-level AUC")
    ax.set_title("Fig.3 — layer-wise transfer, FF++c23 → CDFv2 (frozen CLIP)")
    fig.savefig(PERSIST / "figures" / "fig3_video_layerwise.png", dpi=200); plt.close(fig)

    cross_mu = {L: np.mean([probe[L][s]["cross"] for s in VP.SEEDS]) for L in Ls}
    bestL = max(cross_mu, key=cross_mu.get)
    band  = sorted([L for L in Ls if cross_mu[L] >= cross_mu[bestL] - 0.01])
    print(f"best layer L{bestL:02d} | ±0.01-AUC band: L{band[0]:02d}–L{band[-1]:02d}")

    # ---- head export for av_localize zero-shot (refit best layer, seed 0) ----
    set_seed(VP.SEEDS[0])
    XLb = np.asarray(Xtr[:, bestL], np.float32)
    sc_exp  = StandardScaler().fit(XLb)
    clf_exp = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=VP.SEEDS[0]).fit(sc_exp.transform(XLb), ytr)
    save_probe_head(PERSIST/"ckpts"/"probe_head_video_clip.npz", sc_exp, clf_exp,
                    {"backbone": VP.CLIP_ID, "layer": int(bestL), "pool": VP.POOL,
                     "trained_on": "ffpp_c23_official_train"})
    del XLb

    cy0, cs0, cvids0 = bank[(bestL, VP.SEEDS[0])]
    v_eer, thr = eer_of(cy0, cs0)
    headline = {"backbone": VP.CLIP_ID, "pool": VP.POOL, "layer": int(bestL),
                "band": [int(band[0]), int(band[-1])],
                "n_videos": int(len(cy0)),
                "video_auc": float(roc_auc_score(cy0, cs0)),
                "video_auc_ci": bootstrap_auc_ci(cy0, cs0),
                "video_eer_pct": v_eer * 100,
                "video_eer_ci_pct": [x * 100 for x in bootstrap_eer_ci(cy0, cs0)],
                "eer_threshold": thr}

    cyF, csF, _ = bank[(Ls[-1], VP.SEEDS[0])]
    sig_last = paired_bootstrap_delta(cy0, csF, cs0)
    json.dump(sig_last, open(PERSIST / "results" / "significance_clip_best_vs_last.json", "w"), indent=2)

    sig_effb0 = None
    eff_s = PERSIST / "results" / "cdf_scores_effb0.npy"
    eff_v = PERSIST / "results" / "cdf_vids_effb0.json"
    if eff_s.exists() and eff_v.exists():
        es = np.load(eff_s); ev = json.load(open(eff_v))
        emap = dict(zip(ev, es))
        keep = [i for i, v in enumerate(cvids0) if v in emap]
        if len(keep) > 100:
            sig_effb0 = paired_bootstrap_delta(
                cy0[keep], np.array([emap[cvids0[i]] for i in keep]), cs0[keep])
            json.dump(sig_effb0, open(PERSIST / "results" / "significance_clip_vs_effb0.json", "w"), indent=2)

    np.save(PERSIST / "results" / f"cdf_scores_clip_L{bestL:02d}.npy", cs0)
    np.save(PERSIST / "results" / "cdf_labels_vp.npy", cy0)
    json.dump(cvids0, open(PERSIST / "results" / "cdf_vids_vp.json", "w"))

    SOTA_FFPP_TO_CDF = {"SBI (CVPR22)": 93.2, "LAA-Net (CVPR24)": 95.4,
                        "RAE (ECCV24)": 95.5, "Effort (ICML25)": 95.6,
                        "ForAda (CVPR25)": 95.7, "LNCLIP-DF (2025)": 96.5,
                        "Xception (legacy)": 65.3, "EfficientNet-B4 (legacy)": 64.3}
    headline["_protocol"] = {
        "train": "FF++ c23, OFFICIAL Rossler train split (720 pairs, all 4 manips)",
        "eval_in": "FF++ c23 OFFICIAL test split, video-level",
        "eval_cross": "Celeb-DF-v2 OFFICIAL test list (518 videos)",
        "frames": {"train": VP.TRAIN_FRAMES, "eval": VP.EVAL_FRAMES},
        "video_score": "mean of frame probe probabilities",
        "mirror_check": {"originals": n_r, "manipulated": n_f, "complete": bool(MIRROR_OK)},
        "dropped_videos": {"ffpp_tr": drop_tr, "ffpp_te": drop_te, "cdf": drop_cdf,
                           "min_crops": VP.MIN_CROPS},
        "published_anchors_auc_pct": SOTA_FFPP_TO_CDF}
    json.dump(headline, open(PERSIST / "results" / "headline_video_probe.json", "w"), indent=2)

    print("=" * 66, "\nVIDEO_PROBE VERDICT (frozen CLIP, FF++c23 → CDFv2)")
    print(f"  protocol: OFFICIAL FF++ splits + OFFICIAL CDF test list | "
          f"n={headline['n_videos']} videos | frames eval={VP.EVAL_FRAMES}")
    print("  2025 anchors (video AUC%):",
          "  ".join(f"{k}={v}" for k, v in SOTA_FFPP_TO_CDF.items() if "legacy" not in k))
    print(f"  CLIP-L14 frozen  L{bestL:02d} (band L{band[0]:02d}–L{band[-1]:02d})  "
          f"video-AUC {headline['video_auc']:.3f} "
          f"[{headline['video_auc_ci'][0]:.3f}, {headline['video_auc_ci'][1]:.3f}]  "
          f"EER {headline['video_eer_pct']:.2f}% "
          f"[{headline['video_eer_ci_pct'][0]:.2f}, {headline['video_eer_ci_pct'][1]:.2f}]")
    print(f"  paired ΔAUC (best − last block) = {sig_last['delta_auc_mean']:+.3f} "
          f"CI [{sig_last['delta_ci'][0]:+.3f}, {sig_last['delta_ci'][1]:+.3f}]  "
          f"p={sig_last['p_two_sided']:.4f}")
    if sig_effb0 is not None:
        print(f"  paired ΔAUC (CLIP-probe − EffB0 e2e) = {sig_effb0['delta_auc_mean']:+.3f} "
              f"p={sig_effb0['p_two_sided']:.4f}")
    else:
        print("  (EffB0 control comparison skipped — run STAGE='video_train' once to export its CDF scores)")

    tex = PERSIST / "reports" / "table_row.tex"
    existing = set(open(tex).read().splitlines()) if tex.exists() else set()
    row = (f"FF++c23→CDFv2 & CLIP-L14 frozen L{bestL} (band {band[0]}–{band[-1]}) & "
           f"{headline['video_auc']*100:.1f} & "
           f"[{headline['video_auc_ci'][0]*100:.1f},{headline['video_auc_ci'][1]*100:.1f}] & "
           f"EER {headline['video_eer_pct']:.1f} & n={headline['n_videos']} \\\\")
    if row not in existing:
        with open(tex, "a") as f:
            f.write("% video_probe VERDICT\n" + row + "\n")
        print("  latex: 1 new row →", tex)
    else:
        print("  latex: row already present (identical run) — nothing appended")
    print("  all artifacts →", PERSIST)


# ============================================================================
# ⭐ STAGE D2 (V19): cross_gen_df40 — FF++c23 → DF40 (40 generators)
# ----------------------------------------------------------------------------
# WHY THIS STAGE MATTERS FOR THE PAPER:
#   DF40 (Yan et al. NeurIPS'24) is the 2026 gold-standard for cross-generation
#   evaluation.  40 distinct generators grouped into 4 families:
#     • Face-Swapping (FS):        10 methods incl. SimSwap, InSwap, BlendFace
#     • Face-Reenactment (FR):     13 methods incl. FOMM, HeyGen
#     • Entire-Face-Synthesis (EFS): 12 methods incl. StyleGAN3, DiT, SD-based
#     • Face-Editing (FE):          5 methods
#
#   The paper claim we can make: a frozen CLIP ViT-L/14 probe trained on FF++c23
#   (published anchors: SBI 93.2, Effort 95.6, LNCLIP-DF 96.5 on CDFv2) DOES /
#   DOES NOT generalize to the DF40 spectrum.  If the mean AUC on DF40 is above
#   ~85%, this is a defensible generalization result.  If it drops below 70%, we
#   have honest evidence of the "training-set effect" DF40 was designed to expose.
#
# COMPUTE FOOTPRINT:
#   HuggingFace repo pujanpaudel/deepfake_face_classification is ~3.4 GB of face
#   crops (already MTCNN-processed by the DF40 authors).  32,134 images.  On T4:
#     CLIP-L14 forward @ batch 32:  ~7.5 min for the full set
#     LR head fit (uses your video_probe head, already exported):  <10 s
#   Total wall clock: ~10 min end-to-end after DL.
#
# WHAT WE REPORT:
#   Per-method AUC + EER (a 40-row table for the appendix)
#   Per-family AUC (4-row table for the main text)
#   Bootstrap CIs on the overall AUC
#   Comparison to CLIP head's CDFv2 result to make the "generalization drop"
#   claim honest (paper section: "Beyond CDFv2 — cross-generation transfer").
#
# HONEST CAVEAT:
#   pujanpaudel/deepfake_face_classification bundles all 40 methods without
#   per-method labels in the file paths.  We probe the filesystem structure at
#   run-time, log what we find, and if per-method labels are missing we still
#   report the pooled AUC (this is what most DF40 papers do).
# ============================================================================

if STAGE == "cross_gen_df40":
    import cv2
    from transformers import CLIPVisionModel

    # ---- Sanity: video_probe must have exported its head --------------------
    head_path = PERSIST / "ckpts" / "probe_head_video_clip.npz"
    hjson = PERSIST / "results" / "headline_video_probe.json"
    assert head_path.exists(), (
        "[cross_gen_df40] REFUSING to run — no exported CLIP head at "
        f"{head_path}.  Run STAGE='video_probe' first (this stage piggybacks "
        "on its exported head).")
    assert hjson.exists(), (
        "[cross_gen_df40] no headline_video_probe.json — run video_probe.")

    head = np.load(head_path, allow_pickle=True)
    head_meta = json.loads(str(head["meta"]))
    BEST_LAYER = int(head_meta["layer"])
    print(f"[cross_gen_df40] loaded video_probe head  backbone={head_meta['backbone']}  "
          f"L{BEST_LAYER}  pool={head_meta.get('pool','mean')}  "
          f"trained_on={head_meta['trained_on']}")

    CLIP_M = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    CLIP_S = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    class DF40:
        IMG = 224
        BATCH = 32
        # per-method sub-sampling caps to fit in a T4 session
        MAX_PER_CLASS = 800 if not DEBUG else 40
        MAX_PER_METHOD = 200 if not DEBUG else 20
        BOOT = 1000 if not DEBUG else 200

    # ---- Enumerate DF40 test images -----------------------------------------
    df40_root = DATA / "df40-test"
    assert df40_root.exists() and any(df40_root.rglob("*")), (
        "[cross_gen_df40] DF40 test data not found — enable 'df40_test' in DATASETS.")

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    all_imgs = [p for p in df40_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    print(f"[cross_gen_df40] found {len(all_imgs)} images under {df40_root}")

    # ---- Discover labels + method annotations from the filesystem -----------
    # The pujanpaudel/deepfake_face_classification repo uses a two-level folder
    # convention: <root>/real/... and <root>/fake/... .  We derive method
    # by looking one directory below "fake" when possible.
    def _classify(p):
        parts = [x.lower() for x in p.parts]
        # label
        is_real = any(t in parts for t in ("real", "genuine", "bonafide", "0"))
        is_fake = any(t in parts for t in ("fake", "deepfake", "synth", "synthetic", "1"))
        if is_real == is_fake:
            # fall back to parent dir string search
            joined = str(p).lower()
            if "real" in joined and "fake" not in joined:
                is_real, is_fake = True, False
            elif "fake" in joined and "real" not in joined:
                is_real, is_fake = False, True
            else:
                return None  # ambiguous, skip
        label = 1 if is_fake else 0
        # method (best-effort; only for fakes)
        method = None
        if label == 1:
            # look for a known DF40 method name in the path
            KNOWN = ("fsgan", "faceswap", "simswap", "inswap", "blendface", "uniface",
                     "mobileswap", "e4s", "facedancer", "fomm", "facevid2vid", "wav2lip",
                     "mra", "hyperreenact", "onlysample", "styleface", "sadtalker",
                     "styleheat", "diffface", "dcface", "diffautoenc", "diffswap",
                     "stylegan2", "stylegan3", "stylegan_xl", "vqgan", "sd15", "sd21",
                     "sdxl", "midjourney", "dalle3", "collaborativediffusion", "e4e",
                     "starganv2", "ddim", "pixart", "heygen", "deepfacelab", "codeformer")
            for m in KNOWN:
                if m in "/".join(parts):
                    method = m; break
            if method is None:
                # take the folder right under "fake" as a fallback method name
                try:
                    fi = parts.index("fake")
                    method = parts[fi+1] if fi+1 < len(parts) else "unknown"
                except ValueError:
                    method = "unknown"
        return label, method

    labeled = []
    for p in all_imgs:
        r = _classify(p)
        if r is not None:
            labeled.append((str(p), r[0], r[1] or "real"))
    n_real = sum(1 for _, y, _ in labeled if y == 0)
    n_fake = sum(1 for _, y, _ in labeled if y == 1)
    print(f"[cross_gen_df40] labeled: {n_real} real / {n_fake} fake")
    assert n_real > 100 and n_fake > 100, (
        "[cross_gen_df40] labeling failed — inspect DF40 folder layout; "
        "the current heuristic looks for 'real'/'fake' anywhere in the path.")

    # ---- Balanced subsample (V20: method-stratified, keeps ALL methods) -----
    # The old code did fakes_sub[:MAX_PER_CLASS] which, with 40 methods capped at
    # MAX_PER_METHOD each, silently dropped most methods.  We now keep every
    # detected method (capped per-method) and only cap the TOTAL, scaling every
    # method down proportionally if a budget is exceeded — so per-family/per-method
    # tables retain coverage instead of collapsing to a handful of methods.
    rng = random.Random(1234)
    reals = [t for t in labeled if t[1] == 0]; rng.shuffle(reals)
    fakes = [t for t in labeled if t[1] == 1]; rng.shuffle(fakes)
    reals = reals[:DF40.MAX_PER_CLASS]
    by_method = {}
    for t in fakes:
        by_method.setdefault(t[2], []).append(t)

    # V20: warn loudly if the DF40 mirror exposes no per-method labels — the
    # per-family/per-method claim depends on it.
    n_methods_total = len(by_method)
    n_unknown = len(by_method.get("unknown", []))
    if n_methods_total <= 1 or (n_unknown > 0 and n_unknown >= 0.5 * len(fakes)):
        print("  ⚠️ V20 WARNING: most fakes are labelled 'unknown' — the DF40 mirror has no")
        print("     per-method folders, so per-family/per-method AUC will be meaningless.")
        print("     => Use the OFFICIAL DF40 release (YZY-stack/DF40) for the 40-generator claim,")
        print("        or downgrade the claim to binary real/fake generalization.")
    print(f"[cross_gen_df40] method histogram ({n_methods_total} methods, "
          f"{n_unknown} unknown / {len(fakes)} fakes):")
    for m, lst in sorted(by_method.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"    {m:25s} {len(lst)}")
    if n_methods_total > 12: print(f"    ... +{n_methods_total-12} more methods")

    fakes_sub = []
    for m, lst in by_method.items():
        fakes_sub.extend(lst[:DF40.MAX_PER_METHOD])
    # V20: if a total budget is desired, scale down proportionally rather than slicing.
    if len(fakes_sub) > DF40.MAX_PER_CLASS and DF40.MAX_PER_CLASS > 0:
        keep_frac = DF40.MAX_PER_CLASS / len(fakes_sub)
        fakes_sub = [t for t in fakes_sub if rng.random() < keep_frac]
    eval_set = reals + fakes_sub
    rng.shuffle(eval_set)
    print(f"[cross_gen_df40] eval subsample: {len(eval_set)} images  "
          f"({sum(1 for _,y,_ in eval_set if y==1)} fake across "
          f"{len(set(m for _,y,m in eval_set if y==1))} methods)")

    # ---- CLIP feature extraction --------------------------------------------
    def _load_img(p):
        img = cv2.imread(p)
        if img is None: return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != DF40.IMG or img.shape[1] != DF40.IMG:
            img = cv2.resize(img, (DF40.IMG, DF40.IMG))
        return (((img.astype(np.float32)/255. - CLIP_M) / CLIP_S)
                .transpose(2, 0, 1))

    clip_model = CLIPVisionModel.from_pretrained(head_meta["backbone"]).to(DEVICE).eval().half()

    @torch.no_grad()
    def extract_features(items):
        fp = hashlib.sha256("\n".join(p for p, _, _ in items).encode()).hexdigest()[:10]
        fx = PERSIST/"features"/f"df40__L{BEST_LAYER}__n{len(items)}_{fp}.npy"
        fy = fx.with_name(fx.stem + "__y.npy")
        fm = fx.with_name(fx.stem + "__m.json")
        if fx.exists():
            print(f"[cache] {fx.name}")
            return np.load(fx), np.load(fy), json.load(open(fm))
        X, Y, M, B = [], [], [], []
        t0 = time.time()
        def flush():
            if not B: return
            x = torch.tensor(np.stack([b[0] for b in B]), dtype=torch.float16, device=DEVICE)
            hs = clip_model(pixel_values=x, output_hidden_states=True).hidden_states[BEST_LAYER]
            # match video_probe pool convention (mean over non-CLS tokens)
            feats = hs[:, 1:].mean(1).cpu().numpy().astype(np.float32)
            X.append(feats)
            Y.extend(b[1] for b in B); M.extend(b[2] for b in B); B.clear()
        skipped = 0
        for i, (p, y, m) in enumerate(items):
            img = _load_img(p)
            if img is None:
                skipped += 1; continue
            B.append((img, y, m))
            if len(B) == DF40.BATCH: flush()
            if (i+1) % 640 == 0:
                eta = (time.time()-t0)/(i+1)*(len(items)-i-1)
                print(f"  [df40 clip] {i+1}/{len(items)}  eta {eta/60:.1f}m")
        flush()
        Xa = np.concatenate(X); Ya = np.array(Y, np.int8)
        np.save(fx, Xa); np.save(fy, Ya); json.dump(M, open(fm, "w"))
        print(f"[df40 clip] {len(Y)} feats  (skipped {skipped} unreadable)  → {fx.name}")
        return Xa, Ya, M

    Xdf, ydf, mdf = extract_features(eval_set)
    del clip_model; torch.cuda.empty_cache()

    # ---- Apply the video_probe head (uses your exported LR params) ----------
    s_df = apply_probe_head(head, Xdf)

    # ---- Overall metrics ----------------------------------------------------
    auc_all = float(roc_auc_score(ydf, s_df))
    auc_ci  = bootstrap_auc_ci(ydf, s_df, n=DF40.BOOT)
    eer_all, thr_all = eer_of(ydf, s_df)
    eer_ci  = bootstrap_eer_ci(ydf, s_df, n=DF40.BOOT)
    print(f"[df40 overall]  AUC {auc_all:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]  "
          f"EER {eer_all*100:5.2f}% [{eer_ci[0]*100:5.2f}, {eer_ci[1]*100:5.2f}]  n={len(ydf)}")

    # ---- Per-method breakdown -----------------------------------------------
    # For each fake method: build a binary problem (real vs this method) and
    # report AUC.  Small-n methods are flagged rather than dropped.
    per_method = {}
    real_idx = [i for i, y in enumerate(ydf) if y == 0]
    real_s = s_df[real_idx]
    for m in sorted(set(mdf)):
        if m == "real": continue
        idx_m = [i for i, mm in enumerate(mdf) if mm == m and ydf[i] == 1]
        if len(idx_m) < 20:
            per_method[m] = {"n_fake": len(idx_m), "note": "too few samples", "auc": None}
            continue
        s_m = s_df[idx_m]
        y_bin = np.concatenate([np.zeros(len(real_s)), np.ones(len(s_m))])
        s_bin = np.concatenate([real_s, s_m])
        auc_m = float(roc_auc_score(y_bin, s_bin))
        eer_m, _ = eer_of(y_bin, s_bin)
        per_method[m] = {"n_fake": len(idx_m), "auc": auc_m, "eer_pct": eer_m*100}

    # ---- Family aggregation --------------------------------------------------
    FAMILY = {
        "FS": {"fsgan","faceswap","simswap","inswap","blendface","uniface",
               "mobileswap","e4s","facedancer","deepfacelab","diffswap"},
        "FR": {"fomm","facevid2vid","wav2lip","mra","hyperreenact","onlysample",
               "styleface","sadtalker","styleheat","heygen"},
        "EFS":{"stylegan2","stylegan3","stylegan_xl","vqgan","sd15","sd21","sdxl",
               "midjourney","dalle3","collaborativediffusion","ddim","pixart",
               "diffface","dcface","diffautoenc"},
        "FE": {"e4e","starganv2","styleclip"}}   # V20: codeformer moved out of FR (it is a
               # face-restoration/enhancement tool, not reenactment); styleclip added to FE.
    def fam(m):
        for f, s in FAMILY.items():
            if m in s: return f
        return "OTHER"

    per_family = {}
    for f in list(FAMILY.keys()) + ["OTHER"]:
        methods_in_f = [m for m, r in per_method.items()
                        if fam(m) == f and r["auc"] is not None]
        if not methods_in_f: continue
        aucs = [per_method[m]["auc"] for m in methods_in_f]
        per_family[f] = {"n_methods": len(methods_in_f),
                         "mean_auc": float(np.mean(aucs)),
                         "std_auc":  float(np.std(aucs)),
                         "methods":  methods_in_f}

    headline = {
        "backbone": head_meta["backbone"], "layer": BEST_LAYER,
        "head_trained_on": head_meta["trained_on"],
        "eval_set": "DF40 test (via pujanpaudel/deepfake_face_classification)",
        "n_images": int(len(ydf)),
        "n_methods_detected": int(len(per_method)),
        "overall": {"auc": auc_all, "auc_ci": auc_ci,
                    "eer_pct": eer_all*100, "eer_ci_pct": [x*100 for x in eer_ci]},
        "per_family": per_family,
        "per_method": per_method,
        "_2026_anchors_on_this_axis": ("SBI CDFv2=93.2, Effort CDFv2=95.6, "
                                       "LNCLIP-DF CDFv2=96.5.  DF40 is HARDER "
                                       "than CDFv2 by design — expect drops."),
    }
    json.dump(headline, open(PERSIST/"results"/"headline_cross_gen_df40.json", "w"), indent=2)
    np.save(PERSIST/"results"/"df40_scores.npy", s_df)
    np.save(PERSIST/"results"/"df40_labels.npy", ydf)
    json.dump(mdf, open(PERSIST/"results"/"df40_methods.json", "w"))

    # ---- Verdict + LaTeX ----------------------------------------------------
    print("=" * 66, "\nCROSS_GEN_DF40 VERDICT (video_probe head → DF40 40-generator eval)")
    print(f"  overall  AUC {auc_all:.3f}  EER {eer_all*100:.2f}%  n={len(ydf)}")
    print("  per-family (mean AUC across methods in family):")
    for f, r in per_family.items():
        print(f"    {f:5s}  n_methods={r['n_methods']:2d}  mean AUC {r['mean_auc']:.3f} ± {r['std_auc']:.3f}")
    top = sorted([(r["auc"], m) for m, r in per_method.items() if r["auc"] is not None])
    print(f"  hardest 5 generators (lowest AUC):")
    for auc, m in top[:5]:  print(f"    {m:25s} AUC {auc:.3f}")
    print(f"  easiest 5 generators (highest AUC):")
    for auc, m in top[-5:]: print(f"    {m:25s} AUC {auc:.3f}")

    tex = PERSIST / "reports" / "table_row.tex"
    existing = set(open(tex).read().splitlines()) if tex.exists() else set()
    rows = []
    rows.append(f"FF++c23$\\to$DF40 (all) & CLIP-L14 frozen L{BEST_LAYER} & "
                f"{auc_all*100:.1f} & [{auc_ci[0]*100:.1f},{auc_ci[1]*100:.1f}] & "
                f"EER {eer_all*100:.1f} & n={len(ydf)} \\\\")
    for f, r in per_family.items():
        rows.append(f"FF++c23$\\to$DF40 ({f}) & CLIP-L14 frozen L{BEST_LAYER} & "
                    f"{r['mean_auc']*100:.1f} & ($\\pm${r['std_auc']*100:.1f}) & - & "
                    f"n\\_methods={r['n_methods']} \\\\")
    new_rows = [r for r in rows if r not in existing]
    if new_rows:
        with open(tex, "a") as f:
            f.write("% cross_gen_df40 VERDICT (V19)\n" + "\n".join(new_rows) + "\n")
        print(f"  latex: {len(new_rows)} new row(s) →", tex)
    print("  all artifacts →", PERSIST)


# ============================================================================
# STAGE E: av_localize — Temporal Forgery Localization + modality attribution
#          on LAV-DF, with sliding-window FROZEN probes (XLS-R + CLIP).
#
# THE PRODUCT SPEC THIS IMPLEMENTS: for each video, output
#   [{t_start, t_end, modality: "video"|"audio"|"both", confidence}]
# scored against LAV-DF's ground-truth fake_periods + modify_video/modify_audio.
#
# DESIGN DECISIONS (deliberate — do not "simplify" these away):
#  1) PROBE_SOURCE="lavdf" trains 1s-window probes ON LAV-DF train (fine
#     localization, self-contained). "zeroshot" reuses heads exported by
#     week1_audio / video_probe and AUTO-SWITCHES the audio window to 4s —
#     matching the training protocol (the v17 lesson: never score 1s windows
#     with a 4s-trained head). Coarser localization, stronger transfer claim.
#  2) LAV-DF fake segments average ~0.65s → training windows are labeled fake
#     only if ≥50% overlapped; boundary-ambiguous windows are EXCLUDED from
#     training but ALL windows are scored at eval (no cherry-picking).
#  3) Frames of audio-only-fake videos are visually genuine → they enter the
#     VIDEO probe's training set as REAL (hard negatives), and vice versa.
#     This is what makes per-modality attribution learnable.
#  4) Train sampling is quadrant-stratified (real / V-only / A-only / AV) so
#     the probes see all manipulation types; composition is printed + logged.
#  5) Segment metric is AP@IoU {0.5, 0.75, 0.95} (the LAV-DF/AV1M standard),
#     computed per modality track and on the union track. Detection threshold
#     is chosen on a held-out dev slice at window-level EER — never on test.
#  6) This is a BASELINE skeleton: frozen probes + smoothing + thresholding.
#     Its gap to BA-TFD+ (LAV-DF authors' boundary-aware model, arXiv:2305.01979)
#     and the AV-Deepfake1M++ frontier (top AP@0.95 = 34.27) is the paper-2 /
#     thesis motivation, not a failure. Pull exact BA-TFD+ table values from
#     the paper when writing — deliberately NOT hardcoded here from memory.
# ============================================================================

if STAGE == "av_localize":
    import cv2
    from facenet_pytorch import MTCNN
    from transformers import CLIPVisionModel, Wav2Vec2Model
    from scipy.ndimage import median_filter

    class AL:
        PROBE_SOURCE  = "lavdf"        # "lavdf" | "zeroshot"
        # audio
        A_BACKBONE    = "facebook/wav2vec2-xls-r-300m"
        A_LAYER       = 5              # PAPER1 winner (XLS-R L05); zeroshot reads it from the head
        A_WIN, A_HOP  = 1.0, 0.5       # seconds; zeroshot mode forces A_WIN=4.0 (protocol match)
        A_BATCH       = 16
        # video
        V_BACKBONE    = "openai/clip-vit-large-patch14"
        V_LAYER       = None           # None → use video_probe headline layer if present, else 17
        V_FPS_TRAIN, V_FPS_EVAL = 2.0, 4.0
        V_BATCH, IMG, MARGIN = 32, 224, 0.30
        # sampling / labeling / post-processing
        TRAIN_VIDS, DEV_VIDS, TEST_VIDS = 1200, 200, 600
        POS_OVERLAP   = 0.5            # window fake-fraction to count as positive (train)
        GRID          = 0.5            # common timeline resolution (s)
        SMOOTH_K      = 3              # median filter width on the grid
        MIN_SEG       = 0.5            # discard predicted segments shorter than this (s)
        MERGE_GAP     = 0.5            # merge predicted segments closer than this (s)
        IOUS          = (0.5, 0.75, 0.95)
    if DEBUG:
        AL.TRAIN_VIDS, AL.DEV_VIDS, AL.TEST_VIDS = 30, 10, 15
        AL.V_FPS_TRAIN, AL.V_FPS_EVAL = 1.0, 2.0

    CLIP_M = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    CLIP_S = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    # ---------------- metadata: layout-agnostic discovery --------------------
    LDF = DATA / "lav-df"
    meta_file = next(iter(sorted(LDF.rglob("metadata.json"))), None)
    assert meta_file, f"metadata.json not found under {LDF} — check the lavdf download/extraction"
    META = json.load(open(meta_file))
    if isinstance(META, dict): META = list(META.values())
    print(f"[lavdf] metadata entries: {len(META)} (from {meta_file.relative_to(LDF)})")
    samp = META[0]
    print(f"[lavdf] sample entry keys: {sorted(samp.keys())}")
    for req in ("file",):
        assert req in samp, f"metadata schema surprise — expected key '{req}'; inspect the print above"

    vid_idx = {}
    for p in LDF.rglob("*.mp4"):
        vid_idx[p.name] = p
        vid_idx[str(p.relative_to(LDF))] = p

    def _entry(e):
        f = e.get("file", "")
        p = vid_idx.get(f) or vid_idx.get(Path(f).name)
        if p is None: return None
        periods = [[float(a), float(b)] for a, b in (e.get("fake_periods") or [])]
        mv = bool(e.get("modify_video", False)); ma = bool(e.get("modify_audio", False))
        if not periods: mv = ma = False           # real video
        return {"vid": Path(f).stem, "path": str(p), "split": e.get("split", "train"),
                "periods": periods, "mod_v": mv, "mod_a": ma,
                "dur": float(e.get("duration", 0)) or None}

    ENTRIES = [x for x in (_entry(e) for e in META) if x is not None]
    print(f"[lavdf] resolved on disk: {len(ENTRIES)}/{len(META)}")
    assert ENTRIES, "no LAV-DF videos resolved — extraction incomplete?"

    def quadrant(e):
        return ("real" if not e["periods"] else
                "V"  if e["mod_v"] and not e["mod_a"] else
                "A"  if e["mod_a"] and not e["mod_v"] else "AV")

    def stratified(entries, n, seed):
        by_q = {}
        for e in entries: by_q.setdefault(quadrant(e), []).append(e)
        rng = random.Random(seed)
        for q in by_q: rng.shuffle(by_q[q])
        out, per_q = [], max(1, n // max(1, len(by_q)))
        for q, lst in by_q.items(): out += lst[:per_q]
        rng.shuffle(out); return out[:n]

    tr_all  = [e for e in ENTRIES if e["split"] == "train"]
    te_all  = [e for e in ENTRIES if e["split"] == "test"]
    if not te_all: te_all = [e for e in ENTRIES if e["split"] in ("dev", "val")]
    E_TRAIN = stratified(tr_all, AL.TRAIN_VIDS, 0)
    _train_set = {e["vid"] for e in E_TRAIN}
    E_DEV   = stratified([e for e in tr_all if e["vid"] not in _train_set], AL.DEV_VIDS, 1)
    E_TEST  = stratified(te_all, AL.TEST_VIDS, 2)
    for nm, ee in (("train", E_TRAIN), ("dev", E_DEV), ("test", E_TEST)):
        q = {k: sum(1 for e in ee if quadrant(e) == k) for k in ("real", "V", "A", "AV")}
        print(f"[lavdf:{nm}] {len(ee)} vids | quadrants {q}")
        json.dump({"fingerprint": hashlib.sha256("\n".join(e["vid"] for e in ee).encode()).hexdigest()[:10],
                   "n": len(ee), "quadrants": q, "vids": [e["vid"] for e in ee]},
                  open(PERSIST/"results"/f"manifest_lavdf_{nm}.json", "w"))

    # ---------------- zero-shot heads (optional) ------------------------------
    Z_AUD = PERSIST/"ckpts"/"probe_head_audio_xlsr.npz"
    Z_VID = PERSIST/"ckpts"/"probe_head_video_clip.npz"
    if AL.PROBE_SOURCE == "zeroshot":
        assert Z_AUD.exists() and Z_VID.exists(), \
            "zeroshot heads missing — re-run week1_audio and video_probe (they now export heads), or set PROBE_SOURCE='lavdf'"
        za, zv = np.load(Z_AUD, allow_pickle=True), np.load(Z_VID, allow_pickle=True)
        za_meta, zv_meta = json.loads(str(za["meta"])), json.loads(str(zv["meta"]))
        AL.A_LAYER = int(za_meta["layer"]); AL.A_WIN = float(za_meta.get("window_s", 4.0))
        AL.V_LAYER = int(zv_meta["layer"])
        print(f"[zeroshot] audio head L{AL.A_LAYER} win={AL.A_WIN}s ({za_meta['trained_on']}) | "
              f"video head L{AL.V_LAYER} ({zv_meta['trained_on']})")
        print(f"[zeroshot] NOTE: {AL.A_WIN}s audio windows → localization resolution is coarse by design")
    if AL.V_LAYER is None:
        hjson = PERSIST/"results"/"headline_video_probe.json"
        AL.V_LAYER = (json.load(open(hjson))["layer"] if hjson.exists() else 17)
        print(f"[cfg] video layer = L{AL.V_LAYER}" + ("" if hjson.exists() else " (default — no video_probe headline found)"))

    # ---------------- audio: ffmpeg → wav cache, windows, features -----------
    WAVS = Path("/content/data/lavdf_wav" if IN_COLAB else "./data/lavdf_wav"); WAVS.mkdir(parents=True, exist_ok=True)
    def wav_of(e):
        w = WAVS / (e["vid"] + ".wav")
        if not w.exists():
            subprocess.run(["ffmpeg", "-loglevel", "quiet", "-i", e["path"],
                            "-ar", str(CFG.SR), "-ac", "1", "-y", str(w)], check=False)
        return w if w.exists() and w.stat().st_size > 1000 else None

    def overlap_frac(a, b, periods):
        ov = sum(max(0.0, min(b, q) - max(a, p)) for p, q in periods)
        return ov / max(1e-6, b - a)

    def audio_windows(e, for_train):
        w = wav_of(e)
        if w is None: return []
        try: y, _ = sf.read(str(w), dtype="float32")
        except Exception: return []
        dur = len(y) / CFG.SR
        out, t = [], 0.0
        while t + AL.A_WIN <= dur + 1e-6:
            seg = y[int(t*CFG.SR):int((t+AL.A_WIN)*CFG.SR)]
            m = np.abs(seg).max(); seg = seg/m if m > 0 else seg
            frac = overlap_frac(t, t+AL.A_WIN, e["periods"]) if e["mod_a"] else 0.0
            lab = 1 if frac >= AL.POS_OVERLAP else (0 if frac == 0.0 else -1)  # -1 = ambiguous
            if not for_train or lab >= 0:
                out.append({"y": seg.astype(np.float32), "t": t, "lab": lab, "vid": e["vid"]})
            t += AL.A_HOP
        return out

    def cfg_fp(entries, tag):
        s = "\n".join(e["vid"] for e in entries) + f"|{tag}|{AL.A_WIN}|{AL.A_HOP}|{AL.A_LAYER}|{AL.V_LAYER}|{AL.V_FPS_TRAIN}|{AL.V_FPS_EVAL}"
        return f"n{len(entries)}_{hashlib.sha256(s.encode()).hexdigest()[:10]}"

    @torch.no_grad()
    def extract_audio(entries, split, for_train):
        fp = cfg_fp(entries, "aud")
        fz = PERSIST/"features"/f"lavdf_aud__{split}__{fp}.npz"
        if fz.exists(): print(f"[cache] {fz.name}"); return fz
        model = Wav2Vec2Model.from_pretrained(AL.A_BACKBONE).to(DEVICE).eval().half()
        X, T, L, V, B = [], [], [], [], []
        def flush():
            if not B: return
            x = torch.tensor(np.stack([b["y"] for b in B]), dtype=torch.float16, device=DEVICE)
            hs = model(x, output_hidden_states=True).hidden_states[AL.A_LAYER]
            X.append(hs.mean(1).cpu().numpy().astype(np.float16))
            T.extend(b["t"] for b in B); L.extend(b["lab"] for b in B); V.extend(b["vid"] for b in B)
            B.clear()
        t0 = time.time()
        for j, e in enumerate(entries):
            for wdw in audio_windows(e, for_train):
                B.append(wdw)
                if len(B) == AL.A_BATCH: flush()
            if (j+1) % 50 == 0:
                print(f"  [aud:{split}] {j+1}/{len(entries)} vids  ({(time.time()-t0)/60:.1f}m)")
        flush()
        np.savez(fz, X=np.concatenate(X) if X else np.zeros((0, 1), np.float16),
                 t=np.array(T, np.float32), lab=np.array(L, np.int8),
                 vid=np.array(V))
        del model; torch.cuda.empty_cache()
        print(f"[aud:{split}] {len(T)} windows → {fz.name}")
        return fz

    # ---------------- video: frames at fixed fps, MTCNN, CLIP features -------
    mtcnn = MTCNN(keep_all=False, select_largest=True, post_process=False, device=DEVICE)

    def video_frames(e, fps):
        cap = cv2.VideoCapture(e["path"])
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tot  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur  = tot / vfps if tot > 0 else (e["dur"] or 0)
        out = []
        for t in np.arange(0.0, max(0.0, dur - 1e-3), 1.0/fps)[:120]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*vfps)); ok, fr = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            box, prob = mtcnn.detect(rgb)
            if box is None or prob is None or prob[0] is None or prob[0] < 0.85: continue
            x0, y0, x1, y1 = box[0]; w, h = x1-x0, y1-y0
            x0 = max(int(x0-AL.MARGIN*w), 0); y0 = max(int(y0-AL.MARGIN*h), 0)
            x1 = min(int(x1+AL.MARGIN*w), rgb.shape[1]); y1 = min(int(y1+AL.MARGIN*h), rgb.shape[0])
            crop = cv2.resize(rgb[y0:y1, x0:x1], (AL.IMG, AL.IMG)).astype(np.float32)/255.
            inside = any(p <= t <= q for p, q in e["periods"])
            lab = (1 if (inside and e["mod_v"]) else 0)
            out.append({"img": ((crop-CLIP_M)/CLIP_S).transpose(2, 0, 1),
                        "t": float(t), "lab": lab, "vid": e["vid"]})
        cap.release(); return out

    @torch.no_grad()
    def extract_video(entries, split, fps):
        fp = cfg_fp(entries, f"vid{fps}")
        fz = PERSIST/"features"/f"lavdf_vid__{split}__{fp}.npz"
        if fz.exists(): print(f"[cache] {fz.name}"); return fz
        model = CLIPVisionModel.from_pretrained(AL.V_BACKBONE).to(DEVICE).eval().half()
        X, T, L, V, B = [], [], [], [], []
        def flush():
            if not B: return
            x = torch.tensor(np.stack([b["img"] for b in B]), dtype=torch.float16, device=DEVICE)
            hs = model(pixel_values=x, output_hidden_states=True).hidden_states[AL.V_LAYER]
            X.append(hs[:, 1:].mean(1).cpu().numpy().astype(np.float16))
            T.extend(b["t"] for b in B); L.extend(b["lab"] for b in B); V.extend(b["vid"] for b in B)
            B.clear()
        t0 = time.time()
        for j, e in enumerate(entries):
            for fr in video_frames(e, fps):
                B.append(fr)
                if len(B) == AL.V_BATCH: flush()
            if (j+1) % 25 == 0:
                print(f"  [vid:{split}] {j+1}/{len(entries)} vids  ({(time.time()-t0)/60:.1f}m)")
        flush()
        np.savez(fz, X=np.concatenate(X) if X else np.zeros((0, 1), np.float16),
                 t=np.array(T, np.float32), lab=np.array(L, np.int8),
                 vid=np.array(V))
        del model; torch.cuda.empty_cache()
        print(f"[vid:{split}] {len(T)} frames → {fz.name}")
        return fz

    FZ = {"aud_tr": extract_audio(E_TRAIN, "train", True),
          "aud_dev": extract_audio(E_DEV, "dev", False),
          "aud_te": extract_audio(E_TEST, "test", False),
          "vid_tr": extract_video(E_TRAIN, "train", AL.V_FPS_TRAIN),
          "vid_dev": extract_video(E_DEV, "dev", AL.V_FPS_EVAL),
          "vid_te": extract_video(E_TEST, "test", AL.V_FPS_EVAL)}
    def ld(k):
        z = np.load(FZ[k], allow_pickle=True)
        return z["X"].astype(np.float32), z["t"], z["lab"], z["vid"]

    # ---------------- probes: train on LAV-DF or apply zero-shot heads -------
    def fit_or_load(track):
        if AL.PROBE_SOURCE == "zeroshot":
            npz = np.load(Z_AUD if track == "aud" else Z_VID, allow_pickle=True)
            return ("zeroshot", npz)
        Xtr, _, lab, _ = ld(f"{track}_tr")
        keep = lab >= 0
        assert keep.sum() > 50 and len(np.unique(lab[keep])) == 2, \
            f"{track}: degenerate training windows (n={int(keep.sum())}) — raise TRAIN_VIDS"
        set_seed(0)
        sc  = StandardScaler().fit(Xtr[keep])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                 random_state=0).fit(sc.transform(Xtr[keep]), lab[keep])
        print(f"[probe:{track}] trained on {int(keep.sum())} windows "
              f"({int(lab[keep].sum())} fake) at "
              f"{'L%02d' % (AL.A_LAYER if track=='aud' else AL.V_LAYER)}")
        return ("fitted", (sc, clf))

    HEADS = {t: fit_or_load(t) for t in ("aud", "vid")}
    def score(track, X):
        kind, h = HEADS[track]
        if kind == "zeroshot": return apply_probe_head(h, X)
        sc, clf = h; return clf.predict_proba(sc.transform(X))[:, 1]

    # window-level sanity AUCs + dev-set thresholds (never touched test)
    THR = {}
    for track in ("aud", "vid"):
        Xd, _, labd, _ = ld(f"{track}_dev")
        sd_ = score(track, Xd)
        m = labd >= 0
        if len(np.unique(labd[m])) == 2:
            auc = float(roc_auc_score(labd[m], sd_[m]))
            _, thr = eer_of(labd[m], sd_[m])
        else:
            auc, thr = float("nan"), 0.5
        THR[track] = thr
        print(f"[dev:{track}] window-level AUC={auc:.3f}  EER-threshold={thr:.3f}")

    # ---------------- timelines → segments → AP@IoU --------------------------
    def curves_for(entries, track_key, track):
        X, T, _, V = ld(track_key)
        S = score(track, X)
        by = {}
        for s, t, v in zip(S, T, V): by.setdefault(str(v), []).append((float(t), float(s)))
        cur = {}
        for e in entries:
            pts = sorted(by.get(e["vid"], []))
            dur = (e["dur"] or (pts[-1][0] + 1.0 if pts else 1.0))
            grid = np.arange(0, dur + 1e-6, AL.GRID)
            if not pts:
                cur[e["vid"]] = (grid, np.zeros_like(grid)); continue
            tt = np.array([p[0] + (AL.A_WIN/2 if track == "aud" else 0.0) for p in pts])
            ss = np.array([p[1] for p in pts])
            g = np.interp(grid, tt, ss, left=ss[0], right=ss[-1])
            cur[e["vid"]] = (grid, median_filter(g, size=AL.SMOOTH_K))
        return cur

    def segments(grid, s, thr):
        segs, on, a = [], False, 0.0
        for t, v in zip(grid, s):
            if v >= thr and not on: on, a = True, t
            if v < thr and on:
                on = False
                msk = (grid >= a) & (grid < t)
                segs.append([a, t, float(s[msk].max()) if msk.any() else float(v)])
        if on:
            msk = grid >= a
            segs.append([a, float(grid[-1]) + AL.GRID, float(s[msk].max()) if msk.any() else 0.0])
        merged = []
        for sg in segs:
            if merged and sg[0] - merged[-1][1] <= AL.MERGE_GAP:
                merged[-1][1] = sg[1]; merged[-1][2] = max(merged[-1][2], sg[2])
            else: merged.append(list(sg))
        return [m for m in merged if m[1] - m[0] >= AL.MIN_SEG]

    def iou1d(a, b):
        inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        union = max(a[1], b[1]) - min(a[0], b[0])
        return inter / union if union > 0 else 0.0

    def ap_at_iou(preds, gts, thr_iou):
        # preds: list of (conf, vid, [s,e]) ; gts: dict vid -> list of [s,e]
        n_gt = sum(len(v) for v in gts.values())
        if n_gt == 0: return float("nan")
        used = {v: [False]*len(g) for v, g in gts.items()}
        tp, fp = [], []
        for conf, vid, seg in sorted(preds, key=lambda x: -x[0]):
            cand = gts.get(vid, [])
            j_best, i_best = 0.0, -1
            for i, g in enumerate(cand):
                if used[vid][i]: continue
                j = iou1d(seg, g)
                if j > j_best: j_best, i_best = j, i
            if j_best >= thr_iou and i_best >= 0:
                used[vid][i_best] = True; tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        if not tp: return 0.0
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec = tp / n_gt; prec = tp / np.maximum(1, tp + fp)
        ap = 0.0
        for r in np.arange(0, 1.01, 0.01):     # 101-point interpolation
            p = prec[rec >= r].max() if (rec >= r).any() else 0.0
            ap += p / 101
        return float(ap)

    CUR = {"aud": curves_for(E_TEST, "aud_te", "aud"),
           "vid": curves_for(E_TEST, "vid_te", "vid")}

    preds = {"audio": [], "video": [], "union": []}
    gts   = {"audio": {}, "video": {}, "union": {}}
    attrib_ok = attrib_tot = 0
    demo_pool = []
    for e in E_TEST:
        gts["audio"][e["vid"]] = e["periods"] if e["mod_a"] else []
        gts["video"][e["vid"]] = e["periods"] if e["mod_v"] else []
        gts["union"][e["vid"]] = e["periods"]
        ga, sa = CUR["aud"].get(e["vid"], (np.array([0.]), np.array([0.])))
        gv, sv = CUR["vid"].get(e["vid"], (np.array([0.]), np.array([0.])))
        seg_a = segments(ga, sa, THR["aud"]); seg_v = segments(gv, sv, THR["vid"])
        for s0, s1, c in seg_a: preds["audio"].append((c, e["vid"], [s0, s1]))
        for s0, s1, c in seg_v: preds["video"].append((c, e["vid"], [s0, s1]))
        # union track + modality attribution: overlapping A/V predictions merge to "both"
        tl = ([{"t_start": s0, "t_end": s1, "modality": "audio", "confidence": c} for s0, s1, c in seg_a] +
              [{"t_start": s0, "t_end": s1, "modality": "video", "confidence": c} for s0, s1, c in seg_v])
        tl.sort(key=lambda x: x["t_start"])
        fused = []
        for item in tl:
            if fused and item["t_start"] <= fused[-1]["t_end"] and item["modality"] != fused[-1]["modality"]:
                fused[-1]["t_end"] = max(fused[-1]["t_end"], item["t_end"])
                fused[-1]["modality"] = "both"
                fused[-1]["confidence"] = max(fused[-1]["confidence"], item["confidence"])
            else:
                fused.append(dict(item))
        for f_ in fused: preds["union"].append((f_["confidence"], e["vid"], [f_["t_start"], f_["t_end"]]))
        # attribution accuracy on fused segments that hit a GT period (IoU≥0.5)
        gt_mod = ("both" if e["mod_a"] and e["mod_v"] else "audio" if e["mod_a"]
                  else "video" if e["mod_v"] else None)
        if gt_mod:
            for f_ in fused:
                if any(iou1d([f_["t_start"], f_["t_end"]], g) >= 0.5 for g in e["periods"]):
                    attrib_tot += 1; attrib_ok += int(f_["modality"] == gt_mod)
        # the product artifact: per-video timeline JSON
        json.dump({"vid": e["vid"], "gt_periods": e["periods"],
                   "gt_modality": gt_mod or "real", "predicted": fused},
                  open(PERSIST/"timelines"/f"{e['vid']}.json", "w"), indent=1)
        if len(demo_pool) < 4 and e["periods"]: demo_pool.append(e)

    results = {"probe_source": AL.PROBE_SOURCE,
               "audio": {"backbone": AL.A_BACKBONE, "layer": AL.A_LAYER,
                         "win_s": AL.A_WIN, "hop_s": AL.A_HOP},
               "video": {"backbone": AL.V_BACKBONE, "layer": AL.V_LAYER,
                         "fps_eval": AL.V_FPS_EVAL},
               "n_test_videos": len(E_TEST),
               "attribution_acc": (attrib_ok / attrib_tot) if attrib_tot else None,
               "attribution_n": attrib_tot, "ap": {}}
    for track in ("audio", "video", "union"):
        results["ap"][track] = {f"AP@{i}": ap_at_iou(preds[track], gts[track], i) for i in AL.IOUS}
    json.dump(results, open(PERSIST/"results"/"lavdf_localization.json", "w"), indent=2)

    # ---- demo figure: curves + GT vs predicted segments ----------------------
    if demo_pool:
        fig, axes = plt.subplots(len(demo_pool), 1, figsize=(10, 2.2*len(demo_pool)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, e in zip(axes, demo_pool):
            ga, sa = CUR["aud"][e["vid"]]; gv, sv = CUR["vid"][e["vid"]]
            ax.plot(ga, sa, color="tab:blue", lw=1.2, label="audio score")
            ax.plot(gv, sv, color="tab:red", lw=1.2, label="video score")
            for p, q in e["periods"]:
                ax.axvspan(p, q, color="k", alpha=0.15)
            ax.axhline(THR["aud"], color="tab:blue", ls=":", lw=0.8)
            ax.axhline(THR["vid"], color="tab:red", ls=":", lw=0.8)
            ax.set_ylim(0, 1); ax.set_title(f"{e['vid']}  (GT: {quadrant(e)})", fontsize=9)
        axes[0].legend(fontsize=8, loc="upper right")
        fig.suptitle("Fig.4 — defect timelines (shaded = ground-truth fake periods)")
        fig.tight_layout()
        fig.savefig(PERSIST/"figures"/"fig4_timelines.png", dpi=200); plt.close(fig)

    print("=" * 66, "\nAV_LOCALIZE VERDICT (LAV-DF, frozen probes, "
          f"PROBE_SOURCE={AL.PROBE_SOURCE})")
    print(f"  test: {len(E_TEST)} videos | grid {AL.GRID}s | audio win {AL.A_WIN}s hop {AL.A_HOP}s | video {AL.V_FPS_EVAL} fps")
    for track in ("audio", "video", "union"):
        aps = results["ap"][track]
        print(f"  {track:6s} " + "  ".join(f"{k}={v:.3f}" if v == v else f"{k}=n/a" for k, v in aps.items()))
    if results["attribution_acc"] is not None:
        print(f"  modality attribution acc = {results['attribution_acc']:.3f}  (n={attrib_tot} matched segments)")
    print("  anchors: BA-TFD+ (arXiv:2305.01979, LAV-DF authors) and AV-Deepfake1M++ "
          "frontier (top AP@0.95 = 34.27) — pull exact table values when writing.")
    print(f"  per-video timelines → {PERSIST/'timelines'}  (the product artifact)")

    tex = PERSIST / "reports" / "table_row.tex"
    existing = set(open(tex).read().splitlines()) if tex.exists() else set()
    u = results["ap"]["union"]
    row = (f"LAV-DF TFL & frozen XLS-R L{AL.A_LAYER} + CLIP L{AL.V_LAYER} ({AL.PROBE_SOURCE}) & "
           f"AP@0.5 {u['AP@0.5']:.3f} & AP@0.75 {u['AP@0.75']:.3f} & AP@0.95 {u['AP@0.95']:.3f} & "
           f"n={len(E_TEST)} \\\\")
    if row not in existing:
        with open(tex, "a") as f:
            f.write("% av_localize VERDICT\n" + row + "\n")
        print("  latex: 1 new row →", tex)
    print("  all artifacts →", PERSIST)


# ============================================================================
# STAGE C: eval_dfe2024 — Deepfake-Eval-2024 audio, zero-shot
# ⚠️ KNOWN GAP (G3): ckpts this stage loads are not produced by week1_audio;
# mel config also mismatches Stage A. Skips gracefully; needs a probe re-fit
# rewrite before DFE2024 numbers can go in the paper.
# ============================================================================

if STAGE == "eval_dfe2024":
    HF_TOKEN_DFE    = HF_TOKEN
    DFE_KAGGLE_SLUG = ""
    MAX_FILES       = 4000
    SR, WIN_S       = 16000, 4.0
    DFE = DATA / "dfe2024"

    if not any(DFE.rglob("*")) if DFE.exists() else True:
        DFE.mkdir(parents=True, exist_ok=True)
        if DFE_KAGGLE_SLUG:
            sh(f'kaggle datasets download -d "{DFE_KAGGLE_SLUG}" -p "{DFE}" --unzip')
        else:
            subprocess.run([sys.executable,"-m","pip","install","-q","huggingface_hub"], check=False)
            from huggingface_hub import snapshot_download
            snapshot_download("nuriachandra/Deepfake-Eval-2024", repo_type="dataset",
                              local_dir=str(DFE), token=HF_TOKEN_DFE or None,
                              allow_patterns=["*.csv","*.mp3","*.wav","*.m4a","*.flac","*audio*","*Audio*"])
    print("dfe2024 files:", sum(1 for _ in DFE.rglob("*") if _.is_file()))

    import pandas as pd
    def load_labels(root):
        best = None
        for c in sorted(root.rglob("*.csv")):
            try: df = pd.read_csv(c)
            except Exception: continue
            cols = {k.lower(): k for k in df.columns}
            fcol = next((cols[k] for k in cols if "file" in k or "name" in k or "path" in k), None)
            lcol = next((cols[k] for k in cols if "label" in k or "truth" in k or "fake" in k), None)
            if fcol and lcol and len(df) > 50 and (best is None or len(df) > len(best[0])):
                best = (df, fcol, lcol, c)
        assert best, "no metadata CSV with file+label columns found — check the download"
        df, fcol, lcol, src = best
        print("labels from:", src, "| rows:", len(df), "| cols:", fcol, "/", lcol)
        lab = df[lcol].astype(str).str.lower()
        y = lab.str.contains("fake|spoof|1").astype(int)
        keep = lab.str.contains("fake|spoof|real|bona|0|1")
        return dict(zip(df[fcol].astype(str)[keep], y[keep]))
    label_map = load_labels(DFE)

    exts = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    idx = {p.name: p for p in DFE.rglob("*") if p.suffix.lower() in exts}
    pairs = [(idx[k], v) for k, v in label_map.items() if k in idx]
    if not pairs:
        pairs = [(idx[Path(k).name], v) for k, v in label_map.items() if Path(k).name in idx]
    random.Random(1234).shuffle(pairs)
    if MAX_FILES: pairs = pairs[:MAX_FILES]
    n_fake = sum(v for _, v in pairs)
    print(f"eval set: {len(pairs)} files ({n_fake} fake / {len(pairs)-n_fake} real)")
    assert len(pairs) > 100 and 0 < n_fake < len(pairs), "eval set degenerate — inspect label parsing above"

    def eer(y, s):
        fpr, tpr, _ = roc_curve(y, s)
        return float(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])

    def load_wave(p):
        try:
            w, _ = librosa.load(str(p), sr=SR, mono=True)
            if len(w) < SR: return None
            T = int(SR * WIN_S)
            if len(w) < T: w = np.pad(w, (0, T - len(w)))
            return w[:T].astype(np.float32)
        except Exception:
            return None

    results = {}
    ssl_ckpt = PERSIST / "ckpts" / "ssl_head.pt"
    if ssl_ckpt.exists():
        from transformers import Wav2Vec2Model
        w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE).eval()
        head = nn.Linear(1536, 1).to(DEVICE)
        head.load_state_dict(torch.load(ssl_ckpt, map_location=DEVICE)); head.eval()
        ys, ss = [], []
        with torch.no_grad():
            for p, y in pairs:
                w = load_wave(p)
                if w is None: continue
                h = w2v(torch.from_numpy(w)[None].to(DEVICE)).last_hidden_state
                f = torch.cat([h.mean(1), h.std(1)], -1)
                ss.append(torch.sigmoid(head(f)).item()); ys.append(y)
        results["SSL (wav2vec2-base, frozen)"] = (roc_auc_score(ys, ss), eer(np.array(ys), np.array(ss)), len(ys))
    else:
        print(f"[skip] SSL head not found at {ssl_ckpt} — this stage needs a probe re-fit rewrite (gap G3)")

    mel_ckpt = PERSIST / "ckpts" / "melcnn.pt"
    if mel_ckpt.exists():
        mel_model = torch.load(mel_ckpt, map_location=DEVICE); mel_model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for p, y in pairs:
                w = load_wave(p)
                if w is None: continue
                m = librosa.feature.melspectrogram(y=w, sr=SR, n_mels=128)
                m = torch.from_numpy(np.log1p(m))[None, None].float().to(DEVICE)
                ss.append(torch.sigmoid(mel_model(m).reshape(-1)[0]).item()); ys.append(y)
        results["mel-CNN"] = (roc_auc_score(ys, ss), eer(np.array(ys), np.array(ss)), len(ys))
    else:
        print(f"[skip] mel-CNN ckpt not found at {mel_ckpt} (optional — inversion row)")

    print("\n===== Deepfake-Eval-2024 audio (zero-shot, frozen models) =====")
    print("train: CVoiceFake/ASVspoof-era | eval: real 2024 in-the-wild fakes (arXiv:2503.02857)")
    for k, (auc, e, n) in results.items():
        print(f"{k:38s} AUC {auc:.3f}   EER {e*100:5.1f}%   n={n}")
    out = PERSIST / "results" / "dfe2024_audio.json"
    out.write_text(json.dumps({k: {"auc": a, "eer": e, "n": n} for k, (a, e, n) in results.items()}, indent=2))
    print("saved →", out)
    if results:
        print("\nPaper framing: literature reports ~48% AUC drop for SOTA audio detectors on this set;")
        print("compare your drop vs that band. Collapse == thesis confirmed on 2024 fakes; survival == stronger claim.")


# ============================================================================
# ⭐ STAGE C2 (V19): eval_dfe2024_v2 — FIXED zero-shot on Deepfake-Eval-2024
# ----------------------------------------------------------------------------
# WHAT CHANGED FROM V18's eval_dfe2024:
#   V18 tried to load `ssl_head.pt` and `melcnn.pt` — files that never existed
#   because week1_audio's export path is `probe_head_audio_xlsr.npz` (portable
#   NPZ with coef/intercept/mean/scale/meta).  V18 also mismatched the mel
#   config (n_mels=128 in V18 vs 80 in week1_audio's mel-CNN).
#
# V19 FIXES:
#   1. Load the AUDIO probe head from probe_head_audio_xlsr.npz (correct file).
#   2. Load the HierCon-lite ensemble scores if hiercon_audio ran, for the
#      strongest V19 audio row.
#   3. Add the IMAGE branch (DFE2024 has 1,975 images) using the video_probe's
#      exported CLIP head — same head as cross_gen_df40 uses.
#   4. Use the ACTUAL XLS-R backbone + layer that week1_audio picked (read from
#      the exported head's meta), no hardcoded L05 assumption.
#
# WHAT WE REPORT:
#   Audio branch:  AUC/EER on DFE2024 audio subset
#   Image branch:  AUC/EER on DFE2024 image subset
#   Both vs the "SOTA drops 45-50% AUC on DFE2024" claim in arXiv:2503.02857
#
# HONEST NOTES:
#   • DFE2024 audio is 56.5 hours in 52 languages.  We cap eval at 4000 clips
#     for the T4 session; report the actual n in the JSON.
#   • DFE2024 image count is 1975 (759 real + 1216 fake).  We can eval all.
#   • Video branch is deferred to a future run (requires MTCNN + CLIP frames).
# ============================================================================

if STAGE == "eval_dfe2024_v2":
    import pandas as pd
    import cv2
    from transformers import Wav2Vec2Model, CLIPVisionModel

    HF_TOKEN_DFE = HF_TOKEN
    MAX_AUDIO    = 4000 if not DEBUG else 200
    MAX_IMAGES   = 2000 if not DEBUG else 100
    SR, WIN_S    = 16000, 4.0
    AUDIO_HOP_S  = 2.0          # V20: sliding-window hop for audio scoring
    APPLY_FACE_CROP = True      # V20: MTCNN-crop DFE images to match the video_probe head protocol
    DFE = DATA / "dfe2024"

    # V20: clear stale result files so a partial/failed run can't print old numbers
    for _stale in ("dfe2024_v2_audio.json", "dfe2024_v2_image.json"):
        _p = PERSIST / "results" / _stale
        if _p.exists(): _p.unlink(); print(f"[dfe2024] cleared stale {_stale}")

    # V20: guard against silent label-flips — if AUC < 0.5 the head and labels
    # disagree on direction, so flip and warn (rather than reporting an inverted number).
    def score_with_direction(y, s, tag=""):
        y = np.asarray(y); s = np.asarray(s)
        auc = float(roc_auc_score(y, s))
        if auc < 0.5:
            print(f"  [dfe2024:{tag}] ⚠️ AUC={auc:.3f} < 0.5 — labels look inverted vs the head; flipping scores.")
            s = 1.0 - s
            auc = float(roc_auc_score(y, s))
        return s, auc

    # ---- Download (if not already) ------------------------------------------
    if not DFE.exists() or not any(DFE.rglob("*")):
        DFE.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
        from huggingface_hub import snapshot_download
        try:
            snapshot_download("nuriachandra/Deepfake-Eval-2024", repo_type="dataset",
                              local_dir=str(DFE), token=HF_TOKEN_DFE or None,
                              allow_patterns=["*.csv","*.mp3","*.wav","*.m4a","*.flac","*.ogg",
                                              "*.png","*.jpg","*.jpeg","*.webp",
                                              "*audio*","*Audio*","*image*","*Image*"])
        except Exception as e:
            print(f"[dfe2024] snapshot_download failed: {e}")
            print("[dfe2024] If gated, request access on the HF page and set HF_TOKEN at file top.")
    n_files = sum(1 for _ in DFE.rglob("*") if _.is_file())
    print(f"[dfe2024] files on disk: {n_files}")

    # ---- Label discovery ----------------------------------------------------
    def load_labels(root, kind_hint=None):
        """kind_hint: 'audio' | 'image' | None (best-match).
        V20: sorts by an explicit key (avoids the DataFrame-comparison crash on
        tied scores) and deprioritises the 'with-links' CSVs (which usually hold
        URLs, not local paths)."""
        candidates = []
        for c in sorted(root.rglob("*.csv")):
            try: df = pd.read_csv(c)
            except Exception: continue
            cols = {k.lower(): k for k in df.columns}
            fcol = next((cols[k] for k in cols
                         if any(t in k for t in ("file","name","path","utt"))), None)
            lcol = next((cols[k] for k in cols
                         if any(t in k for t in ("label","truth","fake","class","real"))), None)
            if fcol and lcol and len(df) > 50:
                # if hint says audio/image, prefer csvs whose files match the modality
                name = c.stem.lower()
                score = len(df)
                if kind_hint == "audio" and "audio" in name: score *= 10
                if kind_hint == "image" and "image" in name: score *= 10
                if "with-links" in name or "with_links" in name:
                    score -= 10**9     # V20: URLs not local paths — deprioritise
                candidates.append((score, df, fcol, lcol, c))
        if not candidates:
            print(f"[dfe2024:{kind_hint}] no CSV with file+label columns"); return {}
        candidates.sort(key=lambda x: x[0], reverse=True)   # V20: key-based, crash-safe
        _, df, fcol, lcol, src = candidates[0]
        print(f"[dfe2024:{kind_hint}] labels from {src.name}  ({len(df)} rows, label col={lcol})")
        lab = df[lcol].astype(str).str.lower().str.strip()
        y = lab.str.contains("fake|spoof|synth|1").astype(int)
        keep = lab.str.contains("fake|spoof|synth|real|bona|genuine|0|1")
        return dict(zip(df[fcol].astype(str)[keep], y[keep].astype(int)))

    # =========================== AUDIO BRANCH ================================
    print("\n===== V19 eval_dfe2024_v2 :: AUDIO branch =====")
    audio_labels = load_labels(DFE, kind_hint="audio")
    if audio_labels:
        AUD_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
        aud_idx = {p.name: p for p in DFE.rglob("*") if p.suffix.lower() in AUD_EXTS}
        aud_pairs = [(aud_idx[k], v) for k, v in audio_labels.items() if k in aud_idx]
        if not aud_pairs:
            aud_pairs = [(aud_idx[Path(k).name], v) for k, v in audio_labels.items()
                         if Path(k).name in aud_idx]
        random.Random(1234).shuffle(aud_pairs)
        if MAX_AUDIO: aud_pairs = aud_pairs[:MAX_AUDIO]
        n_fake = sum(v for _, v in aud_pairs)
        print(f"[dfe2024:audio] eval set: {len(aud_pairs)} files ({n_fake} fake / "
              f"{len(aud_pairs)-n_fake} real)")

        audio_head = PERSIST / "ckpts" / "probe_head_audio_xlsr.npz"
        if not audio_head.exists():
            print(f"[dfe2024:audio] SKIP — no exported head at {audio_head}.  "
                  "Run STAGE='week1_audio' first.")
        elif len(aud_pairs) < 20:
            print("[dfe2024:audio] SKIP — too few labeled audio files matched to disk.")
        else:
            head = np.load(audio_head, allow_pickle=True)
            hm = json.loads(str(head["meta"]))
            print(f"[dfe2024:audio] using head: backbone={hm['backbone']}  L{hm['layer']}  "
                  f"win={hm.get('window_s', 4.0)}s  trained_on={hm['trained_on']}")

            w2v = Wav2Vec2Model.from_pretrained(hm["backbone"]).to(DEVICE).eval().half()
            LYR = int(hm["layer"])

            @torch.no_grad()
            def audio_score_all(pairs, batch=8, hop_s=AUDIO_HOP_S):
                """V20: sliding-window scoring. Each file is split into WIN_S windows
                stepped by hop_s; every window is forwarded through XLS-R and scored
                by the head; the file score is the MEAN over windows (MAX also kept
                for a top-1 report). This replaces V19's single 4 s center crop, which
                is not a valid file-level score for variable-length in-the-wild audio."""
                per_file_windows = {}   # file_idx -> list of window scores
                buf = []                # list of (window_np, file_idx)
                def flush():
                    if not buf: return
                    x = torch.tensor(np.stack([b[0] for b in buf]),
                                     dtype=torch.float16, device=DEVICE)
                    hs = w2v(x, output_hidden_states=True).hidden_states[LYR]
                    feats = hs.mean(1).cpu().numpy().astype(np.float32)
                    ws = apply_probe_head(head, feats)
                    for (b, fi), sc in zip(buf, ws):
                        per_file_windows.setdefault(fi, []).append(float(sc))
                    buf.clear()
                T = int(SR * WIN_S)
                hop = max(1, int(SR * hop_s))
                for fi, (p, y) in enumerate(pairs):
                    try:
                        w, _ = librosa.load(str(p), sr=SR, mono=True)
                    except Exception:
                        continue
                    if len(w) < T:
                        w = np.pad(w, (0, T - len(w)))
                    m = np.abs(w).max(); w = w/m if m > 0 else w
                    starts = list(range(0, max(1, len(w) - T + 1), hop))
                    if not starts: starts = [0]
                    for st in starts:
                        buf.append((w[st:st+T].astype(np.float32), fi))
                        if len(buf) == batch: flush()
                    if (fi+1) % 200 == 0: print(f"  [dfe2024:audio] {fi+1}/{len(pairs)} files")
                flush()
                if not per_file_windows:
                    return None
                s_mean, s_max, ys = [], [], []
                for fi, (p, y) in enumerate(pairs):
                    if fi in per_file_windows:
                        wsc = per_file_windows[fi]
                        s_mean.append(float(np.mean(wsc))); s_max.append(float(np.max(wsc)))
                        ys.append(int(y))
                return {"mean": np.array(s_mean), "max": np.array(s_max), "y": np.array(ys),
                        "aggregation": f"sliding-window WIN={WIN_S}s HOP={hop_s}s, mean+max"}

            aud = audio_score_all(aud_pairs)
            del w2v; torch.cuda.empty_cache()
            if aud is not None and len(np.unique(aud["y"])) == 2:
                s_aud, y_aud = aud["mean"], aud["y"]
                s_aud, auc_aud = score_with_direction(y_aud, s_aud, "audio")   # V20
                _smax, auc_aud_max = score_with_direction(y_aud, aud["max"], "audio(max)")
                eer_aud, _ = eer_of(y_aud, s_aud)
                auc_ci = bootstrap_auc_ci(y_aud, s_aud)
                print(f"[dfe2024:audio] EXPORTED FROZEN HEAD (V20):  AUC {auc_aud:.3f} "
                      f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]   EER {eer_aud*100:.2f}%   "
                      f"n={len(y_aud)}  (max-aggr AUC {auc_aud_max:.3f})")
                # also try the HierCon-lite ensemble if available (matched to the same target)
                hc_ens = PERSIST/"results"/"target_scores_hiercon_lite_ens.npy"
                extra = {}
                if hc_ens.exists():
                    # NOTE: these are on the ITW target, NOT on DFE2024 — do not conflate.
                    # For DFE2024 we'd need to re-forward the same clips through the head.
                    # This is left as a TODO — requires loading the hiercon_lite_seed*.pt
                    # models and running inference.  Skipping for now with a note.
                    extra["hiercon_lite_note"] = ("HierCon-lite scores exist for ITW target; "
                        "run hiercon_audio inference on DFE2024 features to add this row.")
                dfe_audio_out = {"n": int(len(y_aud)), "auc": auc_aud, "auc_ci": auc_ci,
                                 "auc_max_aggregation": auc_aud_max,
                                 "aggregation": aud["aggregation"],
                                 "eer_pct": eer_aud*100, "backbone": hm["backbone"],
                                 "layer": LYR, "window_s": hm.get("window_s", 4.0), **extra}
                json.dump(dfe_audio_out, open(PERSIST/"results"/"dfe2024_v2_audio.json", "w"), indent=2)
            else:
                print("[dfe2024:audio] scored 0 valid clips or single-class labels — check labels")
    else:
        print("[dfe2024:audio] no labels — the audio subset may not be present in this snapshot")

    # =========================== IMAGE BRANCH ================================
    print("\n===== V19 eval_dfe2024_v2 :: IMAGE branch =====")
    image_labels = load_labels(DFE, kind_hint="image")
    if image_labels:
        IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        img_idx = {p.name: p for p in DFE.rglob("*") if p.suffix.lower() in IMG_EXTS}
        img_pairs = [(img_idx[k], v) for k, v in image_labels.items() if k in img_idx]
        if not img_pairs:
            img_pairs = [(img_idx[Path(k).name], v) for k, v in image_labels.items()
                         if Path(k).name in img_idx]
        random.Random(1234).shuffle(img_pairs)
        if MAX_IMAGES: img_pairs = img_pairs[:MAX_IMAGES]
        n_fake = sum(v for _, v in img_pairs)
        print(f"[dfe2024:image] eval set: {len(img_pairs)} images ({n_fake} fake / "
              f"{len(img_pairs)-n_fake} real)")

        image_head = PERSIST / "ckpts" / "probe_head_video_clip.npz"
        if not image_head.exists():
            print(f"[dfe2024:image] SKIP — no exported CLIP head at {image_head}.  "
                  "Run STAGE='video_probe' first.")
        elif len(img_pairs) < 20:
            print("[dfe2024:image] SKIP — too few labeled images matched to disk.")
        else:
            head = np.load(image_head, allow_pickle=True)
            hm = json.loads(str(head["meta"]))
            print(f"[dfe2024:image] using head: backbone={hm['backbone']}  L{hm['layer']}  "
                  f"trained_on={hm['trained_on']}")
            CLIP_M2 = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
            CLIP_S2 = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
            clip = CLIPVisionModel.from_pretrained(hm["backbone"]).to(DEVICE).eval().half()
            LYR = int(hm["layer"])
            # V20: MTCNN face-crop to match the video_probe head's training protocol
            # (threshold 0.9, margin 0.30). Falls back to a resize if no face is found.
            mtcnn_img = None
            if APPLY_FACE_CROP:
                try:
                    from facenet_pytorch import MTCNN as _MTCNN
                    mtcnn_img = _MTCNN(keep_all=False, select_largest=True,
                                       post_process=False, device=DEVICE)
                    print("[dfe2024:image] face-crop ON (MTCNN thr 0.9, margin 0.30) — matches video_probe")
                except Exception as _e:
                    print(f"[dfe2024:image] MTCNN unavailable ({_e}); falling back to resize-only")
                    mtcnn_img = None
            _IMG2, _MARG2 = 224, 0.30

            def _prep_image(p):
                img = cv2.imread(str(p))
                if img is None: return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if mtcnn_img is not None:
                    box, prob = mtcnn_img.detect(img)
                    if box is not None and prob is not None and prob[0] is not None and prob[0] >= 0.9:
                        x0,y0,x1,y1 = box[0]; w,h = x1-x0, y1-y0
                        x0=max(int(x0-_MARG2*w),0); y0=max(int(y0-_MARG2*h),0)
                        x1=min(int(x1+_MARG2*w),img.shape[1]); y1=min(int(y1+_MARG2*h),img.shape[0])
                        img = img[y0:y1, x0:x1]
                if img.shape[0] != _IMG2 or img.shape[1] != _IMG2:
                    img = cv2.resize(img, (_IMG2, _IMG2))
                return ((img.astype(np.float32)/255. - CLIP_M2) / CLIP_S2).transpose(2,0,1)

            @torch.no_grad()
            def image_score_all(pairs, batch=16):
                X_feats, Y = [], []
                buf = []
                def flush():
                    if not buf: return
                    x = torch.tensor(np.stack([b[0] for b in buf]),
                                     dtype=torch.float16, device=DEVICE)
                    hs = clip(pixel_values=x, output_hidden_states=True).hidden_states[LYR]
                    X_feats.append(hs[:, 1:].mean(1).cpu().numpy().astype(np.float32))
                    Y.extend(b[1] for b in buf); buf.clear()
                skipped = 0
                for i, (p, y) in enumerate(pairs):
                    v = _prep_image(p)
                    if v is None: skipped += 1; continue
                    buf.append((v, y))
                    if len(buf) == batch: flush()
                    if (i+1) % 200 == 0: print(f"  [dfe2024:image] {i+1}/{len(pairs)}")
                flush()
                if not X_feats: return None, None, skipped
                return np.concatenate(X_feats), np.array(Y), skipped

            X_img, y_img, sk = image_score_all(img_pairs)
            del clip; torch.cuda.empty_cache()
            if X_img is not None and len(np.unique(y_img)) == 2:
                s_img = apply_probe_head(head, X_img)
                s_img, auc_img = score_with_direction(y_img, s_img, "image")   # V20
                eer_img, _ = eer_of(y_img, s_img)
                auc_ci = bootstrap_auc_ci(y_img, s_img)
                print(f"[dfe2024:image] EXPORTED FROZEN HEAD (V20):  AUC {auc_img:.3f} "
                      f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]   EER {eer_img*100:.2f}%   "
                      f"n={len(y_img)}  (skipped {sk} unreadable)  "
                      f"face_crop={'on' if mtcnn_img is not None else 'off'}")
                dfe_img_out = {"n": int(len(y_img)), "auc": auc_img, "auc_ci": auc_ci,
                               "eer_pct": eer_img*100, "backbone": hm["backbone"],
                               "layer": LYR,
                               "face_crop_applied": bool(mtcnn_img is not None)}
                json.dump(dfe_img_out, open(PERSIST/"results"/"dfe2024_v2_image.json", "w"), indent=2)
            else:
                print("[dfe2024:image] scored 0 valid images or single-class labels")
    else:
        print("[dfe2024:image] no labels — the image subset may not be present")

    # =========================== FINAL VERDICT ================================
    print("=" * 66, "\nEVAL_DFE2024_V2 VERDICT (V19 fix of gap G3)")
    ap = PERSIST/"results"/"dfe2024_v2_audio.json"
    ip = PERSIST/"results"/"dfe2024_v2_image.json"
    if ap.exists():
        a = json.load(open(ap))
        print(f"  AUDIO  AUC {a['auc']:.3f} [{a['auc_ci'][0]:.3f}, {a['auc_ci'][1]:.3f}]  "
              f"EER {a['eer_pct']:.2f}%  n={a['n']}")
    if ip.exists():
        i = json.load(open(ip))
        print(f"  IMAGE  AUC {i['auc']:.3f} [{i['auc_ci'][0]:.3f}, {i['auc_ci'][1]:.3f}]  "
              f"EER {i['eer_pct']:.2f}%  n={i['n']}")
    print("  paper framing: arXiv:2503.02857 reports SOTA drops of ~48% AUC (audio) and 45%")
    print("     (image) on DFE2024 vs prior benchmarks.  Compare your drop vs that band.")
    print("     Survival on DFE2024 with a FROZEN probe (no fine-tuning on 2024 data) is a")
    print("     strong story for the paper.  Collapse confirms the thesis (2024 distribution shift).")
    print("  all artifacts →", PERSIST)


print("\nDone. STAGE =", STAGE, "| DEBUG =", DEBUG,
      "\nAll persistent artifacts:", PERSIST,
      "\nNext: flip DEBUG=False for the real run; switch STAGE for the other pipeline.")


# ============================================================================
# PACKAGE — zip PERSIST and download (clip__* caches skipped by default)
# ============================================================================
import datetime, zipfile
_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_zip_path = str(Path("/content" if IN_COLAB else ".") / f"paper1_persist_{STAGE}_{_stamp}.zip")
try:
    n_packed = n_skipped = 0
    with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PERSIST.rglob("*")):
            if p.is_dir(): continue
            if ZIP_SKIP_CLIP_FEATURES and p.name.startswith("clip__"):
                n_skipped += 1; continue
            z.write(p, p.relative_to(PERSIST)); n_packed += 1
    _sz = Path(_zip_path).stat().st_size / (1024*1024)
    print(f"[zip] wrote {_zip_path}  ({_sz:.1f} MB, {n_packed} files"
          + (f", {n_skipped} clip caches skipped" if n_skipped else "") + ")")

    with zipfile.ZipFile(_zip_path) as z:
        names = z.namelist()
    for pat in ("results/", "figures/", "reports/", "ckpts/", "timelines/"):
        hit = [n for n in names if n.startswith(pat)]
        if hit:
            print(f"  {pat} ({len(hit)}):")
            for n in hit[:6]: print(f"    - {n}")
            if len(hit) > 6: print(f"    ... +{len(hit)-6} more")

    if IN_COLAB:
        try:
            from google.colab import files as _f
            _f.download(_zip_path)
            print("[download] browser prompt sent — check your downloads")
        except Exception as e:
            print(f"[download] auto-download failed ({e}); grab it manually from the file browser:")
            print(f"           {_zip_path}")
except Exception as e:
    print(f"[zip] FAILED: {e}")
    print(f"      manual fallback:  !zip -r {_zip_path} {PERSIST}")
