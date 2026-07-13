# experiments/ — exploratory research artifacts

These files are **exploratory scratch work**, kept for provenance. They are
**not** the supported, tested codebase — that lives in [`../src/`](../src) and is
covered by [`../tests/`](../tests) and CI.

Important caveats:

- This directory documents a broader, separate research track (audio + video
  deepfake detection, "PAPER1" Colab/Kaggle iterations). It uses different
  models and datasets than the video detector documented in the top-level
  [`README.md`](../README.md).
- **Numbers reported inside these files are from ad-hoc runs and have not been
  reproduced or validated by this repository's tested pipeline.** Do not cite
  them as results of the `src/` detector. See
  [`14313_report.txt`](14313_report.txt), which itself states the video
  cross-generator performance is "NOT established (n=20, underpowered)".
- Notebooks may reference private paths, gated datasets, and API tokens you
  must supply yourself. No secrets are committed (token fields are empty
  placeholders), but review before running.

Contents:

| File | What it is |
|---|---|
| `PAPER1_COLAB_MASTER.ipynb` | Colab master notebook (data download → pipeline) |
| `PAPER1_COLAB_MASTER_V23_LEAKAGE_FREE_HIERCON_FIXED.py` | Latest single-file Colab pipeline (supersedes earlier V19–V22, which were removed as duplicates) |
| `PAPER1_V19_UPGRADE_NOTES.md` | Notes on the V19 iteration |
| `pipeline_2026.py` | Standalone pipeline draft |
| `baseline-hiercon-mimic.ipynb`, `paper1-week1-inversion-core.ipynb`, `deepfake_detector_kaggle.ipynb` | Assorted experiment notebooks |
| `audio_v16_final.json`, `14313_*.{json,txt}` | Sample run outputs / forensic-report examples |

If you want a clean, reproducible starting point, ignore this directory and
follow the top-level README quick start.
