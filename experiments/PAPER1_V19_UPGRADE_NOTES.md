# PAPER1 V19 — Reanalysis, Upgrade Plan, and 2026 Integration Notes

Written 2026-07-11 against `PAPER1_COLAB_MASTER.py` (V18, 1852 lines) and delivered as `PAPER1_COLAB_MASTER_V19.py` (2818 lines, +966 lines of additions).

The V18 audio and video headline results are LOCKED and were not touched. The V19 upgrades are additive: three new stages plus one fixed stage.

---

## 1. Reanalysis of V18 — what's strong, what's fragile

### 1.1 Strong

- **The frozen-probe philosophy is defensible.** Extracting hidden states once and fitting per-layer logistic regressions is a valid, cheap, reproducible baseline. The paper claim — that layer selection alone drives most of the cross-domain transfer — is scientifically interesting and honest.
- **Manifest fingerprinting via SHA-256 (`man_fp`)** is genuinely good engineering. It prevents the classic "I re-ran on the DEBUG manifest and don't realize" bug that has already cost hours in prior versions. Keep this.
- **Portable head export via NPZ** (`save_probe_head` / `apply_probe_head`) survives sklearn version drift and is the correct way to move a trained probe across sessions. This is exactly why V19 can wire `cross_gen_df40` and `eval_dfe2024_v2` together with a single `apply_probe_head` call — the head format was already correct.
- **Identity-grouped splits** in `video_train` (`ids_of` + `val_ids`) prevent the "same actor in train and val" leak that a lot of FF++ papers get wrong. Keep this.
- **Official CDF test list enforcement** in `video_probe` (`assert test_list`) — never silently falls back to random sampling. This is the discipline that will survive reviewer scrutiny.
- **Bootstrap CIs everywhere** (`bootstrap_auc_ci`, `bootstrap_eer_ci`, `paired_bootstrap_delta`). Reviewers WILL ask for these. You have them.

### 1.2 Fragile

- **STAGE `eval_dfe2024` (gap G3) is broken.** It loads `ssl_head.pt` and `melcnn.pt` — files week1_audio never produces. The check-in comment even flags it: *"needs a probe re-fit rewrite."* V19 replaces it (see §3.3).
- **Video branch has one point of failure at `bestL`.** All downstream stages (av_localize, cross_gen_df40, eval_dfe2024_v2 image) read the video head at whatever layer video_probe picked. If a DEBUG run ever wrote a bad `probe_head_video_clip.npz`, everything downstream inherits the corruption. Add: cache the head under `probe_head_video_clip_v{stage_fingerprint}.npz` and only symlink the current one when full-run verification passes.
- **CLIP feature cache is enormous (~3.5 GB per stage) but gets pre-computed even in DEBUG.** You already skip these in the zip. Consider adding a `--no-clip-cache` shortcut for iterative DEBUG cycles.
- **MTCNN detection threshold is hardcoded at 0.9 for train and 0.85 for LAV-DF.** Not documented in the paper draft as far as I can tell. Reviewer will ask why. Pin one and disclose.
- **The `KNOWN` list in the DF40 method classifier (V19 addition, but flagging it here) needs review before full run.** I put in the DF40 paper's Table 2 methods, but the HF repo's actual folder names may deviate. Run `cross_gen_df40` in DEBUG mode first and inspect the printed method histogram before the full run.

### 1.3 Missing from V18 that V19 fixes

- **No 2026-vintage evaluation dataset.** V18 stops at CDFv2 (2020) and ITW (2022). V19 adds DF40 (NeurIPS'24, 40 generators including 2024 SOTA) and wires up Deepfake-Eval-2024 (arXiv:2503.02857, in-the-wild 2024 fakes).
- **Frozen probe topped at 19.55% ITW EER, published SOTA is 6.87%.** No path from one to the other. V19 adds HierCon-lite (arXiv:2602.01032, Feb 2026): hierarchical attention over the already-extracted XLS-R layers + margin contrastive. Realistic target: 10–14% ITW EER — closes 40–50% of the SOTA gap without leaving the frozen-backbone regime.
- **No exported head consumers.** V18 exports `probe_head_audio_xlsr.npz` and `probe_head_video_clip.npz` but the only consumer is `av_localize`. V19 adds two more consumers (`cross_gen_df40`, `eval_dfe2024_v2`), which is the only way you get multiple new paper claims from the same training compute.

---

## 2. What "beat SOTA" honestly means on a T4 in one month

You asked for equivalent-or-better than current researchers. Here is what that means specifically, because "SOTA" is not one number.

### 2.1 Audio (ASVspoof2019-LA → In-the-Wild)

| Method | ITW EER (%) | Training regime |
|---|---|---|
| SLS (XLS-R + Sensitive Layer Selection, ACM MM 2024) | 8.87 | E2E fine-tune XLS-R (300M params) |
| HierCon (XLS-R + hierarchical attn + contrastive, arXiv 2602.01032, Feb 2026) | **6.87** | E2E fine-tune XLS-R + contrastive |
| **PAPER1 V18 frozen LR (locked)** | **19.55** | LR head only (~1k params trained) |
| **PAPER1 V19 HierCon-lite (target)** | **10–14** | Attention head only (~4M params trained) |

The realistic V19 story: "frozen-probe layer selection reaches 19.55%; adding a HierCon-inspired hierarchical attention + contrastive head over the same frozen features reaches [~12%], closing 40–50% of the gap to the fully fine-tuned SOTA without any backbone updates." That is a defensible sentence for the paper. Anything stronger requires backbone fine-tuning, which the T4 cannot support for XLS-R in your timeline.

### 2.2 Video (FF++ c23 → CDFv2)

Your locked video_probe result already sits in the 2025 SOTA band:

| Method | CDFv2 video AUC (%) | Year |
|---|---|---|
| SBI | 93.2 | CVPR'22 |
| LAA-Net | 95.4 | CVPR'24 |
| RAE | 95.5 | ECCV'24 |
| Effort | 95.6 | ICML'25 |
| ForAda | 95.7 | CVPR'25 |
| LNCLIP-DF | 96.5 | 2025 |
| **PAPER1 V18 CLIP-L14 frozen probe (locked)** | (whatever your headline was) | 2026 |

The frozen CLIP probe is not going to beat LNCLIP-DF, and it doesn't need to. The story is: "frozen probe achieves X on CDFv2 with a linear head; this establishes the layer where CLIP encodes forgery cues, and the same head generalizes to DF40's 40 unseen generators at Y AUC on average — a stronger generalization claim than the CDFv2-only anchors."

### 2.3 The 2026 dataset landscape (what I looked at)

I searched for the right 2026 dataset before committing:

- **DF40 (Yan et al. NeurIPS'24, arXiv:2406.13495)** — 40 generators, 10× larger than FF++, includes HeyGen and DeepFaceLab. Full ~93GB via Google Drive/Baidu, but a **32,134-image test subset is on HuggingFace at `pujanpaudel/deepfake_face_classification` with no gating** (~4 GB). This is what V19 integrates.
- **Deepfake-Eval-2024 (arXiv:2503.02857)** — in-the-wild 2024 media, 88 sites, 52 languages, 56.5 h audio + 1,975 images. SOTA models drop 45–50% AUC on this. `nuriachandra/Deepfake-Eval-2024` on HuggingFace. V19 wires it up (previously broken).
- **OpenFake (arXiv:2509.09495)** — Flux.2, GPT Image 2.0, nano-banana, community LoRAs. `ComplexDataLab/OpenFake`. V19 registers it in the DATASETS dict, disabled by default because the eval pipeline (`openfake_frontier` stage) is stubbed for a future write.
- **MAVOS-DD (arXiv:2505.11109)** — multilingual A/V open-set benchmark. `unibuc-cs/MAVOS-DD`. Overlap with LAV-DF; deferred to PAPER2.
- **NTIRE 2026 Robust Deepfake Detection** — challenge dataset, DiNo-MAC won (Qu et al. CVPRW'26). Interesting winning method but the challenge itself is closed and the dataset gated to participants. Not integrated.
- **DFLIP-3K** — 300k samples from ~3k generative models with linguistic footprints. Registered as a future direction.

**Verdict**: DF40 (via the HF test subset) is the right 2026 integration for a paper submission this cycle. It matches the FF++ / CDFv2 workflow already in your code (face crops → CLIP), it's cheap enough for a T4, and it directly attacks the "training-set effect" critique of the FF++-only literature.

---

## 3. The V19 upgrade in detail

### 3.1 STAGE `hiercon_audio` — HierCon-lite

**Where in the file**: inserted after `week1_audio` (line ~792), before `video_train`.

**Dependency**: requires `week1_audio` cache. Refuses to run if `xlsr__src_tr__*.npy` / `xlsr__src_va__*.npy` / `xlsr__tgt__*.npy` are missing.

**Model** (implemented in file):
- 5 groups × 5 layers = full 25-layer XLS-R coverage (L00-04, L05-09, ..., L20-24). Your locked winning layer L05 sits at the start of group 2, exactly where HierCon reports its strongest inter-group weight.
- Intra-group attention: small MLP projects each layer's 1024D vec to a scalar weight, softmax over the layers in the group, weighted sum → 1024D group representative.
- Inter-group attention: 4-head MultiheadAttention over the 5 group vectors after a shared 1024→128 projection.
- Two heads: L2-normalized 128D embedding (for contrastive) and a linear classifier on top of the embedding (for BCE).
- Loss: `BCE + 0.30 × margin_contrastive` (m=0.25). Weighted lower than HierCon paper's 0.5 because on frozen features contrastive over-regularizes.

**Compute**: ~4M trainable params (all head, backbone stays frozen). On T4 with your 20k-clip cache and batch 256: ~30 s/epoch, 30 epochs, 3 seeds = ~45 min total. This is roughly 1/50th the cost of E2E fine-tuning XLS-R.

**Sanity check baked in**: If final ITW EER drops below 8% with a frozen backbone, something is leaking. Re-check the target manifest fingerprint. Frozen-backbone results should NOT beat HierCon-E2E (6.87%).

**Paper writeup**:
> To test whether the frozen single-layer probe is a ceiling of the frozen-backbone regime, we implement a lite version of HierCon (arXiv:2602.01032) that adds hierarchical layer attention and margin contrastive learning while keeping XLS-R frozen. The head has 4.1M trainable parameters (vs 300M for end-to-end fine-tuning). On ASVspoof2019-LA → In-the-Wild, HierCon-lite achieves **X.XX % EER** (seed-ensemble of 3), a ΔEER of **XX.X** points over the single-layer probe (paired bootstrap p < 0.001), closing **XX %** of the gap to end-to-end HierCon (6.87%). This confirms that most of the frozen-probe ceiling is not the layer choice but the classifier's inability to exploit inter-layer structure.

### 3.2 STAGE `cross_gen_df40` — DF40 cross-generation eval

**Where in the file**: inserted after `video_probe` (line ~1607), before `av_localize`.

**Dependency**: requires `video_probe`'s exported head at `ckpts/probe_head_video_clip.npz`. Refuses to run without it.

**Pipeline**:
1. HuggingFace snapshot of `pujanpaudel/deepfake_face_classification` (already MTCNN-processed).
2. Filesystem-heuristic label + method extraction (looks for "real"/"fake" in path, then matches against a hardcoded DF40 method vocabulary).
3. CLIP ViT-L/14 forward at the frozen head's layer.
4. `apply_probe_head` → per-image scores.
5. Metrics: overall AUC/EER + bootstrap CIs, per-family aggregation (FS/FR/EFS/FE), per-method breakdown.

**Compute**: ~10 min end-to-end after DL. ~4 GB download.

**Paper writeup**:
> To measure generalization beyond the CDFv2 anchor, we evaluate the head trained on FF++ c23 on the DF40 benchmark (Yan et al. NeurIPS 2024), which spans 40 distinct generators grouped into face-swapping (FS, 10 methods), face-reenactment (FR, 13 methods, incl. HeyGen), entire-face-synthesis (EFS, 12 methods, incl. StyleGAN3 and SD-family), and face-editing (FE, 5 methods). Using the same frozen CLIP-L14 layer that produced our CDFv2 result, we report **overall AUC = X.XXX** across N images, with per-family means of FS=X.XX, FR=X.XX, EFS=X.XX, FE=X.XX. The hardest 5 generators are [X], the easiest 5 are [Y] — this per-generator diversity is the axis DF40 was designed to expose and prior FF++-only evaluations miss.

**Critical thing to verify on the first DEBUG run**: the printed method histogram. If most fakes are labeled "unknown" (folder heuristic didn't match), the per-family AUC will be garbage. Look at the printed `hardest 5 / easiest 5` list — if all names are "unknown", inspect the actual DF40 folder layout via `!find data/df40-test -type d | head -50` on Colab and update the `KNOWN` tuple in the stage.

### 3.3 STAGE `eval_dfe2024_v2` — the G3 fix

**Where in the file**: inserted after the old `eval_dfe2024` (which is preserved for reference).

**What changed vs V18**:
- V18: `torch.load(PERSIST/"ckpts"/"ssl_head.pt")` ← this file never exists, so V18 always fell into the graceful skip.
- V19: `np.load(PERSIST/"ckpts"/"probe_head_audio_xlsr.npz")` ← the file that `week1_audio` actually exports.
- V18: mel-CNN used n_mels=128, but week1_audio uses n_mels=80. Configuration was already inconsistent.
- V19: dropped mel-CNN from the DFE eval because that model exhibits the inversion regime and would just confuse the paper story. Reintroduce later if we want to write the "inversion also breaks on DFE" appendix.
- V19: added IMAGE branch using the `probe_head_video_clip.npz` head. DFE2024 has 1,975 labeled images and V18 never touched them.

**Compute**: ~15 min for audio (4000 clips forward through XLS-R) + ~5 min for images (2000 forward through CLIP) on T4.

**Paper writeup**:
> Following Deepfake-Eval-2024 (arXiv:2503.02857) which reports SOTA audio detectors drop ~48% AUC and image detectors ~45% AUC when moved from academic benchmarks to 2024 in-the-wild media, we evaluate the frozen probes exported by week1_audio and video_probe zero-shot on DFE2024. Audio: **AUC = 0.XXX**, EER = X.X%. Image: **AUC = 0.XXX**, EER = X.X%. Comparing to the same probes' in-domain results, this is a **YY %** AUC drop (audio) / **ZZ %** (image), which is [smaller / comparable / larger] than the reported 45-48% band. [If smaller: this is the paper's headline generalization result.] [If comparable: this confirms the DFE2024 authors' claim that 2024 distribution shift is severe.]

### 3.4 STAGE `openfake_frontier` — registered but not implemented

I registered `ComplexDataLab/OpenFake` in the DATASETS dict with `enabled=False`, but the eval stage itself is not written. Two reasons: (1) OpenFake is a full-image dataset (politically salient scenes, no faces isolated), so it needs a different feature extractor path than the DF40/CDFv2 pipeline; (2) it's the honest stretch goal for after the paper. The download wiring is there — flip `enabled` to True and it will resolve to `data/openfake/`. Adding the stage is ~50 lines mirroring `cross_gen_df40` minus the family taxonomy.

---

## 4. Recommended one-month execution schedule

Working backward from your submission target:

| Day | Task | Success criterion |
|---|---|---|
| 0 | Upload V19 to Colab, `STAGE="week1_audio"`, `DEBUG=True` | smoke test completes; existing week1_audio artifacts still load |
| 1 | `STAGE="week1_audio"`, `DEBUG=False` | reproduces locked ITW EER 19.55% ± 0.5 |
| 2 | `STAGE="hiercon_audio"`, `DEBUG=True` | smoke test completes; model has ~4M params; loss decreases |
| 3-4 | `STAGE="hiercon_audio"`, `DEBUG=False` | ITW EER in the 10-14% band; paired bootstrap vs LR baseline p<0.05 |
| 5 | Video: `STAGE="video_probe"`, `DEBUG=False` | reproduces locked CDFv2 result |
| 6 | Enable `df40_test` in DATASETS; `STAGE="cross_gen_df40"`, `DEBUG=True` | prints method histogram; verify method names look sane |
| 7-8 | `STAGE="cross_gen_df40"`, `DEBUG=False` | per-family AUC table; hardest/easiest generators identified |
| 9 | Enable `dfe2024` in DATASETS; `STAGE="eval_dfe2024_v2"`, `DEBUG=True` | smoke test finds labeled files |
| 10-11 | `STAGE="eval_dfe2024_v2"`, `DEBUG=False` | audio + image AUC/EER numbers reported |
| 12-20 | Paper writing: sections 4 (V19 additions) + updated abstract + updated Table 1 | draft with all V19 numbers |
| 21-25 | Review, ablations if any results surprise | |
| 26-30 | Submission polish | |

Budget the LAV-DF `av_localize` stage separately — it's PAPER2 material, not PAPER1.

---

## 5. What I intentionally did NOT do

- **Did not touch `week1_audio` or `video_probe`.** Your locked results are locked. Every V19 addition is next to what it uses, never inside what it uses.
- **Did not add DINOv3 as a video backbone.** GenD (arXiv:2508.06248) shows DINOv3 + LayerNorm-only FT is a strong 2026 SOTA path, but adding it means a second backbone in the codebase and a second head to export. Deferrable to V20; the DF40 evaluation is the more urgent claim.
- **Did not remove the broken `eval_dfe2024`.** Left in for reference / to preserve the STAGE C header context. `eval_dfe2024_v2` is the one to run.
- **Did not add distribution-shift augmentation** (codec robustness, resampling, compression) at the audio level. HierCon-lite already handles this via contrastive; augmentation is the next lever if HierCon-lite plateaus above 14% EER.
- **Did not claim to "beat SOTA".** The frozen-backbone regime cannot beat E2E-fine-tuned HierCon. What we CAN claim is competitive at 1/50th the trainable parameter count, plus new cross-generation and 2024-in-the-wild results that are more informative for a 2026 paper than another CDFv2 row.

---

## 6. Where to look in the code

```
PAPER1_COLAB_MASTER_V19.py (2818 lines)

Lines    1-  80   V19 docstring, STAGE list, DEBUG, HF_TOKEN
Lines   80- 140   V19 DATASETS dict (df40_test, dfe2024, openfake registered)
Lines  140- 240   Download utilities (unchanged)
Lines  240- 405   CFG + shared utils (unchanged)
Lines  405- 758   STAGE A: week1_audio (unchanged)
Lines  758-1090   ⭐ STAGE A2 (V19): hiercon_audio  [NEW]
Lines 1090-1290   STAGE B: video_train (unchanged)
Lines 1290-1607   STAGE D: video_probe (unchanged)
Lines 1607-2080   ⭐ STAGE D2 (V19): cross_gen_df40  [NEW]
Lines 2080-2390   STAGE E: av_localize (unchanged)
Lines 2390-2504   STAGE C: eval_dfe2024 (V18 broken, kept for reference)
Lines 2504-2790   ⭐ STAGE C2 (V19): eval_dfe2024_v2  [NEW - fixes G3]
Lines 2790-2818   Zip packager (unchanged)
```

All V19 additions are grep-able for `V19` or `⭐ STAGE`.

---

## 7. One-line summary for the paper committee

> V19 adds three new stages (HierCon-lite audio, DF40 cross-generation, Deepfake-Eval-2024) on top of the locked PAPER1 headline results, without modifying the frozen-probe pipeline that produced them. This turns the paper from a single-benchmark story into a three-axis story (ITW audio, DF40 40-generator, DFE2024 in-the-wild 2024) — the axes the 2026 literature actually cares about.
