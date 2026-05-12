# LING-230 study reproduction (reviewer scope)

Reproduce the **presentation** (SEP-28k, 2-of-3 vote positives, rules vs Wav2Vec2+SVM, clip-length comparison, pause-based Block) plus the **matched 5s vs 3s** follow-up.

**Convention:** every command below uses **`--label-vote-threshold 2`** (positive when **≥2** annotators agree on that dysfluency column).

**Working directory**

```bash
cd /path/to/DisfluencyDetectModel
```

---

## 0) What to install; what not to commit

- Install: `pip install -r requirements.txt`
- Do **not** commit: `.cache/`, audio trees under `data/`, Hugging Face hub weights (see `artifacts/REGENERATE_CACHES.txt`).

---

## 1) Caches: one clean strategy (read this first)

There are **two different** embedding/feature systems:

| Cache / system | Used by | Notes |
|----------------|---------|--------|
| `.cache/w2v2_embeddings/` + `.cache/prosody_features/` | `tune_thresholds_rules_svm.py`, `scripts/eval_4head_with_block_learned_pause_strict.py`, `scripts/block_only_train_dev_test.py` | Keys include **absolute clip path** and **Wav2Vec2 layer index**. You need a **full pass per `--data-root`** (3s tree vs mixed tree) because paths differ. Per-head configs use **layers 6,7,8** — first full run fills all of those for that root. |
| `.cache/block_pause_features/` | Any script using `_block_pause_matrix` | Same path hash as other caches; **shared** across Block pipelines once built for that root. |
| `disfluency_pipeline.py` BiLSTM optional `--audio-feature-cache-dir` | BiLSTM only | **Separate** from `paper_style_w2v2_svm_test` SVM caches. |

**Recommended order (minimal wasted work)**

1. **Hugging Face weights once** (any machine):  
   `bash scripts/regenerate_hf_and_caches.sh`  
   or the one-liner in `artifacts/REGENERATE_CACHES.txt`.

2. **Per `data-root` you care about**, run **one** full SVM pipeline first (same root for all later SVM steps that day), e.g.  
   `tune_thresholds_rules_svm.py ... --data-root data/SEP-28k_CLIP --label-vote-threshold 2 --max-train 0 --max-dev 0 --max-test 0`  
   then  
   `tune_thresholds_rules_svm.py ... --data-root data/sep28k/clips ...`  
   That warms **Wav2Vec2 + prosody** caches for all heads on each root.

3. Run **old single-layer-8** (`scripts/old_behavior_svm_only_with_optional_pause_selection.py`) second — only adds **layer-8** `.npy` where not already present.

4. Run **matched 5s vs 3s** last — reuses layer-8 caches for both roots; pause cache fills on first Block-related step.

5. **BiLSTM** last or in parallel — it does **not** use the SVM embedding cache above unless you set `--audio-feature-cache-dir` for its own cache.

You do **not** need a separate “regenerate caches only” script beyond step 1–2; the first full extraction run per root builds caches automatically.

---

## 2) Files this study expects in Git

**Core**

- `disfluency_pipeline.py` — splits, labels, BiLSTM train (`python3 -u disfluency_pipeline.py`)
- `paper_style_w2v2_svm_test.py` — SVM Wav2Vec2 + prosody, `extract_embeddings`, bundles
- `tune_thresholds_rules_svm.py` — rules + SVM, dev thresholds, test metrics
- `compare_rules_svm_hybrid.py` — Block vote map, pause matrices (imported helpers)
- `artifacts/svm_clean03_best_configs_full.json` — per-head layer + `C` for **mainline** SVM (not old layer-8-only)
- `svm_clean03_layer_prosody_sweep.py` — optional: regenerate the bundle above if missing
- `zhang_rules.py`, `zhang_full/*.py` — rules stack
- `eval_rules_only.py` — rules-only evaluation (no SVM / no Wav2Vec2); supports `--data-root`
- `SEP-28k-Extended_clips.csv`
- `scripts/eval_4head_with_block_learned_pause_strict.py`
- `scripts/block_only_train_dev_test.py`
- `scripts/run_comparable_sep28kT_D_3s.sh`
- `scripts/compare_matched_5s_vs_3s_pause_pipeline.py`
- `scripts/old_behavior_svm_only_with_optional_pause_selection.py` — old SVM (layer 8, 768-D, no prosody); imported by matched script
- `artifacts/REGENERATE_CACHES.txt`, `scripts/regenerate_hf_and_caches.sh`
- `requirements.txt`, `.gitignore`

**Typical saved outputs** (paths used in commands below)

- `artifacts/tuned_thresholds_rules_svm_*.json`
- `artifacts/error_analysis/four_head_strict_with_learned_block_*.json`
- `artifacts/error_analysis/block_only_train_dev_test_*_learned_pause_strict.json` (3 s and mixed)
- `artifacts/error_analysis/comparable_results_seed42_3s_sep28kT_and_D.json`
- `artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kT.json`, `..._sep28kD.json`
- `artifacts/error_analysis/ling230_old_svm_layer8_*.json` (examples; override `--out-json` as you like)

---

## 3) Data roots

| Condition | `--data-root` |
|-----------|----------------|
| All 3 s clips | `data/SEP-28k_CLIP` |
| Mixed ~5 s + ~3 s | `data/sep28k/clips` |

---

## 4) Rules + mainline SVM (Wav2Vec2 **per-head layers** + prosody)

Train on **train**, tune on **dev**, report **test**. Rules metrics are in the **same** JSON as SVM: `test_metrics_tuned_thresholds.rules_zhang_full_f1`.

**3 s, SEP28k-T**

```bash
python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_3s_sep28kT_strict.json
```

**3 s, SEP28k-D**

```bash
python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/SEP-28k_CLIP \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_3s_sep28kD_strict.json
```

**Mixed 5 s + 3 s, SEP28k-T** (slide “5s and 3s clips” column)

```bash
python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_mix53_sep28kT_strict.json
```

**Rules-only numbers:** open the JSON above; use `test_metrics_tuned_thresholds.rules_zhang_full_f1` (no extra command).

**Rules-only without running SVM / Wav2Vec2** (prints test F1 to stdout; no `artifacts/*.json` unless you redirect):

```bash
python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/SEP-28k_CLIP

python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/sep28k/clips
```

---

## 5) Strict four-head SVM + learned-pause Block override (mainline)

```bash
python3 -u scripts/eval_4head_with_block_learned_pause_strict.py --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --out-json artifacts/error_analysis/four_head_strict_with_learned_block_sep28kT_seed42.json

python3 -u scripts/eval_4head_with_block_learned_pause_strict.py --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/SEP-28k_CLIP \
  --out-json artifacts/error_analysis/four_head_strict_with_learned_block_sep28kD_seed42.json

python3 -u scripts/eval_4head_with_block_learned_pause_strict.py --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --out-json artifacts/error_analysis/four_head_strict_with_learned_block_sep28kT_seed42_mix53.json

python3 -u scripts/eval_4head_with_block_learned_pause_strict.py --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/sep28k/clips \
  --out-json artifacts/error_analysis/four_head_strict_with_learned_block_sep28kD_seed42_mix53.json
```

---

## 6) “Old” SVM pipeline (slides): **layer 8 only**, **768-D**, **no prosody**, **learned Block train selection**, **SEP28k-T**

Same script: four heads on plain `extract_embeddings` (layer 8); optional `--block-learned-pause-selection` retrains **Block** head on mined negatives + dev-tuned Block threshold. JSON includes `svm_f1` and `svm_f1_with_block_override` when enabled.

**3 s clips**

```bash
python3 -u scripts/old_behavior_svm_only_with_optional_pause_selection.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --layer 8 --block-learned-pause-selection \
  --out-json artifacts/error_analysis/ling230_old_svm_layer8_sep28kT_3s_block_learned.json
```

**Mixed 5 s + 3 s clips**

```bash
python3 -u scripts/old_behavior_svm_only_with_optional_pause_selection.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --layer 8 --block-learned-pause-selection \
  --out-json artifacts/error_analysis/ling230_old_svm_layer8_sep28kT_mix53_block_learned.json
```

*(Optional baseline without Block train change: omit `--block-learned-pause-selection`; output is only `svm_f1`.)*

---

## 7) Matched control: **~5 s only** vs **same clips, 3 s only** (length is the only difference)

```bash
python3 -u scripts/compare_matched_5s_vs_3s_pause_pipeline.py \
  --device auto --seed 42 --label-vote-threshold 2 --split-column SEP28k-T \
  --clips-root data/sep28k/clips --clip3-root data/SEP-28k_CLIP \
  --out-json artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kT.json

python3 -u scripts/compare_matched_5s_vs_3s_pause_pipeline.py \
  --device auto --seed 42 --label-vote-threshold 2 --split-column SEP28k-D \
  --clips-root data/sep28k/clips --clip3-root data/SEP-28k_CLIP \
  --out-json artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kD.json
```

---

## 8) Block-only strict: **regular Block SVM** vs **pause-learned natural-v2** (SEP28k-T and SEP28k-D)

One run writes **both**: `results.block_base_train_dev_test` (regular Block head) and `results.block_natural_v2_train_dev_test` (concat pause features + mined negatives; use `--learned-pause-scorer` for logistic pause ranking as in the slides).

**SEP28k-T, 3 s**

```bash
python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kT_learned_pause_strict.json
```

**SEP28k-D, 3 s**

```bash
python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kD_learned_pause_strict.json
```

**Mixed ~5 s + ~3 s, SEP28k-T**

```bash
python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_mix53_sep28kT_learned_pause_strict.json
```

**Mixed ~5 s + ~3 s, SEP28k-D**

```bash
python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/sep28k/clips \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_mix53_sep28kD_learned_pause_strict.json
```

**One-shot bundle** (tune + block-only **3 s** for SEP28k-T and SEP28k-D, then aggregate JSON; does **not** run mixed-clip block-only):

```bash
bash scripts/run_comparable_sep28kT_D_3s.sh
```

---

## 9) BiLSTM (presentation baseline)

Frozen Wav2Vec2 + MFCC → BiLSTM inside `disfluency_pipeline.py`. **Long** on full data (`--max-clips-per-split 0`). Slides used ~20 epochs; adjust if you only smoke-test.

**3 s clips, SEP28k-T** (no rule fusion; backbone only)

```bash
python3 -u disfluency_pipeline.py --mode train --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T \
  --data-root data/SEP-28k_CLIP \
  --max-clips-per-split 0 --epochs 20 \
  --fusion none --rules none \
  --checkpoint-dir checkpoints_ling230_bilstm_3s_sep28kT
```

**Mixed clips, SEP28k-T**

```bash
python3 -u disfluency_pipeline.py --mode train --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T \
  --data-root data/sep28k/clips \
  --max-clips-per-split 0 --epochs 20 \
  --fusion none --rules none \
  --checkpoint-dir checkpoints_ling230_bilstm_mix53_sep28kT
```

Optional repeat runs: add e.g. `--audio-feature-cache-dir .cache/bilstm_audio_features` to reuse frozen features.

---

## 10) Out of scope for this document

Cross-head error scripts, snapshot bundles, block ablation sweeps, and large `*detail.json` dumps — omit from a minimal reviewer branch unless you explicitly want them.

---

## 11) Presentation PDF

Local reference only (not in Git); filename on disk: **LING-230 Presentation.pdf**.
