# Reproducibility Guide

This document captures the exact scripts, artifacts, and commands used to reproduce:

- Rules-only runs (`zhang`, `zhang_full`)
- SVM baseline runs (W2V2 embeddings + SVM)
- Rules/SVM hybrid comparisons
- Threshold tuning (dev -> test)
- Clean `0/3` per-dysfluency SVM sweeps with layer selection and prosodic fusion

## 1) Environment

From repo root:

```bash
cd /Users/Connor/Documents/GitHub/Projects/DysfluencyDetectModel/DisfluencyDetectModel
python3 -m pip install -r requirements.txt
python3 -m pip install librosa
```

## 2) Required data paths

Expected default inputs:

- Labels CSV: `SEP-28k-Extended_clips.csv`
- Audio root: `data/sep28k/clips`
- Split column default: `SEP28k-T`

All scripts below assume those defaults unless overridden.

## 3) Key scripts

- `eval_rules_only.py` — rules-only F1 evaluation
- `paper_style_w2v2_svm_test.py` — W2V2 embedding extraction + SVM baseline
- `compare_rules_svm_hybrid.py` — `zhang`, `zhang_full`, SVM, OR/AND hybrids in one run
- `tune_thresholds_rules_svm.py` — tune thresholds on dev, evaluate on test, save JSON thresholds
- `svm_clean03_layer_prosody_sweep.py` — clean `0/3` per-head sweeps over layers + SVM params + thresholds with prosodic features
- `prolongation_template_eval.py` — template-like prolongation-only (`0` vs `3`) eval
- Rule modules: `zhang_rules.py`, `zhang_full/*`
- Shared split/metric helpers: `disfluency_pipeline.py`

## 4) Cached artifacts

### 4.1 W2V2 embedding cache

- Directory: `.cache/w2v2_embeddings`
- One `.npy` per clip + layer + model + sample-rate + path hash

### 4.2 Prosody feature cache

- Directory: `.cache/prosody_features`
- One `.npy` per clip containing 9-D prosodic feature vector

### 4.3 Threshold/config JSON artifacts

- `artifacts/tuned_thresholds_rules_svm.json`
- `artifacts/svm_clean03_best_configs_full.json`
- `artifacts/svm_clean03_best_configs_smoke.json`

## 5) Exact run commands

## 5.1 Rules-only (full split)

```bash
python3 -u eval_rules_only.py --rules zhang --split test --device auto
python3 -u eval_rules_only.py --rules zhang_full --split test --device auto
```

Optional zhang_full cache:

```bash
python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --zhang-full-cache-dir /path/to/zhang_full_cache
```

## 5.2 Combined SVM + rules + hybrids (one command)

```bash
python3 -u compare_rules_svm_hybrid.py --max-train 0 --max-test 0 --device auto
```

Notes:
- This now auto-loads saved thresholds from:
  `artifacts/tuned_thresholds_rules_svm.json`
- To ignore saved thresholds:

```bash
python3 -u compare_rules_svm_hybrid.py --max-train 0 --max-test 0 --device auto \
  --ignore-saved-thresholds
```

## 5.3 Threshold tuning (train SVM on train, tune on dev, test on test)

```bash
python3 -u tune_thresholds_rules_svm.py --device auto --max-train 0 --max-dev 0 --max-test 0
```

Saves thresholds JSON to:

- `artifacts/tuned_thresholds_rules_svm.json` (default)

Custom output:

```bash
python3 -u tune_thresholds_rules_svm.py --thresholds-out artifacts/my_thresholds.json
```

## 5.4 Clean `0/3` per-dysfluency layer sweep + prosody

Full sweep:

```bash
python3 -u svm_clean03_layer_prosody_sweep.py \
  --device auto \
  --layers 5,6,7,8,9,10,11,12 \
  --batch-size 16 \
  --c-grid 0.1,1,3,10,30 \
  --refresh-embedding-cache \
  --out-json artifacts/svm_clean03_best_configs_full.json
```

Smoke test:

```bash
python3 -u svm_clean03_layer_prosody_sweep.py \
  --device cpu \
  --layers 5 \
  --batch-size 16 \
  --c-grid 1 \
  --out-json artifacts/svm_clean03_best_configs_smoke.json
```

## 5.5 Template-style prolongation-only eval (`0` vs `3`)

```bash
python3 -u prolongation_template_eval.py \
  --csv SEP-28k-Extended_clips.csv \
  --data-root data/sep28k/clips
```

Optional plotting:

```bash
python3 -u prolongation_template_eval.py --plot-index 0
```

## 6) Current known outputs (from prior runs)

- Tuned threshold JSON exists and is used by default in `compare_rules_svm_hybrid.py`
- Full clean `0/3` best-config JSON exists:
  `artifacts/svm_clean03_best_configs_full.json`

## 7) Push checklist

If you want exact reproducibility on another machine/repo clone, include:

- Scripts listed in section 3
- `artifacts/*.json`
- `.cache/w2v2_embeddings/*` (if you want to avoid recomputing embeddings)
- `.cache/prosody_features/*` (if present)
- CSV labels (`SEP-28k-Extended_clips.csv`)
- Any additional data retrieval instructions if audio is not committed

