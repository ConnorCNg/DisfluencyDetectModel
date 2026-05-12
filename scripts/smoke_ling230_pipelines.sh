#!/usr/bin/env bash
# Fast smoke tests for LING-230-style pipelines (small random subsamples per split).
# Requires SEP-28k-Extended_clips.csv and clip trees under data/SEP-28k_CLIP and data/sep28k/clips.
set -euo pipefail
cd "$(dirname "$0")/.."
CAP="${SMOKE_SPLIT_CAP:-256}"
OUT="${SMOKE_OUT_DIR:-artifacts/smoke}"
mkdir -p "$OUT"
DEVICE="${SMOKE_DEVICE:-auto}"
BS="${SMOKE_BATCH_SIZE:-4}"

echo "== eval_rules_only (zhang_full, 3s + mixed) =="
python3 -u eval_rules_only.py --rules zhang_full --split test --device "$DEVICE" \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --max-clips 12 --batch-size "$BS" --no-progress
python3 -u eval_rules_only.py --rules zhang_full --split test --device "$DEVICE" \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/sep28k/clips \
  --max-clips 12 --batch-size "$BS" --no-progress

echo "== tune_thresholds_rules_svm (3s, capped) =="
python3 -u tune_thresholds_rules_svm.py --device "$DEVICE" --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --max-train "$CAP" --max-dev "$CAP" --max-test "$CAP" --batch-size "$BS" \
  --thresholds-out "$OUT/tuned_thresholds_smoke_sep28kT_3s.json"

echo "== old_behavior SVM layer8 + learned block (3s, capped) =="
python3 -u scripts/old_behavior_svm_only_with_optional_pause_selection.py \
  --device "$DEVICE" --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --layer 8 --block-learned-pause-selection \
  --max-train "$CAP" --max-dev "$CAP" --max-test "$CAP" --batch-size "$BS" \
  --out-json "$OUT/old_behavior_smoke_sep28kT_3s.json"

echo "== eval_4head strict + learned block (3s, capped) =="
python3 -u scripts/eval_4head_with_block_learned_pause_strict.py \
  --device "$DEVICE" --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --max-train "$CAP" --max-dev "$CAP" --max-test "$CAP" --batch-size "$BS" \
  --out-json "$OUT/four_head_strict_smoke_sep28kT_3s.json"

echo "== block_only base vs learned pause (3s, capped) =="
python3 -u scripts/block_only_train_dev_test.py --seed 42 --device "$DEVICE" \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --max-train "$CAP" --max-dev "$CAP" --max-test "$CAP" --batch-size "$BS" \
  --out-json "$OUT/block_only_smoke_sep28kT_3s.json"

echo "== compare_matched 5s vs 3s (capped) =="
python3 -u scripts/compare_matched_5s_vs_3s_pause_pipeline.py \
  --device "$DEVICE" --seed 42 --label-vote-threshold 2 --split-column SEP28k-T \
  --clips-root data/sep28k/clips --clip3-root data/SEP-28k_CLIP \
  --max-train "$CAP" --max-dev "$CAP" --max-test "$CAP" --batch-size "$BS" \
  --out-json "$OUT/matched_5s_vs_3s_smoke_sep28kT.json"

echo "== disfluency_pipeline BiLSTM (tiny) =="
python3 -u disfluency_pipeline.py --mode train --device "$DEVICE" \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --max-clips-per-split 6 --epochs 1 --batch-size 2 \
  --fusion none --rules none --no-progress \
  --checkpoint-dir "$OUT/checkpoints_bilstm_smoke"

echo "All smoke steps finished. Outputs under $OUT"
