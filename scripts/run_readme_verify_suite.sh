#!/usr/bin/env bash
# Rerun README pipelines (no --refresh* flags; uses existing .cache).
# BiLSTM uses small caps for a lighter pass (README full run uses max-clips-per-split 0).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/verify_run/logs artifacts/error_analysis

need() { test -e "$1" || { echo "Missing required path: $1" >&2; exit 1; }; }

need SEP-28k-Extended_clips.csv
need artifacts/svm_clean03_best_configs_full.json
need data/SEP-28k_CLIP
need data/sep28k/clips

echo "== Rules-only (logs under artifacts/verify_run/logs/) =="
python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --no-progress 2>&1 | tee artifacts/verify_run/logs/rules_eval_3s_SEP28k-T.txt

python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/sep28k/clips \
  --no-progress 2>&1 | tee artifacts/verify_run/logs/rules_eval_mix_SEP28k-T.txt

echo "== tune_thresholds_rules_svm (3s T, 3s D, mix T) =="
python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_3s_sep28kT_strict.json

python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/SEP-28k_CLIP \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_3s_sep28kD_strict.json

python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_mix53_sep28kT_strict.json

echo "== eval_4head strict + learned Block =="
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

echo "== old_behavior layer-8 SVM + learned Block (SEP28k-T) =="
python3 -u scripts/old_behavior_svm_only_with_optional_pause_selection.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --layer 8 --block-learned-pause-selection \
  --out-json artifacts/error_analysis/ling230_old_svm_layer8_sep28kT_3s_block_learned.json

python3 -u scripts/old_behavior_svm_only_with_optional_pause_selection.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --layer 8 --block-learned-pause-selection \
  --out-json artifacts/error_analysis/ling230_old_svm_layer8_sep28kT_mix53_block_learned.json

echo "== matched 5s vs 3s =="
python3 -u scripts/compare_matched_5s_vs_3s_pause_pipeline.py \
  --device auto --seed 42 --label-vote-threshold 2 --split-column SEP28k-T \
  --clips-root data/sep28k/clips --clip3-root data/SEP-28k_CLIP \
  --out-json artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kT.json

python3 -u scripts/compare_matched_5s_vs_3s_pause_pipeline.py \
  --device auto --seed 42 --label-vote-threshold 2 --split-column SEP28k-D \
  --clips-root data/sep28k/clips --clip3-root data/SEP-28k_CLIP \
  --out-json artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kD.json

echo "== block_only (4 runs) =="
python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kT_learned_pause_strict.json

python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kD_learned_pause_strict.json

python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_mix53_sep28kT_learned_pause_strict.json

python3 -u scripts/block_only_train_dev_test.py --seed 42 --device auto --label-vote-threshold 2 \
  --split-column SEP28k-D --data-root data/sep28k/clips \
  --learned-pause-scorer --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_mix53_sep28kD_learned_pause_strict.json

echo "== run_comparable_sep28kT_D_3s.sh (re-tunes 3s T/D + block-only + merge JSON) =="
bash scripts/run_comparable_sep28kT_D_3s.sh

echo "== BiLSTM (capped: 128 clips/split, 3 epochs; logs in artifacts/verify_run/logs/) =="
python3 -u disfluency_pipeline.py --mode train --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T \
  --data-root data/SEP-28k_CLIP \
  --max-clips-per-split 128 --epochs 3 --batch-size 8 \
  --fusion none --rules none --no-progress \
  --checkpoint-dir artifacts/verify_run/checkpoints_bilstm_3s_SEP28k-T \
  2>&1 | tee artifacts/verify_run/logs/bilstm_3s_SEP28k-T.log

python3 -u disfluency_pipeline.py --mode train --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T \
  --data-root data/sep28k/clips \
  --max-clips-per-split 128 --epochs 3 --batch-size 8 \
  --fusion none --rules none --no-progress \
  --checkpoint-dir artifacts/verify_run/checkpoints_bilstm_mix_SEP28k-T \
  2>&1 | tee artifacts/verify_run/logs/bilstm_mix_SEP28k-T.log

echo "== Summary tables =="
python3 -u scripts/summarize_readme_verify_results.py --write-md artifacts/verify_run/README_verify_summary.md

echo "Done. Root: $ROOT"
