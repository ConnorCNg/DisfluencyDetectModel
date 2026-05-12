#!/usr/bin/env bash
set -euo pipefail

# Comparable protocol:
# - data-root: 3s clips
# - seed: 42
# - 4-head: train on train, tune thresholds on dev, evaluate on test (tune_thresholds_rules_svm.py)
# - Block-only: strict train/dev/test with learned pause scorer

python3 -u tune_thresholds_rules_svm.py \
  --device auto \
  --seed 42 \
  --label-vote-threshold 2 \
  --split-column SEP28k-T \
  --data-root data/SEP-28k_CLIP \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_3s_sep28kT_strict.json

python3 -u tune_thresholds_rules_svm.py \
  --device auto \
  --seed 42 \
  --label-vote-threshold 2 \
  --split-column SEP28k-D \
  --data-root data/SEP-28k_CLIP \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_3s_sep28kD_strict.json

python3 -u scripts/block_only_train_dev_test.py \
  --seed 42 \
  --device auto \
  --label-vote-threshold 2 \
  --split-column SEP28k-T \
  --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer \
  --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kT_learned_pause_strict.json

python3 -u scripts/block_only_train_dev_test.py \
  --seed 42 \
  --device auto \
  --label-vote-threshold 2 \
  --split-column SEP28k-D \
  --data-root data/SEP-28k_CLIP \
  --learned-pause-scorer \
  --pause-low-vote-mode le1 \
  --out-json artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kD_learned_pause_strict.json

python3 - <<'PY'
import json, os

def load_json(p):
    with open(p,'r',encoding='utf-8') as f:
        return json.load(f)

T = load_json('artifacts/tuned_thresholds_rules_svm_3s_sep28kT_strict.json')
D = load_json('artifacts/tuned_thresholds_rules_svm_3s_sep28kD_strict.json')
BT = load_json('artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kT_learned_pause_strict.json')
BD = load_json('artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kD_learned_pause_strict.json')

summary = {
  'experiment':'comparable_train_dev_test_4head_and_block_learned_pause',
  'seed':42,
  'label_vote_threshold': 2,
  'data_root':'data/SEP-28k_CLIP',
  'protocol':'train on train, tune thresholds on dev, evaluate on test',
  'splits':{
    'SEP28k-T':{
      'four_head_from_tune_thresholds':{
        'thresholds_json':'artifacts/tuned_thresholds_rules_svm_3s_sep28kT_strict.json',
        'svm_f1':T['test_metrics_tuned_thresholds']['svm_f1'],
        'rules_f1':T['test_metrics_tuned_thresholds']['rules_zhang_full_f1'],
      },
      'block_only_learned_pause_strict':{
        'json':'artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kT_learned_pause_strict.json',
        'base':BT['results']['block_base_train_dev_test'],
        'natural_v2_learned_pause':BT['results']['block_natural_v2_train_dev_test'],
        'delta':BT['results']['delta_natural_minus_base'],
      }
    },
    'SEP28k-D':{
      'four_head_from_tune_thresholds':{
        'thresholds_json':'artifacts/tuned_thresholds_rules_svm_3s_sep28kD_strict.json',
        'svm_f1':D['test_metrics_tuned_thresholds']['svm_f1'],
        'rules_f1':D['test_metrics_tuned_thresholds']['rules_zhang_full_f1'],
      },
      'block_only_learned_pause_strict':{
        'json':'artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kD_learned_pause_strict.json',
        'base':BD['results']['block_base_train_dev_test'],
        'natural_v2_learned_pause':BD['results']['block_natural_v2_train_dev_test'],
        'delta':BD['results']['delta_natural_minus_base'],
      }
    }
  }
}

out='artifacts/error_analysis/comparable_results_seed42_3s_sep28kT_and_D.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out,'w',encoding='utf-8') as f:
    json.dump(summary,f,indent=2)
print(out)
PY

echo "Done."
