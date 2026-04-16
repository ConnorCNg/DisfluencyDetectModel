#!/usr/bin/env bash
# Rules half only (same defaults as run_ablation_baseline_vs_rules.sh rules stage).
set -euo pipefail
cd "$(dirname "$0")"

EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-auto}"
RULES="${RULES:-zhang}"
FUSION="${FUSION:-gated}"
LOGDIR="${LOGDIR:-ablation_logs}"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)

declare -a extra_rules=()
if [[ "$RULES" == "zhang_full" && -n "${ZHANG_CACHE:-}" ]]; then
  extra_rules=(--zhang-full-cache-dir "$ZHANG_CACHE")
fi

echo "=== Rules only ($RULES + $FUSION) ===" | tee "$LOGDIR/rules_only_${STAMP}.log"
python3 -u disfluency_pipeline.py --mode train \
  --max-clips-per-split 0 --epochs "$EPOCHS" --batch-size "$BATCH" \
  --fusion "$FUSION" --rules "$RULES" --no-whisper --no-progress \
  --device "$DEVICE" \
  "${extra_rules[@]}" \
  --checkpoint-dir checkpoints_ablation_rules_full_compare \
  2>&1 | tee -a "$LOGDIR/rules_only_${STAMP}.log"

echo "Done. Log: $LOGDIR/rules_only_${STAMP}.log"
