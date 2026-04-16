#!/usr/bin/env bash
# Full-clip train + test: baseline vs rules+fusion (SEP-28k, max-clips-per-split=0).
# Default rules mode: zhang (fast). Set RULES=zhang_full and ZHANG_CACHE=... for full cascade + cache.
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

# Always defined (bash + `set -u` otherwise errors on "${extra_rules[@]}" when empty/unset).
declare -a extra_rules=()
if [[ "$RULES" == "zhang_full" && -n "${ZHANG_CACHE:-}" ]]; then
  extra_rules=(--zhang-full-cache-dir "$ZHANG_CACHE")
fi

echo "=== Baseline (no fusion, no rules) ===" | tee "$LOGDIR/full_${STAMP}_baseline.log"
python3 -u disfluency_pipeline.py --mode train \
  --max-clips-per-split 0 --epochs "$EPOCHS" --batch-size "$BATCH" \
  --fusion none --rules none --no-whisper --no-progress \
  --device "$DEVICE" \
  --checkpoint-dir checkpoints_ablation_baseline_full_compare \
  2>&1 | tee -a "$LOGDIR/full_${STAMP}_baseline.log"

echo "=== Rules ($RULES + $FUSION) ===" | tee "$LOGDIR/full_${STAMP}_rules.log"
python3 -u disfluency_pipeline.py --mode train \
  --max-clips-per-split 0 --epochs "$EPOCHS" --batch-size "$BATCH" \
  --fusion "$FUSION" --rules "$RULES" --no-whisper --no-progress \
  --device "$DEVICE" \
  "${extra_rules[@]}" \
  --checkpoint-dir checkpoints_ablation_rules_full_compare \
  2>&1 | tee -a "$LOGDIR/full_${STAMP}_rules.log"

echo "Done. Logs: $LOGDIR/full_${STAMP}_baseline.log , $LOGDIR/full_${STAMP}_rules.log"
