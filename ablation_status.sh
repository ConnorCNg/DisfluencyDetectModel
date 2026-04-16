#!/usr/bin/env bash
# Show training progress (epoch / val / test / saves) from the latest ablation log(s).
# Usage: ./ablation_status.sh [N]   — N = number of matching lines to show (default 40)
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-40}"
PATTERN='Epoch [0-9]+/[0-9]+ · train loss|val loss|val F1|Test loss|Test F1|^===|saved'

pick_latest() {
  ls -t "$@" 2>/dev/null | head -1
}

LOG=""
if MASTER=$(pick_latest ablation_logs/nohup_master_*.log) && [[ -n "$MASTER" && -f "$MASTER" ]]; then
  LOG="$MASTER"
elif [[ -f ablation_full_run.log ]]; then
  LOG="ablation_full_run.log"
else
  # Sequential runs: concatenate latest baseline then latest rules (same stamp if possible)
  B=$(pick_latest ablation_logs/full_*_baseline.log)
  R=$(pick_latest ablation_logs/full_*_rules.log)
  if [[ -n "${B:-}" && -f "$B" ]]; then
    if [[ -n "${R:-}" && -f "$R" ]]; then
      LOG=$(mktemp)
      trap 'rm -f "$LOG"' EXIT
      cat "$B" "$R" >"$LOG"
    else
      LOG="$B"
    fi
  fi
fi

if [[ -z "${LOG:-}" || ! -f "$LOG" ]]; then
  echo "No log found. Expected one of:" >&2
  echo "  ablation_logs/nohup_master_*.log" >&2
  echo "  ablation_logs/full_*_{baseline,rules}.log" >&2
  echo "  ablation_full_run.log" >&2
  exit 1
fi

echo "---- $(basename "$LOG") (last $N hits) ----"
grep -E "$PATTERN" "$LOG" | tail -n "$N"
