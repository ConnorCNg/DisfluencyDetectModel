#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TRAIN="${TRAIN:-2000}"
TEST="${TEST:-500}"
BATCH="${BATCH:-16}"
LAYER="${LAYER:-8}"
VOTE_THRESHOLD="${VOTE_THRESHOLD:-2}"
DEVICE="${DEVICE:-cpu}"
CACHE_DIR="${CACHE_DIR:-.cache/w2v2_embeddings}"

python3 -u paper_style_w2v2_svm_test.py \
  --device "$DEVICE" \
  --max-train "$TRAIN" \
  --max-test "$TEST" \
  --batch-size "$BATCH" \
  --layer "$LAYER" \
  --vote-threshold "$VOTE_THRESHOLD" \
  --embedding-cache-dir "$CACHE_DIR"
