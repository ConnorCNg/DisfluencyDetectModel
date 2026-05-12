#!/usr/bin/env bash
# Download Wav2Vec2 weights into the default Hugging Face cache, then remind how
# to fill .cache/w2v2_embeddings and .cache/prosody_features/ (run compare/tune).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Downloading facebook/wav2vec2-base-960h via transformers (one-time cache)"
python3 -c "
from transformers import Wav2Vec2Model
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
print('OK: model weights cached (see HF_HOME / ~/.cache/huggingface/hub by default).')
"

echo ""
echo "==> Project .cache (embeddings + prosody) is NOT downloaded here."
echo "    Run a full compare or tune once, for example:"
echo "    python3 -u compare_rules_svm_hybrid.py --device auto --seed 42 --max-train 0 --max-test 0 \\"
echo "      --thresholds-json artifacts/tuned_thresholds_rules_svm.json"
echo ""
echo "    See artifacts/REGENERATE_CACHES.txt for details."
