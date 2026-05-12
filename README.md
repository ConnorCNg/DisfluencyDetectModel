# DisfluencyDetectModel

SEP-28k dysfluency experiments: rules, Wav2Vec2+SVM, Block variants, and a small BiLSTM baseline. This README is only what I actually ran for the main SEP-28k numbers plus the follow-up where I compared **matched ~5 s clips** to the **same clip IDs cut to 3 s** (length is the only thing that changes there).

**Labels:** every command below uses `--label-vote-threshold 2`, i.e. a head is positive for a clip if **at least two of three** annotators voted that dysfluency on.

```bash
pip install -r requirements.txt
cd /path/to/DisfluencyDetectModel
```

Do not commit `.cache/`, downloaded audio, or HF hub weights. If you need a clean machine to pull weights once, `artifacts/REGENERATE_CACHES.txt` and `bash scripts/regenerate_hf_and_caches.sh` are there for that.

---

## Clip roots

| What | `--data-root` |
|------|----------------|
| All clips forced to ~3 s | `data/SEP-28k_CLIP` |
| Mixed ~5 s + ~3 s (default extract layout) | `data/sep28k/clips` |

You need `SEP-28k-Extended_clips.csv` at repo root and the wavs under those folders.

---

## Caches (so you are not surprised)

- SVM / four-head / block-only scripts share **Wav2Vec2 + 9-D prosody** caches under `.cache/w2v2_embeddings/` and `.cache/prosody_features/`. Keys include the **absolute path** to the wav, so you pay the extraction cost **once per `--data-root`** (3 s tree vs mixed tree are different paths).
- Block pause features: `.cache/block_pause_features/` (same idea: per path).
- `disfluency_pipeline.py` training can use `--audio-feature-cache-dir` for its own frozen features; that is **not** the same cache as the SVM pipeline above.

## Optional: one-shot verify (local only)

`artifacts/verify_run/` is gitignored. From repo root, after you have both clip trees and caches warmed:

```bash
bash scripts/run_readme_verify_suite.sh
```

That reruns the README commands (no `--refresh-*` flags), runs BiLSTM with **128 clips per split and 3 epochs** instead of a full 20-epoch pass, then writes `artifacts/verify_run/README_verify_summary.md` and prints the same tables to stdout. For numbers that match the paper README exactly, use the same caps as in the copy-paste blocks above for BiLSTM.

Practical order: run `tune_thresholds_rules_svm.py` once per root you care about with `--max-train 0 --max-dev 0 --max-test 0` first so later scripts mostly hit disk cache. Old layer-8-only code mostly adds missing layer-8 `.npy` files if the mainline run already filled other layers.

---

## Full reruns (copy-paste)

### Rules + mainline SVM (per-head W2V2 layers + prosody)

Train SVM on train, tune score thresholds on dev, numbers on test. Rules F1 is in the same JSON as SVM: `test_metrics_tuned_thresholds.rules_zhang_full_f1`.

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

**Mixed 5 s + 3 s, SEP28k-T**

```bash
python3 -u tune_thresholds_rules_svm.py \
  --device auto --seed 42 --label-vote-threshold 2 \
  --split-column SEP28k-T --data-root data/sep28k/clips \
  --max-train 0 --max-dev 0 --max-test 0 \
  --thresholds-out artifacts/tuned_thresholds_rules_svm_mix53_sep28kT_strict.json
```

**Rules only, no SVM load** (prints test F1 to stdout):

```bash
python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/SEP-28k_CLIP

python3 -u eval_rules_only.py --rules zhang_full --split test --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T --data-root data/sep28k/clips
```

### Four-head SVM + learned-pause Block override (mainline bundle)

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

### Layer-8 only, 768-D, no prosody, learned Block train selection (SEP28k-T)

```bash
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
```

Drop `--block-learned-pause-selection` if you only want plain four-head SVM F1.

### Matched ~5 s vs same clip 3 s (the extra control)

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

### Block-only: vanilla Block SVM vs pause-heavy “natural v2” (both in one JSON)

`results.block_base_train_dev_test` vs `results.block_natural_v2_train_dev_test` with `--learned-pause-scorer`.

```bash
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
```

**One shell script** that only does tune + block-only on 3 s for T and D and then merges one summary JSON:

```bash
bash scripts/run_comparable_sep28kT_D_3s.sh
```

### BiLSTM baseline (no rule fusion)

Slow on full data (`--max-clips-per-split 0`). I used 20 epochs in the main runs; drop that for a dry run.

```bash
python3 -u disfluency_pipeline.py --mode train --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T \
  --data-root data/SEP-28k_CLIP \
  --max-clips-per-split 0 --epochs 20 \
  --fusion none --rules none \
  --checkpoint-dir checkpoints_bilstm_3s_sep28kT

python3 -u disfluency_pipeline.py --mode train --device auto \
  --label-vote-threshold 2 --split-column SEP28k-T \
  --data-root data/sep28k/clips \
  --max-clips-per-split 0 --epochs 20 \
  --fusion none --rules none \
  --checkpoint-dir checkpoints_bilstm_mix53_sep28kT
```

Optional: `--audio-feature-cache-dir .cache/bilstm_audio_features` to reuse frozen features between runs.

---

## `artifacts/svm_clean03_best_configs_full.json`

The mainline SVM stack reads per-head **Wav2Vec2 layer** and **C** from that file. It was produced by `svm_clean03_layer_prosody_sweep.py`, which trains on **unanimous** labels only (vote exactly 0 vs exactly 3) to pick layer/C. That is **not** the same labeling rule as the 2/3 study runs above; the JSON is just a frozen hyperparameter bundle for features and SVM capacity.

---

## Downloading and cutting audio

You need Python 3, NumPy (`download_audio.py`), pandas + scipy (`extract_clips.py`).

| Script | Input CSV | What it does |
|--------|-----------|--------------|
| `download_audio.py` | `SEP-28k_episodes.csv` | Downloads 16 kHz mono episode wavs. |
| `extract_clips.py` | `SEP-28k-Extended_clips.csv` | Cuts **5 s** clips from `Start_5_sec` / `Stop_5_sec`. |

Do not pass the extended clips CSV to `download_audio.py`.

Default layout for SEP-28k in this repo: episodes `data/sep28k/wavs`, clips `data/sep28k/clips`.

```bash
python download_audio.py
python extract_clips.py --progress
```

Optional explicit paths:

```bash
python download_audio.py --episodes SEP-28k_episodes.csv --wavs data/sep28k/wavs
python extract_clips.py --labels SEP-28k-Extended_clips.csv --wavs data/sep28k/wavs --clips data/sep28k/clips --progress
```

Some URLs fail; the downloader skips and continues. Missing episode wavs show up as skipped rows on extract.

Rough sizes: ~32 GB raw episodes, ~2.6 GB for 5 s clips.

FluencyBank: point `--episodes` / `--labels` / `--wavs` / `--clips` at your own paths (same pattern as in the table).

---

## `disfluency_pipeline.py` (BiLSTM + four heads)

Loads `SEP-28k-Extended_clips.csv`, wav2vec2 + MFCC, optional Whisper, BiLSTM, four heads. Splits from `SEP28k-T` or `SEP28k-D` (`--split-column`); `dev` in the CSV is validation. Missing files are skipped; if a split is empty the code re-splits 70/15/15 with a warning.

First run pulls wav2vec2 (and whisper if used) from Hugging Face.

```bash
python disfluency_pipeline.py --max-clips 64 --batch-size 4
python disfluency_pipeline.py --mode train --epochs 2 --batch-size 4 --max-clips-per-split 48
python disfluency_pipeline.py --split-column SEP28k-D --max-clips 32
python disfluency_pipeline.py --mode train --resume checkpoints/checkpoint_last.pt --epochs 1 --max-clips-per-split 48
```

Checkpoints under `--checkpoint-dir` (default `checkpoints/`): `checkpoint_last.pt`, `checkpoint_best.pt`.

Four heads only; training loss is BCE on those four. `presence_F1` is derived (any gold label on vs any head above `--f1-threshold`, default 0.5). `type_F1` is per-class.

`--max-clips-per-split 0` means use every clip found on disk under the active `--data-root`. Use a small number for a quick test.

Clips load with soundfile. On Apple Silicon, architecture mismatches usually mean reinstalling arm64 torch/soundfile/etc. HF `masked_spec_embed` warning is ignorable here; mel MFCC warnings are usually harmless.
