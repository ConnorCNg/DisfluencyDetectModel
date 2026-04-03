# DisfluencyDetectModel

## Downloading & processing audio

You need **Python 3** with **NumPy** (`download_audio.py`) and **pandas + scipy** (`extract_clips.py`). Install once:

```bash
pip install -r requirements.txt
```

If `import numpy` fails with an **architecture** error on macOS (Intel vs ARM wheels), reinstall for the same interpreter you use to run the scripts:

```bash
python -m pip install --force-reinstall numpy
```

### What each script uses

| Script | Input CSV | What it does |
|--------|-----------|----------------|
| `download_audio.py` | `SEP-28k_episodes.csv` (or `fluencybank_episodes.csv`) | Downloads each episode URL and saves **16 kHz mono** `.wav` files under a show folder. |
| `extract_clips.py` | `SEP-28k-Extended_clips.csv` (or `fluencybank_labels.csv`) | Reads `Start_5_sec` / `Stop_5_sec` and writes **5 s** `.wav` clips. |

The episodes CSV and the extended clips CSV are **different** files: do not pass the extended clips file to `download_audio.py`.

### SEP-28k (defaults in this repo)

From the project root, full episodes go to **`data/sep28k/wavs`** and clips to **`data/sep28k/clips`**.

```bash
# 1) Download & convert full episodes (long; large disk use ~32 GB raw)
python download_audio.py

# 2) Cut 5 s clips (needs the episode .wav from step 1 for each clip row)
python extract_clips.py --progress
```

Optional explicit paths (same as defaults):

```bash
python download_audio.py --episodes SEP-28k_episodes.csv --wavs data/sep28k/wavs
python extract_clips.py --labels SEP-28k-Extended_clips.csv --wavs data/sep28k/wavs --clips data/sep28k/clips --progress
```

Some URLs in the dataset may be dead or blocked; the downloader **skips** failures and continues. Rows whose episode `.wav` is missing are skipped during extraction (you will see `Missing ...` lines).

For long downloads, you can run the command in the background and append output to a log, e.g. `nohup python download_audio.py >> data/sep28k/download_audio.log 2>&1 &` — use `tail -f` on that log to watch progress; **Ctrl+C** only stops `tail`, not the Python job.

### FluencyBank

Use the FluencyBank episode and labels CSVs and set `--wavs` / `--clips` to your own folders, for example:

```bash
python download_audio.py --episodes fluencybank_episodes.csv --wavs [WAV_DIR]
python extract_clips.py --labels fluencybank_labels.csv --wavs [WAV_DIR] --clips [CLIP_DIR] --progress
```

### Sizes (SEP-28k, approximate)

- Raw episode `.wav` files: ~32 GB  
- 5 s clips: ~2.6 GB  

You can point extract at different clip CSV splits (e.g. train/dev/test) and a separate `--clips` directory if you want those splits in different folders.

## Model pipeline (`disfluency_pipeline.py`)

After clips exist under **`data/sep28k/clips`**, the script loads **`SEP-28k-Extended_clips.csv`**, builds **wav2vec2 + MFCC** features, optional **Whisper** (encoder frozen, small trainable projection), then a **BiLSTM** and **four heads** (prolongation, repetition, interjection, block).

**Splits:** use **`SEP28k-T`** or **`SEP28k-D`** (`--split-column`) — CSV values `train` / `dev` / `test` (`dev` → validation). Rows without a clip file on disk are skipped. If any split ends up empty after that, all available clips are **re-split** 70% / 15% / 15% (with a warning).

Install PyTorch / Transformers / soundfile from `requirements.txt`. First run downloads **wav2vec2** and **whisper-tiny** from Hugging Face (needs network once).

```bash
# Demo (default): one batch, small clip cap
python disfluency_pipeline.py --max-clips 64 --batch-size 4

# Quick training test (low epochs, cap per split)
python disfluency_pipeline.py --mode train --epochs 2 --batch-size 4 --max-clips-per-split 48

# Use SEP28k-D for splits instead of SEP28k-T
python disfluency_pipeline.py --split-column SEP28k-D --max-clips 32

# Resume training (only two checkpoint files are kept: last + best val)
python disfluency_pipeline.py --mode train --resume checkpoints/checkpoint_last.pt --epochs 1 --max-clips-per-split 48
```

Checkpoints (under `--checkpoint-dir`, default `checkpoints/`):

- **`checkpoint_last.pt`** — overwritten each epoch (resume here).
- **`checkpoint_best.pt`** — overwritten only when validation loss improves (if a validation set exists).

**Metrics:** The model has **four heads only** (one per dysfluency type). Training loss is **BCE on those four** — there is **no** extra “any dysfluency” class or head competing with them.

**Reporting:** **`presence_F1`** is a **derived** binary score (not trained): ground truth = any of the four labels positive; prediction = any head’s sigmoid ≥ `--f1-threshold`. **`type_F1`** lists per-type F1 (Prolongation, Repetition, Interjection, Block). Default threshold is **0.5**; override with `--f1-threshold`. Demo mode prints F1 on a **single batch** (illustrative only).

**How much data runs:** **`--max-clips-per-split` defaults to `0`** — each split uses **every** clip that exists under `data/sep28k/clips` (after dropping CSV rows with missing files). Pass e.g. **`--max-clips-per-split 48`** for a fast smoke test (first 48 rows per split in CSV order). The “Skipped N CSV rows” message means those rows have no matching file on disk.

**Progress:** Training shows a **tqdm** bar per epoch (`Epoch i/N · train`) with batch progress and running loss; validation/test use bars with `Epoch i/N · val` or `Test`. Use **`--no-progress`** when redirecting logs. **`--device auto`** (default) uses **CUDA** if PyTorch sees a GPU — GPUs are usually **much faster** than CPU for wav2vec/Whisper/training (CPU is not faster for this workload). Force **`--device cpu`** or **`--device cuda`** as needed.

**Notes**

- Clips load via **soundfile** (avoids some **torchaudio** installs requiring **torchcodec**).
- On **Apple Silicon**, if you see **architecture** errors, reinstall **arm64** wheels, e.g.  
  `arch -arm64 python3 -m pip install --force-reinstall torch torchaudio soundfile cffi transformers scikit-learn`
- Hugging Face may warn about `masked_spec_embed`; safe to ignore for this project. A **mel filterbank** MFCC warning is usually harmless.
