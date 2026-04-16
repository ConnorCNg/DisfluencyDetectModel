#!/usr/bin/env python3
"""
Offline: compute Zhang-full rule logits (+ optional Whisper word timestamps)
and write JSON caches for training with --rules zhang_full.

Example:
  python precompute_zhang_full_cache.py --out-dir data/sep28k/zhang_full_cache \\
    --max-clips 200 --whisper --device cuda

Requires HuggingFace transformers + torch for --whisper. Without --whisper,
only acoustic cascade logits are stored (interjection column stays weak).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from disfluency_pipeline import Config, _apply_max_clips, build_split_lists
from zhang_full.cache import cache_filename_for_path, write_cache
from zhang_full.cascade import compute_rule_logits_mono


def _load_mono(path: str, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        import torchaudio

        t = torch.from_numpy(x).unsqueeze(0)
        t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=target_sr)
        x = t.numpy().reshape(-1).astype(np.float32)
    return x


def _whisper_word_segments(
    mono: np.ndarray,
    sample_rate: int,
    model_name: str,
    device: str,
) -> List[Tuple[str, float, float]]:
    try:
        from transformers import pipeline
    except ImportError:
        print("[Warn] transformers not installed; skipping Whisper.", file=sys.stderr)
        return []

    device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
    if device == "cpu":
        device_id = -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=device_id,
    )
    out: Dict[str, Any] = pipe(
        {"array": mono, "sampling_rate": sample_rate},
        return_timestamps="word",
    )
    words: List[Tuple[str, float, float]] = []
    chunks = out.get("chunks") or []
    for ch in chunks:
        text = (ch.get("text") or "").strip()
        ts = ch.get("timestamp")
        if not text or ts is None:
            continue
        if isinstance(ts, (list, tuple)) and len(ts) == 2:
            t0, t1 = float(ts[0] or 0.0), float(ts[1] or 0.0)
        else:
            continue
        words.append((text, t0, t1))
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute Zhang-full rule logits JSON cache")
    ap.add_argument("--data-root", type=str, default="data/sep28k/clips")
    ap.add_argument("--label-csv", type=str, default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--split-column", type=str, default="SEP28k-T")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all clips from CSV")
    ap.add_argument("--whisper", action="store_true", help="Run Whisper for word timestamps")
    ap.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-tiny",
        help="ASR model for --whisper",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device

    cfg = Config()
    cfg.data_root = args.data_root
    cfg.label_csv = args.label_csv
    cfg.split_column = args.split_column
    cfg.sample_rate = args.sample_rate

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    paths = tp + vp + ep
    labels_dummy = tl + vl + el
    cap = args.max_clips
    if cap > 0:
        paths, _ = _apply_max_clips(paths, labels_dummy, cap)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, path in enumerate(paths):
        abs_p = os.path.abspath(path)
        dest = os.path.join(args.out_dir, cache_filename_for_path(abs_p))
        if os.path.isfile(dest):
            continue
        mono = _load_mono(path, cfg.sample_rate)
        words: Optional[List[Tuple[str, float, float]]] = None
        meta: Dict[str, Any] = {"whisper": False}
        if args.whisper:
            words = _whisper_word_segments(
                mono, cfg.sample_rate, args.whisper_model, dev
            )
            meta["whisper"] = True
            meta["whisper_model"] = args.whisper_model
            meta["n_word_chunks"] = len(words)
        logits = compute_rule_logits_mono(mono, cfg.sample_rate, words).tolist()
        # Recompute interjection / word rep with words inside compute_rule_logits_mono
        # (already applied). logits is final.
        write_cache(args.out_dir, abs_p, logits, words=words, meta=meta)
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[{i+1}/{len(paths)}] wrote {dest}", flush=True)
    print(f"Done. Processed {len(paths)} clip(s); cache dir: {args.out_dir}")


if __name__ == "__main__":
    main()
