#!/usr/bin/env python3
"""
Evaluate heuristic / rule-based logits only (no neural model, no fusion).

Uses the same clip splits + label binarization as disfluency_pipeline.Config
(row_to_label_vector + label_vote_threshold).

The tqdm progress bar is written to **stderr** only when stderr is a **real
terminal** (``sys.stderr.isatty()``). If you redirect stderr to a log file
(e.g. ``2>&1 | tee``), tqdm is **disabled** so the log stays summary-only; run
without redirecting stderr to still see the bar in your terminal.

The bar shows batch progress, ``elapsed < remaining`` (ETA to finish all
batches), and a postfix ``clips done/total · N left``.

Use ``--no-progress`` to hide the bar even in an interactive terminal.

If you run from an environment where **no bar appears** (Cursor task output,
some CI runners, etc.), stderr is often **not** a TTY so tqdm is skipped. Fix:

- Run the same command in **Cursor’s Terminal tab** (or Terminal.app), not
  only via the agent “run” capture; or
- Pass ``--force-progress`` to enable tqdm anyway (do **not** redirect stderr
  to a log file if you use this).

Example:
  python3 -u eval_rules_only.py --rules zhang --split test --device mps
  python3 -u eval_rules_only.py --rules zhang_full --split test --device mps
  python3 -u eval_rules_only.py --rules zhang_full --split test --zhang-full-cache-dir /path/to/cache

``zhang_full`` runs MFCC/RMS/ACF on the selected device; DTW uses a small CPU
NumPy scan per clip (unchanged algorithm). Use ``--device auto`` for CUDA/MPS.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from disfluency_pipeline import (
    Config,
    DisfluencyDataset,
    F1_CLASS_NAMES,
    _apply_max_clips,
    build_split_lists,
    collate_batch,
    compute_f1_metrics,
    format_f1_metrics,
)
from zhang_full.rule_module import ZhangFullRuleLogits
from zhang_rules import ZhangStyleRuleLogits


def _split_loader(
    cfg: Config,
    split: str,
    max_clips: int,
    batch_size: int,
) -> DataLoader:
    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    if split == "train":
        paths, labels = tp, tl
    elif split == "val":
        paths, labels = vp, vl
    elif split == "test":
        paths, labels = ep, el
    else:
        raise ValueError("split must be train|val|test")

    paths, labels = _apply_max_clips(paths, labels, max_clips)
    if not paths:
        raise RuntimeError(f"No clips for split={split!r}")

    ds = DisfluencyDataset(cfg, paths, labels)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )


def build_rules(
    rule_mode: str,
    device: torch.device,
    sample_rate: int,
    zhang_full_cache_dir: str,
) -> torch.nn.Module:
    if rule_mode == "zhang":
        return ZhangStyleRuleLogits(sample_rate=sample_rate).to(device).eval()
    if rule_mode == "zhang_full":
        cdir = (zhang_full_cache_dir or "").strip() or None
        return ZhangFullRuleLogits(sample_rate=sample_rate, cache_dir=cdir).to(
            device
        ).eval()
    raise ValueError("rules must be zhang or zhang_full")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-only dysfluency F1 evaluation.")
    ap.add_argument("--rules", choices=("zhang", "zhang_full"), required=True)
    ap.add_argument("--split", choices=("train", "val", "test"), default="test")
    ap.add_argument("--max-clips", type=int, default=0, help="0 = all clips in split.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--label-vote-threshold", type=int, default=2)
    ap.add_argument("--f1-threshold", type=float, default=0.5)
    ap.add_argument(
        "--zhang-full-cache-dir",
        default="",
        help="Optional cache dir for zhang_full JSON caches.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Never show tqdm, even when stderr is a terminal.",
    )
    ap.add_argument(
        "--force-progress",
        action="store_true",
        help="Show tqdm even when stderr is not a TTY (e.g. some IDE runners).",
    )
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)

    cfg = Config()
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))

    loader = _split_loader(cfg, args.split, args.max_clips, args.batch_size)
    rule_mod = build_rules(
        args.rules,
        dev,
        cfg.sample_rate,
        args.zhang_full_cache_dir,
    )

    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    def _forward_one(batch: dict) -> int:
        waveforms = batch["waveform"].to(dev)
        labels_t = torch.clamp(batch["labels"].to(dev), 0.0, 1.0)
        paths: Optional[List[str]] = batch.get("paths")
        logits = rule_mod(waveforms, paths)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(labels_t.cpu())
        return int(waveforms.shape[0])

    # tqdm: off when --no-progress; else on if stderr is a TTY or --force-progress.
    use_tqdm = (not args.no_progress) and (
        sys.stderr.isatty() or args.force_progress
    )
    n_clips_total = len(loader.dataset)
    n_batches = len(loader)

    with torch.no_grad():
        if use_tqdm:
            clips_done = 0
            pbar = tqdm(
                loader,
                total=n_batches,
                desc=f"Rules-only ({args.rules}) · {args.split}",
                unit="batch",
                dynamic_ncols=True,
                file=sys.stderr,
                smoothing=0.05,
            )
            for batch in pbar:
                clips_done += _forward_one(batch)
                left = n_clips_total - clips_done
                pbar.set_postfix_str(
                    f"clips {clips_done}/{n_clips_total} · {left} left",
                    refresh=True,
                )
        else:
            for batch in loader:
                _forward_one(batch)

    probs_cat = torch.cat(all_probs, dim=0).numpy()
    labels_cat = torch.cat(all_labels, dim=0).numpy()
    metrics = compute_f1_metrics(labels_cat, probs_cat, args.f1_threshold)

    print(f"Using device: {dev}", flush=True)
    print(
        f"Rules: {args.rules}  split={args.split}  clips={len(loader.dataset)}  "
        f"vote_threshold={cfg.label_vote_threshold}  f1_threshold={args.f1_threshold}",
        flush=True,
    )
    if args.rules == "zhang_full" and (args.zhang_full_cache_dir or "").strip():
        print(f"Zhang-full cache: {args.zhang_full_cache_dir}", flush=True)
    print(f"Per-class order: {F1_CLASS_NAMES}", flush=True)
    print(f"F1: {format_f1_metrics(metrics)}", flush=True)


if __name__ == "__main__":
    main()
