#!/usr/bin/env python3
"""
Bayerl-style W2V2 supervised fine-tuning baseline (single model, multi-label heads).

This trains a wav2vec2 encoder + linear 4-head classifier on SEP-28k train split,
selects best checkpoint by dev macro-F1, and reports test metrics.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from disfluency_pipeline import (
    Config,
    DisfluencyDataset,
    F1_CLASS_NAMES,
    build_split_lists,
    collate_batch,
    compute_f1_metrics,
    format_f1_metrics,
)


def _pick_device(which: str) -> torch.device:
    if which == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(which)


def _subsample(
    paths: List[str], labels: List[np.ndarray], cap: int, seed: int
) -> Tuple[List[str], List[np.ndarray]]:
    if cap <= 0 or len(paths) <= cap:
        return paths, labels
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    idx = idx[:cap]
    return [paths[i] for i in idx], [labels[i] for i in idx]


class W2V2Head(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name, local_files_only=True)
        self.w2v2 = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True)
        self.cls = nn.Linear(self.w2v2.config.hidden_size, 4)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B,1,T)
        wavs = []
        for w in waveforms:
            a = np.asarray(w[0].detach().cpu().numpy(), dtype=np.float32)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            wavs.append(a)
        ins = self.fe(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        ins = {k: v.to(waveforms.device) for k, v in ins.items()}
        out = self.w2v2(**ins)
        x = out.last_hidden_state.mean(dim=1)
        return self.cls(x)


@dataclass
class SplitLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _make_loaders(
    cfg: Config,
    batch_size: int,
    max_train: int,
    max_val: int,
    max_test: int,
    seed: int,
) -> SplitLoaders:
    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    tp, tl = _subsample(tp, tl, max_train, seed + 1)
    vp, vl = _subsample(vp, vl, max_val, seed + 2)
    ep, el = _subsample(ep, el, max_test, seed + 3)
    if not tp or not vp or not ep:
        raise RuntimeError("Need non-empty train/dev/test.")
    tr = DataLoader(DisfluencyDataset(cfg, tp, tl), batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    va = DataLoader(DisfluencyDataset(cfg, vp, vl), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    te = DataLoader(DisfluencyDataset(cfg, ep, el), batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    return SplitLoaders(train=tr, val=va, test=te)


def _eval(model: nn.Module, dl: DataLoader, dev: torch.device, thr: float) -> dict:
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for b in dl:
            w = b["waveform"].to(dev)
            y = b["labels"].cpu().numpy()
            logits = model(w)
            probs = torch.sigmoid(logits).cpu().numpy()
            ys.append(y)
            ps.append(probs)
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return compute_f1_metrics(y, p, thr)


def _macro(m: dict) -> float:
    return float(np.mean([m[k] for k in F1_CLASS_NAMES]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune W2V2 + 4-head classifier.")
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument(
        "--label-vote-threshold",
        type=int,
        default=3,
        help="SEP-28k-Extended: present when vote count >= this (3 = unanimous).",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--model-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--f1-threshold", type=float, default=0.5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/w2v2_bayerl_style.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = _pick_device(args.device)
    print(f"Using device: {dev}", flush=True)

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))

    ldrs = _make_loaders(
        cfg, args.batch_size, args.max_train, args.max_val, args.max_test, args.seed
    )
    print(
        f"Loaders: train={len(ldrs.train.dataset)} dev={len(ldrs.val.dataset)} test={len(ldrs.test.dataset)}",
        flush=True,
    )

    model = W2V2Head(args.model_name).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_macro = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for b in tqdm(ldrs.train, desc=f"Train epoch {ep}", unit="batch"):
            w = b["waveform"].to(dev)
            y = b["labels"].to(dev).float()
            opt.zero_grad(set_to_none=True)
            z = model(w)
            z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss_fn(z, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        m_val = _eval(model, ldrs.val, dev, args.f1_threshold)
        val_macro = _macro(m_val)
        print(
            f"Epoch {ep}: train_loss={np.mean(losses):.4f}  dev_macro={val_macro:.4f}  dev={format_f1_metrics(m_val)}",
            flush=True,
        )
        if val_macro > best_macro:
            best_macro = val_macro
            torch.save({"model": model.state_dict(), "dev_macro": best_macro}, args.out)
            print(f"[Saved best] {args.out}", flush=True)

    ck = torch.load(args.out, map_location=dev)
    model.load_state_dict(ck["model"])
    m_test = _eval(model, ldrs.test, dev, args.f1_threshold)
    print(f"Best dev macro={best_macro:.4f}", flush=True)
    print(f"Test: {format_f1_metrics(m_test)}", flush=True)
    print(f"Test macro={_macro(m_test):.4f}", flush=True)


if __name__ == "__main__":
    main()
