#!/usr/bin/env python3
"""
Tune rule and SVM thresholds on dev, then evaluate test.

Protocol:
- Labels/splits from disfluency_pipeline (4 heads, SEP28k-T by default).
- Train SVM on train split only.
- Tune per-head thresholds on dev:
  - Rules: threshold over sigmoid probs in [0.05..0.95]
  - SVM: threshold over decision_function scores (quantile grid from dev scores)
- Report test metrics with tuned thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from disfluency_pipeline import (
    Config,
    DisfluencyDataset,
    F1_CLASS_NAMES,
    build_split_lists,
    collate_batch,
    compute_f1_metrics,
    format_f1_metrics,
)
from paper_style_w2v2_svm_test import ClipRecord, extract_embeddings
from zhang_full.rule_module import ZhangFullRuleLogits
from zhang_rules import ZhangStyleRuleLogits


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


def _to_records(paths: Sequence[str], labels: Sequence[np.ndarray], split: str) -> List[ClipRecord]:
    out: List[ClipRecord] = []
    for p, y in zip(paths, labels):
        out.append(
            ClipRecord(
                path=p,
                split=split,
                labels=(np.asarray(y).reshape(-1) > 0.5).astype(np.int32),
            )
        )
    return out


def _collect_rule_probs(
    cfg: Config,
    paths: Sequence[str],
    labels: Sequence[np.ndarray],
    rule_mod: torch.nn.Module,
    dev: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = DisfluencyDataset(cfg, list(paths), list(labels))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    chunks: List[np.ndarray] = []
    it = tqdm(dl, desc="Rules", unit="batch")
    with torch.no_grad():
        for b in it:
            w = b["waveform"].to(dev)
            logits = rule_mod(w, b.get("paths"))
            chunks.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@dataclass
class SVMHead:
    pipe: object


def _fit_svm_heads(x_tr: np.ndarray, y_tr: np.ndarray) -> List[SVMHead]:
    heads: List[SVMHead] = []
    for i in range(y_tr.shape[1]):
        p = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
        )
        p.fit(x_tr, y_tr[:, i])
        heads.append(SVMHead(pipe=p))
    return heads


def _svm_scores(heads: List[SVMHead], x: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], len(heads)), dtype=np.float64)
    for i, h in enumerate(heads):
        out[:, i] = h.pipe.decision_function(x)
    return out


def _tune_thresholds_probs(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 37)
    th = np.zeros(y_true.shape[1], dtype=np.float64)
    yb = (y_true > 0.5).astype(np.int32)
    for i in range(y_true.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            p = (probs[:, i] >= t).astype(np.int32)
            f = f1_score(yb[:, i], p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        th[i] = best_t
    return th


def _tune_thresholds_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    th = np.zeros(y_true.shape[1], dtype=np.float64)
    yb = (y_true > 0.5).astype(np.int32)
    for i in range(y_true.shape[1]):
        qs = np.linspace(0.05, 0.95, 37)
        grid = np.unique(np.quantile(scores[:, i], qs))
        best_f1 = -1.0
        best_t = 0.0
        for t in grid:
            p = (scores[:, i] >= t).astype(np.int32)
            f = f1_score(yb[:, i], p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)
        th[i] = best_t
    return th


def _apply_thresholds(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    return (x >= th.reshape(1, -1)).astype(np.float64)


def _fmt_head_thresholds(th: np.ndarray) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in zip(F1_CLASS_NAMES, th))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune rules/SVM thresholds on dev.")
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--label-vote-threshold", type=int, default=2)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--zhang-full-cache-dir", default="")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--thresholds-out",
        default="artifacts/tuned_thresholds_rules_svm.json",
        help="Path to write tuned thresholds JSON.",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = _pick_device(args.device)
    print(f"Using device: {dev}", flush=True)

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    tp, tl = _subsample(tp, tl, args.max_train, args.seed + 1)
    vp, vl = _subsample(vp, vl, args.max_dev, args.seed + 2)
    ep, el = _subsample(ep, el, args.max_test, args.seed + 3)

    if not tp or not vp or not ep:
        raise RuntimeError("Need non-empty train/dev/test after caps.")

    print(f"Counts: train={len(tp)} dev={len(vp)} test={len(ep)}", flush=True)

    # SVM side
    tr_recs = _to_records(tp, tl, "train")
    dv_recs = _to_records(vp, vl, "val")
    te_recs = _to_records(ep, el, "test")
    x_tr, y_tr = extract_embeddings(
        tr_recs,
        args.w2v2_name,
        args.layer,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
    )
    x_dv, y_dv = extract_embeddings(
        dv_recs,
        args.w2v2_name,
        args.layer,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
    )
    x_te, y_te = extract_embeddings(
        te_recs,
        args.w2v2_name,
        args.layer,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
    )
    svm_heads = _fit_svm_heads(x_tr, y_tr)
    svm_dv_scores = _svm_scores(svm_heads, x_dv)
    svm_te_scores = _svm_scores(svm_heads, x_te)
    svm_th = _tune_thresholds_scores(y_dv, svm_dv_scores)
    svm_te_bin = _apply_thresholds(svm_te_scores, svm_th)
    m_svm = compute_f1_metrics(y_te.astype(np.float64), svm_te_bin.astype(np.float64), 0.5)

    # Rules side (zhang and zhang_full)
    zhang_mod = ZhangStyleRuleLogits(sample_rate=cfg.sample_rate).to(dev).eval()
    zfull_mod = ZhangFullRuleLogits(
        sample_rate=cfg.sample_rate,
        cache_dir=(args.zhang_full_cache_dir.strip() or None),
    ).to(dev).eval()

    z_dv = _collect_rule_probs(cfg, vp, vl, zhang_mod, dev, args.batch_size)
    z_te = _collect_rule_probs(cfg, ep, el, zhang_mod, dev, args.batch_size)
    z_th = _tune_thresholds_probs(np.asarray(vl), z_dv)
    z_te_bin = _apply_thresholds(z_te, z_th)
    m_zhang = compute_f1_metrics(np.asarray(el), z_te_bin.astype(np.float64), 0.5)

    zf_dv = _collect_rule_probs(cfg, vp, vl, zfull_mod, dev, args.batch_size)
    zf_te = _collect_rule_probs(cfg, ep, el, zfull_mod, dev, args.batch_size)
    zf_th = _tune_thresholds_probs(np.asarray(vl), zf_dv)
    zf_te_bin = _apply_thresholds(zf_te, zf_th)
    m_zfull = compute_f1_metrics(np.asarray(el), zf_te_bin.astype(np.float64), 0.5)

    print("\nTuned thresholds (per head):", flush=True)
    print(f"SVM score threshold:        {_fmt_head_thresholds(svm_th)}", flush=True)
    print(f"Rules zhang prob threshold: {_fmt_head_thresholds(z_th)}", flush=True)
    print(f"Rules zfull prob threshold: {_fmt_head_thresholds(zf_th)}", flush=True)

    print("\nTest metrics with tuned thresholds:", flush=True)
    print(f"SVM (tuned):        {format_f1_metrics(m_svm)}", flush=True)
    print(f"Rules zhang (tuned):{format_f1_metrics(m_zhang)}", flush=True)
    print(f"Rules zfull (tuned):{format_f1_metrics(m_zfull)}", flush=True)

    out_fp = (args.thresholds_out or "").strip()
    if out_fp:
        out_dir = os.path.dirname(out_fp)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "split_column": args.split_column,
            "label_vote_threshold": int(args.label_vote_threshold),
            "seed": int(args.seed),
            "heads": list(F1_CLASS_NAMES),
            "svm_score_thresholds": {
                k: float(v) for k, v in zip(F1_CLASS_NAMES, svm_th)
            },
            "rules_zhang_prob_thresholds": {
                k: float(v) for k, v in zip(F1_CLASS_NAMES, z_th)
            },
            "rules_zhang_full_prob_thresholds": {
                k: float(v) for k, v in zip(F1_CLASS_NAMES, zf_th)
            },
        }
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved thresholds: {out_fp}", flush=True)


if __name__ == "__main__":
    main()
