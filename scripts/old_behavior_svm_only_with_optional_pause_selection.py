#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare_rules_svm_hybrid import _block_pause_matrix, _path_to_block_vote, _pick_device
from disfluency_pipeline import Config, F1_CLASS_NAMES, build_split_lists, compute_f1_metrics
from paper_style_w2v2_svm_test import ClipRecord, extract_embeddings


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


@dataclass
class SVMHead:
    pipe: object


def _fit_svm_heads(x_tr: np.ndarray, y_tr: np.ndarray, c_value: float) -> List[SVMHead]:
    heads: List[SVMHead] = []
    for i in range(y_tr.shape[1]):
        p = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=float(c_value), gamma="scale", class_weight="balanced"),
        )
        p.fit(x_tr, y_tr[:, i])
        heads.append(SVMHead(pipe=p))
    return heads


def _tune_threshold_single(y_true: np.ndarray, scores: np.ndarray) -> float:
    yb = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int32)
    sc = np.asarray(scores).reshape(-1)
    qs = np.linspace(0.05, 0.95, 37)
    grid = np.unique(np.quantile(sc, qs))
    best_f1 = -1.0
    best_t = 0.0
    for t in grid:
        p = (sc >= t).astype(np.int32)
        f = f1_score(yb, p, zero_division=0)
        if f > best_f1:
            best_f1 = float(f)
            best_t = float(t)
    return best_t


def _tune_thresholds_scores(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    th = np.zeros(y_true.shape[1], dtype=np.float64)
    for i in range(y_true.shape[1]):
        th[i] = _tune_threshold_single(y_true[:, i], scores[:, i])
    return th


def _apply_thresholds(x: np.ndarray, th: np.ndarray) -> np.ndarray:
    return (x >= th.reshape(1, -1)).astype(np.float64)


def _svm_scores(heads: List[SVMHead], x: np.ndarray) -> np.ndarray:
    out = np.zeros((x.shape[0], len(heads)), dtype=np.float64)
    for i, h in enumerate(heads):
        out[:, i] = h.pipe.decision_function(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Old-behavior SVM-only tuner with optional learned pause selection for Block.")
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--label-vote-threshold", type=int, default=2)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--svm-c", type=float, default=10.0)
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--pause-cache-dir", default=".cache/block_pause_features")
    ap.add_argument("--refresh-block-pause-cache", action="store_true")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-learned-pause-selection", action="store_true")
    ap.add_argument("--block-np-quantile", type=float, default=0.8)
    ap.add_argument("--pause-low-vote-mode", choices=("le1", "eq1"), default="le1")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = _pick_device(args.device)

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

    tr_recs = _to_records(tp, tl, "train")
    dv_recs = _to_records(vp, vl, "dev")
    te_recs = _to_records(ep, el, "test")
    x_tr, y_tr = extract_embeddings(
        tr_recs, args.w2v2_name, args.layer, args.batch_size, dev, args.embedding_cache_dir.strip()
    )
    x_dv, y_dv = extract_embeddings(
        dv_recs, args.w2v2_name, args.layer, args.batch_size, dev, args.embedding_cache_dir.strip()
    )
    x_te, y_te = extract_embeddings(
        te_recs, args.w2v2_name, args.layer, args.batch_size, dev, args.embedding_cache_dir.strip()
    )

    heads = _fit_svm_heads(x_tr, y_tr, c_value=args.svm_c)
    dv_scores = _svm_scores(heads, x_dv)
    te_scores = _svm_scores(heads, x_te)
    th = _tune_thresholds_scores(y_dv, dv_scores)
    te_bin = _apply_thresholds(te_scores, th)
    m_base = compute_f1_metrics(y_te.astype(np.float64), te_bin.astype(np.float64), 0.5)

    out = {
        "seed": int(args.seed),
        "split_column": args.split_column,
        "data_root": args.data_root,
        "label_vote_threshold": int(args.label_vote_threshold),
        "svm_mode": "old_behavior_single_layer",
        "layer": int(args.layer),
        "svm_c": float(args.svm_c),
        "thresholds": {k: float(v) for k, v in zip(F1_CLASS_NAMES, th)},
        "svm_f1": {k: float(v) for k, v in m_base.items()},
    }

    if args.block_learned_pause_selection:
        yb_tr = (y_tr[:, 3] > 0.5).astype(np.int32)
        yb_dv = (y_dv[:, 3] > 0.5).astype(np.int32)
        votes_map = _path_to_block_vote(cfg.label_csv, cfg.data_root)
        vb_tr = np.array([int(votes_map.get(os.path.abspath(p), -1)) for p in tp], dtype=np.int32)
        p_tr = _block_pause_matrix(tp, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)

        if args.pause_low_vote_mode == "eq1":
            pos_for_scorer = np.flatnonzero((yb_tr == 0) & (vb_tr == 1))
        else:
            pos_for_scorer = np.flatnonzero((yb_tr == 0) & (vb_tr <= 1))
        neg_for_scorer = np.flatnonzero(yb_tr == 1)
        if len(pos_for_scorer) > 0 and len(neg_for_scorer) > 0:
            x_s = np.concatenate([p_tr[pos_for_scorer], p_tr[neg_for_scorer]], axis=0)
            y_s = np.concatenate(
                [np.ones(len(pos_for_scorer), dtype=np.int32), np.zeros(len(neg_for_scorer), dtype=np.int32)],
                axis=0,
            )
            pause_lr = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            )
            pause_lr.fit(x_s, y_s)
            pause_score = pause_lr.predict_proba(p_tr)[:, 1]

            fit_pos = np.flatnonzero(yb_tr == 1)
            fit_neg = np.flatnonzero(yb_tr == 0)
            if args.pause_low_vote_mode == "eq1":
                low_vote_neg = fit_neg[vb_tr[fit_neg] == 1]
            else:
                low_vote_neg = fit_neg[vb_tr[fit_neg] <= 1]
            if len(low_vote_neg) > 0:
                qthr = float(np.quantile(pause_score[low_vote_neg], float(args.block_np_quantile)))
                nat_pause_neg = low_vote_neg[pause_score[low_vote_neg] >= qthr]
            else:
                qthr = float("nan")
                nat_pause_neg = np.array([], dtype=np.int64)
            rng = np.random.default_rng(int(args.seed) + 808)
            n_bg = min(len(fit_neg), max(len(fit_pos), len(nat_pause_neg)))
            bg_neg = rng.choice(fit_neg, size=n_bg, replace=False) if n_bg > 0 else np.array([], dtype=np.int64)
            fit_sel = np.unique(np.concatenate([fit_pos, nat_pause_neg, bg_neg]))

            p = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=float(args.svm_c), gamma="scale", class_weight="balanced"),
            )
            p.fit(x_tr[fit_sel], yb_tr[fit_sel])
            block_thr = _tune_threshold_single(yb_dv, p.decision_function(x_dv))
            block_te = (p.decision_function(x_te) >= block_thr).astype(np.float64)

            te_bin_np = te_bin.copy()
            te_bin_np[:, 3] = block_te
            m_np = compute_f1_metrics(y_te.astype(np.float64), te_bin_np.astype(np.float64), 0.5)
            out["block_learned_pause_selection"] = {
                "enabled": True,
                "pause_low_vote_mode": args.pause_low_vote_mode,
                "block_np_quantile": float(args.block_np_quantile),
                "selection_counts": {
                    "fit_pos": int(len(fit_pos)),
                    "fit_neg": int(len(fit_neg)),
                    "low_vote_neg": int(len(low_vote_neg)),
                    "nat_pause_neg": int(len(nat_pause_neg)),
                    "bg_neg": int(len(bg_neg)),
                    "fit_sel": int(len(fit_sel)),
                },
                "q_threshold": qthr,
                "override_block_threshold": float(block_thr),
                "svm_f1_with_block_override": {k: float(v) for k, v in m_np.items()},
            }
        else:
            out["block_learned_pause_selection"] = {
                "enabled": False,
                "reason": "insufficient positives/negatives for pause scorer",
            }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(args.out_json)


if __name__ == "__main__":
    main()
