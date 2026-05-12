#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare_rules_svm_hybrid import _balanced_row_indices_y_bin, _block_pause_matrix, _path_to_block_vote, _pick_device
from disfluency_pipeline import Config, F1_CLASS_NAMES, build_split_lists, compute_f1_metrics
from paper_style_w2v2_svm_test import ClipRecord, extract_per_head_w2v2_prosody, load_svm_clean03_head_bundle


def _subsample(
    paths: List[str], labels: List[np.ndarray], cap: int, seed: int
) -> Tuple[List[str], List[np.ndarray]]:
    if cap <= 0 or len(paths) <= cap:
        return list(paths), list(labels)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    idx = idx[:cap]
    return [paths[int(i)] for i in idx], [labels[int(i)] for i in idx]


def _to_records(paths: Sequence[str], labels: Sequence[np.ndarray], split: str) -> List[ClipRecord]:
    out: List[ClipRecord] = []
    for p, y in zip(paths, labels):
        out.append(
            ClipRecord(path=p, split=split, labels=(np.asarray(y).reshape(-1) > 0.5).astype(np.int32))
        )
    return out


def _tune_threshold_like_tune_script(y_true: np.ndarray, scores: np.ndarray) -> float:
    yb = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int32)
    sc = np.asarray(scores).reshape(-1)
    qs = np.linspace(0.05, 0.95, 37)
    grid = np.unique(np.quantile(sc, qs))
    best_f1 = -1.0
    best_t = 0.0
    for t in grid:
        p = (sc >= t).astype(np.int32)
        tp = float(np.sum((p == 1) & (yb == 1)))
        fp = float(np.sum((p == 1) & (yb == 0)))
        fn = float(np.sum((p == 0) & (yb == 1)))
        denom = (2.0 * tp) + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0.0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _fit_svm_binary(x: np.ndarray, y: np.ndarray, c_value: float, seed: int, balance: bool) -> object:
    yi = y.astype(np.int32).reshape(-1)
    if balance:
        idx = _balanced_row_indices_y_bin(yi, seed)
        x_fit = x[idx]
        y_fit = yi[idx]
    else:
        x_fit = x
        y_fit = yi
    m = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(c_value), gamma="scale", class_weight="balanced"),
    )
    m.fit(x_fit, y_fit)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict train/dev/test 4-head eval + learned-pause Block override.")
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/SEP-28k_CLIP")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--label-vote-threshold", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--svm-head-config-json", default="artifacts/svm_clean03_best_configs_full.json")
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--prosody-cache-dir", default=".cache/prosody_features")
    ap.add_argument("--pause-cache-dir", default=".cache/block_pause_features")
    ap.add_argument("--refresh-prosody-cache", action="store_true")
    ap.add_argument("--refresh-block-pause-cache", action="store_true")
    ap.add_argument("--no-balance-svm-train", action="store_true")
    ap.add_argument("--block-np-quantile", type=float, default=0.8)
    ap.add_argument("--pause-low-vote-mode", choices=("le1", "eq1"), default="le1")
    ap.add_argument("--max-train", type=int, default=0, help="0 = use all train clips (else random subsample cap).")
    ap.add_argument("--max-dev", type=int, default=0, help="0 = use all dev clips.")
    ap.add_argument("--max-test", type=int, default=0, help="0 = use all test clips.")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = _pick_device(args.device)

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = int(args.label_vote_threshold)
    cfg.seed = int(args.seed)

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    tp, tl = _subsample(tp, tl, int(args.max_train), int(args.seed) + 1)
    vp, vl = _subsample(vp, vl, int(args.max_dev), int(args.seed) + 2)
    ep, el = _subsample(ep, el, int(args.max_test), int(args.seed) + 3)
    tr_recs = _to_records(tp, tl, "train")
    dv_recs = _to_records(vp, vl, "dev")
    te_recs = _to_records(ep, el, "test")

    layers_t, c_t, _ = load_svm_clean03_head_bundle(args.svm_head_config_json)
    c_list = list(c_t)

    x_tr_h, y_tr = extract_per_head_w2v2_prosody(
        tr_recs, args.w2v2_name, layers_t, args.batch_size, dev, args.embedding_cache_dir.strip(),
        cfg.sample_rate, args.prosody_cache_dir.strip(), args.refresh_prosody_cache
    )
    x_dv_h, y_dv = extract_per_head_w2v2_prosody(
        dv_recs, args.w2v2_name, layers_t, args.batch_size, dev, args.embedding_cache_dir.strip(),
        cfg.sample_rate, args.prosody_cache_dir.strip(), args.refresh_prosody_cache
    )
    x_te_h, y_te = extract_per_head_w2v2_prosody(
        te_recs, args.w2v2_name, layers_t, args.batch_size, dev, args.embedding_cache_dir.strip(),
        cfg.sample_rate, args.prosody_cache_dir.strip(), args.refresh_prosody_cache
    )

    # Baseline 4-head strict
    pred_te = np.zeros((len(ep), 4), dtype=np.float64)
    th = np.zeros(4, dtype=np.float64)
    for i in range(4):
        yi_tr = (y_tr[:, i] > 0.5).astype(np.int32)
        yi_dv = (y_dv[:, i] > 0.5).astype(np.int32)
        m = _fit_svm_binary(
            x_tr_h[i], yi_tr, c_list[i], seed=int(args.seed) + 100_003 * i, balance=not args.no_balance_svm_train
        )
        sc_dv = m.decision_function(x_dv_h[i])
        t = _tune_threshold_like_tune_script(yi_dv, sc_dv)
        th[i] = t
        pred_te[:, i] = (m.decision_function(x_te_h[i]) >= t).astype(np.float64)

    y_true = y_te.astype(np.float64)
    m_base = compute_f1_metrics(y_true, pred_te, 0.5)

    # Learned-pause Block override (strict)
    yb_tr = (y_tr[:, 3] > 0.5).astype(np.int32)
    yb_dv = (y_dv[:, 3] > 0.5).astype(np.int32)
    votes_map = _path_to_block_vote(cfg.label_csv, cfg.data_root)
    vb_tr = np.array([int(votes_map.get(os.path.abspath(p), -1)) for p in tp], dtype=np.int32)
    p_tr = _block_pause_matrix(tp, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)
    p_dv = _block_pause_matrix(vp, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)
    p_te = _block_pause_matrix(ep, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)
    xb_tr = np.concatenate([x_tr_h[3], p_tr], axis=1)
    xb_dv = np.concatenate([x_dv_h[3], p_dv], axis=1)
    xb_te = np.concatenate([x_te_h[3], p_te], axis=1)

    if args.pause_low_vote_mode == "eq1":
        pos_for_scorer = np.flatnonzero((yb_tr == 0) & (vb_tr == 1))
    else:
        pos_for_scorer = np.flatnonzero((yb_tr == 0) & (vb_tr <= 1))
    neg_for_scorer = np.flatnonzero(yb_tr == 1)
    x_s = np.concatenate([p_tr[pos_for_scorer], p_tr[neg_for_scorer]], axis=0)
    y_s = np.concatenate([np.ones(len(pos_for_scorer), dtype=np.int32), np.zeros(len(neg_for_scorer), dtype=np.int32)], axis=0)
    pause_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))
    pause_lr.fit(x_s, y_s)
    pause_score = pause_lr.predict_proba(p_tr)[:, 1]

    fit_pos = np.flatnonzero(yb_tr == 1)
    fit_neg = np.flatnonzero(yb_tr == 0)
    if args.pause_low_vote_mode == "eq1":
        low_vote_neg = fit_neg[vb_tr[fit_neg] == 1]
    else:
        low_vote_neg = fit_neg[vb_tr[fit_neg] <= 1]
    qthr = float(np.quantile(pause_score[low_vote_neg], float(args.block_np_quantile)))
    nat_pause_neg = low_vote_neg[pause_score[low_vote_neg] >= qthr]
    rng = np.random.default_rng(int(args.seed) + 808)
    n_bg = min(len(fit_neg), max(len(fit_pos), len(nat_pause_neg)))
    bg_neg = rng.choice(fit_neg, size=n_bg, replace=False) if n_bg > 0 else np.array([], dtype=np.int64)
    fit_sel = np.unique(np.concatenate([fit_pos, nat_pause_neg, bg_neg]))

    m_block = _fit_svm_binary(xb_tr[fit_sel], yb_tr[fit_sel], c_list[3], seed=int(args.seed) + 404_041, balance=False)
    t_block = _tune_threshold_like_tune_script(yb_dv, m_block.decision_function(xb_dv))
    pred_block_te = (m_block.decision_function(xb_te) >= t_block).astype(np.float64)

    pred_te_mix = pred_te.copy()
    pred_te_mix[:, 3] = pred_block_te
    m_mix = compute_f1_metrics(y_true, pred_te_mix, 0.5)

    out = {
        "seed": int(args.seed),
        "split_column": args.split_column,
        "data_root": args.data_root,
        "protocol": "strict_train_dev_test",
        "baseline_4head_svm_f1": {k: float(v) for k, v in m_base.items()},
        "with_learned_pause_block_override_f1": {k: float(v) for k, v in m_mix.items()},
        "delta_override_minus_base": {k: float(m_mix[k] - m_base[k]) for k in m_base.keys()},
        "thresholds": {
            "baseline_per_head": {name: float(th[i]) for i, name in enumerate(F1_CLASS_NAMES)},
            "override_block": float(t_block),
        },
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(args.out_json)
    print("baseline:", out["baseline_4head_svm_f1"])
    print("override:", out["with_learned_pause_block_override_f1"])


if __name__ == "__main__":
    main()

