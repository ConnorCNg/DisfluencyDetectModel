#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from compare_rules_svm_hybrid import (  # noqa: E402
    _balanced_row_indices_y_bin,
    _block_pause_matrix,
    _path_to_block_vote,
    _pick_device,
)
from disfluency_pipeline import Config, build_split_lists  # noqa: E402
from paper_style_w2v2_svm_test import (  # noqa: E402
    ClipRecord,
    extract_per_head_w2v2_prosody,
    load_svm_clean03_head_bundle,
)


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


def _to_records(paths: List[str], labels: List[np.ndarray], split: str) -> List[ClipRecord]:
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


def _bin_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(
        y_true.astype(np.int32),
        y_pred.astype(np.int32),
        average="binary",
        zero_division=0,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def _tune_threshold_single_like_tune_script(y_true: np.ndarray, scores: np.ndarray) -> float:
    # Match tune_thresholds_rules_svm.py behavior for score thresholds.
    yb = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int32)
    sc = np.asarray(scores).reshape(-1)
    qs = np.linspace(0.05, 0.95, 37)
    grid = np.unique(np.quantile(sc, qs))
    best_f1 = -1.0
    best_t = 0.0
    for t in grid:
        p = (sc >= t).astype(np.int32)
        f = precision_recall_fscore_support(
            yb,
            p,
            average="binary",
            zero_division=0,
        )[2]
        if float(f) > best_f1:
            best_f1 = float(f)
            best_t = float(t)
    return best_t


def _fit_block_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    seed: int,
    balance_train: bool,
) -> object:
    yi = y_train.astype(np.int32).reshape(-1)
    if balance_train:
        idx = _balanced_row_indices_y_bin(yi, seed)
        x_fit = x_train[idx]
        y_fit = yi[idx]
    else:
        x_fit = x_train
        y_fit = yi
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(c_value), gamma="scale", class_weight="balanced"),
    )
    model.fit(x_fit, y_fit)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Block-only strict train/dev/test pipeline.")
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
    ap.add_argument("--refresh-prosody-cache", action="store_true")
    ap.add_argument("--pause-cache-dir", default=".cache/block_pause_features")
    ap.add_argument("--refresh-block-pause-cache", action="store_true")
    ap.add_argument("--no-balance-svm-train", action="store_true")
    ap.add_argument("--block-np-quantile", type=float, default=0.8)
    ap.add_argument("--block-np-c", type=float, default=0.1)
    ap.add_argument(
        "--block-np-balance-total-neg-to-pos",
        action="store_true",
        help=(
            "For Block natural-v2 fit set, enforce total negatives "
            "(natural-pause + background) ~= number of positives."
        ),
    )
    ap.add_argument(
        "--learned-pause-scorer",
        action="store_true",
        help=(
            "Replace heuristic pause score with learned linear scorer: fit logistic regression "
            "on train pause features (positive=low-vote non-block, negative=block positive)."
        ),
    )
    ap.add_argument(
        "--pause-low-vote-mode",
        choices=("le1", "eq1"),
        default="le1",
        help="Define low-vote non-block pool as vote<=1 (le1) or vote==1 (eq1).",
    )
    ap.add_argument(
        "--include-ri-negatives",
        action="store_true",
        help=(
            "Force inclusion of train clips with Repetition or Interjection label=1 and Block label=0 "
            "in the Block-v2 negative fit set."
        ),
    )
    ap.add_argument(
        "--out-json",
        default="artifacts/error_analysis/block_only_train_dev_test_seed42_3s.json",
    )
    ap.add_argument("--max-train", type=int, default=0, help="0 = use all train clips (else random subsample cap).")
    ap.add_argument("--max-dev", type=int, default=0, help="0 = use all dev clips.")
    ap.add_argument("--max-test", type=int, default=0, help="0 = use all test clips.")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = _pick_device(args.device)

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))
    cfg.seed = int(args.seed)

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    tp, tl = _subsample(tp, tl, int(args.max_train), int(args.seed) + 1)
    vp, vl = _subsample(vp, vl, int(args.max_dev), int(args.seed) + 2)
    ep, el = _subsample(ep, el, int(args.max_test), int(args.seed) + 3)
    if not tp or not vp or not ep:
        raise RuntimeError("Need non-empty train/dev/test splits.")

    tr_recs = _to_records(tp, tl, "train")
    dv_recs = _to_records(vp, vl, "val")
    te_recs = _to_records(ep, el, "test")

    layers_t, c_t, _ = load_svm_clean03_head_bundle(args.svm_head_config_json)
    c_block_from_bundle = float(c_t[3])

    x_tr_h, y_tr = extract_per_head_w2v2_prosody(
        tr_recs,
        args.w2v2_name,
        layers_t,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
        cfg.sample_rate,
        args.prosody_cache_dir.strip(),
        args.refresh_prosody_cache,
    )
    x_dv_h, y_dv = extract_per_head_w2v2_prosody(
        dv_recs,
        args.w2v2_name,
        layers_t,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
        cfg.sample_rate,
        args.prosody_cache_dir.strip(),
        args.refresh_prosody_cache,
    )
    x_te_h, y_te = extract_per_head_w2v2_prosody(
        te_recs,
        args.w2v2_name,
        layers_t,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
        cfg.sample_rate,
        args.prosody_cache_dir.strip(),
        args.refresh_prosody_cache,
    )

    xb_tr = x_tr_h[3]
    xb_dv = x_dv_h[3]
    xb_te = x_te_h[3]
    yb_tr = (y_tr[:, 3] > 0.5).astype(np.int32)
    yb_dv = (y_dv[:, 3] > 0.5).astype(np.int32)
    yb_te = (y_te[:, 3] > 0.5).astype(np.int32)

    # A) Base block SVM (strict train/dev/test)
    base_model = _fit_block_svm(
        xb_tr,
        yb_tr,
        c_block_from_bundle,
        seed=int(args.seed) + 300_009,
        balance_train=not args.no_balance_svm_train,
    )
    base_scores_dv = base_model.decision_function(xb_dv)
    base_thr = _tune_threshold_single_like_tune_script(yb_dv, base_scores_dv)
    base_pred_te = (base_model.decision_function(xb_te) >= base_thr).astype(np.int32)
    base_metrics = _bin_metrics(yb_te, base_pred_te)
    base_metrics["threshold"] = float(base_thr)

    # B) Block natural-v2 (strict train/dev/test; no internal split)
    votes_map = _path_to_block_vote(cfg.label_csv, cfg.data_root)
    vb_tr = np.array([int(votes_map.get(os.path.abspath(p), -1)) for p in tp], dtype=np.int32)

    p_tr = _block_pause_matrix(tp, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)
    p_dv = _block_pause_matrix(vp, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)
    p_te = _block_pause_matrix(ep, cfg.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache)
    x2_tr = np.concatenate([xb_tr, p_tr], axis=1)
    x2_dv = np.concatenate([xb_dv, p_dv], axis=1)
    x2_te = np.concatenate([xb_te, p_te], axis=1)

    idx = np.arange(len(yb_tr))
    fit_pos = idx[yb_tr == 1]
    fit_neg = idx[yb_tr == 0]
    if args.pause_low_vote_mode == "eq1":
        low_vote_neg = fit_neg[vb_tr[fit_neg] == 1]
    else:
        low_vote_neg = fit_neg[vb_tr[fit_neg] <= 1]
    base_neg_pool = low_vote_neg if low_vote_neg.size > 0 else fit_neg
    pause_scorer_info: Dict[str, object] = {"mode": "heuristic"}
    if args.learned_pause_scorer:
        # Fit pause scorer on train only: + class is low-vote non-block, - class is block positive.
        if args.pause_low_vote_mode == "eq1":
            low_vote_neg_all = np.flatnonzero((yb_tr == 0) & (vb_tr == 1))
        else:
            low_vote_neg_all = np.flatnonzero((yb_tr == 0) & (vb_tr <= 1))
        pos_for_scorer = low_vote_neg_all
        neg_for_scorer = np.flatnonzero(yb_tr == 1)
        if len(pos_for_scorer) == 0 or len(neg_for_scorer) == 0:
            pause_score = p_tr[:, 0] + 0.15 * p_tr[:, 1] + 0.35 * p_tr[:, 2] - 0.01 * p_tr[:, 3]
            pause_scorer_info["fallback"] = "heuristic_due_to_empty_class"
        else:
            x_s = np.concatenate([p_tr[pos_for_scorer], p_tr[neg_for_scorer]], axis=0)
            y_s = np.concatenate(
                [
                    np.ones(len(pos_for_scorer), dtype=np.int32),
                    np.zeros(len(neg_for_scorer), dtype=np.int32),
                ],
                axis=0,
            )
            pause_lr = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
            )
            pause_lr.fit(x_s, y_s)
            pause_score = pause_lr.predict_proba(p_tr)[:, 1]
            coef = pause_lr.named_steps["logisticregression"].coef_[0].tolist()
            intercept = float(pause_lr.named_steps["logisticregression"].intercept_[0])
            pause_scorer_info = {
                "mode": "learned_logreg",
                "train_pos_low_vote_nonblock": int(len(pos_for_scorer)),
                "train_neg_block_positive": int(len(neg_for_scorer)),
                "coef_order": ["silence_ratio", "pause_count", "long_pause_count", "rms_mean_db_rel"],
                "coef": [float(v) for v in coef],
                "intercept": intercept,
            }
    else:
        pause_score = p_tr[:, 0] + 0.15 * p_tr[:, 1] + 0.35 * p_tr[:, 2] - 0.01 * p_tr[:, 3]
    qthr = float(np.quantile(pause_score[base_neg_pool], float(args.block_np_quantile)))
    nat_pause_neg = base_neg_pool[pause_score[base_neg_pool] >= qthr]

    rng = np.random.default_rng(int(args.seed) + 808)
    if args.block_np_balance_total_neg_to_pos:
        # Enforce total negatives ~= positives in fit selection.
        n_pos_target = int(len(fit_pos))
        if n_pos_target <= 0:
            sel_nat = np.array([], dtype=np.int64)
            bg_neg = np.array([], dtype=np.int64)
        else:
            if len(nat_pause_neg) >= n_pos_target:
                sel_nat = rng.choice(nat_pause_neg, size=n_pos_target, replace=False)
                bg_neg = np.array([], dtype=np.int64)
            else:
                sel_nat = nat_pause_neg
                n_bg = n_pos_target - len(sel_nat)
                bg_pool = np.setdiff1d(fit_neg, sel_nat, assume_unique=False)
                n_bg = min(int(n_bg), int(len(bg_pool)))
                bg_neg = (
                    rng.choice(bg_pool, size=n_bg, replace=False)
                    if n_bg > 0
                    else np.array([], dtype=np.int64)
                )
    else:
        n_bg = min(len(fit_neg), max(len(fit_pos), len(nat_pause_neg)))
        bg_neg = rng.choice(fit_neg, size=n_bg, replace=False) if n_bg > 0 else np.array([], dtype=np.int64)
        sel_nat = nat_pause_neg
    fit_sel = np.unique(np.concatenate([fit_pos, sel_nat, bg_neg]))

    ri_neg = np.array([], dtype=np.int64)
    if args.include_ri_negatives:
        rep_or_int = ((y_tr[:, 1] > 0.5) | (y_tr[:, 2] > 0.5)).astype(np.int32)
        ri_neg = np.flatnonzero((yb_tr == 0) & (rep_or_int == 1)).astype(np.int64)
        if ri_neg.size > 0:
            fit_sel = np.unique(np.concatenate([fit_sel, ri_neg]))

    np_model = _fit_block_svm(
        x2_tr[fit_sel],
        yb_tr[fit_sel],
        float(args.block_np_c),
        seed=int(args.seed) + 404_041,
        balance_train=False,
    )
    np_scores_dv = np_model.decision_function(x2_dv)
    np_thr = _tune_threshold_single_like_tune_script(yb_dv, np_scores_dv)
    np_pred_te = (np_model.decision_function(x2_te) >= np_thr).astype(np.int32)
    np_metrics = _bin_metrics(yb_te, np_pred_te)
    np_metrics["threshold"] = float(np_thr)

    out = {
        "config": {
            "seed": int(args.seed),
            "data_root": args.data_root,
            "split_column": args.split_column,
            "label_vote_threshold": int(args.label_vote_threshold),
            "balance_svm_train": bool(not args.no_balance_svm_train),
            "block_layer_from_bundle": int(layers_t[3]),
            "block_C_from_bundle": float(c_block_from_bundle),
            "block_np_quantile": float(args.block_np_quantile),
            "block_np_C": float(args.block_np_c),
            "pause_low_vote_mode": str(args.pause_low_vote_mode),
        },
        "counts": {
            "train": int(len(tp)),
            "dev": int(len(vp)),
            "test": int(len(ep)),
            "train_block_pos": int(yb_tr.sum()),
            "dev_block_pos": int(yb_dv.sum()),
            "test_block_pos": int(yb_te.sum()),
            "natural_pause_neg_pool": int(len(base_neg_pool)),
            "natural_pause_neg_selected": int(len(nat_pause_neg)),
            "natural_pause_neg_used_in_fit": int(len(sel_nat)),
            "fit_pos": int(len(fit_pos)),
            "fit_bg_neg": int(len(bg_neg)),
            "fit_ri_neg_included": int(len(ri_neg)),
            "fit_sel_total": int(len(fit_sel)),
        },
        "results": {
            "block_base_train_dev_test": base_metrics,
            "block_natural_v2_train_dev_test": np_metrics,
            "delta_natural_minus_base": {
                "f1": float(np_metrics["f1"] - base_metrics["f1"]),
                "precision": float(np_metrics["precision"] - base_metrics["precision"]),
                "recall": float(np_metrics["recall"] - base_metrics["recall"]),
            },
        },
        "pause_scorer": pause_scorer_info,
    }

    out_fp = args.out_json.strip()
    os.makedirs(os.path.dirname(out_fp) or ".", exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(out_fp)
    print(
        "Block-only strict train/dev/test\n"
        f"base:      F1={base_metrics['f1']:.4f} P={base_metrics['precision']:.4f} R={base_metrics['recall']:.4f} thr={base_metrics['threshold']:.4f}\n"
        f"naturalv2: F1={np_metrics['f1']:.4f} P={np_metrics['precision']:.4f} R={np_metrics['recall']:.4f} thr={np_metrics['threshold']:.4f}\n"
        f"delta:     F1={out['results']['delta_natural_minus_base']['f1']:+.4f} "
        f"P={out['results']['delta_natural_minus_base']['precision']:+.4f} "
        f"R={out['results']['delta_natural_minus_base']['recall']:+.4f}"
    )


if __name__ == "__main__":
    main()

