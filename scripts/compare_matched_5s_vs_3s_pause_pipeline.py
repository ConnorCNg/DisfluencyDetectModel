#!/usr/bin/env python3
"""
Matched-pair 5s vs 3s evaluation: same clip IDs / splits / labels; only audio root differs.

Uses the same protocol as scripts/old_behavior_svm_only_with_optional_pause_selection.py:
- frozen Wav2Vec2, single layer (default 8), mean-pooled 768-D
- one SVM per head, C fixed
- dev threshold tuning (quantile grid 5%..95%)
- optional learned-pause-based Block training row selection (same as old_behavior script)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import soundfile as sf
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
from paper_style_w2v2_svm_test import extract_embeddings

_ob_path = os.path.join(REPO_ROOT, "scripts", "old_behavior_svm_only_with_optional_pause_selection.py")
_spec = importlib.util.spec_from_file_location("old_behavior_svm_only", _ob_path)
_ob = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["old_behavior_svm_only"] = _ob
_spec.loader.exec_module(_ob)
_apply_thresholds = _ob._apply_thresholds
_fit_svm_heads = _ob._fit_svm_heads
_svm_scores = _ob._svm_scores
_to_records = _ob._to_records
_tune_threshold_single = _ob._tune_threshold_single
_tune_thresholds_scores = _ob._tune_thresholds_scores


def _is_about_5s(path: str) -> bool:
    info = sf.info(path)
    dur = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
    return 4.5 <= dur < 5.5


def _map_clips_to_3s_path(path_5s: str, clips_root: str, clip3_root: str) -> str:
    rel = os.path.relpath(os.path.abspath(path_5s), os.path.abspath(clips_root))
    return os.path.join(os.path.abspath(clip3_root), rel)


def _filter_matched_pairs(
    paths: List[str],
    labels: List[np.ndarray],
    clips_root: str,
    clip3_root: str,
) -> Tuple[List[str], List[str], List[np.ndarray]]:
    p5: List[str] = []
    p3: List[str] = []
    lab: List[np.ndarray] = []
    for p, y in zip(paths, labels):
        if not _is_about_5s(p):
            continue
        q = _map_clips_to_3s_path(p, clips_root, clip3_root)
        if not os.path.isfile(q):
            continue
        p5.append(p)
        p3.append(q)
        lab.append(y)
    return p5, p3, lab


def _subsample_matched(
    p5: List[str],
    p3: List[str],
    lab: List[np.ndarray],
    cap: int,
    seed: int,
) -> Tuple[List[str], List[str], List[np.ndarray]]:
    if cap <= 0 or len(p5) <= cap:
        return p5, p3, lab
    rng = np.random.default_rng(seed)
    idx = np.arange(len(p5))
    rng.shuffle(idx)
    idx = idx[:cap]
    return [p5[int(i)] for i in idx], [p3[int(i)] for i in idx], [lab[int(i)] for i in idx]


def _run_condition(
    name: str,
    tp: List[str],
    vp: List[str],
    ep: List[str],
    tl: List[np.ndarray],
    vl: List[np.ndarray],
    el: List[np.ndarray],
    cfg_for_votes: Config,
    dev: torch.device,
    args: argparse.Namespace,
) -> dict:
    tr_recs = _to_records(tp, tl, "train")
    dv_recs = _to_records(vp, vl, "dev")
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

    heads = _fit_svm_heads(x_tr, y_tr, c_value=args.svm_c)
    dv_scores = _svm_scores(heads, x_dv)
    te_scores = _svm_scores(heads, x_te)
    th = _tune_thresholds_scores(y_dv, dv_scores)
    te_bin = _apply_thresholds(te_scores, th)
    m_base = compute_f1_metrics(y_te.astype(np.float64), te_bin.astype(np.float64), 0.5)

    out: dict = {
        "condition": name,
        "n_train": len(tp),
        "n_dev": len(vp),
        "n_test": len(ep),
        "svm_f1": {k: float(v) for k, v in m_base.items()},
        "thresholds": {k: float(v) for k, v in zip(F1_CLASS_NAMES, th)},
    }

    if args.block_learned_pause_selection:
        yb_tr = (y_tr[:, 3] > 0.5).astype(np.int32)
        yb_dv = (y_dv[:, 3] > 0.5).astype(np.int32)
        votes_map = _path_to_block_vote(cfg_for_votes.label_csv, cfg_for_votes.data_root)
        vb_tr = np.array([int(votes_map.get(os.path.abspath(p), -1)) for p in tp], dtype=np.int32)
        p_tr = _block_pause_matrix(
            tp, cfg_for_votes.sample_rate, args.pause_cache_dir.strip(), args.refresh_block_pause_cache
        )

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
                "svm_f1_with_block_override": {k: float(v) for k, v in m_np.items()},
                "override_block_threshold": float(block_thr),
                "q_threshold": qthr,
            }
        else:
            out["block_learned_pause_selection"] = {"enabled": False, "reason": "insufficient scorer data"}

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--clips-root", default="data/sep28k/clips", help="Mixed root; ~5s files live here.")
    ap.add_argument("--clip3-root", default="data/SEP-28k_CLIP", help="All-3s mirror of clip layout.")
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-learned-pause-selection", action="store_true", default=True)
    ap.add_argument("--no-block-learned-pause-selection", action="store_false", dest="block_learned_pause_selection")
    ap.add_argument("--block-np-quantile", type=float, default=0.8)
    ap.add_argument("--pause-low-vote-mode", choices=("le1", "eq1"), default="le1")
    ap.add_argument("--out-json", default="artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline.json")
    ap.add_argument("--max-train", type=int, default=0, help="0 = all matched train clips after filter.")
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = _pick_device(args.device)

    clips_root = os.path.abspath(args.clips_root)
    clip3_root = os.path.abspath(args.clip3_root)

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = clips_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)

    def filt(p: List[str], l: List[np.ndarray]) -> Tuple[List[str], List[str], List[np.ndarray]]:
        return _filter_matched_pairs(p, l, clips_root, clip3_root)

    tp5, tp3, tl_m = filt(tp, tl)
    vp5, vp3, vl_m = filt(vp, vl)
    ep5, ep3, el_m = filt(ep, el)

    tp5, tp3, tl_m = _subsample_matched(tp5, tp3, tl_m, int(args.max_train), int(args.seed) + 1)
    vp5, vp3, vl_m = _subsample_matched(vp5, vp3, vl_m, int(args.max_dev), int(args.seed) + 2)
    ep5, ep3, el_m = _subsample_matched(ep5, ep3, el_m, int(args.max_test), int(args.seed) + 3)

    if not tp5 or not vp5 or not ep5:
        raise RuntimeError("After ~5s + 3s mirror filter, a split is empty. Check roots and duration buckets.")

    cfg5 = Config()
    cfg5.label_csv = args.csv
    cfg5.data_root = clips_root
    cfg5.split_column = args.split_column
    cfg5.label_vote_threshold = cfg.label_vote_threshold

    cfg3 = Config()
    cfg3.label_csv = args.csv
    cfg3.data_root = clip3_root
    cfg3.split_column = args.split_column
    cfg3.label_vote_threshold = cfg.label_vote_threshold

    r5 = _run_condition("5s_paths_matched_subset", tp5, vp5, ep5, tl_m, vl_m, el_m, cfg5, dev, args)
    r3 = _run_condition("3s_paths_same_clip_ids", tp3, vp3, ep3, tl_m, vl_m, el_m, cfg3, dev, args)

    delta = {k: float(r3["svm_f1"][k] - r5["svm_f1"][k]) for k in r5["svm_f1"].keys()}
    out = {
        "seed": int(args.seed),
        "split_column": args.split_column,
        "label_vote_threshold": int(args.label_vote_threshold),
        "clips_root": clips_root,
        "clip3_root": clip3_root,
        "matched_counts": {
            "train": len(tp5),
            "dev": len(vp5),
            "test": len(ep5),
        },
        "five_second_run": r5,
        "three_second_run": r3,
        "delta_3s_minus_5s_svm_f1": delta,
    }
    if "block_learned_pause_selection" in r5 and r5["block_learned_pause_selection"].get("enabled"):
        b5 = r5["block_learned_pause_selection"]["svm_f1_with_block_override"]
        b3 = r3["block_learned_pause_selection"]["svm_f1_with_block_override"]
        out["delta_3s_minus_5s_block_override_f1"] = {k: float(b3[k] - b5[k]) for k in b5.keys()}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
