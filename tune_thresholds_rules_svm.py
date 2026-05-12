#!/usr/bin/env python3
"""
Tune rule and SVM thresholds on dev, then evaluate test.

Protocol:
- Labels/splits from disfluency_pipeline (4 heads, SEP28k-T by default).
- Train SVM on train split only (by default: per-head W2V2 layer + prosody from the
  layer-sweep JSON, same as Bayer-style compare_rules_svm_hybrid). Training rows
  are undersampled per head to 50% positive / 50% negative (match minority count).
- Tune per-head thresholds on dev:
  - Rules: threshold over sigmoid probs in [0.05..0.95]
  - SVM: threshold over decision_function scores (quantile grid from dev scores)
- Report test metrics with tuned thresholds.
- Writes JSON including ``split_column``, ``metrics_evaluated_on_split`` (``test``),
  ``seed`` / ``subsample_seeds``, SVM balance flags, and a ``test_metrics_tuned_thresholds`` snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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
from paper_style_w2v2_svm_test import (
    ClipRecord,
    extract_embeddings,
    extract_per_head_w2v2_prosody,
    load_svm_clean03_head_bundle,
)
from zhang_full.rule_module import ZhangFullRuleLogits

DEFAULT_SVM_HEAD_JSON = "artifacts/svm_clean03_best_configs_full.json"


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


def _balanced_row_indices_y_bin(y_col: np.ndarray, seed: int) -> np.ndarray:
    """Undersample to min(n_pos, n_neg) of each class; if a class is empty, use all rows."""
    y_col = np.asarray(y_col, dtype=np.int32).reshape(-1)
    pos = np.flatnonzero(y_col == 1)
    neg = np.flatnonzero(y_col == 0)
    if pos.size == 0 or neg.size == 0:
        return np.arange(y_col.shape[0], dtype=np.int64)
    n = int(min(pos.size, neg.size))
    rng = np.random.default_rng(int(seed))
    pos_take = rng.choice(pos, size=n, replace=False)
    neg_take = rng.choice(neg, size=n, replace=False)
    out = np.concatenate([pos_take, neg_take])
    rng.shuffle(out)
    return out.astype(np.int64)


def _fit_svm_heads(
    x_tr: Union[np.ndarray, List[np.ndarray]],
    y_tr: np.ndarray,
    c_per_head: Optional[List[float]] = None,
    *,
    balance_train: bool = True,
    balance_seed: int = 0,
) -> List[SVMHead]:
    heads: List[SVMHead] = []
    n = y_tr.shape[1]
    if isinstance(x_tr, list):
        for i in range(n):
            c = float(c_per_head[i]) if c_per_head is not None else 10.0
            yi = (y_tr[:, i] > 0.5).astype(np.int32)
            if balance_train:
                idx = _balanced_row_indices_y_bin(yi, balance_seed + 100_003 * i)
                xi = x_tr[i][idx]
                yi_fit = yi[idx]
                print(
                    f"  [SVM train balanced] {F1_CLASS_NAMES[i]}: n={len(idx)} "
                    f"pos={int(yi_fit.sum())} neg={int(len(yi_fit) - yi_fit.sum())}",
                    flush=True,
                )
            else:
                xi = x_tr[i]
                yi_fit = yi
            p = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced"),
            )
            p.fit(xi, yi_fit)
            heads.append(SVMHead(pipe=p))
        return heads
    for i in range(n):
        c = float(c_per_head[i]) if c_per_head is not None else 10.0
        yi = (y_tr[:, i] > 0.5).astype(np.int32)
        if balance_train:
            idx = _balanced_row_indices_y_bin(yi, balance_seed + 100_003 * i)
            xi = x_tr[idx]
            yi_fit = yi[idx]
            print(
                f"  [SVM train balanced] {F1_CLASS_NAMES[i]}: n={len(idx)} "
                f"pos={int(yi_fit.sum())} neg={int(len(yi_fit) - yi_fit.sum())}",
                flush=True,
            )
        else:
            xi = x_tr
            yi_fit = yi
        p = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced"),
        )
        p.fit(xi, yi_fit)
        heads.append(SVMHead(pipe=p))
    return heads


def _svm_scores(heads: List[SVMHead], x: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    if isinstance(x, list):
        n = x[0].shape[0]
    else:
        n = x.shape[0]
    out = np.zeros((n, len(heads)), dtype=np.float64)
    for i, h in enumerate(heads):
        xi = x[i] if isinstance(x, list) else x
        out[:, i] = h.pipe.decision_function(xi)
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
    ap.add_argument(
        "--label-vote-threshold",
        type=int,
        default=3,
        help="SEP-28k-Extended: present when vote count >= this (3 = unanimous).",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Legacy mode: single W2V2 layer index.",
    )
    ap.add_argument(
        "--svm-head-config-json",
        default="",
        help=(
            "svm_clean03_layer_prosody_sweep JSON (per-head layer, C). "
            "Empty: use file if it exists, else legacy single-layer SVM."
        ),
    )
    ap.add_argument(
        "--svm-legacy-single-layer",
        action="store_true",
        help="768-D embedding from --layer only; C=10 per head.",
    )
    ap.add_argument("--prosody-cache-dir", default=".cache/prosody_features")
    ap.add_argument("--refresh-prosody-cache", action="store_true")
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--zhang-full-cache-dir", default="")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-balance-svm-train",
        action="store_true",
        help="Train each head SVM on all train rows (default: 50/50 pos/neg per head).",
    )
    ap.add_argument(
        "--block-natural-pause-v2",
        action="store_true",
        default=True,
        help=(
            "Record that compare_rules_svm_hybrid.py uses Block natural-pause v2 replacement "
            "by default (metadata only in this thresholds artifact)."
        ),
    )
    ap.add_argument(
        "--no-block-natural-pause-v2",
        action="store_false",
        dest="block_natural_pause_v2",
        help="Record Block natural-pause v2 as disabled in thresholds metadata.",
    )
    ap.add_argument(
        "--block-np-quantile",
        type=float,
        default=0.80,
        help="Block natural-pause v2 quantile metadata for reproducibility.",
    )
    ap.add_argument(
        "--block-np-c",
        type=float,
        default=0.10,
        help="Block natural-pause v2 SVC C metadata for reproducibility.",
    )
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

    explicit_json = (args.svm_head_config_json or "").strip()
    if args.svm_legacy_single_layer:
        use_per_head = False
        head_json = ""
    elif explicit_json:
        use_per_head = True
        head_json = explicit_json
        if not os.path.isfile(head_json):
            raise SystemExit(f"Missing --svm-head-config-json file: {head_json!r}")
    else:
        head_json = DEFAULT_SVM_HEAD_JSON
        use_per_head = os.path.isfile(head_json)
        if use_per_head:
            print(f"SVM: per-head W2V2+prosody from {head_json}", flush=True)
        else:
            print(
                f"[Info] {DEFAULT_SVM_HEAD_JSON!r} not found; SVM uses legacy "
                f"single layer={args.layer}.",
                flush=True,
            )

    layers_t: Optional[Tuple[int, int, int, int]] = None
    c_list: Optional[List[float]] = None
    if use_per_head:
        layers_t, c_t, _pd = load_svm_clean03_head_bundle(head_json)
        c_list = list(c_t)

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
    if use_per_head:
        assert layers_t is not None and c_list is not None
        x_tr, y_tr = extract_per_head_w2v2_prosody(
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
        x_dv, y_dv = extract_per_head_w2v2_prosody(
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
        x_te, y_te = extract_per_head_w2v2_prosody(
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
        svm_heads = _fit_svm_heads(
            x_tr,
            y_tr,
            c_per_head=c_list,
            balance_train=not args.no_balance_svm_train,
            balance_seed=int(args.seed),
        )
        svm_dv_scores = _svm_scores(svm_heads, x_dv)
        svm_te_scores = _svm_scores(svm_heads, x_te)
    else:
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
        svm_heads = _fit_svm_heads(
            x_tr,
            y_tr,
            balance_train=not args.no_balance_svm_train,
            balance_seed=int(args.seed),
        )
        svm_dv_scores = _svm_scores(svm_heads, x_dv)
        svm_te_scores = _svm_scores(svm_heads, x_te)
    svm_th = _tune_thresholds_scores(y_dv, svm_dv_scores)
    svm_te_bin = _apply_thresholds(svm_te_scores, svm_th)
    m_svm = compute_f1_metrics(y_te.astype(np.float64), svm_te_bin.astype(np.float64), 0.5)

    # Rules side (zhang_full only; naive zhang removed)
    zfull_mod = ZhangFullRuleLogits(
        sample_rate=cfg.sample_rate,
        cache_dir=(args.zhang_full_cache_dir.strip() or None),
    ).to(dev).eval()

    zf_dv = _collect_rule_probs(cfg, vp, vl, zfull_mod, dev, args.batch_size)
    zf_te = _collect_rule_probs(cfg, ep, el, zfull_mod, dev, args.batch_size)
    zf_th = _tune_thresholds_probs(np.asarray(vl), zf_dv)
    zf_te_bin = _apply_thresholds(zf_te, zf_th)
    m_zfull = compute_f1_metrics(np.asarray(el), zf_te_bin.astype(np.float64), 0.5)

    print("\nTuned thresholds (per head):", flush=True)
    print(f"SVM score threshold:        {_fmt_head_thresholds(svm_th)}", flush=True)
    print(f"Rules zfull prob threshold: {_fmt_head_thresholds(zf_th)}", flush=True)

    print("\nTest metrics with tuned thresholds:", flush=True)
    print(f"SVM (tuned):        {format_f1_metrics(m_svm)}", flush=True)
    print(f"Rules zfull (tuned):{format_f1_metrics(m_zfull)}", flush=True)

    out_fp = (args.thresholds_out or "").strip()
    if out_fp:
        out_dir = os.path.dirname(out_fp)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "split_column": args.split_column,
            "label_csv": os.path.basename(args.csv),
            "label_vote_threshold": int(args.label_vote_threshold),
            "seed": int(args.seed),
            "metrics_evaluated_on_split": "test",
            "threshold_tuned_on_split": "dev",
            "svm_trained_on_split": "train",
            "subsample_caps": {
                "max_train": int(args.max_train),
                "max_dev": int(args.max_dev),
                "max_test": int(args.max_test),
            },
            "subsample_seeds": {
                "train": int(args.seed) + 1,
                "dev": int(args.seed) + 2,
                "test": int(args.seed) + 3,
            },
            "svm_train_balanced_per_head": not bool(args.no_balance_svm_train),
            "svm_train_balance_seed": int(args.seed),
            "block_natural_pause_v2": {
                "enabled_by_default_in_compare": bool(args.block_natural_pause_v2),
                "pause_neg_quantile": float(args.block_np_quantile),
                "svm_C": float(args.block_np_c),
                "notes": (
                    "This thresholds file stores metadata for Block natural-pause v2. "
                    "The Block-v2 decision threshold is learned internally in compare_rules_svm_hybrid.py."
                ),
            },
            "heads": list(F1_CLASS_NAMES),
            "svm_feature_mode": (
                "per_head_w2v2_prosody" if use_per_head else "single_layer_w2v2"
            ),
            "svm_head_config_json": (
                os.path.abspath(head_json) if use_per_head and head_json else ""
            ),
            "svm_score_thresholds": {
                k: float(v) for k, v in zip(F1_CLASS_NAMES, svm_th)
            },
            "rules_zhang_full_prob_thresholds": {
                k: float(v) for k, v in zip(F1_CLASS_NAMES, zf_th)
            },
        }
        if use_per_head and layers_t is not None and c_list is not None:
            payload["per_head_layers"] = {
                k: int(v) for k, v in zip(F1_CLASS_NAMES, layers_t)
            }
            payload["per_head_C"] = {
                k: float(v) for k, v in zip(F1_CLASS_NAMES, c_list)
            }
        payload["test_metrics_tuned_thresholds"] = {
            "svm_f1": {k: float(v) for k, v in m_svm.items()},
            "rules_zhang_full_f1": {k: float(v) for k, v in m_zfull.items()},
        }
        with open(out_fp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved thresholds: {out_fp}", flush=True)


if __name__ == "__main__":
    main()
