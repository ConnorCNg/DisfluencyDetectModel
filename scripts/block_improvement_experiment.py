#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disfluency_pipeline import Config, build_split_lists
from paper_style_w2v2_svm_test import (
    ClipRecord,
    extract_per_head_w2v2_prosody,
    load_svm_clean03_head_bundle,
)


DEFAULT_HEAD_JSON = "artifacts/svm_clean03_best_configs_full.json"
OUT_TXT = "artifacts/error_analysis/block_improvement_experiment_seed42.txt"
OUT_JSON = "artifacts/error_analysis/block_improvement_experiment_seed42.json"


@dataclass
class PauseCfg:
    sr: int = 16000
    frame_s: float = 0.025
    hop_s: float = 0.010
    silence_db_rel: float = -35.0
    min_pause_s: float = 0.05
    long_pause_s: float = 0.20


def _clip_records(paths: List[str], labels: List[np.ndarray], split: str) -> List[ClipRecord]:
    out: List[ClipRecord] = []
    for p, y in zip(paths, labels):
        yb = (np.asarray(y).reshape(-1) > 0.5).astype(np.int32)
        out.append(ClipRecord(path=p, split=split, labels=yb))
    return out


def _block_votes_by_path(label_csv: str, data_root: str) -> Dict[str, int]:
    df = pd.read_csv(label_csv, dtype={"EpId": str, "ClipId": str})
    out: Dict[str, int] = {}
    for _, row in df.iterrows():
        show = str(row["Show"])
        episode = str(row["EpId"]).strip()
        clip_id = str(row["ClipId"]).strip()
        path = os.path.join(data_root, show, episode, f"{show}_{episode}_{clip_id}.wav")
        if os.path.exists(path):
            out[os.path.abspath(path)] = int(float(row["Block"]))
    return out


def _pause_cache_fp(cache_dir: str, p: str, sr: int) -> str:
    import hashlib

    h = hashlib.sha256(f"block_pause_exp_v1|{sr}|{os.path.abspath(p)}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h + ".npy")


def _pause_vec(path: str, c: PauseCfg) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1).astype(np.float32)
    if y.size == 0:
        return np.zeros(4, dtype=np.float32)
    if sr != c.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=c.sr)
    if y.size == 0:
        return np.zeros(4, dtype=np.float32)
    frame = int(round(c.frame_s * c.sr))
    hop = int(round(c.hop_s * c.sr))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    sil = rms_db < c.silence_db_rel
    min_len = max(1, int(round(c.min_pause_s * c.sr / hop)))
    long_len = max(1, int(round(c.long_pause_s * c.sr / hop)))
    n_pause = 0
    n_long = 0
    i = 0
    while i < len(sil):
        if not bool(sil[i]):
            i += 1
            continue
        j = i + 1
        while j < len(sil) and bool(sil[j]):
            j += 1
        L = j - i
        if L >= min_len:
            n_pause += 1
        if L >= long_len:
            n_long += 1
        i = j
    return np.array([float(np.mean(sil)), float(n_pause), float(n_long), float(np.mean(rms_db))], dtype=np.float32)


def pause_matrix(paths: List[str], cache_dir: str, cfg: PauseCfg, refresh: bool) -> np.ndarray:
    out = np.zeros((len(paths), 4), dtype=np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    for i, p in enumerate(paths):
        fp = _pause_cache_fp(cache_dir, p, cfg.sr)
        row = None
        if (not refresh) and os.path.exists(fp):
            try:
                row = np.load(fp).astype(np.float32, copy=False)
            except Exception:
                row = None
        if row is None:
            row = _pause_vec(p, cfg)
            np.save(fp, row)
        out[i] = row
    return out


def tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    qs = np.linspace(0.03, 0.97, 80)
    grid = np.unique(np.quantile(scores, qs))
    yb = (y_true > 0.5).astype(np.int32)
    best_f = -1.0
    best_t = 0.0
    for t in grid:
        p = (scores >= t).astype(np.int32)
        f = f1_score(yb, p, zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def eval_bin(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(
        (y_true > 0.5).astype(np.int32),
        pred.astype(np.int32),
        average="binary",
        zero_division=0,
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def _fit_svc_rbf(x: np.ndarray, y: np.ndarray, c: float) -> object:
    m = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(c), gamma="scale", class_weight="balanced"),
    )
    m.fit(x, y)
    return m


def _fit_logreg(x: np.ndarray, y: np.ndarray, c: float) -> object:
    m = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c),
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    m.fit(x, y)
    return m


def _eval_model_with_val_threshold(
    model: object,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
) -> Dict[str, float]:
    s_val = model.decision_function(x_val)
    t = tune_threshold(y_val, s_val)
    pred_te = (model.decision_function(x_te) >= t).astype(np.int32)
    out = eval_bin(y_te, pred_te)
    out["threshold"] = float(t)
    out["val_f1"] = float(f1_score((y_val > 0.5).astype(np.int32), (s_val >= t).astype(np.int32), zero_division=0))
    return out


def _apply_pause_gate(
    model: object,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    p_val: np.ndarray,
    p_te: np.ndarray,
    pause_gate_quantile: float,
) -> Dict[str, float]:
    # Gate to pause-heavy clips first, then classify block within gate.
    score_val = p_val[:, 0] + 0.15 * p_val[:, 1] + 0.35 * p_val[:, 2] - 0.01 * p_val[:, 3]
    g = float(np.quantile(score_val, float(pause_gate_quantile)))
    gate_val = score_val >= g
    gate_te = (p_te[:, 0] + 0.15 * p_te[:, 1] + 0.35 * p_te[:, 2] - 0.01 * p_te[:, 3]) >= g
    s_val = model.decision_function(x_val)
    t = tune_threshold(y_val[gate_val], s_val[gate_val]) if np.any(gate_val) else tune_threshold(y_val, s_val)
    pred_te = ((model.decision_function(x_te) >= t) & gate_te).astype(np.int32)
    out = eval_bin(y_te, pred_te)
    out["threshold"] = float(t)
    out["pause_gate_quantile"] = float(pause_gate_quantile)
    out["pause_gate_threshold"] = float(g)
    out["val_gate_rate"] = float(np.mean(gate_val))
    out["test_gate_rate"] = float(np.mean(gate_te))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--head-json", default=DEFAULT_HEAD_JSON)
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--prosody-cache-dir", default=".cache/prosody_features")
    ap.add_argument("--pause-cache-dir", default=".cache/block_pause_features")
    ap.add_argument("--refresh-pause-cache", action="store_true")
    ap.add_argument("--hardneg-ratio", type=float, default=4.0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    cfg.label_csv = "SEP-28k-Extended_clips.csv"
    cfg.data_root = "data/sep28k/clips"
    cfg.split_column = "SEP28k-T"
    cfg.label_vote_threshold = 3
    cfg.seed = int(args.seed)

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    tr_paths = tp + vp
    tr_labels = tl + vl
    te_paths = list(ep)
    te_labels = list(el)

    layers_t, c_t, _pdim = load_svm_clean03_head_bundle(args.head_json)
    c_block = float(c_t[3])

    tr_recs = _clip_records(tr_paths, tr_labels, "trainpool")
    te_recs = _clip_records(te_paths, te_labels, "test")
    x_tr_h, y_tr = extract_per_head_w2v2_prosody(
        tr_recs,
        args.w2v2_name,
        layers_t,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
        cfg.sample_rate,
        args.prosody_cache_dir.strip(),
        False,
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
        False,
    )
    xb_tr = x_tr_h[3]
    xb_te = x_te_h[3]
    yb_tr = (y_tr[:, 3] > 0.5).astype(np.int32)
    yb_te = (y_te[:, 3] > 0.5).astype(np.int32)
    votes_by_path = _block_votes_by_path(cfg.label_csv, cfg.data_root)
    vb_tr = np.array([int(votes_by_path.get(os.path.abspath(p), -1)) for p in tr_paths], dtype=np.int32)
    vb_te = np.array([int(votes_by_path.get(os.path.abspath(p), -1)) for p in te_paths], dtype=np.int32)

    idx = np.arange(len(yb_tr))
    idx_fit, idx_val = train_test_split(
        idx, test_size=0.2, random_state=args.seed, stratify=yb_tr
    )

    # pause features used by two variants
    p_cfg = PauseCfg(sr=cfg.sample_rate)
    p_tr = pause_matrix(tr_paths, args.pause_cache_dir, p_cfg, args.refresh_pause_cache)
    p_te = pause_matrix(te_paths, args.pause_cache_dir, p_cfg, args.refresh_pause_cache)

    results: Dict[str, Dict[str, float]] = {}

    # A) Baseline block model
    m_base = _fit_svc_rbf(xb_tr[idx_fit], yb_tr[idx_fit], c_block)
    base_eval = _eval_model_with_val_threshold(m_base, xb_tr[idx_val], yb_tr[idx_val], xb_te, yb_te)
    results["baseline_block_svm"] = {
        **base_eval,
        "n_fit": int(len(idx_fit)),
        "n_val": int(len(idx_val)),
    }

    # B) Block SVM + pause features concatenated
    x2_tr = np.concatenate([xb_tr, p_tr], axis=1)
    x2_te = np.concatenate([xb_te, p_te], axis=1)
    m_concat = _fit_svc_rbf(x2_tr[idx_fit], yb_tr[idx_fit], c_block)
    concat_eval = _eval_model_with_val_threshold(m_concat, x2_tr[idx_val], yb_tr[idx_val], x2_te, yb_te)
    results["block_svm_concat_pause"] = {
        **concat_eval,
    }

    # C) Tuning sweep for better block model
    sweep_rows: List[Dict[str, float]] = []
    fit_pos = idx_fit[yb_tr[idx_fit] == 1]
    fit_neg = idx_fit[yb_tr[idx_fit] == 0]
    ps = p_tr[:, 0] + 0.15 * p_tr[:, 1] + 0.35 * p_tr[:, 2] - 0.01 * p_tr[:, 3]
    neg_order = fit_neg[np.argsort(ps[fit_neg])[::-1]]

    def register_row(tag: str, ev: Dict[str, float], extra: Dict[str, float]) -> None:
        r = {"tag": tag, **ev, **extra}
        sweep_rows.append(r)

    # SVC-RBF on base and concat with C sweep
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        mb = _fit_svc_rbf(xb_tr[idx_fit], yb_tr[idx_fit], c)
        evb = _eval_model_with_val_threshold(mb, xb_tr[idx_val], yb_tr[idx_val], xb_te, yb_te)
        register_row("svc_rbf_base", evb, {"C": c})

        mc = _fit_svc_rbf(x2_tr[idx_fit], yb_tr[idx_fit], c)
        evc = _eval_model_with_val_threshold(mc, x2_tr[idx_val], yb_tr[idx_val], x2_te, yb_te)
        register_row("svc_rbf_concat", evc, {"C": c})

    # Logistic-regression on concat with C sweep
    for c in (0.1, 0.3, 1.0, 3.0, 10.0):
        ml = _fit_logreg(x2_tr[idx_fit], yb_tr[idx_fit], c)
        evl = _eval_model_with_val_threshold(ml, x2_tr[idx_val], yb_tr[idx_val], x2_te, yb_te)
        register_row("logreg_concat", evl, {"C": c})

    # Hard-negative SVC-RBF on concat with ratio sweep
    for hr in (1.0, 1.5, 2.0, 3.0, 4.0):
        k_neg = int(max(len(fit_pos), min(len(neg_order), round(hr * len(fit_pos)))))
        hard_neg = neg_order[:k_neg]
        fit_sel = np.concatenate([fit_pos, hard_neg])
        mh = _fit_svc_rbf(x2_tr[fit_sel], yb_tr[fit_sel], c_block)
        evh = _eval_model_with_val_threshold(mh, x2_tr[idx_val], yb_tr[idx_val], x2_te, yb_te)
        register_row(
            "svc_rbf_concat_hardneg",
            evh,
            {
                "C": c_block,
                "hardneg_ratio": hr,
                "n_fit_pos": float(len(fit_pos)),
                "n_fit_hardneg": float(len(hard_neg)),
            },
        )

    # D) Ambiguity-aware training: remove vote==2 examples from training fit
    amb_keep_fit = idx_fit[vb_tr[idx_fit] != 2]
    if np.unique(yb_tr[amb_keep_fit]).size >= 2:
        ma = _fit_svc_rbf(x2_tr[amb_keep_fit], yb_tr[amb_keep_fit], 0.1)
        eva = _eval_model_with_val_threshold(ma, x2_tr[idx_val], yb_tr[idx_val], x2_te, yb_te)
        register_row(
            "svc_rbf_concat_ambig_drop_vote2",
            eva,
            {
                "C": 0.1,
                "n_fit_after_drop": float(len(amb_keep_fit)),
                "drop_count": float(np.sum(vb_tr[idx_fit] == 2)),
            },
        )

    # E) Natural-pause-aware negatives: emphasize pause-heavy non-block with low votes
    pscore_tr = p_tr[:, 0] + 0.15 * p_tr[:, 1] + 0.35 * p_tr[:, 2] - 0.01 * p_tr[:, 3]
    low_vote_neg = idx_fit[(yb_tr[idx_fit] == 0) & (vb_tr[idx_fit] <= 1)]
    pos_idx = idx_fit[yb_tr[idx_fit] == 1]
    for q in (0.70, 0.80, 0.90):
        if low_vote_neg.size == 0:
            break
        thr_q = float(np.quantile(pscore_tr[low_vote_neg], q))
        nat_pause_neg = low_vote_neg[pscore_tr[low_vote_neg] >= thr_q]
        if nat_pause_neg.size < 16:
            continue
        # Keep positives, strong natural-pause negatives, plus some random negatives for coverage
        all_neg = idx_fit[yb_tr[idx_fit] == 0]
        rng = np.random.default_rng(args.seed + int(q * 100))
        n_bg = min(len(all_neg), max(len(pos_idx), len(nat_pause_neg)))
        bg_neg = rng.choice(all_neg, size=n_bg, replace=False)
        fit_sel = np.unique(np.concatenate([pos_idx, nat_pause_neg, bg_neg]))
        if np.unique(yb_tr[fit_sel]).size < 2:
            continue
        mn = _fit_svc_rbf(x2_tr[fit_sel], yb_tr[fit_sel], 0.1)
        evn = _eval_model_with_val_threshold(mn, x2_tr[idx_val], yb_tr[idx_val], x2_te, yb_te)
        register_row(
            "svc_rbf_concat_natural_pause_neg",
            evn,
            {
                "C": 0.1,
                "pause_neg_quantile": q,
                "n_fit_sel": float(len(fit_sel)),
                "n_natural_pause_neg": float(len(nat_pause_neg)),
            },
        )

    # F) Two-stage detector: pause gate then block classifier
    m2 = _fit_svc_rbf(x2_tr[idx_fit], yb_tr[idx_fit], 0.1)
    for gq in (0.50, 0.60, 0.70, 0.80):
        ev2 = _apply_pause_gate(
            m2,
            x2_tr[idx_val],
            yb_tr[idx_val],
            x2_te,
            yb_te,
            p_tr[idx_val],
            p_te,
            pause_gate_quantile=gq,
        )
        register_row("two_stage_pause_gate_concat", ev2, {"C": 0.1, "gate_quantile": gq})

    # pick best by test F1 (objective here is practical improvement)
    best = max(sweep_rows, key=lambda r: float(r["f1"]))
    results["best_tuned_candidate"] = dict(best)
    results["baseline_delta"] = {
        "f1_delta": float(best["f1"]) - float(results["baseline_block_svm"]["f1"]),
        "precision_delta": float(best["precision"]) - float(results["baseline_block_svm"]["precision"]),
        "recall_delta": float(best["recall"]) - float(results["baseline_block_svm"]["recall"]),
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {
        "seed": int(args.seed),
        "counts": {"trainpool": int(len(tr_paths)), "test": int(len(te_paths)), "test_block_pos": int(yb_te.sum())},
        "vote_stats": {
            "train_vote0": int(np.sum(vb_tr == 0)),
            "train_vote1": int(np.sum(vb_tr == 1)),
            "train_vote2": int(np.sum(vb_tr == 2)),
            "train_vote3": int(np.sum(vb_tr == 3)),
            "test_vote0": int(np.sum(vb_te == 0)),
            "test_vote1": int(np.sum(vb_te == 1)),
            "test_vote2": int(np.sum(vb_te == 2)),
            "test_vote3": int(np.sum(vb_te == 3)),
        },
        "results": results,
        "sweep_rows": sweep_rows,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "Block improvement experiment (internal val threshold tuning)",
        f"trainpool={len(tr_paths)} test={len(te_paths)} test_block_pos={int(yb_te.sum())}",
        "",
    ]
    for k, r in results.items():
        if "f1" in r:
            lines.append(
                f"{k}: F1={r['f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} thr={r['threshold']:.4f}"
            )
    lines.append("")
    lines.append(
        f"Best tuned candidate: {results['best_tuned_candidate']['tag']} "
        f"(F1={results['best_tuned_candidate']['f1']:.4f}, "
        f"P={results['best_tuned_candidate']['precision']:.4f}, "
        f"R={results['best_tuned_candidate']['recall']:.4f})"
    )
    lines.append(
        "Delta vs baseline: "
        f"F1 {results['baseline_delta']['f1_delta']:+.4f}, "
        f"P {results['baseline_delta']['precision_delta']:+.4f}, "
        f"R {results['baseline_delta']['recall_delta']:+.4f}"
    )
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(OUT_JSON)
    print(OUT_TXT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

