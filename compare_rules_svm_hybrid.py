#!/usr/bin/env python3
"""
Compare ``zhang_full`` rules vs W2V2+SVM on the **same clips and labels**.

- Labels: same 4 heads as ``disfluency_pipeline`` (Repetition = max(SoundRep, WordRep)
  votes ≥ threshold), from ``build_split_lists`` / ``row_to_label_vector``.
- Split: default ``SEP28k-T`` (``dev`` → internal ``val``); train SVM on train+val pool;
  metrics on the ``test`` rows of that column.
- Rules: evaluates ``zhang_full`` logits only (naive ``zhang`` removed) → sigmoid.
- SVM: one binary RBF SVC per head. By default uses ``artifacts/svm_clean03_best_configs_full.json``
  (Bayer-style): per-dysfluency Wav2Vec2 layer + 9-D prosody + sweep ``C``, reusing the same
  embedding cache keys as ``svm_clean03_layer_prosody_sweep.py``. Pass ``--svm-legacy-single-layer``
  for a single hidden layer (``--layer``) and fixed ``C=10`` on raw 768-D embeddings only.

Use ``--smoke`` for a tiny subsample (quick sanity check). Use ``--max-train 0 --max-test 0``
for full train+val / full test pools.

Examples::

  python3 -u compare_rules_svm_hybrid.py --smoke --device auto
  python3 -u compare_rules_svm_hybrid.py --max-train 0 --max-test 0 --device auto
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from tqdm import tqdm

from disfluency_pipeline import (
    Config,
    DisfluencyDataset,
    F1_CLASS_NAMES,
    collate_batch,
    compute_f1_metrics,
    format_f1_metrics,
    build_split_lists,
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
    dev = torch.device(which)
    if dev.type == "mps":
        try:
            _ = torch.zeros(1).to(dev)
        except Exception:
            return torch.device("cpu")
    return dev


def _subsample(
    paths: List[str],
    labels: List[np.ndarray],
    max_n: int,
    rng: random.Random,
) -> Tuple[List[str], List[np.ndarray]]:
    if max_n <= 0 or len(paths) <= max_n:
        return paths, labels
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    idx = idx[:max_n]
    return [paths[i] for i in idx], [labels[i] for i in idx]


def _to_clip_records(paths: List[str], labels: List[np.ndarray], split_tag: str) -> List[ClipRecord]:
    out: List[ClipRecord] = []
    for p, lab in zip(paths, labels):
        y = (np.asarray(lab, dtype=np.float32).reshape(-1) > 0.5).astype(np.int32)
        if y.shape != (4,):
            raise ValueError(f"Expected 4-dim label, got shape {y.shape} for {p}")
        out.append(ClipRecord(path=p, split=split_tag, labels=y))
    return out


@dataclass
class SVMHead:
    pipe: Pipeline


def _balanced_row_indices_y_bin(y_col: np.ndarray, seed: int) -> np.ndarray:
    """
    Undersample majority class so positives and negatives are equal in count
    (min(n_pos, n_neg) of each). If one class is missing, return all rows.
    """
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


def _fit_svm_ovr(
    x_train: Union[np.ndarray, List[np.ndarray]],
    y_train: np.ndarray,
    c_per_head: Optional[List[float]] = None,
    *,
    balance_train: bool = True,
    balance_seed: int = 0,
):
    heads: List[SVMHead] = []
    n_heads = y_train.shape[1]
    if isinstance(x_train, list):
        for i in range(n_heads):
            c = float(c_per_head[i]) if c_per_head is not None else 10.0
            yi = (y_train[:, i] > 0.5).astype(np.int32)
            if balance_train:
                idx = _balanced_row_indices_y_bin(yi, balance_seed + 100_003 * i)
                xi = x_train[i][idx]
                yi_fit = yi[idx]
                print(
                    f"  [SVM train balanced] {F1_CLASS_NAMES[i]}: n={len(idx)} "
                    f"pos={int(yi_fit.sum())} neg={int(len(yi_fit) - yi_fit.sum())}",
                    flush=True,
                )
            else:
                xi = x_train[i]
                yi_fit = yi
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced"),
            )
            clf.fit(xi, yi_fit)
            heads.append(SVMHead(pipe=clf))
        return heads
    for i in range(n_heads):
        c = float(c_per_head[i]) if c_per_head is not None else 10.0
        yi = (y_train[:, i] > 0.5).astype(np.int32)
        if balance_train:
            idx = _balanced_row_indices_y_bin(yi, balance_seed + 100_003 * i)
            xi = x_train[idx]
            yi_fit = yi[idx]
            print(
                f"  [SVM train balanced] {F1_CLASS_NAMES[i]}: n={len(idx)} "
                f"pos={int(yi_fit.sum())} neg={int(len(yi_fit) - yi_fit.sum())}",
                flush=True,
            )
        else:
            xi = x_train
            yi_fit = yi
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced"),
        )
        clf.fit(xi, yi_fit)
        heads.append(SVMHead(pipe=clf))
    return heads


def _svm_predict(
    heads: List[SVMHead], x: Union[np.ndarray, List[np.ndarray]]
) -> np.ndarray:
    if isinstance(x, list):
        n = x[0].shape[0]
    else:
        n = x.shape[0]
    pred = np.zeros((n, len(heads)), dtype=np.int32)
    for i, h in enumerate(heads):
        xi = x[i] if isinstance(x, list) else x
        pred[:, i] = h.pipe.predict(xi).astype(np.int32)
    return pred


def _svm_scores(
    heads: List[SVMHead], x: Union[np.ndarray, List[np.ndarray]]
) -> np.ndarray:
    if isinstance(x, list):
        n = x[0].shape[0]
    else:
        n = x.shape[0]
    sc = np.zeros((n, len(heads)), dtype=np.float64)
    for i, h in enumerate(heads):
        xi = x[i] if isinstance(x, list) else x
        sc[:, i] = h.pipe.decision_function(xi)
    return sc


def _collect_rule_probs(
    cfg: Config,
    paths: List[str],
    labels: List[np.ndarray],
    rule_mod: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    ds = DisfluencyDataset(cfg, paths, labels)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    chunks: List[np.ndarray] = []
    use_bar = sys.stderr.isatty()
    it = tqdm(loader, desc="Rules", unit="batch", file=sys.stderr) if use_bar else loader
    rule_mod.eval()
    with torch.no_grad():
        for batch in it:
            w = batch["waveform"].to(device)
            paths_b = batch.get("paths")
            logits = rule_mod(w, paths_b)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            chunks.append(probs)
    return np.concatenate(chunks, axis=0)


def _macro_f1(metrics: dict) -> float:
    return float(np.mean([metrics[k] for k in F1_CLASS_NAMES if k in metrics]))


def _block_pause_cache_fp(cache_dir: str, path: str, sr: int) -> str:
    key = f"block_pause_v1|{sr}|{os.path.abspath(path)}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h + ".npy")


def _compute_block_pause_vec(path: str, sr_target: int) -> np.ndarray:
    """
    Block-oriented pause features for a single clip:
      [silence_ratio_rel, pause_count, long_pause_count, rms_mean_db_rel]
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1).astype(np.float32)
    if y.size == 0:
        return np.zeros(4, dtype=np.float32)
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
    if y.size == 0:
        return np.zeros(4, dtype=np.float32)
    frame = int(round(0.025 * sr_target))
    hop = int(round(0.010 * sr_target))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    sil = rms_db < -35.0  # dB relative to clip max
    min_len = max(1, int(round(0.05 * sr_target / hop)))   # >= 50ms
    long_len = max(1, int(round(0.20 * sr_target / hop)))  # >= 200ms
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
    return np.array(
        [
            float(np.mean(sil)),
            float(n_pause),
            float(n_long),
            float(np.mean(rms_db)),
        ],
        dtype=np.float32,
    )


def _block_pause_matrix(
    paths: List[str],
    sample_rate: int,
    cache_dir: str,
    refresh: bool,
) -> np.ndarray:
    out = np.zeros((len(paths), 4), dtype=np.float32)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for i, p in enumerate(paths):
        fp = _block_pause_cache_fp(cache_dir, p, sample_rate) if cache_dir else ""
        row: Optional[np.ndarray] = None
        if fp and (not refresh) and os.path.exists(fp):
            try:
                row = np.load(fp).astype(np.float32, copy=False)
            except Exception:
                row = None
        if row is None:
            row = _compute_block_pause_vec(p, sample_rate)
            if fp:
                np.save(fp, row.astype(np.float32))
        out[i] = row
    return out


def _fit_block_pause_calibrator(
    svm_block_scores_train: np.ndarray,
    y_block_train: np.ndarray,
    pause_feats_train: np.ndarray,
) -> Pipeline:
    x = np.column_stack([svm_block_scores_train.reshape(-1, 1), pause_feats_train])
    y = (y_block_train > 0.5).astype(np.int32).reshape(-1)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    pipe.fit(x, y)
    return pipe


def _path_to_block_vote(label_csv: str, data_root: str) -> dict:
    df = pd.read_csv(label_csv, dtype={"EpId": str, "ClipId": str})
    out = {}
    for _, row in df.iterrows():
        show = str(row["Show"])
        episode = str(row["EpId"]).strip()
        clip_id = str(row["ClipId"]).strip()
        path = os.path.join(data_root, show, episode, f"{show}_{episode}_{clip_id}.wav")
        if os.path.exists(path):
            out[os.path.abspath(path)] = int(float(row["Block"]))
    return out


def _tune_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    qs = np.linspace(0.03, 0.97, 80)
    grid = np.unique(np.quantile(scores, qs))
    yb = (y_true > 0.5).astype(np.int32)
    best_f = -1.0
    best_t = 0.0
    for t in grid:
        p = (scores >= t).astype(np.int32)
        tp = float(np.sum((p == 1) & (yb == 1)))
        fp = float(np.sum((p == 1) & (yb == 0)))
        fn = float(np.sum((p == 0) & (yb == 1)))
        denom = (2.0 * tp) + fp + fn
        f = (2.0 * tp / denom) if denom > 0.0 else 0.0
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def _fit_block_natural_pause_v2(
    x_block_train: np.ndarray,
    y_train: np.ndarray,
    train_paths: List[str],
    x_block_test: np.ndarray,
    test_paths: List[str],
    cfg: Config,
    seed: int,
    pause_cache_dir: str,
    refresh_pause_cache: bool,
    c_block: float,
    pause_neg_quantile: float,
) -> Tuple[np.ndarray, dict]:
    yb = (y_train[:, 3] > 0.5).astype(np.int32)
    votes_map = _path_to_block_vote(cfg.label_csv, cfg.data_root)
    vb = np.array([int(votes_map.get(os.path.abspath(p), -1)) for p in train_paths], dtype=np.int32)
    p_tr = _block_pause_matrix(train_paths, cfg.sample_rate, pause_cache_dir, refresh_pause_cache)
    p_te = _block_pause_matrix(test_paths, cfg.sample_rate, pause_cache_dir, refresh_pause_cache)
    x2_tr = np.concatenate([x_block_train, p_tr], axis=1)
    x2_te = np.concatenate([x_block_test, p_te], axis=1)

    idx = np.arange(len(yb))
    idx_fit, idx_val = train_test_split(
        idx, test_size=0.2, random_state=int(seed), stratify=yb
    )
    fit_pos = idx_fit[yb[idx_fit] == 1]
    fit_neg = idx_fit[yb[idx_fit] == 0]
    low_vote_neg = fit_neg[vb[fit_neg] <= 1]
    base_neg_pool = low_vote_neg if low_vote_neg.size > 0 else fit_neg

    ps = p_tr[:, 0] + 0.15 * p_tr[:, 1] + 0.35 * p_tr[:, 2] - 0.01 * p_tr[:, 3]
    qthr = float(np.quantile(ps[base_neg_pool], float(pause_neg_quantile)))
    nat_pause_neg = base_neg_pool[ps[base_neg_pool] >= qthr]
    rng = np.random.default_rng(int(seed) + 808)
    n_bg = min(len(fit_neg), max(len(fit_pos), len(nat_pause_neg)))
    bg_neg = (
        rng.choice(fit_neg, size=n_bg, replace=False)
        if n_bg > 0
        else np.array([], dtype=np.int64)
    )
    fit_sel = np.unique(np.concatenate([fit_pos, nat_pause_neg, bg_neg]))

    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=float(c_block), gamma="scale", class_weight="balanced"),
    )
    clf.fit(x2_tr[fit_sel], yb[fit_sel])
    s_val = clf.decision_function(x2_tr[idx_val])
    thr = _tune_threshold(yb[idx_val], s_val)
    s_te = clf.decision_function(x2_te)
    pred_te = (s_te >= thr).astype(np.int32)
    meta = {
        "n_fit_pos": int(len(fit_pos)),
        "n_fit_neg": int(len(fit_neg)),
        "n_natural_pause_neg": int(len(nat_pause_neg)),
        "n_fit_sel": int(len(fit_sel)),
        "pause_neg_quantile": float(pause_neg_quantile),
        "pause_q_threshold": float(qthr),
        "decision_threshold": float(thr),
        "C": float(c_block),
    }
    return pred_te, meta


def _clip_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _write_svm_rules_error_analysis(
    test_paths: List[str],
    y_true: np.ndarray,
    svm_bin: np.ndarray,
    rules_bin: np.ndarray,
    out_prefix: str,
) -> None:
    """
    Per-head binary errors: FP (pred=1,y=0), FN (pred=0,y=1).
    Overlap: both wrong on (clip, head); only SVM wrong; only rules wrong.
    """
    yb = (y_true > 0.5).astype(np.int32)
    ps = (svm_bin > 0.5).astype(np.int32)
    pr = (rules_bin > 0.5).astype(np.int32)
    n = len(test_paths)
    names = [_clip_basename(p) for p in test_paths]

    detail: dict = {"per_head": {}, "meta": {"n_test_clips": n}}
    lines: List[str] = [
        "SVM vs zhang_full (rules) — per-head clip basenames",
        f"N test clips: {n}",
        "",
    ]

    for hi, hname in enumerate(F1_CLASS_NAMES):
        yt = yb[:, hi]
        sv = ps[:, hi]
        ru = pr[:, hi]
        svm_fp = [names[i] for i in range(n) if sv[i] == 1 and yt[i] == 0]
        svm_fn = [names[i] for i in range(n) if sv[i] == 0 and yt[i] == 1]
        rules_fp = [names[i] for i in range(n) if ru[i] == 1 and yt[i] == 0]
        rules_fn = [names[i] for i in range(n) if ru[i] == 0 and yt[i] == 1]
        wrong_svm = sv != yt
        wrong_rules = ru != yt
        both_wrong = [names[i] for i in range(n) if wrong_svm[i] and wrong_rules[i]]
        svm_only = [names[i] for i in range(n) if wrong_svm[i] and not wrong_rules[i]]
        rules_only = [names[i] for i in range(n) if wrong_rules[i] and not wrong_svm[i]]

        detail["per_head"][hname] = {
            "svm_false_positives": sorted(svm_fp),
            "svm_false_negatives": sorted(svm_fn),
            "rules_false_positives": sorted(rules_fp),
            "rules_false_negatives": sorted(rules_fn),
            "both_wrong_clip": sorted(both_wrong),
            "svm_wrong_rules_correct": sorted(svm_only),
            "rules_wrong_svm_correct": sorted(rules_only),
            "counts": {
                "svm_fp": len(svm_fp),
                "svm_fn": len(svm_fn),
                "rules_fp": len(rules_fp),
                "rules_fn": len(rules_fn),
                "both_wrong": len(both_wrong),
                "svm_only_wrong": len(svm_only),
                "rules_only_wrong": len(rules_only),
            },
        }

        lines.append(f"=== {hname} ===")
        c = detail["per_head"][hname]["counts"]
        lines.append(
            f"  SVM   FP={c['svm_fp']} FN={c['svm_fn']}  |  "
            f"Rules FP={c['rules_fp']} FN={c['rules_fn']}"
        )
        lines.append(
            f"  Overlap: both_wrong={c['both_wrong']}  "
            f"svm_only={c['svm_only_wrong']}  rules_only={c['rules_only_wrong']}"
        )
        lines.append("")

    out_json = out_prefix + "_detail.json"
    out_txt = out_prefix + "_summary.txt"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        for hname in F1_CLASS_NAMES:
            block = detail["per_head"][hname]
            f.write(f"\n--- {hname}: full lists ---\n")
            f.write(f"SVM FP ({len(block['svm_false_positives'])}):\n")
            f.write("\n".join(block["svm_false_positives"]) + "\n")
            f.write(f"\nSVM FN ({len(block['svm_false_negatives'])}):\n")
            f.write("\n".join(block["svm_false_negatives"]) + "\n")
            f.write(f"\nRules FP ({len(block['rules_false_positives'])}):\n")
            f.write("\n".join(block["rules_false_positives"]) + "\n")
            f.write(f"\nRules FN ({len(block['rules_false_negatives'])}):\n")
            f.write("\n".join(block["rules_false_negatives"]) + "\n")
            f.write(f"\nBoth wrong ({len(block['both_wrong_clip'])}):\n")
            f.write("\n".join(block["both_wrong_clip"]) + "\n")
            f.write(f"\nSVM wrong, rules correct ({len(block['svm_wrong_rules_correct'])}):\n")
            f.write("\n".join(block["svm_wrong_rules_correct"]) + "\n")
            f.write(f"\nRules wrong, SVM correct ({len(block['rules_wrong_svm_correct'])}):\n")
            f.write("\n".join(block["rules_wrong_svm_correct"]) + "\n")

    print(
        f"\n[Error analysis] Wrote {out_json} and {out_txt}",
        flush=True,
    )


def _load_thresholds(path: str) -> Optional[dict]:
    p = (path or "").strip()
    if not p:
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _ordered_thresholds(d: Optional[dict], key: str) -> Optional[np.ndarray]:
    if not d or key not in d:
        return None
    try:
        row = d[key]
        arr = np.array([float(row[h]) for h in F1_CLASS_NAMES], dtype=np.float64)
        return arr
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="zhang_full rules vs SVM on aligned 4-head SEP-28k labels."
    )
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument(
        "--label-vote-threshold",
        type=int,
        default=3,
        help="SEP-28k-Extended: present when vote count >= this (3 = unanimous).",
    )
    ap.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Cap train+val clips for SVM (0 = all with files).",
    )
    ap.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="Cap test clips (0 = all with files).",
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny caps for a quick run.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Legacy single-layer mode: W2V2 hidden index (ignored when per-head JSON is used).",
    )
    ap.add_argument(
        "--svm-head-config-json",
        default="",
        help=(
            "JSON from svm_clean03_layer_prosody_sweep (per-head layer, C). "
            "Empty: use file if "
            f"{DEFAULT_SVM_HEAD_JSON!r} exists, else legacy single-layer SVM."
        ),
    )
    ap.add_argument(
        "--svm-legacy-single-layer",
        action="store_true",
        help="768-D embedding from --layer only (no prosody); C=10 per head.",
    )
    ap.add_argument(
        "--prosody-cache-dir",
        default=".cache/prosody_features",
        help="Used with per-head W2V2+prosody features (same cache as layer sweep).",
    )
    ap.add_argument(
        "--refresh-prosody-cache",
        action="store_true",
        help="Recompute prosody .npy caches under --prosody-cache-dir.",
    )
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--zhang-full-cache-dir", default="")
    ap.add_argument("--f1-threshold", type=float, default=0.5)
    ap.add_argument(
        "--thresholds-json",
        default="artifacts/tuned_thresholds_rules_svm.json",
        help="Optional thresholds JSON from tune_thresholds_rules_svm.py.",
    )
    ap.add_argument(
        "--ignore-saved-thresholds",
        action="store_true",
        help="Ignore thresholds JSON and use defaults (rules=0.5, svm=predict).",
    )
    ap.add_argument(
        "--no-balance-svm-train",
        action="store_true",
        help="Train each head SVM on all train rows (default: 50/50 pos/neg per head).",
    )
    ap.add_argument(
        "--block-pause-calibration",
        action="store_true",
        help=(
            "Improve Block head with a train-fitted logistic calibrator over "
            "[SVM Block score + pause features]. Other heads unchanged."
        ),
    )
    ap.add_argument(
        "--block-pause-cache-dir",
        default=".cache/block_pause_features",
        help="Cache dir for Block pause features used by --block-pause-calibration.",
    )
    ap.add_argument(
        "--refresh-block-pause-cache",
        action="store_true",
        help="Recompute block pause feature cache files.",
    )
    ap.add_argument(
        "--block-natural-pause-v2",
        action="store_true",
        default=True,
        help=(
            "Block-only replacement head: train on [block embedding + pause features] with "
            "natural-pause negative mining (low-vote non-blocks), threshold tuned on internal val."
        ),
    )
    ap.add_argument(
        "--no-block-natural-pause-v2",
        action="store_false",
        dest="block_natural_pause_v2",
        help="Disable Block natural-pause v2 replacement head.",
    )
    ap.add_argument(
        "--block-np-quantile",
        type=float,
        default=0.80,
        help="Natural-pause negative mining quantile over pause score on low-vote non-blocks.",
    )
    ap.add_argument(
        "--block-np-c",
        type=float,
        default=0.10,
        help="SVC C for --block-natural-pause-v2 block head.",
    )
    ap.add_argument(
        "--error-analysis-out",
        default="",
        help=(
            "If set, write FP/FN clip lists per head for SVM vs rules, plus overlap sets, "
            "to <prefix>_detail.json and <prefix>_summary.txt (UTF-8)."
        ),
    )
    args = ap.parse_args()

    if args.smoke:
        if args.max_train <= 0:
            args.max_train = 48
        if args.max_test <= 0:
            args.max_test = 32

    rng = random.Random(args.seed)
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
                f"single layer={args.layer} (768-D, C=10).",
                flush=True,
            )

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))
    cfg.seed = int(args.seed)

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    train_paths = tp + vp
    train_labels = tl + vl
    test_paths = list(ep)
    test_labels = list(el)

    train_paths, train_labels = _subsample(train_paths, train_labels, args.max_train, rng)
    test_paths, test_labels = _subsample(test_paths, test_labels, args.max_test, rng)

    if not train_paths or not test_paths:
        raise RuntimeError("Empty train or test after subsample; check CSV and data_root.")

    train_recs = _to_clip_records(train_paths, train_labels, "trainpool")
    test_recs = _to_clip_records(test_paths, test_labels, "test")

    print(
        f"Splits: column={cfg.split_column!r}  train+val={len(train_recs)}  "
        f"test={len(test_recs)}  vote_threshold={cfg.label_vote_threshold}",
        flush=True,
    )

    cdir = (args.zhang_full_cache_dir or "").strip() or None
    zhang_full_rule_mod = ZhangFullRuleLogits(
        sample_rate=cfg.sample_rate, cache_dir=cdir
    ).to(dev).eval()

    if use_per_head:
        layers_t, c_t, _pdim = load_svm_clean03_head_bundle(head_json)
        c_list = list(c_t)
        print(
            "Extracting per-head W2V2+prosody (train then test; reuses layer caches)…",
            flush=True,
        )
        x_train_h, y_train = extract_per_head_w2v2_prosody(
            train_recs,
            args.w2v2_name,
            layers_t,
            args.batch_size,
            dev,
            args.embedding_cache_dir.strip(),
            cfg.sample_rate,
            args.prosody_cache_dir.strip(),
            args.refresh_prosody_cache,
        )
        x_test_h, y_test = extract_per_head_w2v2_prosody(
            test_recs,
            args.w2v2_name,
            layers_t,
            args.batch_size,
            dev,
            args.embedding_cache_dir.strip(),
            cfg.sample_rate,
            args.prosody_cache_dir.strip(),
            args.refresh_prosody_cache,
        )
        print(
            f"Features: per-head D={x_train_h[0].shape[1]}  "
            f"layers={dict(zip(F1_CLASS_NAMES, layers_t))}  C={dict(zip(F1_CLASS_NAMES, c_list))}",
            flush=True,
        )
        print("Training SVMs (4 heads)…", flush=True)
        svm_heads = _fit_svm_ovr(
            x_train_h,
            y_train,
            c_per_head=c_list,
            balance_train=not args.no_balance_svm_train,
            balance_seed=int(args.seed),
        )
        svm_pred = _svm_predict(svm_heads, x_test_h)
        svm_scores_raw = _svm_scores(svm_heads, x_test_h)
    else:
        print("Extracting W2V2 embeddings (train then test)…", flush=True)
        x_train, y_train = extract_embeddings(
            train_recs,
            args.w2v2_name,
            args.layer,
            args.batch_size,
            dev,
            args.embedding_cache_dir.strip(),
        )
        x_test, y_test = extract_embeddings(
            test_recs,
            args.w2v2_name,
            args.layer,
            args.batch_size,
            dev,
            args.embedding_cache_dir.strip(),
        )
        print(f"Embeddings: train={x_train.shape} test={x_test.shape}", flush=True)
        print("Training SVMs (4 heads)…", flush=True)
        svm_heads = _fit_svm_ovr(
            x_train,
            y_train,
            balance_train=not args.no_balance_svm_train,
            balance_seed=int(args.seed),
        )
        svm_pred = _svm_predict(svm_heads, x_test)
        svm_scores_raw = _svm_scores(svm_heads, x_test)

    print("Running rules (zhang_full) on test waveforms…", flush=True)
    zhang_full_probs = _collect_rule_probs(
        cfg, test_paths, test_labels, zhang_full_rule_mod, dev, args.batch_size
    )

    saved = None
    if not args.ignore_saved_thresholds:
        saved = _load_thresholds(args.thresholds_json)
        if saved is not None:
            sthr = saved.get("label_vote_threshold")
            if sthr is not None and int(sthr) != int(cfg.label_vote_threshold):
                print(
                    f"[Warn] Ignoring {args.thresholds_json!r}: its label_vote_threshold={int(sthr)} "
                    f"does not match this run ({cfg.label_vote_threshold}). "
                    "Saved SVM/rule thresholds were tuned under the other policy — "
                    "re-run tune_thresholds_rules_svm.py or pass matching --label-vote-threshold.",
                    file=sys.stderr,
                    flush=True,
                )
                saved = None
            elif saved is not None:
                scol = saved.get("split_column")
                if isinstance(scol, str) and scol != cfg.split_column:
                    print(
                        f"[Warn] Thresholds JSON split_column={scol!r} differs from this run "
                        f"({cfg.split_column!r}); saved thresholds may be misaligned.",
                        file=sys.stderr,
                        flush=True,
                    )
                doc_seed = saved.get("seed")
                if doc_seed is not None and int(doc_seed) != int(args.seed):
                    print(
                        f"[Warn] Thresholds JSON seed={int(doc_seed)} differs from this run "
                        f"(--seed {int(args.seed)}); subsampling / SVM balance may not match tuning.",
                        file=sys.stderr,
                        flush=True,
                    )
    svm_th = _ordered_thresholds(saved, "svm_score_thresholds")
    if use_per_head and svm_th is not None and saved is not None:
        if saved.get("svm_feature_mode") != "per_head_w2v2_prosody":
            print(
                "[Warn] Ignoring saved svm_score_thresholds: they were tuned for a "
                "different SVM feature mode. Re-run tune_thresholds_rules_svm.py.",
                file=sys.stderr,
                flush=True,
            )
            svm_th = None
    if (
        (not use_per_head)
        and svm_th is not None
        and saved is not None
        and saved.get("svm_feature_mode") == "per_head_w2v2_prosody"
    ):
        print(
            "[Warn] Ignoring saved svm_score_thresholds: tuned on per-head W2V2+prosody "
            "but this run uses --svm-legacy-single-layer.",
            file=sys.stderr,
            flush=True,
        )
        svm_th = None
    zf_th = _ordered_thresholds(saved, "rules_zhang_full_prob_thresholds")

    if zf_th is not None:
        zhang_full_bin = (zhang_full_probs >= zf_th.reshape(1, -1)).astype(np.float32)
    else:
        zhang_full_bin = (zhang_full_probs >= float(args.f1_threshold)).astype(np.float32)

    y_true = y_test.astype(np.float64)
    if svm_th is not None:
        svm_scores = (svm_scores_raw >= svm_th.reshape(1, -1)).astype(np.float64)
        svm_mode = "scores+saved-thresholds"
    else:
        svm_scores = svm_pred.astype(np.float64)
        svm_mode = "predict() default boundary"

    # Optional Block-only improvement: calibrate Block from SVM score + pause features.
    if args.block_pause_calibration:
        print("Fitting Block pause calibrator (train+val pool)…", flush=True)
        yb_train = (y_train[:, 3] > 0.5).astype(np.int32)
        if use_per_head:
            x_train_block_score = _svm_scores(svm_heads, x_train_h)[:, 3]
        else:
            x_train_block_score = _svm_scores(svm_heads, x_train)[:, 3]
        pause_train = _block_pause_matrix(
            train_paths,
            cfg.sample_rate,
            args.block_pause_cache_dir.strip(),
            args.refresh_block_pause_cache,
        )
        pause_test = _block_pause_matrix(
            test_paths,
            cfg.sample_rate,
            args.block_pause_cache_dir.strip(),
            args.refresh_block_pause_cache,
        )
        cal = _fit_block_pause_calibrator(x_train_block_score, yb_train, pause_train)
        x_test_cal = np.column_stack([svm_scores_raw[:, 3].reshape(-1, 1), pause_test])
        p_block = cal.predict_proba(x_test_cal)[:, 1]
        svm_scores_block_cal = svm_scores.copy()
        svm_scores_block_cal[:, 3] = (p_block >= 0.5).astype(np.float64)
        m_svm_block_cal = compute_f1_metrics(y_true, svm_scores_block_cal, 0.5)
    else:
        svm_scores_block_cal = None
        m_svm_block_cal = None

    # Optional Block-only replacement: natural-pause-aware model from experiment.
    if args.block_natural_pause_v2:
        print("Fitting Block natural-pause v2 head (train internal val-tuned)…", flush=True)
        if use_per_head:
            x_block_train = x_train_h[3]
            x_block_test = x_test_h[3]
        else:
            x_block_train = x_train
            x_block_test = x_test
        block_np_pred, block_np_meta = _fit_block_natural_pause_v2(
            x_block_train=x_block_train,
            y_train=y_train,
            train_paths=train_paths,
            x_block_test=x_block_test,
            test_paths=test_paths,
            cfg=cfg,
            seed=int(args.seed),
            pause_cache_dir=args.block_pause_cache_dir.strip(),
            refresh_pause_cache=args.refresh_block_pause_cache,
            c_block=float(args.block_np_c),
            pause_neg_quantile=float(args.block_np_quantile),
        )
        svm_scores_block_np = svm_scores.copy()
        svm_scores_block_np[:, 3] = block_np_pred.astype(np.float64)
        m_svm_block_np = compute_f1_metrics(y_true, svm_scores_block_np, 0.5)
    else:
        svm_scores_block_np = None
        m_svm_block_np = None
        block_np_meta = None

    zhang_full_scores = zhang_full_bin.astype(np.float64)
    m_zhang_full = compute_f1_metrics(y_true, zhang_full_scores, 0.5)
    m_svm = compute_f1_metrics(y_true, svm_scores, 0.5)

    print("", flush=True)
    if saved is not None:
        print(f"Thresholds loaded from: {args.thresholds_json}", flush=True)
    else:
        print("Thresholds loaded from: (none) using defaults", flush=True)
    print(f"Per-type order: {F1_CLASS_NAMES}", flush=True)
    print(f"Rules (zhang_full):      {format_f1_metrics(m_zhang_full)}", flush=True)
    print(f"SVM ({svm_mode}):        {format_f1_metrics(m_svm)}", flush=True)
    if m_svm_block_cal is not None:
        print(
            f"SVM + Block pause-cal:   {format_f1_metrics(m_svm_block_cal)}",
            flush=True,
        )
        db = m_svm_block_cal.get("Block", 0.0) - m_svm.get("Block", 0.0)
        print(f"Block F1 delta (cal - base): {db:+.4f}", flush=True)
    if m_svm_block_np is not None:
        print(
            f"SVM + Block natural-v2:  {format_f1_metrics(m_svm_block_np)}",
            flush=True,
        )
        db2 = m_svm_block_np.get("Block", 0.0) - m_svm.get("Block", 0.0)
        print(f"Block F1 delta (natural-v2 - base): {db2:+.4f}", flush=True)
        print(f"Block natural-v2 meta: {block_np_meta}", flush=True)
    print(
        f"Macro-F1 (4 types):  zhang_full={_macro_f1(m_zhang_full):.4f}  "
        f"svm={_macro_f1(m_svm):.4f}"
        + (
            f"  svm_block_cal={_macro_f1(m_svm_block_cal):.4f}"
            if m_svm_block_cal is not None
            else ""
        )
        + (
            f"  svm_block_natural_v2={_macro_f1(m_svm_block_np):.4f}"
            if m_svm_block_np is not None
            else ""
        ),
        flush=True,
    )
    print(
        f"Protocol: split_column={cfg.split_column!r}  metrics_on=test  "
        f"SVM_train_pool=train+dev  rng_seed={int(args.seed)}",
        flush=True,
    )
    err_out = (args.error_analysis_out or "").strip()
    if err_out:
        _write_svm_rules_error_analysis(
            test_paths,
            y_true,
            svm_scores.astype(np.float64),
            zhang_full_scores.astype(np.float64),
            err_out,
        )
        if svm_scores_block_cal is not None:
            _write_svm_rules_error_analysis(
                test_paths,
                y_true,
                svm_scores_block_cal.astype(np.float64),
                zhang_full_scores.astype(np.float64),
                err_out + "_svm_block_pause_cal",
            )
        if svm_scores_block_np is not None:
            _write_svm_rules_error_analysis(
                test_paths,
                y_true,
                svm_scores_block_np.astype(np.float64),
                zhang_full_scores.astype(np.float64),
                err_out + "_svm_block_natural_v2",
            )
    if args.smoke or len(train_recs) < 500:
        print(
            "[Note] Small train sets often give unreliable SVM F1; "
            "use --max-train 0 --max-test 0 (omit --smoke) for a serious comparison.",
            flush=True,
        )


if __name__ == "__main__":
    main()
