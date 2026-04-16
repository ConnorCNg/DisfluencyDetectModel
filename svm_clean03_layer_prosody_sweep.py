#!/usr/bin/env python3
"""
Per-dysfluency SVM with clean 0/3 labels, layer sweep, and prosodic features.

Steps:
1) Build train/dev/test pools from SEP-28k split column.
2) For each head, keep only clips whose target vote is exactly 0 or 3.
3) (Optionally) refresh/re-cache wav2vec2 embeddings for layers 5..12.
4) Add cached prosodic features per clip.
5) Per head: tune layer + C + decision threshold on dev, then evaluate test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch

from paper_style_w2v2_svm_test import ClipRecord, extract_embeddings


HEADS = ("Prolongation", "Repetition", "Interjection", "Block")
VOTE_COLS = ("Prolongation", "SoundRep", "WordRep", "Interjection", "Block")


@dataclass
class RowRec:
    path: str
    split: str  # train|val|test
    votes: Dict[str, int]


def _norm_split(s: str) -> str | None:
    t = str(s).strip().lower()
    if t == "dev":
        return "val"
    if t in ("train", "val", "test"):
        return t
    return None


def _target_vote(v: Dict[str, int], head: str) -> int:
    if head == "Prolongation":
        return int(v["Prolongation"])
    if head == "Repetition":
        return max(int(v["SoundRep"]), int(v["WordRep"]))
    if head == "Interjection":
        return int(v["Interjection"])
    if head == "Block":
        return int(v["Block"])
    raise ValueError(head)


def load_rows(csv_path: str, data_root: str, split_column: str) -> List[RowRec]:
    out: List[RowRec] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sp = _norm_split(row.get(split_column, ""))
            if sp is None:
                continue
            show = str(row["Show"])
            ep = str(row["EpId"]).strip()
            cid = str(row["ClipId"]).strip()
            path = os.path.join(data_root, show, ep, f"{show}_{ep}_{cid}.wav")
            if not os.path.exists(path):
                continue
            votes = {k: int(float(row[k])) for k in VOTE_COLS}
            out.append(RowRec(path=path, split=sp, votes=votes))
    return out


def split_for_head(rows: Sequence[RowRec], head: str) -> Tuple[List[RowRec], List[RowRec], List[RowRec]]:
    tr, dv, te = [], [], []
    for r in rows:
        t = _target_vote(r.votes, head)
        if t not in (0, 3):
            continue
        if r.split == "train":
            tr.append(r)
        elif r.split == "val":
            dv.append(r)
        elif r.split == "test":
            te.append(r)
    return tr, dv, te


def to_clip_records(rows: Sequence[RowRec], head: str) -> List[ClipRecord]:
    recs: List[ClipRecord] = []
    for r in rows:
        y = 1 if _target_vote(r.votes, head) == 3 else 0
        recs.append(ClipRecord(path=r.path, split=r.split, labels=np.array([y], dtype=np.int32)))
    return recs


def _prosody_cache_fp(cache_dir: str, path: str, sr: int) -> str:
    k = hashlib.sha256(f"prosody|sr={sr}|{os.path.abspath(path)}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{k}.npy")


def _emb_cache_fp(cache_dir: str, model_name: str, layer: int, sample_rate: int, path: str) -> str:
    key = hashlib.sha256(
        f"{model_name}|layer={layer}|sr={sample_rate}|{os.path.abspath(path)}".encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def _compute_prosody(path: str, target_sr: int) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1).astype(np.float32)
    if y.size == 0:
        return np.zeros(9, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.size == 0:
        return np.zeros(9, dtype=np.float32)
    eps = 1e-8
    # 25ms / 10ms as in many speech setups
    frame_length = int(0.025 * target_sr)
    hop = int(0.010 * target_sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + eps, ref=2e-5)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop, center=True)[0]
    # Pitch statistics via YIN (robust enough for short clips)
    try:
        f0 = librosa.yin(y, fmin=80, fmax=400, sr=target_sr, frame_length=frame_length, hop_length=hop)
        f0 = f0[np.isfinite(f0)]
    except Exception:
        f0 = np.array([], dtype=np.float32)
    if f0.size == 0:
        f0_mean = 0.0
        f0_std = 0.0
        voiced_ratio = 0.0
    else:
        f0_mean = float(np.mean(f0))
        f0_std = float(np.std(f0))
        voiced_ratio = float(min(1.0, f0.size / max(1, rms.size)))
    silence_ratio = float(np.mean(rms_db < -35.0))
    feat = np.array(
        [
            float(len(y) / target_sr),
            float(np.mean(rms_db)),
            float(np.std(rms_db)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            f0_mean,
            f0_std,
            voiced_ratio,
            silence_ratio,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)


def prosody_features(rows: Sequence[RowRec], cache_dir: str, sr: int, refresh: bool) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    out: List[np.ndarray] = []
    for r in rows:
        fp = _prosody_cache_fp(cache_dir, r.path, sr)
        if (not refresh) and os.path.exists(fp):
            out.append(np.load(fp))
            continue
        f = _compute_prosody(r.path, sr)
        np.save(fp, f)
        out.append(f)
    if not out:
        return np.zeros((0, 9), dtype=np.float32)
    return np.stack(out, axis=0)


def tune_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    # Quantile sweep for robust threshold search
    q = np.linspace(0.05, 0.95, 81)
    grid = np.unique(np.quantile(score, q))
    best_t = 0.0
    best_f = -1.0
    for t in grid:
        pred = (score >= t).astype(np.int32)
        f = f1_score(y_true, pred, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t


def main() -> None:
    ap = argparse.ArgumentParser(description="0/3 per-head SVM with layer+prosody sweep.")
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--layers", default="5,6,7,8,9,10,11,12")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--embedding-cache-dir", default=".cache/w2v2_embeddings")
    ap.add_argument("--prosody-cache-dir", default=".cache/prosody_features")
    ap.add_argument("--refresh-embedding-cache", action="store_true")
    ap.add_argument("--refresh-prosody-cache", action="store_true")
    ap.add_argument("--c-grid", default="0.1,1,3,10,30")
    ap.add_argument("--out-json", default="artifacts/svm_clean03_best_configs.json")
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

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    c_grid = [float(x.strip()) for x in args.c_grid.split(",") if x.strip()]
    rows = load_rows(args.csv, args.data_root, args.split_column)
    if not rows:
        raise RuntimeError("No rows found after split/path filtering.")

    print(f"Using device: {dev}", flush=True)
    print(f"Rows loaded: {len(rows)}", flush=True)
    print(f"Layers: {layers}", flush=True)
    print(f"C grid: {c_grid}", flush=True)

    # Pre-cache embeddings across all rows for each layer (requested 5..12 recache)
    all_recs = [ClipRecord(path=r.path, split=r.split, labels=np.array([0], dtype=np.int32)) for r in rows]
    for layer in layers:
        if args.refresh_embedding_cache and args.embedding_cache_dir.strip():
            n_del = 0
            for r in rows:
                fp = _emb_cache_fp(args.embedding_cache_dir.strip(), args.w2v2_name, layer, 16000, r.path)
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                        n_del += 1
                    except OSError:
                        pass
            print(f"[Cache] removed {n_del} embedding cache files for layer {layer}", flush=True)
        print(f"[Cache] W2V2 layer {layer}", flush=True)
        _ = extract_embeddings(
            all_recs,
            args.w2v2_name,
            layer,
            args.batch_size,
            dev,
            args.embedding_cache_dir.strip(),
        )

    results = {}
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    for head in HEADS:
        tr_rows, dv_rows, te_rows = split_for_head(rows, head)
        if not tr_rows or not dv_rows or not te_rows:
            print(f"[Skip] {head} (empty split after 0/3 filtering)", flush=True)
            continue
        print(
            f"\n[{head}] clean 0/3 counts train={len(tr_rows)} dev={len(dv_rows)} test={len(te_rows)}",
            flush=True,
        )

        y_tr = np.array([1 if _target_vote(r.votes, head) == 3 else 0 for r in tr_rows], dtype=np.int32)
        y_dv = np.array([1 if _target_vote(r.votes, head) == 3 else 0 for r in dv_rows], dtype=np.int32)
        y_te = np.array([1 if _target_vote(r.votes, head) == 3 else 0 for r in te_rows], dtype=np.int32)

        # Prosody once per head/split
        p_tr = prosody_features(tr_rows, args.prosody_cache_dir, 16000, args.refresh_prosody_cache)
        p_dv = prosody_features(dv_rows, args.prosody_cache_dir, 16000, args.refresh_prosody_cache)
        p_te = prosody_features(te_rows, args.prosody_cache_dir, 16000, args.refresh_prosody_cache)

        best = {
            "layer": None,
            "C": None,
            "threshold": None,
            "dev_f1": -1.0,
            "test_f1": None,
            "test_precision": None,
            "test_recall": None,
            "n_train": int(len(tr_rows)),
            "n_dev": int(len(dv_rows)),
            "n_test": int(len(te_rows)),
        }

        tr_recs = to_clip_records(tr_rows, head)
        dv_recs = to_clip_records(dv_rows, head)
        te_recs = to_clip_records(te_rows, head)

        for layer in layers:
            x_tr_emb, _ = extract_embeddings(
                tr_recs, args.w2v2_name, layer, args.batch_size, dev, args.embedding_cache_dir.strip()
            )
            x_dv_emb, _ = extract_embeddings(
                dv_recs, args.w2v2_name, layer, args.batch_size, dev, args.embedding_cache_dir.strip()
            )
            x_te_emb, _ = extract_embeddings(
                te_recs, args.w2v2_name, layer, args.batch_size, dev, args.embedding_cache_dir.strip()
            )
            x_tr = np.concatenate([x_tr_emb, p_tr], axis=1)
            x_dv = np.concatenate([x_dv_emb, p_dv], axis=1)
            x_te = np.concatenate([x_te_emb, p_te], axis=1)

            for c in c_grid:
                clf = make_pipeline(
                    StandardScaler(),
                    SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced"),
                )
                clf.fit(x_tr, y_tr)
                dv_score = clf.decision_function(x_dv)
                th = tune_threshold(y_dv, dv_score)
                dv_pred = (dv_score >= th).astype(np.int32)
                dv_f1 = float(f1_score(y_dv, dv_pred, zero_division=0))
                if dv_f1 > best["dev_f1"]:
                    te_score = clf.decision_function(x_te)
                    te_pred = (te_score >= th).astype(np.int32)
                    tp = int(np.sum((te_pred == 1) & (y_te == 1)))
                    fp = int(np.sum((te_pred == 1) & (y_te == 0)))
                    fn = int(np.sum((te_pred == 0) & (y_te == 1)))
                    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    te_f1 = float(f1_score(y_te, te_pred, zero_division=0))
                    best.update(
                        {
                            "layer": int(layer),
                            "C": float(c),
                            "threshold": float(th),
                            "dev_f1": float(dv_f1),
                            "test_f1": float(te_f1),
                            "test_precision": float(prec),
                            "test_recall": float(rec),
                            "feature_dim": int(x_tr.shape[1]),
                        }
                    )
        results[head] = best
        print(
            f"[{head}] best layer={best['layer']} C={best['C']} th={best['threshold']:.4f} "
            f"dev_f1={best['dev_f1']:.4f} test_f1={best['test_f1']:.4f}",
            flush=True,
        )

    payload = {
        "split_column": args.split_column,
        "protocol": "clean_0_vs_3_per_head",
        "layers": layers,
        "c_grid": c_grid,
        "prosody_dim": 9,
        "results": results,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {args.out_json}", flush=True)

    print("\nSummary (test F1):", flush=True)
    for h in HEADS:
        if h in results:
            r = results[h]
            print(
                f"  {h}: F1={r['test_f1']:.4f}  (layer={r['layer']} C={r['C']} th={r['threshold']:.4f})",
                flush=True,
            )


if __name__ == "__main__":
    main()
