"""
Small paper-style experiment:
- Extract clip embeddings from wav2vec2 hidden states (mean-pooled over time)
- Train one-vs-rest binary SVMs for the same four dysfluency heads as ``disfluency_pipeline``
  (Repetition = max(SoundRep, WordRep) binarized like ``row_to_label_vector``).
- Report per-label F1 on a held-out test subset

Labels and splits must come from ``SEP-28k-Extended_clips.csv`` (enforced).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from disfluency_pipeline import (
    Config,
    F1_CLASS_NAMES,
    assert_sep28k_extended_dysfluency_csv,
    row_to_label_vector,
)


@dataclass
class ClipRecord:
    path: str
    split: str
    labels: np.ndarray  # binary, shape (4,) in F1_CLASS_NAMES order


def normalize_split(s: str) -> str | None:
    t = str(s).strip().lower()
    if t == "dev":
        return "val"
    if t in ("train", "val", "test"):
        return t
    return None


def load_records(
    csv_path: str,
    data_root: str,
    split_column: str,
    vote_threshold: int,
) -> List[ClipRecord]:
    assert_sep28k_extended_dysfluency_csv(csv_path)
    vote_cfg = Config()
    vote_cfg.label_vote_threshold = max(0, int(vote_threshold))

    df = pd.read_csv(csv_path, dtype={"EpId": str, "ClipId": str})
    required = {
        "Show",
        "EpId",
        "ClipId",
        split_column,
        "Prolongation",
        "SoundRep",
        "WordRep",
        "Interjection",
        "Block",
    }
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out: List[ClipRecord] = []
    for _, row in df.iterrows():
        split = normalize_split(row[split_column])
        if split is None:
            continue
        show = str(row["Show"])
        ep = str(row["EpId"]).strip()
        cid = str(row["ClipId"]).strip()
        path = os.path.join(data_root, show, ep, f"{show}_{ep}_{cid}.wav")
        if not os.path.exists(path):
            continue

        lab = row_to_label_vector(row, vote_cfg)
        y = (lab >= 0.5).astype(np.int32)
        out.append(ClipRecord(path=path, split=split, labels=y))
    return out


def sample_split(
    records: List[ClipRecord],
    max_train: int,
    max_test: int,
    seed: int,
) -> Tuple[List[ClipRecord], List[ClipRecord]]:
    train_pool = [r for r in records if r.split in ("train", "val")]
    test_pool = [r for r in records if r.split == "test"]
    rng = random.Random(seed)
    rng.shuffle(train_pool)
    rng.shuffle(test_pool)
    tr = train_pool[:max_train] if max_train > 0 else train_pool
    te = test_pool[:max_test] if max_test > 0 else test_pool
    return tr, te


def load_svm_clean03_head_bundle(path: str) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float], int]:
    """
    Load per-head Wav2Vec2 layer, SVM C, and prosody width from ``svm_clean03_layer_prosody_sweep`` JSON.

    Returns ``(layers, C_values, prosody_dim)`` with tuple order matching ``F1_CLASS_NAMES``.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    res = d.get("results")
    if not isinstance(res, dict):
        raise ValueError(f"{path!r}: expected a 'results' object")
    layers: List[int] = []
    cs: List[float] = []
    for h in F1_CLASS_NAMES:
        if h not in res:
            raise ValueError(f"{path!r}: missing results[{h!r}]")
        block = res[h]
        layers.append(int(block["layer"]))
        cs.append(float(block["C"]))
    pdim = int(d.get("prosody_dim", 9))
    return (tuple(layers), tuple(cs), pdim)


def _prosody_cache_fp(cache_dir: str, path: str, sr: int) -> str:
    k = hashlib.sha256(f"prosody|sr={sr}|{os.path.abspath(path)}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{k}.npy")


def _compute_prosody(path: str, target_sr: int) -> np.ndarray:
    """9-D clip-level prosody vector (aligned with svm_clean03_layer_prosody_sweep)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1).astype(np.float32)
    if y.size == 0:
        return np.zeros(9, dtype=np.float32)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.size == 0:
        return np.zeros(9, dtype=np.float32)
    eps = 1e-8
    frame_length = int(0.025 * target_sr)
    hop = int(0.010 * target_sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + eps, ref=2e-5)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop, center=True)[0]
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


def prosody_matrix_for_paths(
    paths: Sequence[str], cache_dir: str, sr: int, refresh: bool
) -> np.ndarray:
    if not paths:
        return np.zeros((0, 9), dtype=np.float32)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    out: List[np.ndarray] = []
    for p in paths:
        if cache_dir:
            fp = _prosody_cache_fp(cache_dir, p, sr)
            if (not refresh) and os.path.exists(fp):
                out.append(np.load(fp))
                continue
            f = _compute_prosody(p, sr)
            np.save(fp, f)
            out.append(f)
        else:
            out.append(_compute_prosody(p, sr))
    return np.stack(out, axis=0)


def _w2v2_layer_cache_fp(
    cache_dir: str, model_name: str, layer: int, sample_rate: int, path: str
) -> str:
    key = hashlib.sha256(
        f"{model_name}|layer={layer}|sr={sample_rate}|{os.path.abspath(path)}".encode("utf-8")
    ).hexdigest()
    return os.path.join(cache_dir, f"{key}.npy")


def load_waveform(path: str, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(data.T)[0:1, :]
    if sr != target_sr:
        w = torchaudio.functional.resample(w, orig_freq=sr, new_freq=target_sr)
    return w.squeeze(0)


def extract_embeddings(
    records: List[ClipRecord],
    model_name: str,
    layer: int,
    batch_size: int,
    device: torch.device,
    cache_dir: str,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True).to(device).eval()

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    cache_hits = 0
    cache_misses = 0

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for i in tqdm(range(0, len(records), batch_size), desc="Extract embeddings", unit="batch"):
        chunk = records[i : i + batch_size]
        emb_np: List[np.ndarray | None] = []
        uncached_wavs: List[np.ndarray] = []
        uncached_idx: List[int] = []
        for j, r in enumerate(chunk):
            if cache_dir:
                fp = _w2v2_layer_cache_fp(cache_dir, model_name, layer, sample_rate, r.path)
                if os.path.exists(fp):
                    emb_np.append(np.load(fp))
                    cache_hits += 1
                    continue
            emb_np.append(None)
            uncached_idx.append(j)
            uncached_wavs.append(load_waveform(r.path, sample_rate).numpy())

        if uncached_wavs:
            inputs = fe(uncached_wavs, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("hidden_states not returned")
            if layer < 0 or layer >= len(hs):
                raise ValueError(f"layer={layer} out of range [0, {len(hs)-1}]")
            uncached_emb = hs[layer].mean(dim=1).cpu().numpy()  # (U, D)
            for k, j in enumerate(uncached_idx):
                emb_np[j] = uncached_emb[k]
                if cache_dir:
                    np.save(
                        _w2v2_layer_cache_fp(
                            cache_dir, model_name, layer, sample_rate, chunk[j].path
                        ),
                        uncached_emb[k],
                    )
                cache_misses += 1

        xs.append(np.stack([e for e in emb_np if e is not None], axis=0))
        ys.append(np.stack([r.labels for r in chunk], axis=0))

    x = np.concatenate(xs, axis=0) if xs else np.zeros((0, 768), dtype=np.float32)
    _ldim = int(records[0].labels.shape[0]) if records else len(F1_CLASS_NAMES)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0, _ldim), dtype=np.int32)
    if cache_dir:
        print(
            f"Embedding cache: hits={cache_hits} misses={cache_misses} "
            f"(dir={cache_dir})",
            flush=True,
        )
    return x, y


def extract_per_head_w2v2_prosody(
    records: List[ClipRecord],
    model_name: str,
    layers_by_head: Tuple[int, int, int, int],
    batch_size: int,
    device: torch.device,
    embedding_cache_dir: str,
    sample_rate: int,
    prosody_cache_dir: str,
    refresh_prosody: bool,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    For each dysfluency head, build ``concat(mean-pool W2V2 layer L_h, prosody_9)`` per clip.

    Uses the same per-layer ``.npy`` cache keys as :func:`extract_embeddings` / the layer sweep.
    ``layers_by_head`` follows ``F1_CLASS_NAMES`` order.
    """
    if not records:
        z = np.zeros((0, len(F1_CLASS_NAMES)), dtype=np.int32)
        return ([np.zeros((0, 768 + 9), dtype=np.float32) for _ in F1_CLASS_NAMES], z)

    paths = [r.path for r in records]
    pros = prosody_matrix_for_paths(paths, prosody_cache_dir, sample_rate, refresh_prosody)
    feat_dim = 768 + int(pros.shape[1])
    unique_layers = sorted(set(int(x) for x in layers_by_head))

    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(model_name, local_files_only=True).to(device).eval()

    n = len(records)
    out_heads = [np.zeros((n, feat_dim), dtype=np.float32) for _ in F1_CLASS_NAMES]

    cache_hits = 0
    cache_misses = 0
    if embedding_cache_dir:
        os.makedirs(embedding_cache_dir, exist_ok=True)

    for i0 in tqdm(
        range(0, n, batch_size), desc="Extract per-head W2V2+prosody", unit="batch"
    ):
        chunk = records[i0 : i0 + batch_size]
        b = len(chunk)
        pros_b = pros[i0 : i0 + b].astype(np.float32, copy=False)

        # emb_slot[L][j_local] = (768,) or None if still need fill from batch forward
        emb_slot: Dict[int, List[np.ndarray | None]] = {
            L: [None] * b for L in unique_layers
        }

        need_forward_idx: List[int] = []
        forward_wavs: List[np.ndarray] = []

        for j, r in enumerate(chunk):
            need_forward_row = False
            for L in unique_layers:
                if embedding_cache_dir:
                    fp = _w2v2_layer_cache_fp(
                        embedding_cache_dir, model_name, L, sample_rate, r.path
                    )
                    if os.path.exists(fp):
                        emb_slot[L][j] = np.load(fp).astype(np.float32, copy=False)
                        cache_hits += 1
                        continue
                need_forward_row = True
            if need_forward_row:
                need_forward_idx.append(j)
                forward_wavs.append(load_waveform(r.path, sample_rate).numpy())

        if forward_wavs:
            inputs = fe(forward_wavs, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            if hs is None:
                raise RuntimeError("hidden_states not returned")
            for L in unique_layers:
                if L < 0 or L >= len(hs):
                    raise ValueError(f"layer={L} out of range [0, {len(hs) - 1}]")
            for k, j in enumerate(need_forward_idx):
                r = chunk[j]
                for L in unique_layers:
                    if emb_slot[L][j] is not None:
                        continue
                    vec = hs[L][k].mean(dim=0).detach().float().cpu().numpy()
                    emb_slot[L][j] = vec
                    cache_misses += 1
                    if embedding_cache_dir:
                        fp = _w2v2_layer_cache_fp(
                            embedding_cache_dir, model_name, L, sample_rate, r.path
                        )
                        np.save(fp, vec)

        for j in range(b):
            for hi, _ in enumerate(F1_CLASS_NAMES):
                Lh = int(layers_by_head[hi])
                wvec = emb_slot[Lh][j]
                if wvec is None:
                    raise RuntimeError("internal: missing embedding slot")
                out_heads[hi][i0 + j] = np.concatenate(
                    [wvec.astype(np.float32, copy=False), pros_b[j]], axis=0
                )

    y = np.stack([r.labels for r in records], axis=0)
    if embedding_cache_dir:
        print(
            f"Per-head embedding cache: hits={cache_hits} misses={cache_misses} "
            f"(dir={embedding_cache_dir})",
            flush=True,
        )
    return (out_heads, y)


def run_svm_ovr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, label in enumerate(F1_CLASS_NAMES):
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
        )
        clf.fit(x_train, y_train[:, i])
        pred = clf.predict(x_test)
        out[label] = f1_score(y_test[:, i], pred, average="binary", zero_division=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Small paper-style W2V2+SVM test.")
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument(
        "--vote-threshold",
        type=int,
        default=3,
        help="Binarize SEP-28k-Extended vote counts: present if count >= threshold (3 = unanimous).",
    )
    ap.add_argument("--max-train", type=int, default=800)
    ap.add_argument("--max-test", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--w2v2-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--layer", type=int, default=8, help="W2V2 hidden-state layer index.")
    ap.add_argument(
        "--embedding-cache-dir",
        default=".cache/w2v2_embeddings",
        help="Directory for per-clip cached W2V2 embeddings. Empty string disables cache.",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
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
    if dev.type == "mps":
        # Some local Python/PyTorch builds report mps but cannot move modules to it.
        try:
            _ = torch.zeros(1).to(dev)
        except Exception:
            print("MPS unavailable in this runtime; falling back to CPU.", flush=True)
            dev = torch.device("cpu")

    print(f"Using device: {dev}", flush=True)
    records = load_records(args.csv, args.data_root, args.split_column, args.vote_threshold)
    tr, te = sample_split(records, args.max_train, args.max_test, args.seed)
    print(
        f"Records: total={len(records)} train_pool={len([r for r in records if r.split in ('train','val')])} "
        f"test_pool={len([r for r in records if r.split=='test'])}",
        flush=True,
    )
    print(
        f"Sampled: train={len(tr)} test={len(te)} vote_threshold={args.vote_threshold} layer={args.layer}",
        flush=True,
    )
    if not tr or not te:
        raise RuntimeError("Empty train/test sample; adjust split/sample params.")

    print(
        "SVM params: kernel=rbf, C=10.0, gamma=scale, class_weight=balanced",
        flush=True,
    )
    x_train, y_train = extract_embeddings(
        tr,
        args.w2v2_name,
        args.layer,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
    )
    x_test, y_test = extract_embeddings(
        te,
        args.w2v2_name,
        args.layer,
        args.batch_size,
        dev,
        args.embedding_cache_dir.strip(),
    )
    print(f"Embedding shape: train={x_train.shape} test={x_test.shape}", flush=True)

    f1 = run_svm_ovr(x_train, y_train, x_test, y_test)
    print("Per-label test F1:", flush=True)
    for k in F1_CLASS_NAMES:
        print(f"  {k}: {f1[k]:.4f}", flush=True)
    print(f"Macro-F1: {float(np.mean([f1[k] for k in F1_CLASS_NAMES])):.4f}", flush=True)


if __name__ == "__main__":
    main()
