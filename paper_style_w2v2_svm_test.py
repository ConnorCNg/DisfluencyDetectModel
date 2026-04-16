"""
Small paper-style experiment:
- Extract clip embeddings from wav2vec2 hidden states (mean-pooled over time)
- Train one-vs-rest binary SVMs for 5 dysfluency labels
- Report per-label F1 on a held-out test subset
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


LABELS = ("Block", "Prolongation", "SoundRep", "WordRep", "Interjection")


@dataclass
class ClipRecord:
    path: str
    split: str
    labels: np.ndarray  # binary, shape (5,) for five-head mode or (4,) for pipeline order


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
    df = pd.read_csv(csv_path, dtype={"EpId": str, "ClipId": str})
    required = {"Show", "EpId", "ClipId", split_column, *LABELS}
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

        vals = np.array(
            [float(row["Block"]), float(row["Prolongation"]), float(row["SoundRep"]), float(row["WordRep"]), float(row["Interjection"])],
            dtype=np.float32,
        )
        y = (vals >= float(vote_threshold)).astype(np.int32)
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

    def cache_fp(path: str) -> str:
        key = hashlib.sha256(
            f"{model_name}|layer={layer}|sr={sample_rate}|{os.path.abspath(path)}".encode("utf-8")
        ).hexdigest()
        return os.path.join(cache_dir, f"{key}.npy")

    for i in tqdm(range(0, len(records), batch_size), desc="Extract embeddings", unit="batch"):
        chunk = records[i : i + batch_size]
        emb_np: List[np.ndarray | None] = []
        uncached_wavs: List[np.ndarray] = []
        uncached_idx: List[int] = []
        for j, r in enumerate(chunk):
            if cache_dir:
                fp = cache_fp(r.path)
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
                    np.save(cache_fp(chunk[j].path), uncached_emb[k])
                cache_misses += 1

        xs.append(np.stack([e for e in emb_np if e is not None], axis=0))
        ys.append(np.stack([r.labels for r in chunk], axis=0))

    x = np.concatenate(xs, axis=0) if xs else np.zeros((0, 768), dtype=np.float32)
    _ldim = int(records[0].labels.shape[0]) if records else len(LABELS)
    y = np.concatenate(ys, axis=0) if ys else np.zeros((0, _ldim), dtype=np.int32)
    if cache_dir:
        print(
            f"Embedding cache: hits={cache_hits} misses={cache_misses} "
            f"(dir={cache_dir})",
            flush=True,
        )
    return x, y


def run_svm_ovr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for i, label in enumerate(LABELS):
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
    ap.add_argument("--vote-threshold", type=int, default=2)
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
    for k in LABELS:
        print(f"  {k}: {f1[k]:.4f}", flush=True)
    print(f"Macro-F1: {float(np.mean([f1[k] for k in LABELS])):.4f}", flush=True)


if __name__ == "__main__":
    main()
