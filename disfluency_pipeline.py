"""
Disfluency pipeline: 5 s clips → wav2vec2 + MFCC (+ Whisper) → BiLSTM → 4 heads.

SEP-28k: clips under data/sep28k/clips; labels in SEP-28k-Extended_clips.csv.
Splits: column SEP28k-T or SEP28k-D (values: train / dev / test). dev → validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import soundfile as sf
import torchaudio
from torchaudio.transforms import MFCC
from tqdm import tqdm

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    WhisperModel,
    WhisperProcessor,
)


# ----------------- CONFIG -----------------


@dataclass
class Config:
    data_root: str = "data/sep28k/clips"
    label_csv: str = "SEP-28k-Extended_clips.csv"
    label_source: Literal["sep28k_extended", "direct"] = "sep28k_extended"
    label_columns: Tuple[str, str, str, str] = (
        "Prolongation",
        "Repetition",
        "Interjection",
        "Block",
    )

    # Which CSV column defines train / dev / test (SEP28k-T or SEP28k-D)
    split_column: str = "SEP28k-T"

    sample_rate: int = 16000
    wav2vec2_name: str = "facebook/wav2vec2-base-960h"
    mfcc_n_mfcc: int = 40
    max_frames: int = 400

    whisper_name: str = "openai/whisper-tiny"
    use_whisper: bool = True
    whisper_proj_dim: int = 64

    hidden_size: int = 128
    num_layers: int = 1

    batch_size: int = 8
    # Resolved in main(): "cuda" if available unless forced to "cpu"
    device: str = "cpu"
    show_progress: bool = True  # tqdm bars during train/val/test
    max_clips: int = 0  # demo: cap total clips (0 = no cap)
    # quick train: cap per split (0 = use all in that split)
    max_clips_per_split: int = 0

    lr: float = 1e-3
    epochs: int = 2
    seed: int = 42

    checkpoint_dir: str = "checkpoints"
    checkpoint_last_name: str = "checkpoint_last.pt"
    checkpoint_best_name: str = "checkpoint_best.pt"

    resume_path: Optional[str] = None

    # Binary prediction threshold for F1 (sigmoid prob >= threshold → positive)
    f1_threshold: float = 0.5


CFG = Config()


def _print_device_info(device: torch.device) -> None:
    """Clarify GPU vs CPU: neural-net training is usually much faster on GPU."""
    if device.type == "cuda":
        print(
            f"Using device: {device} (CUDA GPU — typical for this model; "
            "matrix-heavy ops are usually faster than CPU.)"
        )
    else:
        print(
            f"Using device: {device} (CPU — PyTorch did not select CUDA; "
            "a GPU is usually faster for wav2vec/Whisper/training, not slower.)"
        )


# Order for printing / dict keys (these match the four model heads — no separate "any" head)
F1_CLASS_NAMES: Tuple[str, ...] = (
    "Prolongation",
    "Repetition",
    "Interjection",
    "Block",
)

# Derived metric only: binary F1 for "dysfluency present" (not trained as a class)
F1_PRESENCE_KEY = "presence"


def row_to_label_vector(row: pd.Series, cfg: Config) -> np.ndarray:
    if cfg.label_source == "sep28k_extended":
        p = float(row["Prolongation"])
        r = max(float(row["SoundRep"]), float(row["WordRep"]))
        interj = float(row["Interjection"])
        blk = float(row["Block"])
        return np.array([p, r, interj, blk], dtype=np.float32)
    return row[list(cfg.label_columns)].to_numpy(dtype=np.float32)


def _normalize_split_name(s: str) -> Optional[str]:
    s = str(s).strip().lower()
    if s == "" or s == "nan":
        return None
    if s == "dev":
        return "val"
    if s in ("train", "val", "test"):
        return s
    return None


def build_split_lists(cfg: Config) -> Tuple[
    List[str],
    List[np.ndarray],
    List[str],
    List[np.ndarray],
    List[str],
    List[np.ndarray],
    int,
]:
    """Paths + labels per split from CSV; only rows with an existing clip file."""
    df = pd.read_csv(cfg.label_csv, dtype={"EpId": str, "ClipId": str})
    if cfg.split_column not in df.columns:
        raise ValueError(
            f"Column {cfg.split_column!r} not in CSV. Use SEP28k-T or SEP28k-D."
        )
    if cfg.label_source == "sep28k_extended":
        need = ("Prolongation", "SoundRep", "WordRep", "Interjection", "Block")
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")

    train_p, train_l = [], []
    val_p, val_l = [], []
    test_p, test_l = [], []
    n_missing_file = 0

    for _, row in df.iterrows():
        sp = _normalize_split_name(row[cfg.split_column])
        if sp is None:
            continue
        show = row["Show"]
        episode = str(row["EpId"]).strip()
        clip_id = str(row["ClipId"]).strip()
        path = os.path.join(
            cfg.data_root, show, episode, f"{show}_{episode}_{clip_id}.wav"
        )
        if not os.path.exists(path):
            n_missing_file += 1
            continue
        lab = row_to_label_vector(row, cfg)
        if sp == "train":
            train_p.append(path)
            train_l.append(lab)
        elif sp == "val":
            val_p.append(path)
            val_l.append(lab)
        else:
            test_p.append(path)
            test_l.append(lab)

    if n_missing_file:
        print(
            f"[Info] Skipped {n_missing_file} CSV rows (no file under {cfg.data_root})"
        )

    (
        train_p,
        train_l,
        val_p,
        val_l,
        test_p,
        test_l,
        used_fallback,
    ) = rebalance_splits_if_needed(
        train_p, train_l, val_p, val_l, test_p, test_l, cfg.seed
    )
    if used_fallback:
        print(
            "[Warning] At least one split was empty after filtering; "
            "re-split all available clips 70% / 15% / 15% (train / val / test)."
        )

    return train_p, train_l, val_p, val_l, test_p, test_l, n_missing_file


def rebalance_splits_if_needed(
    train_p: List[str],
    train_l: List[np.ndarray],
    val_p: List[str],
    val_l: List[np.ndarray],
    test_p: List[str],
    test_l: List[np.ndarray],
    seed: int,
) -> Tuple[
    List[str],
    List[np.ndarray],
    List[str],
    List[np.ndarray],
    List[str],
    List[np.ndarray],
    bool,
]:
    """If train, val, or test is empty, pool all clips and re-split 70/15/15."""
    if min(len(train_p), len(val_p), len(test_p)) > 0:
        return train_p, train_l, val_p, val_l, test_p, test_l, False

    pool_p = train_p + val_p + test_p
    pool_l = train_l + val_l + test_l
    n = len(pool_p)
    if n == 0:
        raise RuntimeError("No clips found on disk for any split.")

    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    # Tiny pools: duplicate clips so train/val/test are all non-empty (needed for training).
    if n == 1:
        i = order[0]
        p, lab = pool_p[i], pool_l[i]
        return [p], [lab], [p], [lab], [p], [lab], True
    if n == 2:
        a, b = order[0], order[1]
        return (
            [pool_p[a]],
            [pool_l[a]],
            [pool_p[b]],
            [pool_l[b]],
            [pool_p[b]],
            [pool_l[b]],
            True,
        )

    n_tr = max(1, int(round(0.70 * n)))
    n_va = max(1, int(round(0.15 * n)))
    n_te = n - n_tr - n_va
    if n_te < 1:
        n_te = 1
        n_tr = max(1, n - n_va - n_te)
    assert n_tr + n_va + n_te == n

    tr_i = order[:n_tr]
    va_i = order[n_tr : n_tr + n_va]
    te_i = order[n_tr + n_va :]

    def take(idxs):
        return [pool_p[i] for i in idxs], [pool_l[i] for i in idxs]

    tp, tl = take(tr_i)
    vp, vl = take(va_i)
    ep, el = take(te_i)
    return tp, tl, vp, vl, ep, el, True


class DisfluencyDataset(Dataset):
    def __init__(self, cfg: Config, paths: List[str], labels: List[np.ndarray]):
        self.cfg = cfg
        self.paths = list(paths)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        labels = self.labels[idx]
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
        if sr != self.cfg.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=self.cfg.sample_rate
            )
        waveform = waveform[0:1, :]
        return {
            "waveform": waveform,
            "labels": torch.from_numpy(labels),
        }


def collate_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    max_len = max(item["waveform"].shape[-1] for item in batch)
    waveforms = []
    labels = []
    for item in batch:
        w = item["waveform"]
        pad_amount = max_len - w.shape[-1]
        if pad_amount > 0:
            w = torch.nn.functional.pad(w, (0, pad_amount))
        else:
            w = w[:, :max_len]
        waveforms.append(w)
        labels.append(item["labels"])
    return {
        "waveform": torch.stack(waveforms, dim=0),
        "labels": torch.stack(labels, dim=0),
    }


# ----------------- MODULES -----------------


class WhisperEncoderEmbedder(nn.Module):
    """Frozen Whisper encoder + trainable projection."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.processor = WhisperProcessor.from_pretrained(cfg.whisper_name)
        w = WhisperModel.from_pretrained(cfg.whisper_name)
        self.encoder = w.encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        d_model = self.encoder.config.d_model
        self.proj = nn.Linear(d_model, cfg.whisper_proj_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        B = waveform.shape[0]
        wav_list = [waveform[i, 0].detach().cpu().numpy() for i in range(B)]
        inputs = self.processor(
            wav_list,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(waveform.device)
        # Whisper was trained on 30 s windows; pad / trim mel time dim to 3000.
        t = input_features.shape[-1]
        if t < 3000:
            input_features = torch.nn.functional.pad(
                input_features, (0, 3000 - t)
            )
        elif t > 3000:
            input_features = input_features[..., :3000]
        with torch.no_grad():
            enc_out = self.encoder(input_features).last_hidden_state
        pooled = enc_out.mean(dim=1)
        return self.proj(pooled)


class MultiFeatureExtractor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.wav2vec2_name
        )
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(cfg.wav2vec2_name)
        for p in self.wav2vec2_model.parameters():
            p.requires_grad = False
        self.mfcc = MFCC(sample_rate=cfg.sample_rate, n_mfcc=cfg.mfcc_n_mfcc)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        B, _, _ = waveform.shape
        wav_flat = waveform.squeeze(1).float()
        batch_inputs = [wav_flat[i].detach().cpu().numpy() for i in range(B)]
        inputs = self.wav2vec2_feature_extractor(
            batch_inputs,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(waveform.device) for k, v in inputs.items()}
        w2v_out = self.wav2vec2_model(**inputs).last_hidden_state
        mfcc = self.mfcc(waveform)
        if mfcc.dim() == 4:
            mfcc = mfcc.squeeze(1)
        t1 = w2v_out.shape[1]
        mfcc_m = torch.nn.functional.interpolate(
            mfcc, size=t1, mode="linear", align_corners=False
        ).transpose(1, 2)
        feat = torch.cat([w2v_out, mfcc_m], dim=-1)
        return self._time_resize(feat, self.cfg.max_frames)

    @staticmethod
    def _time_resize(seq: torch.Tensor, target_len: int) -> torch.Tensor:
        b, t, d = seq.shape
        if t == target_len:
            return seq
        if t > target_len:
            seq = seq.transpose(1, 2)
            seq = torch.nn.functional.interpolate(
                seq, size=target_len, mode="linear", align_corners=False
            )
            return seq.transpose(1, 2)
        pad_t = target_len - t
        pad = torch.zeros(b, pad_t, d, device=seq.device, dtype=seq.dtype)
        return torch.cat([seq, pad], dim=1)


class TemporalDisfluencyModel(nn.Module):
    def __init__(self, cfg: Config, input_dim: int, whisper_dim: int):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=0.1)
        h2 = cfg.hidden_size * 2
        fusion = h2 + whisper_dim
        self.head_p = nn.Linear(fusion, 1)
        self.head_r = nn.Linear(fusion, 1)
        self.head_i = nn.Linear(fusion, 1)
        self.head_b = nn.Linear(fusion, 1)

    def forward(
        self, features: torch.Tensor, whisper_emb: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out, _ = self.lstm(features)
        pooled = out.mean(dim=1)
        if whisper_emb is not None:
            pooled = torch.cat([pooled, whisper_emb], dim=-1)
        pooled = self.dropout(pooled)
        return {
            "prolongation": self.head_p(pooled),
            "repetition": self.head_r(pooled),
            "interjection": self.head_i(pooled),
            "block": self.head_b(pooled),
        }


def logits_to_tensor(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            outputs["prolongation"],
            outputs["repetition"],
            outputs["interjection"],
            outputs["block"],
        ],
        dim=1,
    )


def compute_f1_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    y_true, y_probs: (N, 4) aligned with F1_CLASS_NAMES.

    Per-type: binary F1 per head (which dysfluency types are present).

    Presence (key F1_PRESENCE_KEY): single aggregate F1 — not a model output.
    Ground truth positive if any of the four labels is positive; prediction
    positive if any head's sigmoid is >= threshold. Same four heads, OR'd for
    this score only; it does not add a loss term or a fifth class.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    if y_true.size == 0:
        return {}
    y_bin = (y_true > 0.5).astype(np.int32)
    pred_bin = (y_probs >= threshold).astype(np.int32)
    out: Dict[str, float] = {}
    for i, name in enumerate(F1_CLASS_NAMES):
        out[name] = f1_score(
            y_bin[:, i], pred_bin[:, i], average="binary", zero_division=0
        )
    true_any = (y_bin.sum(axis=1) > 0).astype(np.int32)
    pred_any = (pred_bin.sum(axis=1) > 0).astype(np.int32)
    out[F1_PRESENCE_KEY] = f1_score(
        true_any, pred_any, average="binary", zero_division=0
    )
    return out


def format_f1_metrics(m: Dict[str, float]) -> str:
    if not m:
        return "(no F1)"
    pres = m.get(F1_PRESENCE_KEY)
    if pres is None:
        return "(no F1)"
    types = "  ".join(
        f"{name}={m[name]:.4f}" for name in F1_CLASS_NAMES if name in m
    )
    return (
        f"presence_F1={pres:.4f} (any type vs any head ≥ threshold)  |  "
        f"type_F1: {types}"
    )


# ----------------- DEMO & TRAIN -----------------


def _apply_max_clips(
    paths: List[str], labels: List[np.ndarray], cap: int
) -> Tuple[List[str], List[np.ndarray]]:
    if cap <= 0 or len(paths) <= cap:
        return paths, labels
    return paths[:cap], labels[:cap]


def run_demo(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    _print_device_info(device)

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    paths = tp + vp + ep
    labels = tl + vl + el
    paths, labels = _apply_max_clips(paths, labels, cfg.max_clips)
    if len(paths) == 0:
        raise RuntimeError("No clips to demo.")

    ds = DisfluencyDataset(cfg, paths, labels)
    print(f"Demo dataset size: {len(ds)} (combined splits, cap={cfg.max_clips or 'none'})")
    loader = DataLoader(
        ds,
        batch_size=min(cfg.batch_size, len(ds)),
        shuffle=True,
        collate_fn=collate_batch,
    )
    batch = next(iter(loader))
    waveforms = batch["waveform"].to(device)
    labels_t = batch["labels"].to(device)
    labels_t = torch.clamp(labels_t, 0.0, 1.0)

    print(f"Waveforms shape: {waveforms.shape}")
    print(f"Labels shape: {labels_t.shape}")

    fe = MultiFeatureExtractor(cfg).to(device).eval()
    whisper_mod: Optional[WhisperEncoderEmbedder] = None
    if cfg.use_whisper:
        whisper_mod = WhisperEncoderEmbedder(cfg).to(device).eval()
        w_emb = whisper_mod(waveforms)
        print(f"Whisper embedding shape: {w_emb.shape}")
    else:
        w_emb = None
        print("Whisper: disabled")

    with torch.no_grad():
        audio_feat = fe(waveforms)
    print(f"Audio feature sequence shape: {audio_feat.shape}")

    wdim = cfg.whisper_proj_dim if cfg.use_whisper else 0
    model = TemporalDisfluencyModel(
        cfg, input_dim=audio_feat.shape[-1], whisper_dim=wdim
    ).to(device)
    logits = logits_to_tensor(model(audio_feat, w_emb))
    print(f"Logits shape: {logits.shape}")
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels_t)
    print(f"Example batch loss: {loss.item():.4f}")
    print("Sigmoid probs (first row):", torch.sigmoid(logits[0]).detach().cpu().numpy())
    probs_np = torch.sigmoid(logits).detach().cpu().numpy()
    f1_demo = compute_f1_metrics(
        labels_t.cpu().numpy(), probs_np, cfg.f1_threshold
    )
    print(f"F1 (this batch only): {format_f1_metrics(f1_demo)}")


def _run_epoch(
    model: TemporalDisfluencyModel,
    fe: MultiFeatureExtractor,
    whisper_mod: Optional[WhisperEncoderEmbedder],
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
    *,
    epoch_idx: int = 1,
    epochs_total: int = 1,
    show_progress: bool = True,
) -> float:
    if train:
        model.train()
        if whisper_mod:
            whisper_mod.train()
    else:
        model.eval()
        if whisper_mod:
            whisper_mod.eval()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.BCEWithLogitsLoss()
    desc = f"Epoch {epoch_idx}/{epochs_total} · train"
    iterator: object = loader
    if train and show_progress:
        iterator = tqdm(
            loader,
            desc=desc,
            leave=True,
            unit="batch",
        )
    for batch in iterator:
        waveforms = batch["waveform"].to(device)
        labels_t = torch.clamp(batch["labels"].to(device), 0.0, 1.0)
        if train and optimizer:
            optimizer.zero_grad()
        with torch.no_grad():
            audio_feat = fe(waveforms)
        w_emb = whisper_mod(waveforms) if whisper_mod else None
        logits = logits_to_tensor(model(audio_feat, w_emb))
        loss = criterion(logits, labels_t)
        if train and optimizer:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if train and show_progress and isinstance(iterator, tqdm):
            iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n_batches, 1)


def evaluate_split(
    model: TemporalDisfluencyModel,
    fe: MultiFeatureExtractor,
    whisper_mod: Optional[WhisperEncoderEmbedder],
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    *,
    pbar_desc: str = "val",
    show_progress: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Average loss + F1 (presence + per-type) over the full loader."""
    model.eval()
    if whisper_mod:
        whisper_mod.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0
    iterator: object = loader
    if show_progress:
        iterator = tqdm(loader, desc=pbar_desc, leave=False, unit="batch")
    with torch.no_grad():
        for batch in iterator:
            waveforms = batch["waveform"].to(device)
            labels_t = torch.clamp(batch["labels"].to(device), 0.0, 1.0)
            audio_feat = fe(waveforms)
            w_emb = whisper_mod(waveforms) if whisper_mod else None
            logits = logits_to_tensor(model(audio_feat, w_emb))
            loss = criterion(logits, labels_t)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels_t.cpu())
            if show_progress and isinstance(iterator, tqdm):
                iterator.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = total_loss / max(n_batches, 1)
    if not all_logits:
        return avg_loss, {}
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits_cat).numpy()
    metrics = compute_f1_metrics(labels_cat.numpy(), probs, threshold)
    return avg_loss, metrics


def save_checkpoint(
    path: str,
    cfg: Config,
    next_epoch: int,
    model: TemporalDisfluencyModel,
    whisper_mod: Optional[WhisperEncoderEmbedder],
    optimizer: torch.optim.Optimizer,
    best_val: float,
) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(
        {
            "next_epoch": next_epoch,
            "cfg": cfg.__dict__,
            "model": model.state_dict(),
            "whisper": whisper_mod.state_dict() if whisper_mod else None,
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: TemporalDisfluencyModel,
    whisper_mod: Optional[WhisperEncoderEmbedder],
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if whisper_mod and ckpt.get("whisper"):
        whisper_mod.load_state_dict(ckpt["whisper"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("next_epoch", 0)), float(ckpt.get("best_val", float("inf")))


def run_train(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)
    _print_device_info(device)

    tp, tl, vp, vl, ep, el, _ = build_split_lists(cfg)
    cap = cfg.max_clips_per_split
    tp, tl = _apply_max_clips(tp, tl, cap)
    vp, vl = _apply_max_clips(vp, vl, cap)
    ep, el = _apply_max_clips(ep, el, cap)

    print(
        f"Train / val / test sizes: {len(tp)} / {len(vp)} / {len(ep)} "
        f"(per-split cap={cap or 'none'})"
    )
    if len(tp) == 0:
        raise RuntimeError("No training clips.")

    train_ds = DisfluencyDataset(cfg, tp, tl)
    val_ds = DisfluencyDataset(cfg, vp, vl) if vp else None
    test_ds = DisfluencyDataset(cfg, ep, el) if ep else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        if val_ds and len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        if test_ds and len(test_ds) > 0
        else None
    )

    fe = MultiFeatureExtractor(cfg).to(device).eval()
    whisper_mod = (
        WhisperEncoderEmbedder(cfg).to(device) if cfg.use_whisper else None
    )

    # Infer dims
    sample_w = next(iter(train_loader))["waveform"].to(device)
    with torch.no_grad():
        adim = fe(sample_w).shape[-1]
    wdim = cfg.whisper_proj_dim if cfg.use_whisper else 0

    model = TemporalDisfluencyModel(cfg, input_dim=adim, whisper_dim=wdim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + (list(whisper_mod.proj.parameters()) if whisper_mod else []),
        lr=cfg.lr,
    )

    start_epoch = 0
    best_val = float("inf")
    ckpt_dir = cfg.checkpoint_dir
    last_path = os.path.join(ckpt_dir, cfg.checkpoint_last_name)
    best_path = os.path.join(ckpt_dir, cfg.checkpoint_best_name)

    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        start_epoch, best_val = load_checkpoint(
            cfg.resume_path, model, whisper_mod, optimizer
        )
        model.to(device)
        if whisper_mod:
            whisper_mod.to(device)
        print(
            f"Resumed from {cfg.resume_path} next_epoch={start_epoch}, best_val={best_val}"
        )

    for run_step, epoch in enumerate(
        range(start_epoch, start_epoch + cfg.epochs), start=1
    ):
        tr_loss = _run_epoch(
            model,
            fe,
            whisper_mod,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch_idx=run_step,
            epochs_total=cfg.epochs,
            show_progress=cfg.show_progress,
        )
        print(
            f"Epoch {run_step}/{cfg.epochs} · train loss (mean over batches): {tr_loss:.4f}"
        )

        if val_loader:
            val_loss, val_f1 = evaluate_split(
                model,
                fe,
                whisper_mod,
                val_loader,
                device,
                cfg.f1_threshold,
                pbar_desc=f"Epoch {run_step}/{cfg.epochs} · val",
                show_progress=cfg.show_progress,
            )
            print(f"  val loss: {val_loss:.4f}")
            print(f"  val F1:   {format_f1_metrics(val_f1)}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    cfg,
                    epoch + 1,
                    model,
                    whisper_mod,
                    optimizer,
                    best_val,
                )
                print(f"  saved best → {best_path}")

        save_checkpoint(
            last_path,
            cfg,
            epoch + 1,
            model,
            whisper_mod,
            optimizer,
            best_val,
        )
        print(f"  saved last → {last_path}")

    if test_loader:
        te_loss, te_f1 = evaluate_split(
            model,
            fe,
            whisper_mod,
            test_loader,
            device,
            cfg.f1_threshold,
            pbar_desc="Test",
            show_progress=cfg.show_progress,
        )
        print(f"Test loss: {te_loss:.4f}")
        print(f"Test F1:   {format_f1_metrics(te_f1)}")


# ----------------- CLI -----------------


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was set but torch.cuda.is_available() is False."
        )
    return choice


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Disfluency pipeline")
    p.add_argument(
        "--mode",
        choices=("demo", "train"),
        default="demo",
        help="demo: one batch forward; train: short training loop",
    )
    p.add_argument("--max-clips", type=int, default=32, help="Demo: max clips total")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--split-column", type=str, default="SEP28k-T")
    p.add_argument("--no-whisper", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--max-clips-per-split",
        type=int,
        default=0,
        help="Train: max clips per split after CSV order (0 = use all on disk).",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--f1-threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for per-type F1 and for OR-ing heads into presence F1.",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto: use CUDA if available; otherwise force cpu or cuda.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. for log files).",
    )

    args = p.parse_args()
    cfg = Config()
    cfg.device = _resolve_device(args.device)
    cfg.show_progress = not args.no_progress
    cfg.split_column = args.split_column
    cfg.use_whisper = not args.no_whisper
    cfg.f1_threshold = args.f1_threshold
    if args.mode == "demo":
        cfg.max_clips = args.max_clips
        cfg.batch_size = args.batch_size
        run_demo(cfg)
    else:
        cfg.epochs = args.epochs
        cfg.batch_size = args.batch_size
        cfg.lr = args.lr
        cfg.max_clips_per_split = args.max_clips_per_split
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.resume_path = args.resume or None
        run_train(cfg)


if __name__ == "__main__":
    main()
