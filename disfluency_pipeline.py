"""
Disfluency pipeline: 5 s clips → wav2vec2 + MFCC → BiLSTM → 4 heads.

SEP-28k: clips under data/sep28k/clips; dysfluency votes only from SEP-28k-Extended_clips.csv
(not SEP-28k_episodes.csv, which is for URLs/downloads).
Splits: column SEP28k-T or SEP28k-D (values: train / dev / test). dev → validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import hashlib
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
    # SEP-28k-Extended: each dysfluency column is an integer vote count in {0,1,2,3} from
    # three annotators. Positive when count >= label_vote_threshold; use 3 for strict
    # unanimous positives only (aligns with "0 vs 3" cleanness on the vote scale).
    label_vote_threshold: int = 3

    # Stacking-style logit fusion (rule logits detached; no gradients into rules).
    # "none" = identical to backbone-only training/inference.
    fusion_mode: Literal["none", "linear", "gated"] = "none"
    # For the first N whole epochs (1-based), train on backbone logits only; then
    # BCE on fused logits so the fusion layer can calibrate. 0 = always use fused
    # loss when fusion_mode is not none.
    fusion_nn_only_epochs: int = 1

    # Rule logits fused with the neural heads (see zhang_rules.py, zhang_full/).
    rule_mode: Literal["none", "zhang", "zhang_full"] = "none"
    # JSON cache directory from precompute_zhang_full_cache.py (empty = on-the-fly acoustic only).
    zhang_full_cache_dir: str = ""
    # Optional cache directories for frozen embeddings (empty = disabled).
    audio_feature_cache_dir: str = ""


CFG = Config()

# Official per-clip annotator vote table (Show, EpId, ClipId, dysfluency columns, splits).
SEP28K_EXTENDED_CLIPS_CSV = "SEP-28k-Extended_clips.csv"


def assert_sep28k_extended_dysfluency_csv(csv_path: str) -> None:
    """
    Enforce that SEP-28k dysfluency annotations are read from the Extended clips CSV only.

    ``SEP-28k_episodes.csv`` is for downloads only; it does not carry per-clip dysfluency votes.
    """
    name = os.path.basename(os.path.normpath(csv_path))
    if name != SEP28K_EXTENDED_CLIPS_CSV:
        raise ValueError(
            f"Dysfluency labels must be loaded from {SEP28K_EXTENDED_CLIPS_CSV!r} "
            f"(got filename {name!r} from path {csv_path!r}). "
            "SEP-28k-Extended_clips.csv has Prolongation, Block, SoundRep, WordRep, Interjection, "
            "and split columns SEP28k-T / SEP28k-D."
        )


def _print_device_info(device: torch.device) -> None:
    """Clarify GPU vs CPU: neural-net training is usually much faster on GPU."""
    if device.type == "cuda":
        print(
            f"Using device: {device} (CUDA GPU — typical for this model; "
            "matrix-heavy ops are usually faster than CPU.)"
        )
    elif device.type == "mps":
        print(
            f"Using device: {device} (Apple Silicon GPU via MPS — typically faster "
            "than CPU for this model.)"
        )
    else:
        print(
            f"Using device: {device} (CPU — PyTorch did not select CUDA; "
            "a GPU is usually faster for wav2vec/training, not slower.)"
        )


# Order for printing / dict keys (these match the four model heads — no separate "any" head)
F1_CLASS_NAMES: Tuple[str, ...] = (
    "Prolongation",
    "Repetition",
    "Interjection",
    "Block",
)

# `TemporalDisfluencyModel` output keys (column order for logits tensor)
HEAD_LOGIT_KEYS: Tuple[str, ...] = (
    "prolongation",
    "repetition",
    "interjection",
    "block",
)

# Derived metric only: binary F1 for "dysfluency present" (not trained as a class)
F1_PRESENCE_KEY = "presence"


def row_to_label_vector(row: pd.Series, cfg: Config) -> np.ndarray:
    """
    Build the 4-head training label vector for SEP-28k-Extended (same order as F1_CLASS_NAMES).

    Per-head vote columns (0–3 annotator counts):
    - Prolongation ← ``Prolongation``
    - Repetition ← ``max(SoundRep, WordRep)`` (two columns, one repetition head)
    - Interjection ← ``Interjection``
    - Block ← ``Block``

    Each head is binarized with ``cfg.label_vote_threshold`` (default 3 = positive only
    when all three annotators agreed).
    """
    if cfg.label_source == "sep28k_extended":
        thr = float(cfg.label_vote_threshold)
        p = 1.0 if float(row["Prolongation"]) >= thr else 0.0
        r = (
            1.0
            if max(float(row["SoundRep"]), float(row["WordRep"])) >= thr
            else 0.0
        )
        interj = 1.0 if float(row["Interjection"]) >= thr else 0.0
        blk = 1.0 if float(row["Block"]) >= thr else 0.0
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
    if cfg.label_source == "sep28k_extended":
        assert_sep28k_extended_dysfluency_csv(cfg.label_csv)
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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
            "path": path,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(item["waveform"].shape[-1] for item in batch)
    waveforms = []
    labels = []
    paths_out: List[str] = []
    for item in batch:
        w = item["waveform"]
        pad_amount = max_len - w.shape[-1]
        if pad_amount > 0:
            w = torch.nn.functional.pad(w, (0, pad_amount))
        else:
            w = w[:, :max_len]
        waveforms.append(w)
        labels.append(item["labels"])
        paths_out.append(str(item["path"]))
    return {
        "waveform": torch.stack(waveforms, dim=0),
        "labels": torch.stack(labels, dim=0),
        "paths": paths_out,
    }


# ----------------- MODULES -----------------


class MultiFeatureExtractor(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wav2vec2_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            cfg.wav2vec2_name, local_files_only=True
        )
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(
            cfg.wav2vec2_name, local_files_only=True
        )
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
    def __init__(self, cfg: Config, input_dim: int):
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
        self.head_p = nn.Linear(h2, 1)
        self.head_r = nn.Linear(h2, 1)
        self.head_i = nn.Linear(h2, 1)
        self.head_b = nn.Linear(h2, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        out, _ = self.lstm(features)
        pooled = out.mean(dim=1)
        pooled = self.dropout(pooled)
        return {
            "prolongation": self.head_p(pooled),
            "repetition": self.head_r(pooled),
            "interjection": self.head_i(pooled),
            "block": self.head_b(pooled),
            "pooled": pooled,
        }


def logits_to_tensor(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([outputs[k] for k in HEAD_LOGIT_KEYS], dim=1)


class ZeroRuleLogits(nn.Module):
    """Placeholder rule head: (B, 4) zeros on the batch device. Rules are not trained."""

    def forward(
        self,
        waveform: torch.Tensor,
        paths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        b = waveform.shape[0]
        return torch.zeros(
            b, len(F1_CLASS_NAMES), device=waveform.device, dtype=waveform.dtype
        )


def build_rule_module(cfg: Config, device: torch.device) -> nn.Module:
    if cfg.rule_mode == "zhang":
        from zhang_rules import ZhangStyleRuleLogits

        return ZhangStyleRuleLogits(sample_rate=cfg.sample_rate).to(device)
    if cfg.rule_mode == "zhang_full":
        from zhang_full.rule_module import ZhangFullRuleLogits

        cdir = (cfg.zhang_full_cache_dir or "").strip() or None
        return ZhangFullRuleLogits(
            sample_rate=cfg.sample_rate,
            cache_dir=cdir,
        ).to(device)
    return ZeroRuleLogits().to(device)


class LogitFusion(nn.Module):
    """
    Per-class fusion on logits. Rule branch is detached inside forward (no grad).

    linear: z_k = w_nn_k * z_nn_k + w_r_k * z_rule_k + b_k
    gated:  alpha_k = sigmoid(g_k(pooled)); z_k = alpha_k * z_nn_k + (1-alpha_k) * z_rule_k
    """

    def __init__(self, pooled_dim: int, mode: Literal["linear", "gated"]):
        super().__init__()
        self.mode = mode
        if mode == "linear":
            self.w_nn = nn.Parameter(torch.ones(len(F1_CLASS_NAMES)))
            self.w_r = nn.Parameter(torch.zeros(len(F1_CLASS_NAMES)))
            self.bias = nn.Parameter(torch.zeros(len(F1_CLASS_NAMES)))
        elif mode == "gated":
            self.gate = nn.Linear(pooled_dim, len(F1_CLASS_NAMES))
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, 2.0)
        else:
            raise ValueError(f"Unknown fusion mode: {mode}")

    def forward(
        self,
        z_nn: torch.Tensor,
        z_rule: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        z_r = z_rule.detach()
        if self.mode == "linear":
            return self.w_nn * z_nn + self.w_r * z_r + self.bias
        alpha = torch.sigmoid(self.gate(pooled))
        return alpha * z_nn + (1.0 - alpha) * z_r


def fused_logits(
    raw: Dict[str, torch.Tensor],
    waveform: torch.Tensor,
    rule_mod: nn.Module,
    fusion: Optional[LogitFusion],
    fusion_mode: Literal["none", "linear", "gated"],
    paths: Optional[List[str]] = None,
) -> torch.Tensor:
    z_nn = logits_to_tensor(raw)
    if fusion_mode == "none" or fusion is None:
        return z_nn
    pooled = raw["pooled"]
    z_rule = rule_mod(waveform, paths)
    return fusion(z_nn, z_rule, pooled)


def training_logits(
    raw: Dict[str, torch.Tensor],
    waveform: torch.Tensor,
    rule_mod: nn.Module,
    fusion: Optional[LogitFusion],
    cfg: Config,
    epoch_idx: int,
    paths: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    During the first fusion_nn_only_epochs epochs (1-based), optimize backbone
    on z_nn only; afterwards use fused logits (still includes detached rule path).
    """
    z_nn = logits_to_tensor(raw)
    if cfg.fusion_mode == "none" or fusion is None:
        return z_nn
    if epoch_idx <= cfg.fusion_nn_only_epochs:
        return z_nn
    return fused_logits(
        raw, waveform, rule_mod, fusion, cfg.fusion_mode, paths
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


def _cache_fp(cache_dir: str, path: str, tag: str) -> str:
    ap = os.path.abspath(path)
    key = hashlib.sha256(f"{tag}|{ap}".encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.pt")


def _load_cached_tensor(cache_fp: str, device: torch.device) -> Optional[torch.Tensor]:
    if not os.path.isfile(cache_fp):
        return None
    try:
        try:
            obj = torch.load(cache_fp, map_location=device, weights_only=False)
        except TypeError:
            obj = torch.load(cache_fp, map_location=device)
        if isinstance(obj, dict):
            t = obj.get("tensor")
            if isinstance(t, torch.Tensor):
                return t.to(device=device, dtype=torch.float32)
        if isinstance(obj, torch.Tensor):
            return obj.to(device=device, dtype=torch.float32)
    except Exception:
        return None
    return None


def _maybe_cached_audio_features(
    fe: MultiFeatureExtractor,
    waveforms: torch.Tensor,
    paths: Optional[List[str]],
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    cdir = (cfg.audio_feature_cache_dir or "").strip()
    tag = f"audio:{cfg.wav2vec2_name}:{cfg.mfcc_n_mfcc}:{cfg.max_frames}:{cfg.sample_rate}"
    if (not cdir) or (not paths) or (len(paths) != waveforms.shape[0]):
        with torch.no_grad():
            return fe(waveforms)
    os.makedirs(cdir, exist_ok=True)
    cached: List[Optional[torch.Tensor]] = []
    all_hit = True
    for p in paths:
        t = _load_cached_tensor(_cache_fp(cdir, p, tag), device)
        cached.append(t)
        if t is None:
            all_hit = False
    if all_hit:
        return torch.stack([x for x in cached if x is not None], dim=0)
    with torch.no_grad():
        feats = fe(waveforms)
    for i, p in enumerate(paths):
        if cached[i] is None:
            torch.save({"tensor": feats[i].detach().cpu().to(torch.float16)}, _cache_fp(cdir, p, tag))
    return feats


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
    batch_paths: Optional[List[str]] = batch.get("paths")
    labels_t = batch["labels"].to(device)
    labels_t = torch.clamp(labels_t, 0.0, 1.0)

    print(f"Waveforms shape: {waveforms.shape}")
    print(f"Labels shape: {labels_t.shape}")

    fe = MultiFeatureExtractor(cfg).to(device).eval()
    audio_feat = _maybe_cached_audio_features(fe, waveforms, batch_paths, cfg, device)
    print(f"Audio feature sequence shape: {audio_feat.shape}")

    model = TemporalDisfluencyModel(cfg, input_dim=audio_feat.shape[-1]).to(device)
    rule_mod = build_rule_module(cfg, device)
    fusion: Optional[LogitFusion] = None
    if cfg.fusion_mode == "linear":
        fusion = LogitFusion(pooled_dim=cfg.hidden_size * 2, mode="linear").to(device)
    elif cfg.fusion_mode == "gated":
        fusion = LogitFusion(pooled_dim=cfg.hidden_size * 2, mode="gated").to(device)
    raw = model(audio_feat)
    logits = fused_logits(
        raw, waveforms, rule_mod, fusion, cfg.fusion_mode, batch_paths
    )
    print(f"Logits shape: {logits.shape}")
    if cfg.fusion_mode != "none":
        rdesc = (
            "Zhang-style acoustic heuristics"
            if cfg.rule_mode == "zhang"
            else (
                "Zhang-full cascade (+ cache if set)"
                if cfg.rule_mode == "zhang_full"
                else "zeros (no rules)"
            )
        )
        print(f"Fusion: mode={cfg.fusion_mode} (rule logits: {rdesc})")
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
    fusion: Optional[LogitFusion],
    rule_mod: nn.Module,
    cfg: Config,
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
        if fusion is not None:
            fusion.train()
    else:
        model.eval()
        if fusion is not None:
            fusion.eval()
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
        batch_paths: Optional[List[str]] = batch.get("paths")
        labels_t = torch.clamp(batch["labels"].to(device), 0.0, 1.0)
        if train and optimizer:
            optimizer.zero_grad()
        audio_feat = _maybe_cached_audio_features(fe, waveforms, batch_paths, cfg, device)
        raw = model(audio_feat)
        logits = training_logits(
            raw, waveforms, rule_mod, fusion, cfg, epoch_idx, batch_paths
        )
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
    fusion: Optional[LogitFusion],
    rule_mod: nn.Module,
    cfg: Config,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    *,
    use_nn_logits: bool = False,
    pbar_desc: str = "val",
    show_progress: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Average loss + F1 (presence + per-type) over the full loader."""
    model.eval()
    if fusion:
        fusion.eval()
    rule_mod.eval()
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
            batch_paths: Optional[List[str]] = batch.get("paths")
            labels_t = torch.clamp(batch["labels"].to(device), 0.0, 1.0)
            audio_feat = _maybe_cached_audio_features(
                fe, waveforms, batch_paths, cfg, device
            )
            raw = model(audio_feat)
            if (
                use_nn_logits
                or cfg.fusion_mode == "none"
                or fusion is None
            ):
                logits = logits_to_tensor(raw)
            else:
                logits = fused_logits(
                    raw,
                    waveforms,
                    rule_mod,
                    fusion,
                    cfg.fusion_mode,
                    batch_paths,
                )
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
    fusion: Optional[LogitFusion],
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
            "fusion": fusion.state_dict() if fusion is not None else None,
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: TemporalDisfluencyModel,
    fusion: Optional[LogitFusion],
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if fusion is not None and ckpt.get("fusion"):
        fusion.load_state_dict(ckpt["fusion"])
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
        f"(per-split cap={cap or 'none'})",
        flush=True,
    )
    nt, nv, ne = len(tp), len(vp), len(ep)
    n_unique = nt + nv + ne
    n_passes = cfg.epochs * (nt + nv) + ne
    print(
        f"Clips — unique across splits: {n_unique}  |  "
        f"this run ({cfg.epochs} epoch(s)): "
        f"{cfg.epochs * nt} train + {cfg.epochs * nv} val + {ne} test "
        f"clip passes (test once after last epoch).",
        flush=True,
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
    # Infer dims
    sample_b = next(iter(train_loader))
    sample_w = sample_b["waveform"].to(device)
    sample_paths: Optional[List[str]] = sample_b.get("paths")
    adim = _maybe_cached_audio_features(fe, sample_w, sample_paths, cfg, device).shape[-1]

    model = TemporalDisfluencyModel(cfg, input_dim=adim).to(device)
    pooled_dim = cfg.hidden_size * 2
    fusion: Optional[LogitFusion] = None
    if cfg.fusion_mode == "linear":
        fusion = LogitFusion(pooled_dim, "linear").to(device)
    elif cfg.fusion_mode == "gated":
        fusion = LogitFusion(pooled_dim, "gated").to(device)
    rule_mod = build_rule_module(cfg, device)

    opt_params: List[nn.Parameter] = list(model.parameters())
    if fusion is not None:
        opt_params += list(fusion.parameters())
    optimizer = torch.optim.Adam(opt_params, lr=cfg.lr)

    start_epoch = 0
    best_val = float("inf")
    ckpt_dir = cfg.checkpoint_dir
    last_path = os.path.join(ckpt_dir, cfg.checkpoint_last_name)
    best_path = os.path.join(ckpt_dir, cfg.checkpoint_best_name)

    if cfg.fusion_mode != "none" or cfg.rule_mode != "none":
        print(
            f"Rules: mode={cfg.rule_mode}  |  "
            f"Fusion: mode={cfg.fusion_mode}, "
            f"fusion_nn_only_epochs={cfg.fusion_nn_only_epochs}",
            flush=True,
        )
        if cfg.rule_mode != "none" and cfg.fusion_mode == "none":
            print(
                "  (Rule logits are only blended when --fusion linear|gated; "
                "currently training on neural logits only.)",
                flush=True,
            )
        if cfg.rule_mode == "zhang_full":
            c = (cfg.zhang_full_cache_dir or "").strip()
            if c:
                print(f"  Zhang-full cache: {c}", flush=True)
            else:
                print(
                    "  Zhang-full: no --zhang-full-cache-dir (on-the-fly acoustic rules; "
                    "slow vs precompute_zhang_full_cache.py).",
                    flush=True,
                )

    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        start_epoch, best_val = load_checkpoint(
            cfg.resume_path, model, fusion, optimizer
        )
        model.to(device)
        print(
            f"Resumed from {cfg.resume_path} next_epoch={start_epoch}, best_val={best_val}",
            flush=True,
        )

    for run_step, epoch in enumerate(
        range(start_epoch, start_epoch + cfg.epochs), start=1
    ):
        tr_loss = _run_epoch(
            model,
            fe,
            fusion,
            rule_mod,
            cfg,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch_idx=run_step,
            epochs_total=cfg.epochs,
            show_progress=cfg.show_progress,
        )
        print(
            f"Epoch {run_step}/{cfg.epochs} · train loss (mean over batches): {tr_loss:.4f}",
            flush=True,
        )

        if val_loader:
            val_use_nn = (
                cfg.fusion_mode != "none"
                and fusion is not None
                and run_step <= cfg.fusion_nn_only_epochs
            )
            val_loss, val_f1 = evaluate_split(
                model,
                fe,
                fusion,
                rule_mod,
                cfg,
                val_loader,
                device,
                cfg.f1_threshold,
                use_nn_logits=val_use_nn,
                pbar_desc=f"Epoch {run_step}/{cfg.epochs} · val",
                show_progress=cfg.show_progress,
            )
            print(f"  val loss: {val_loss:.4f}", flush=True)
            print(f"  val F1:   {format_f1_metrics(val_f1)}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    best_path,
                    cfg,
                    epoch + 1,
                    model,
                    fusion,
                    optimizer,
                    best_val,
                )
                print(f"  saved best → {best_path}", flush=True)

        save_checkpoint(
            last_path,
            cfg,
            epoch + 1,
            model,
            fusion,
            optimizer,
            best_val,
        )
        print(f"  saved last → {last_path}", flush=True)

    if test_loader:
        te_loss, te_f1 = evaluate_split(
            model,
            fe,
            fusion,
            rule_mod,
            cfg,
            test_loader,
            device,
            cfg.f1_threshold,
            use_nn_logits=False,
            pbar_desc="Test",
            show_progress=cfg.show_progress,
        )
        print(f"Test loss: {te_loss:.4f}", flush=True)
        print(f"Test F1:   {format_f1_metrics(te_f1)}", flush=True)


# ----------------- CLI -----------------


def _resolve_device(choice: str) -> str:
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was set but torch.cuda.is_available() is False."
        )
    if choice == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit(
                "--device mps was set but torch.backends.mps.is_available() is False."
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
        "--label-vote-threshold",
        type=int,
        default=3,
        help="SEP-28k-Extended: mark dysfluency present when vote count >= this (3 = unanimous).",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="",
        help="Override clip WAV root (default: Config.data_root, usually data/sep28k/clips).",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto: use CUDA, then MPS (Apple Silicon), else CPU; or force cpu/cuda/mps.",
    )
    p.add_argument(
        "--audio-feature-cache-dir",
        type=str,
        default="",
        help="Optional cache dir for frozen wav2vec2+MFCC features keyed by clip path.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. for log files).",
    )
    p.add_argument(
        "--fusion",
        choices=("none", "linear", "gated"),
        default="none",
        help="Stacking-style logit fusion after the four heads.",
    )
    p.add_argument(
        "--rules",
        choices=("none", "zhang", "zhang_full"),
        default="none",
        help="none: zeros; zhang: lightweight heuristics; zhang_full: cascade+DTW "
        "+ optional Whisper cache (precompute_zhang_full_cache.py).",
    )
    p.add_argument(
        "--zhang-full-cache-dir",
        type=str,
        default="",
        help="With zhang_full: directory of JSON caches from precompute_zhang_full_cache.py.",
    )
    p.add_argument(
        "--fusion-nn-only-epochs",
        type=int,
        default=1,
        help="Train on backbone logits only for the first N epochs (1-based), then BCE on fused logits.",
    )

    args = p.parse_args()
    cfg = Config()
    cfg.device = _resolve_device(args.device)
    cfg.show_progress = not args.no_progress
    cfg.split_column = args.split_column
    cfg.f1_threshold = args.f1_threshold
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))
    if (args.data_root or "").strip():
        cfg.data_root = (args.data_root or "").strip()
    cfg.audio_feature_cache_dir = args.audio_feature_cache_dir or ""
    if args.mode == "demo":
        cfg.max_clips = args.max_clips
        cfg.batch_size = args.batch_size
        cfg.fusion_mode = args.fusion  # type: ignore[assignment]
        cfg.rule_mode = args.rules  # type: ignore[assignment]
        cfg.zhang_full_cache_dir = args.zhang_full_cache_dir or ""
        run_demo(cfg)
    else:
        cfg.epochs = args.epochs
        cfg.batch_size = args.batch_size
        cfg.lr = args.lr
        cfg.max_clips_per_split = args.max_clips_per_split
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.resume_path = args.resume or None
        cfg.fusion_mode = args.fusion  # type: ignore[assignment]
        cfg.rule_mode = args.rules  # type: ignore[assignment]
        cfg.zhang_full_cache_dir = args.zhang_full_cache_dir or ""
        cfg.fusion_nn_only_epochs = max(0, args.fusion_nn_only_epochs)
        run_train(cfg)


if __name__ == "__main__":
    main()
