#!/usr/bin/env python3
"""
Diagnose Block errors with pause/spectral features on the full test split.

Reads Block group clip lists from:
  artifacts/error_analysis/seed42_full_block_characterization.json

Computes per-clip diagnostics from waveform:
  - relative silence ratio (RMS dB vs clip max)
  - pause count / duration stats (contiguous silence segments)
  - pre-pause energy slope and "burst-before-drop"
  - pre-pause spectral centroid / flatness changes

Writes:
  artifacts/error_analysis/seed42_full_block_pause_hypothesis.json
  artifacts/error_analysis/seed42_full_block_pause_hypothesis.txt
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disfluency_pipeline import Config, build_split_lists


IN_JSON = "artifacts/error_analysis/seed42_full_block_characterization.json"
OUT_JSON = "artifacts/error_analysis/seed42_full_block_pause_hypothesis.json"
OUT_TXT = "artifacts/error_analysis/seed42_full_block_pause_hypothesis.txt"


@dataclass
class DiagCfg:
    sr: int = 16000
    frame_length: float = 0.025
    hop: float = 0.010
    silence_db_rel: float = -35.0
    min_pause_s: float = 0.05
    pre_window_s: float = 0.08


def _contiguous_segments(mask: np.ndarray) -> List[tuple[int, int]]:
    """Return [(start_idx, end_idx_exclusive)] for True runs."""
    out: List[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(mask[j]):
            j += 1
        out.append((i, j))
        i = j
    return out


def _safe_mean(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0


def _safe_median(x: List[float]) -> float:
    return float(np.median(x)) if x else 0.0


def _safe_std(x: List[float]) -> float:
    return float(np.std(x)) if x else 0.0


def clip_diag(path: str, cfg: DiagCfg) -> Dict[str, float]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1).astype(np.float32)
    if y.size == 0:
        return {
            "silence_ratio_rel": 0.0,
            "pause_count": 0.0,
            "pause_mean_s": 0.0,
            "pause_median_s": 0.0,
            "pre_rms_slope_db_per_frame_mean": 0.0,
            "pre_burst_drop_db_mean": 0.0,
            "pre_centroid_drop_hz_mean": 0.0,
            "pre_flatness_drop_mean": 0.0,
            "zcr_mean": 0.0,
        }
    if sr != cfg.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr)
    if y.size == 0:
        return {
            "silence_ratio_rel": 0.0,
            "pause_count": 0.0,
            "pause_mean_s": 0.0,
            "pause_median_s": 0.0,
            "pre_rms_slope_db_per_frame_mean": 0.0,
            "pre_burst_drop_db_mean": 0.0,
            "pre_centroid_drop_hz_mean": 0.0,
            "pre_flatness_drop_mean": 0.0,
            "zcr_mean": 0.0,
        }

    frame = int(round(cfg.frame_length * cfg.sr))
    hop = int(round(cfg.hop * cfg.sr))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    sil = rms_db < cfg.silence_db_rel
    segs = _contiguous_segments(sil)
    min_len = max(1, int(round(cfg.min_pause_s * cfg.sr / hop)))
    segs = [(a, b) for (a, b) in segs if (b - a) >= min_len]
    pause_durs = [float((b - a) * hop / cfg.sr) for (a, b) in segs]

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame, hop_length=hop, center=True
    )[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=cfg.sr, n_fft=frame, hop_length=hop)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame, hop_length=hop)[0]

    pre_w = max(2, int(round(cfg.pre_window_s * cfg.sr / hop)))
    pre_slopes: List[float] = []
    burst_drop: List[float] = []
    cent_drop: List[float] = []
    flat_drop: List[float] = []
    for a, _b in segs:
        lo = max(0, a - pre_w)
        hi = a
        if hi - lo < 2:
            continue
        rr = rms_db[lo:hi]
        xx = np.arange(rr.size, dtype=np.float64)
        # Least-squares slope (dB/frame) before pause onset.
        denom = float(np.sum((xx - xx.mean()) ** 2)) + 1e-9
        slope = float(np.sum((xx - xx.mean()) * (rr - rr.mean())) / denom)
        pre_slopes.append(slope)
        burst_drop.append(float(np.max(rr) - rms_db[a]))

        cc = centroid[lo:hi]
        ff = flatness[lo:hi]
        cent_drop.append(float(np.mean(cc) - centroid[a]))
        flat_drop.append(float(np.mean(ff) - flatness[a]))

    return {
        "silence_ratio_rel": float(np.mean(sil)),
        "pause_count": float(len(segs)),
        "pause_mean_s": _safe_mean(pause_durs),
        "pause_median_s": _safe_median(pause_durs),
        "pre_rms_slope_db_per_frame_mean": _safe_mean(pre_slopes),
        "pre_burst_drop_db_mean": _safe_mean(burst_drop),
        "pre_centroid_drop_hz_mean": _safe_mean(cent_drop),
        "pre_flatness_drop_mean": _safe_mean(flat_drop),
        "zcr_mean": float(np.mean(zcr)) if zcr.size else 0.0,
    }


def summarize_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.array([float(r[k]) for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        }
    return out


def _clip_base(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main() -> None:
    with open(IN_JSON, "r", encoding="utf-8") as f:
        block = json.load(f)["lists"]

    groups = {
        "svm_fp": set(block["svm_block_false_positives"]),
        "svm_fn": set(block["svm_block_false_negatives"]),
        "svm_correct": set(block["svm_block_correct"]),
        "rules_fp": set(block["rules_block_false_positives"]),
        "rules_fn": set(block["rules_block_false_negatives"]),
        "rules_correct": set(block["rules_block_correct"]),
    }

    cfg = Config()
    cfg.label_csv = "SEP-28k-Extended_clips.csv"
    cfg.data_root = "data/sep28k/clips"
    cfg.split_column = "SEP28k-T"
    cfg.label_vote_threshold = 3
    cfg.seed = 42
    _tp, _tl, _vp, _vl, ep, _el, _ = build_split_lists(cfg)

    names = [_clip_base(p) for p in ep]
    diag_cfg = DiagCfg(sr=cfg.sample_rate)

    per_group_rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in groups}
    per_group_names: Dict[str, List[str]] = {k: [] for k in groups}

    for p, n in tqdm(list(zip(ep, names)), desc="Block pause diagnostics", unit="clip"):
        feats = clip_diag(p, diag_cfg)
        for gname, gset in groups.items():
            if n in gset:
                per_group_rows[gname].append(feats)
                per_group_names[gname].append(n)

    out = {
        "input_error_json": IN_JSON,
        "groups_count": {k: len(v) for k, v in per_group_names.items()},
        "diag_cfg": {
            "sr": diag_cfg.sr,
            "frame_length_s": diag_cfg.frame_length,
            "hop_s": diag_cfg.hop,
            "silence_db_rel": diag_cfg.silence_db_rel,
            "min_pause_s": diag_cfg.min_pause_s,
            "pre_window_s": diag_cfg.pre_window_s,
        },
        "group_feature_summary": {
            k: summarize_rows(v) for k, v in per_group_rows.items()
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    def med(g: str, feat: str) -> float:
        return out["group_feature_summary"].get(g, {}).get(feat, {}).get("median", 0.0)

    lines = [
        "Block hypothesis diagnostics (full test split)",
        f"Groups: {out['groups_count']}",
        "",
        "Median feature values:",
        (
            "SVM   FP/FN/correct silence_ratio_rel: "
            f"{med('svm_fp','silence_ratio_rel'):.4f} / "
            f"{med('svm_fn','silence_ratio_rel'):.4f} / "
            f"{med('svm_correct','silence_ratio_rel'):.4f}"
        ),
        (
            "SVM   FP/FN/correct pause_count: "
            f"{med('svm_fp','pause_count'):.2f} / "
            f"{med('svm_fn','pause_count'):.2f} / "
            f"{med('svm_correct','pause_count'):.2f}"
        ),
        (
            "SVM   FP/FN/correct pre_burst_drop_db_mean: "
            f"{med('svm_fp','pre_burst_drop_db_mean'):.3f} / "
            f"{med('svm_fn','pre_burst_drop_db_mean'):.3f} / "
            f"{med('svm_correct','pre_burst_drop_db_mean'):.3f}"
        ),
        (
            "SVM   FP/FN/correct pre_centroid_drop_hz_mean: "
            f"{med('svm_fp','pre_centroid_drop_hz_mean'):.3f} / "
            f"{med('svm_fn','pre_centroid_drop_hz_mean'):.3f} / "
            f"{med('svm_correct','pre_centroid_drop_hz_mean'):.3f}"
        ),
        (
            "Rules FP/FN/correct silence_ratio_rel: "
            f"{med('rules_fp','silence_ratio_rel'):.4f} / "
            f"{med('rules_fn','silence_ratio_rel'):.4f} / "
            f"{med('rules_correct','silence_ratio_rel'):.4f}"
        ),
        (
            "Rules FP/FN/correct pause_count: "
            f"{med('rules_fp','pause_count'):.2f} / "
            f"{med('rules_fn','pause_count'):.2f} / "
            f"{med('rules_correct','pause_count'):.2f}"
        ),
        "",
        "See JSON for full mean/median/p25/p75 per feature per group.",
    ]
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(OUT_JSON)
    print(OUT_TXT)
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()

