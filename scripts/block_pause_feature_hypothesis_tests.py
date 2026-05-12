#!/usr/bin/env python3
"""
Hypothesis tests for differentiating natural pauses vs true Block clips.

Uses full-test run outputs:
  - artifacts/error_analysis/seed42_full_detail.json
  - split/labels from SEP-28k-Extended (build_split_lists)

Outputs:
  - artifacts/error_analysis/seed42_block_hypothesis_tests.json
  - artifacts/error_analysis/seed42_block_hypothesis_tests.txt
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disfluency_pipeline import Config, build_split_lists

IN_DETAIL = "artifacts/error_analysis/seed42_full_detail.json"
OUT_JSON = "artifacts/error_analysis/seed42_block_hypothesis_tests.json"
OUT_TXT = "artifacts/error_analysis/seed42_block_hypothesis_tests.txt"


@dataclass
class DiagCfg:
    sr: int = 16000
    frame_length_s: float = 0.025
    hop_s: float = 0.010
    silence_db_rel: float = -35.0
    min_pause_s: float = 0.05
    long_pause_s: float = 0.20
    pre_window_s: float = 0.08


def _clip_base(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    i = 0
    n = len(mask)
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


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average rank for ties, ranks start at 1."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        r = (i + 1 + j) * 0.5
        ranks[order[i:j]] = r
        i = j
    return ranks


def auc_1d(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    y = np.concatenate([np.ones(len(pos), dtype=np.int32), np.zeros(len(neg), dtype=np.int32)])
    s = np.concatenate([pos, neg]).astype(np.float64)
    r = _rankdata_average(s)
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    sum_r_pos = float(np.sum(r[y == 1]))
    u = sum_r_pos - n_pos * (n_pos + 1.0) * 0.5
    return float(u / (n_pos * n_neg))


def effect_median_scale(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    m1 = float(np.median(pos))
    m0 = float(np.median(neg))
    q1 = float(np.percentile(pos, 75) - np.percentile(pos, 25))
    q0 = float(np.percentile(neg, 75) - np.percentile(neg, 25))
    scale = 0.5 * (q1 + q0) + 1e-9
    return (m1 - m0) / scale


def clip_features(path: str, c: DiagCfg) -> Dict[str, float]:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1).astype(np.float32)
    if y.size == 0:
        return {}
    if sr != c.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=c.sr)
    if y.size == 0:
        return {}

    n_fft = int(round(c.frame_length_s * c.sr))
    hop = int(round(c.hop_s * c.sr))
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-8, ref=np.max)
    sil = rms_db < c.silence_db_rel
    segs_all = _segments(sil)
    min_len = max(1, int(round(c.min_pause_s * c.sr / hop)))
    long_len = max(1, int(round(c.long_pause_s * c.sr / hop)))
    segs = [(a, b) for (a, b) in segs_all if (b - a) >= min_len]
    segs_long = [(a, b) for (a, b) in segs if (b - a) >= long_len]

    pause_durs = np.array([(b - a) * hop / c.sr for (a, b) in segs], dtype=np.float64)
    long_durs = np.array([(b - a) * hop / c.sr for (a, b) in segs_long], dtype=np.float64)
    onset = librosa.onset.onset_strength(y=y, sr=c.sr, hop_length=hop).astype(np.float64)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    centroid = librosa.feature.spectral_centroid(S=S, sr=c.sr).reshape(-1).astype(np.float64)
    flatness = librosa.feature.spectral_flatness(S=S).reshape(-1).astype(np.float64)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop, center=True)[0]
    flux = np.sqrt(np.sum(np.diff(S, axis=1, prepend=S[:, :1]) ** 2, axis=0)).astype(np.float64)

    pre_w = max(2, int(round(c.pre_window_s * c.sr / hop)))
    pre_drop_db: List[float] = []
    pre_slope_db: List[float] = []
    pre_flux: List[float] = []
    pre_centroid_drop: List[float] = []
    pre_flatness_rise: List[float] = []
    pre_onset: List[float] = []
    for a, _b in segs:
        lo = max(0, a - pre_w)
        if a - lo < 2:
            continue
        rr = rms_db[lo:a]
        xx = np.arange(rr.size, dtype=np.float64)
        denom = float(np.sum((xx - xx.mean()) ** 2)) + 1e-9
        slope = float(np.sum((xx - xx.mean()) * (rr - rr.mean())) / denom)
        pre_slope_db.append(slope)
        pre_drop_db.append(float(np.max(rr) - rms_db[a]))
        pre_flux.append(float(np.mean(flux[lo:a])))
        pre_centroid_drop.append(float(np.mean(centroid[lo:a]) - centroid[a]))
        pre_flatness_rise.append(float(flatness[a] - np.mean(flatness[lo:a])))
        pre_onset.append(float(np.mean(onset[lo:a])))

    dur_s = float(len(y) / c.sr)
    return {
        "duration_s": dur_s,
        "silence_ratio_rel": float(np.mean(sil)),
        "pause_count": float(len(segs)),
        "long_pause_count": float(len(segs_long)),
        "pause_mean_s": float(np.mean(pause_durs)) if pause_durs.size else 0.0,
        "pause_max_s": float(np.max(pause_durs)) if pause_durs.size else 0.0,
        "long_pause_mean_s": float(np.mean(long_durs)) if long_durs.size else 0.0,
        "rms_mean_db_rel": float(np.mean(rms_db)),
        "rms_std_db_rel": float(np.std(rms_db)),
        "zcr_mean": float(np.mean(zcr)) if zcr.size else 0.0,
        "pre_pause_rms_slope_db_per_frame": float(np.mean(pre_slope_db)) if pre_slope_db else 0.0,
        "pre_pause_burst_drop_db": float(np.mean(pre_drop_db)) if pre_drop_db else 0.0,
        "pre_pause_flux": float(np.mean(pre_flux)) if pre_flux else 0.0,
        "pre_pause_centroid_drop_hz": float(np.mean(pre_centroid_drop)) if pre_centroid_drop else 0.0,
        "pre_pause_flatness_rise": float(np.mean(pre_flatness_rise)) if pre_flatness_rise else 0.0,
        "pre_pause_onset_strength": float(np.mean(pre_onset)) if pre_onset else 0.0,
    }


def main() -> None:
    with open(IN_DETAIL, "r", encoding="utf-8") as f:
        det = json.load(f)
    block = det["per_head"]["Block"]
    svm_fp = set(block["svm_false_positives"])
    svm_fn = set(block["svm_false_negatives"])

    cfg = Config()
    cfg.label_csv = "SEP-28k-Extended_clips.csv"
    cfg.data_root = "data/sep28k/clips"
    cfg.split_column = "SEP28k-T"
    cfg.label_vote_threshold = 3
    cfg.seed = 42
    _tp, _tl, _vp, _vl, ep, el, _ = build_split_lists(cfg)

    names = [_clip_base(p) for p in ep]
    y_block = np.array([1 if float(v[3]) > 0.5 else 0 for v in el], dtype=np.int32)

    c = DiagCfg(sr=cfg.sample_rate)
    feats: Dict[str, Dict[str, float]] = {}
    for p, n in tqdm(list(zip(ep, names)), desc="Block hypothesis features", unit="clip"):
        f = clip_features(p, c)
        if f:
            feats[n] = f

    feature_names = sorted(next(iter(feats.values())).keys())
    all_names = [n for n in names if n in feats]
    pos_names = [n for i, n in enumerate(names) if y_block[i] == 1 and n in feats]
    neg_names = [n for i, n in enumerate(names) if y_block[i] == 0 and n in feats]

    svm_fp_names = [n for n in all_names if n in svm_fp]
    svm_fn_names = [n for n in all_names if n in svm_fn]
    svm_tp_names = [n for i, n in enumerate(names) if y_block[i] == 1 and n not in svm_fn and n in feats]
    svm_tn_names = [n for i, n in enumerate(names) if y_block[i] == 0 and n not in svm_fp and n in feats]

    def vec(group: List[str], feat: str) -> np.ndarray:
        return np.array([feats[n][feat] for n in group], dtype=np.float64)

    tests: Dict[str, Dict[str, Dict[str, float]]] = {
        "true_block_vs_nonblock": {},
        "svm_fp_vs_svm_tn_natural_pause_confusion": {},
        "svm_fn_vs_svm_tp_missed_blocks": {},
    }
    for ft in feature_names:
        p = vec(pos_names, ft)
        n = vec(neg_names, ft)
        auc = auc_1d(p, n)
        tests["true_block_vs_nonblock"][ft] = {
            "auc_pos_gt_neg": auc,
            "effect_median_scaled": effect_median_scale(p, n),
            "median_pos": float(np.median(p)) if p.size else 0.0,
            "median_neg": float(np.median(n)) if n.size else 0.0,
        }

        p = vec(svm_fp_names, ft)
        n = vec(svm_tn_names, ft)
        auc = auc_1d(p, n)
        tests["svm_fp_vs_svm_tn_natural_pause_confusion"][ft] = {
            "auc_fp_gt_tn": auc,
            "effect_median_scaled": effect_median_scale(p, n),
            "median_fp": float(np.median(p)) if p.size else 0.0,
            "median_tn": float(np.median(n)) if n.size else 0.0,
        }

        p = vec(svm_fn_names, ft)
        n = vec(svm_tp_names, ft)
        auc = auc_1d(p, n)
        tests["svm_fn_vs_svm_tp_missed_blocks"][ft] = {
            "auc_fn_gt_tp": auc,
            "effect_median_scaled": effect_median_scale(p, n),
            "median_fn": float(np.median(p)) if p.size else 0.0,
            "median_tp": float(np.median(n)) if n.size else 0.0,
        }

    def topk(section: str, k: int = 8) -> List[Tuple[str, float]]:
        rows = tests[section]
        scored = []
        for ft, r in rows.items():
            # Distance from random 0.5
            score = abs(float(r[list(r.keys())[0]]) - 0.5)
            scored.append((ft, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    out = {
        "input_detail_json": IN_DETAIL,
        "n_with_features": len(all_names),
        "group_counts": {
            "true_block_pos": len(pos_names),
            "true_block_neg": len(neg_names),
            "svm_fp": len(svm_fp_names),
            "svm_tn": len(svm_tn_names),
            "svm_fn": len(svm_fn_names),
            "svm_tp": len(svm_tp_names),
        },
        "diag_cfg": {
            "sr": c.sr,
            "frame_length_s": c.frame_length_s,
            "hop_s": c.hop_s,
            "silence_db_rel": c.silence_db_rel,
            "min_pause_s": c.min_pause_s,
            "long_pause_s": c.long_pause_s,
            "pre_window_s": c.pre_window_s,
        },
        "tests": tests,
        "top_features": {
            "true_block_vs_nonblock": topk("true_block_vs_nonblock"),
            "svm_fp_vs_svm_tn_natural_pause_confusion": topk("svm_fp_vs_svm_tn_natural_pause_confusion"),
            "svm_fn_vs_svm_tp_missed_blocks": topk("svm_fn_vs_svm_tp_missed_blocks"),
        },
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    def render_top(section: str, auc_key: str) -> List[str]:
        lines: List[str] = [f"{section}:"]
        for ft, _score in out["top_features"][section]:
            row = out["tests"][section][ft]
            lines.append(
                f"  - {ft}: {auc_key}={row[auc_key]:.3f} effect={row['effect_median_scaled']:.3f}"
            )
        return lines

    txt_lines: List[str] = []
    txt_lines.append("Block hypothesis tests (seed42 full test)")
    txt_lines.append(f"Group counts: {out['group_counts']}")
    txt_lines.append("")
    txt_lines.extend(render_top("true_block_vs_nonblock", "auc_pos_gt_neg"))
    txt_lines.append("")
    txt_lines.extend(
        render_top("svm_fp_vs_svm_tn_natural_pause_confusion", "auc_fp_gt_tn")
    )
    txt_lines.append("")
    txt_lines.extend(render_top("svm_fn_vs_svm_tp_missed_blocks", "auc_fn_gt_tp"))
    txt_lines.append("")
    txt_lines.append("Interpretation guide:")
    txt_lines.append("- AUC near 0.5 means weak separation.")
    txt_lines.append("- AUC >0.6 or <0.4 suggests useful one-feature signal.")
    txt_lines.append("- Positive effect means first group has larger median.")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    print(OUT_JSON)
    print(OUT_TXT)
    for ln in txt_lines:
        print(ln)


if __name__ == "__main__":
    main()

