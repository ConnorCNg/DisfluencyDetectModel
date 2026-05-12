#!/usr/bin/env python3
"""
Cross-head SVM error trends from an error-analysis detail JSON.

Uses per-head SVM FP/FN clip basenames plus ground-truth vectors from the same
split as compare (default: full test pool from build_split_lists).

Metrics:
  (1) |FP_i ∩ FN_j| — same clip: false positive on head i, false negative on j.
      Interpret as a "wrong-on / missed-on" pairing (not causal, but useful).
  (2) Among clips with FP on i, fraction with y_j = 1 (true type j present while
      i is wrongly fired).

Example:
  python3 scripts/cross_head_svm_error_cooccurrence.py \\
    --detail-json artifacts/error_analysis/seed42_full_detail.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from disfluency_pipeline import Config, F1_CLASS_NAMES, build_split_lists


def _basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _load_truth_by_basename(cfg: Config) -> Dict[str, np.ndarray]:
    _tp, _tl, _vp, _vl, ep, el, _ = build_split_lists(cfg)
    out: Dict[str, np.ndarray] = {}
    for p, lab in zip(ep, el):
        out[_basename(p)] = (np.asarray(lab, dtype=np.float64).reshape(-1) > 0.5).astype(np.int32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-json", required=True)
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--label-vote-threshold", type=int, default=3)
    ap.add_argument("--out-txt", default="")
    args = ap.parse_args()

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))

    truth = _load_truth_by_basename(cfg)

    with open(args.detail_json, "r", encoding="utf-8") as f:
        detail = json.load(f)

    per = detail["per_head"]
    fp_sets: Dict[str, Set[str]] = {}
    fn_sets: Dict[str, Set[str]] = {}
    for h in F1_CLASS_NAMES:
        fp_sets[h] = set(per[h]["svm_false_positives"])
        fn_sets[h] = set(per[h]["svm_false_negatives"])

    lines: List[str] = []
    lines.append("Cross-head SVM error co-occurrence (test clips)")
    lines.append(
        f"detail_json={args.detail_json}  label_csv={cfg.label_csv}  "
        f"data_root={cfg.data_root}  split_column={cfg.split_column}  thr={cfg.label_vote_threshold}"
    )
    lines.append(f"test clips with labels loaded: {len(truth)}")
    lines.append("")

    # --- (1) FP_i ∩ FN_j counts ---
    lines.append("=== |FP_i ∩ FN_j| (row = FP head i, col = FN head j) ===")
    mat = np.zeros((4, 4), dtype=np.int64)
    for ri, hi in enumerate(F1_CLASS_NAMES):
        for ci, hj in enumerate(F1_CLASS_NAMES):
            inter = fp_sets[hi] & fn_sets[hj]
            mat[ri, ci] = len(inter)
    header = "FP\\\\FN".ljust(14) + "".join(f"{h[:4]:>12}" for h in F1_CLASS_NAMES)
    lines.append(header)
    for ri, hi in enumerate(F1_CLASS_NAMES):
        row = f"{hi[:12]:<14}" + "".join(f"{int(mat[ri, ci]):>12}" for ci in range(4))
        lines.append(row)
    lines.append("")

    # Row-normalized: among FP_i, share that also miss j
    lines.append("=== P(FN_j | FP_i) = |FP_i∩FN_j| / |FP_i| ===")
    lines.append(header)
    for ri, hi in enumerate(F1_CLASS_NAMES):
        denom = max(1, len(fp_sets[hi]))
        row = f"{hi[:12]:<14}" + "".join(f"{mat[ri, ci] / denom:>11.3f}" for ci in range(4))
        lines.append(row)
    lines.append("")

    # --- (2) Among FP_i clips, how often is y_j == 1 ---
    lines.append("=== Among SVM FP clips on head i, fraction with y_j = 1 ===")
    header2 = "FP_i\\y_j".ljust(14) + "".join(f"{h[:4]:>12}" for h in F1_CLASS_NAMES)
    lines.append(header2)
    for hi in F1_CLASS_NAMES:
        clips = fp_sets[hi]
        if not clips:
            lines.append(f"{hi[:12]:<14}" + "   (no FP)")
            continue
        counts = np.zeros(4, dtype=np.int64)
        miss = 0
        for c in clips:
            y = truth.get(c)
            if y is None:
                miss += 1
                continue
            counts += y.astype(np.int64)
        denom = max(1, len(clips) - miss)
        row = f"{hi[:12]:<14}" + "".join(f"{counts[j] / denom:>11.3f}" for j in range(4))
        if miss:
            row += f"  (missing_truth={miss})"
        lines.append(row)
    lines.append("")

    # Sample clips for a few high-signal cells (optional storytelling)
    pairs: List[Tuple[str, str]] = [
        ("Block", "Prolongation"),
        ("Prolongation", "Block"),
        ("Block", "Repetition"),
        ("Repetition", "Block"),
        ("Interjection", "Block"),
        ("Block", "Interjection"),
    ]
    lines.append("=== Example clips: FP_i ∩ FN_j (up to 12 each) ===")
    for hi, hj in pairs:
        inter = sorted(fp_sets[hi] & fn_sets[hj])[:12]
        lines.append(f"{hi} FP ∩ {hj} FN (n={len(fp_sets[hi] & fn_sets[hj])}): {', '.join(inter)}")

    text = "\n".join(lines) + "\n"
    out_txt = (args.out_txt or "").strip()
    if not out_txt:
        base = os.path.splitext(os.path.basename(args.detail_json))[0]
        out_txt = os.path.join("artifacts", "error_analysis", f"{base}_cross_head.txt")
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(out_txt)
    print(text)


if __name__ == "__main__":
    main()
