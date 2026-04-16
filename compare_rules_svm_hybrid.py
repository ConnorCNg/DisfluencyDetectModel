#!/usr/bin/env python3
"""
Compare rule-only vs W2V2+SVM vs simple hybrids on the **same clips and labels**.

- Labels: same 4 heads as ``disfluency_pipeline`` (Repetition = max(SoundRep, WordRep)
  votes ≥ threshold), from ``build_split_lists`` / ``row_to_label_vector``.
- Split: default ``SEP28k-T`` (``dev`` → internal ``val``); train SVM on train+val pool.
- Rules: evaluates both ``zhang`` and ``zhang_full`` logits → sigmoid → ``--f1-threshold``.
- SVM: one binary RBF SVC per head (same as paper_style script), ``predict``.
- Hybrids (on binary preds): OR = max(rule, svm), AND = rule & svm.

Use ``--smoke`` for a tiny subsample (quick sanity check). Use ``--max-train 0 --max-test 0``
for full train+val / full test pools.

Examples::

  python3 -u compare_rules_svm_hybrid.py --smoke --device auto
  python3 -u compare_rules_svm_hybrid.py --max-train 0 --max-test 0 --device auto
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
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
from paper_style_w2v2_svm_test import ClipRecord, extract_embeddings
from zhang_full.rule_module import ZhangFullRuleLogits
from zhang_rules import ZhangStyleRuleLogits


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


def _fit_svm_ovr(
    x_train: np.ndarray,
    y_train: np.ndarray,
):
    heads: List[SVMHead] = []
    for i in range(y_train.shape[1]):
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
        )
        clf.fit(x_train, y_train[:, i])
        heads.append(SVMHead(pipe=clf))
    return heads


def _svm_predict(heads: List[SVMHead], x: np.ndarray) -> np.ndarray:
    pred = np.zeros((x.shape[0], len(heads)), dtype=np.int32)
    for i, h in enumerate(heads):
        pred[:, i] = h.pipe.predict(x).astype(np.int32)
    return pred


def _svm_scores(heads: List[SVMHead], x: np.ndarray) -> np.ndarray:
    sc = np.zeros((x.shape[0], len(heads)), dtype=np.float64)
    for i, h in enumerate(heads):
        sc[:, i] = h.pipe.decision_function(x)
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
        description="Rules vs SVM vs hybrid on aligned 4-head SEP-28k labels."
    )
    ap.add_argument("--csv", default="SEP-28k-Extended_clips.csv")
    ap.add_argument("--data-root", default="data/sep28k/clips")
    ap.add_argument("--split-column", default="SEP28k-T")
    ap.add_argument("--label-vote-threshold", type=int, default=2)
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
    ap.add_argument("--layer", type=int, default=8)
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

    cfg = Config()
    cfg.label_csv = args.csv
    cfg.data_root = args.data_root
    cfg.split_column = args.split_column
    cfg.label_vote_threshold = max(0, int(args.label_vote_threshold))

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

    zhang_rule_mod = ZhangStyleRuleLogits(sample_rate=cfg.sample_rate).to(dev).eval()
    cdir = (args.zhang_full_cache_dir or "").strip() or None
    zhang_full_rule_mod = ZhangFullRuleLogits(
        sample_rate=cfg.sample_rate, cache_dir=cdir
    ).to(dev).eval()

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
    svm_heads = _fit_svm_ovr(x_train, y_train)
    svm_pred = _svm_predict(svm_heads, x_test)
    svm_scores_raw = _svm_scores(svm_heads, x_test)

    print("Running rules (zhang) on test waveforms…", flush=True)
    zhang_probs = _collect_rule_probs(
        cfg, test_paths, test_labels, zhang_rule_mod, dev, args.batch_size
    )
    print("Running rules (zhang_full) on test waveforms…", flush=True)
    zhang_full_probs = _collect_rule_probs(
        cfg, test_paths, test_labels, zhang_full_rule_mod, dev, args.batch_size
    )

    saved = None
    if not args.ignore_saved_thresholds:
        saved = _load_thresholds(args.thresholds_json)
    svm_th = _ordered_thresholds(saved, "svm_score_thresholds")
    z_th = _ordered_thresholds(saved, "rules_zhang_prob_thresholds")
    zf_th = _ordered_thresholds(saved, "rules_zhang_full_prob_thresholds")

    if z_th is not None:
        zhang_bin = (zhang_probs >= z_th.reshape(1, -1)).astype(np.float32)
    else:
        zhang_bin = (zhang_probs >= float(args.f1_threshold)).astype(np.float32)

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
    zhang_or_scores = np.maximum(zhang_bin, svm_pred.astype(np.float32)).astype(np.float64)
    zhang_and_scores = (zhang_bin * svm_pred.astype(np.float32)).astype(np.float64)
    zhang_full_or_scores = np.maximum(zhang_full_bin, svm_pred.astype(np.float32)).astype(np.float64)
    zhang_full_and_scores = (zhang_full_bin * svm_pred.astype(np.float32)).astype(np.float64)

    m_zhang = compute_f1_metrics(y_true, zhang_probs, float(args.f1_threshold))
    m_zhang_full = compute_f1_metrics(y_true, zhang_full_probs, float(args.f1_threshold))
    m_svm = compute_f1_metrics(y_true, svm_scores, 0.5)
    m_zhang_or = compute_f1_metrics(y_true, zhang_or_scores, 0.5)
    m_zhang_and = compute_f1_metrics(y_true, zhang_and_scores, 0.5)
    m_zhang_full_or = compute_f1_metrics(y_true, zhang_full_or_scores, 0.5)
    m_zhang_full_and = compute_f1_metrics(y_true, zhang_full_and_scores, 0.5)

    print("", flush=True)
    if saved is not None:
        print(f"Thresholds loaded from: {args.thresholds_json}", flush=True)
    else:
        print("Thresholds loaded from: (none) using defaults", flush=True)
    print(f"Per-type order: {F1_CLASS_NAMES}", flush=True)
    print(f"Rules (zhang):           {format_f1_metrics(m_zhang)}", flush=True)
    print(f"Rules (zhang_full):      {format_f1_metrics(m_zhang_full)}", flush=True)
    print(f"SVM ({svm_mode}):        {format_f1_metrics(m_svm)}", flush=True)
    print(f"Hybrid OR (zhang,SVM):   {format_f1_metrics(m_zhang_or)}", flush=True)
    print(f"Hybrid AND (zhang,SVM):  {format_f1_metrics(m_zhang_and)}", flush=True)
    print(f"Hybrid OR (zfull,SVM):   {format_f1_metrics(m_zhang_full_or)}", flush=True)
    print(f"Hybrid AND (zfull,SVM):  {format_f1_metrics(m_zhang_full_and)}", flush=True)
    print(
        f"Macro-F1 (4 types):  zhang={_macro_f1(m_zhang):.4f}  "
        f"zhang_full={_macro_f1(m_zhang_full):.4f}  svm={_macro_f1(m_svm):.4f}  "
        f"or(zhang)={_macro_f1(m_zhang_or):.4f}  and(zhang)={_macro_f1(m_zhang_and):.4f}  "
        f"or(zfull)={_macro_f1(m_zhang_full_or):.4f}  and(zfull)={_macro_f1(m_zhang_full_and):.4f}",
        flush=True,
    )
    if args.smoke or len(train_recs) < 500:
        print(
            "[Note] Small train sets often give unreliable SVM F1; "
            "use --max-train 0 --max-test 0 (omit --smoke) for a serious comparison.",
            flush=True,
        )


if __name__ == "__main__":
    main()
