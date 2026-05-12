#!/usr/bin/env python3
"""Load JSON outputs from README verify runs and print Markdown tables."""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADS = ("Prolongation", "Repetition", "Interjection", "Block", "presence")


def _load(path: str) -> Optional[Dict[str, Any]]:
    p = path if os.path.isabs(path) else os.path.join(REPO, path)
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    def line(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    sep = "| " + " | ".join("-" * w[i] for i in range(len(headers))) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def _parse_rules_log(text: str) -> Tuple[Optional[str], Optional[str]]:
    presence = None
    types = None
    for line in text.splitlines():
        if "presence_F1=" in line:
            m = re.search(r"presence_F1=([0-9.]+)", line)
            if m:
                presence = m.group(1)
        if "type_F1:" in line:
            types = line.strip()
    return presence, types


def _parse_bilstm_test_f1(log_text: str) -> Optional[str]:
    last = None
    for line in log_text.splitlines():
        if "Test F1:" in line or "Test F1" in line:
            last = line.strip()
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-md", default="", help="Optional path to write full Markdown report (repo-relative ok).")
    args = ap.parse_args()

    sections: List[str] = []

    # --- Tune thresholds (SVM + rules zhang_full on test) ---
    tune_specs = [
        ("3s SEP28k-T", "artifacts/tuned_thresholds_rules_svm_3s_sep28kT_strict.json"),
        ("3s SEP28k-D", "artifacts/tuned_thresholds_rules_svm_3s_sep28kD_strict.json"),
        ("mix SEP28k-T", "artifacts/tuned_thresholds_rules_svm_mix53_sep28kT_strict.json"),
    ]
    rows = []
    for label, rel in tune_specs:
        j = _load(rel)
        if not j:
            rows.append([label, "—", "—", "—"])
            continue
        svm = j.get("test_metrics_tuned_thresholds", {}).get("svm_f1", {})
        rules = j.get("test_metrics_tuned_thresholds", {}).get("rules_zhang_full_f1", {})
        thr = j.get("label_vote_threshold", "")
        rows.append(
            [
                label,
                f"{float(svm.get('presence', 0)):.4f}",
                f"{float(rules.get('presence', 0)):.4f}",
                str(thr),
            ]
        )
    h = ["Run", "SVM presence", "Rules presence", "label_vote_threshold in file"]
    sections.append("### tune_thresholds_rules_svm.py (test, tuned on dev)\n")
    sections.append(_md_table(h, rows))

    head_row = ["Run"] + list(HEADS)
    svm_rows2: List[List[str]] = []
    for label, rel in tune_specs:
        j = _load(rel)
        if not j:
            svm_rows2.append([label] + ["—"] * len(HEADS))
            continue
        svm = j.get("test_metrics_tuned_thresholds", {}).get("svm_f1", {})
        svm_rows2.append([label] + [f"{float(svm.get(h, 0)):.4f}" for h in HEADS])
    sections.append("\n### SVM F1 per head (tune script, test)\n")
    sections.append(_md_table(head_row, svm_rows2))

    # --- Rules-only logs ---
    sections.append("\n### eval_rules_only.py (zhang_full, test split)\n")
    log_specs = [
        ("3s", "artifacts/verify_run/logs/rules_eval_3s_SEP28k-T.txt"),
        ("mix", "artifacts/verify_run/logs/rules_eval_mix_SEP28k-T.txt"),
    ]
    rr = []
    for label, rel in log_specs:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            rr.append([label, "log missing", ""])
            continue
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
        pres, typ = _parse_rules_log(txt)
        rr.append([label, pres or "?", typ or ""])
    sections.append(_md_table(["Data root", "presence_F1", "tail type_F1 line"], [[a, b, c[:80] + ("…" if len(c) > 80 else "")] for a, b, c in rr]))

    # --- Four-head strict ---
    sections.append("\n### eval_4head_with_block_learned_pause_strict.py\n")
    four = [
        ("3s T", "artifacts/error_analysis/four_head_strict_with_learned_block_sep28kT_seed42.json"),
        ("3s D", "artifacts/error_analysis/four_head_strict_with_learned_block_sep28kD_seed42.json"),
        ("mix T", "artifacts/error_analysis/four_head_strict_with_learned_block_sep28kT_seed42_mix53.json"),
        ("mix D", "artifacts/error_analysis/four_head_strict_with_learned_block_sep28kD_seed42_mix53.json"),
    ]
    fr = []
    for label, rel in four:
        j = _load(rel)
        if not j:
            fr.append([label, "—", "—"])
            continue
        b = j.get("baseline_4head_svm_f1", {})
        o = j.get("with_learned_pause_block_override_f1", {})
        fr.append([label, f"{float(b.get('presence', 0)):.4f}", f"{float(o.get('presence', 0)):.4f}"])
    sections.append(_md_table(["Run", "baseline presence", "Block-override presence"], fr))

    # --- Old layer-8 SVM ---
    sections.append("\n### old_behavior_svm_only_with_optional_pause_selection.py (SEP28k-T)\n")
    old_specs = [
        ("3s", "artifacts/error_analysis/ling230_old_svm_layer8_sep28kT_3s_block_learned.json"),
        ("mix", "artifacts/error_analysis/ling230_old_svm_layer8_sep28kT_mix53_block_learned.json"),
    ]
    orows = []
    for label, rel in old_specs:
        j = _load(rel)
        if not j:
            orows.append([label, "—", "—"])
            continue
        base = j.get("svm_f1", {})
        blk = j.get("block_learned_pause_selection", {})
        ov = blk.get("svm_f1_with_block_override") if isinstance(blk, dict) else None
        orows.append(
            [
                label,
                f"{float(base.get('presence', 0)):.4f}",
                f"{float(ov.get('presence', 0)):.4f}" if ov else "—",
            ]
        )
    sections.append(_md_table(["Run", "SVM presence", "with Block override presence"], orows))

    # --- Matched 5s vs 3s ---
    sections.append("\n### compare_matched_5s_vs_3s_pause_pipeline.py\n")
    ms = []
    for split, rel in [("SEP28k-T", "artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kT.json"), ("SEP28k-D", "artifacts/error_analysis/matched_5s_vs_3s_pause_pipeline_sep28kD.json")]:
        j = _load(rel)
        if not j:
            ms.append([split, "—", "—", "—"])
            continue
        f5 = j["five_second_run"]["svm_f1"]["presence"]
        f3 = j["three_second_run"]["svm_f1"]["presence"]
        d = j["delta_3s_minus_5s_svm_f1"]["presence"]
        ms.append([split, f"{float(f5):.4f}", f"{float(f3):.4f}", f"{float(d):+.4f}"])
    sections.append(_md_table(["Split", "5s matched presence", "3s same-ID presence", "delta 3s−5s"], ms))

    # --- Block-only ---
    sections.append("\n### block_only_train_dev_test.py (Block head binary F1)\n")
    bo = [
        ("3s T", "artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kT_learned_pause_strict.json"),
        ("3s D", "artifacts/error_analysis/block_only_train_dev_test_seed42_3s_sep28kD_learned_pause_strict.json"),
        ("mix T", "artifacts/error_analysis/block_only_train_dev_test_seed42_mix53_sep28kT_learned_pause_strict.json"),
        ("mix D", "artifacts/error_analysis/block_only_train_dev_test_seed42_mix53_sep28kD_learned_pause_strict.json"),
    ]
    br = []
    for label, rel in bo:
        j = _load(rel)
        if not j:
            br.append([label, "—", "—", "—"])
            continue
        res = j.get("results", {})
        base = res.get("block_base_train_dev_test", {})
        nat = res.get("block_natural_v2_train_dev_test", {})
        br.append(
            [
                label,
                f"{float(base.get('f1', 0)):.4f}",
                f"{float(nat.get('f1', 0)):.4f}",
                f"{float(res.get('delta_natural_minus_base', {}).get('f1', 0)):+.4f}",
            ]
        )
    sections.append(_md_table(["Run", "base Block F1", "natural v2 F1", "delta F1"], br))

    # --- Comparable bundle ---
    sections.append("\n### run_comparable_sep28kT_D_3s.sh merged JSON\n")
    cj = _load("artifacts/error_analysis/comparable_results_seed42_3s_sep28kT_and_D.json")
    if cj:
        for split in ("SEP28k-T", "SEP28k-D"):
            s = cj["splits"][split]
            svm = s["four_head_from_tune_thresholds"]["svm_f1"]["presence"]
            rules = s["four_head_from_tune_thresholds"]["rules_f1"]["presence"]
            b0 = s["block_only_learned_pause_strict"]["base"]["f1"]
            b1 = s["block_only_learned_pause_strict"]["natural_v2_learned_pause"]["f1"]
            sections.append(f"- **{split}** tune SVM presence={float(svm):.4f}, rules presence={float(rules):.4f}; block-only base F1={float(b0):.4f}, natural v2 F1={float(b1):.4f}\n")
    else:
        sections.append("_comparable_results JSON missing._\n")

    # --- BiLSTM ---
    sections.append("\n### disfluency_pipeline.py BiLSTM (verify run caps)\n")
    bl = []
    for label, rel in [
        ("3s", "artifacts/verify_run/logs/bilstm_3s_SEP28k-T.log"),
        ("mix", "artifacts/verify_run/logs/bilstm_mix_SEP28k-T.log"),
    ]:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            bl.append([label, "log missing"])
            continue
        with open(p, "r", encoding="utf-8") as f:
            line = _parse_bilstm_test_f1(f.read())
        bl.append([label, line or "(no Test F1 line found)"])
    sections.append(_md_table(["Data root", "last Test F1 line"], bl))

    report = "\n".join(sections)
    print(report)
    if args.write_md.strip():
        out = args.write_md.strip()
        outp = out if os.path.isabs(out) else os.path.join(REPO, out)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            f.write("# README verify run summary\n\n")
            f.write(report)
        print(f"\nWrote {outp}", flush=True)


if __name__ == "__main__":
    main()
