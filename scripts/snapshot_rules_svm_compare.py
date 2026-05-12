#!/usr/bin/env python3
"""
Copy artifacts and source files needed to reproduce a rules+SVM compare run.

Does not copy audio clips (too large); manifest records data_root and CSV path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]

# Python modules that compare_rules_svm_hybrid.py imports (direct + important transitive).
SOURCE_FILES = [
    "compare_rules_svm_hybrid.py",
    "tune_thresholds_rules_svm.py",
    "disfluency_pipeline.py",
    "paper_style_w2v2_svm_test.py",
    "zhang_full/__init__.py",
    "zhang_full/rule_module.py",
    "zhang_full/cascade.py",
    "zhang_full/interjection_lex.py",
    "zhang_full/cache.py",
    "zhang_full/dtw.py",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info(cwd: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"git_available": False}
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        out["git_available"] = True
        out["commit"] = rev
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        )
        out["dirty_files"] = [ln for ln in dirty.splitlines() if ln.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return out


def _copy_file(src: Path, dst_dir: Path, subpath: str) -> Dict[str, Any]:
    dst = dst_dir / subpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rel = str(Path(subpath))
    return {
        "path": rel,
        "source": str(src.resolve()),
        "bytes": dst.stat().st_size,
        "sha256": _sha256(dst),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Snapshot thresholds, SVM head JSON, label CSV, and key Python sources."
    )
    ap.add_argument(
        "--thresholds-json",
        default="artifacts/tuned_thresholds_rules_svm.json",
        help="Thresholds JSON from tune_thresholds_rules_svm.py",
    )
    ap.add_argument(
        "--csv",
        default="SEP-28k-Extended_clips.csv",
        help="Label CSV (same basename as enforced for dysfluency labels).",
    )
    ap.add_argument(
        "--data-root",
        default="data/sep28k/clips",
        help="Clip root (recorded in manifest only; wav files are not copied).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Destination directory. Default: artifacts/snapshots/<utc>_<tag>",
    )
    ap.add_argument(
        "--tag",
        default="compare_seed42",
        help="Short label for the default out-dir name.",
    )
    ap.add_argument(
        "--run-command",
        default="",
        help="Exact shell command for this run (stored in run_command.txt).",
    )
    args = ap.parse_args()

    root = REPO_ROOT
    os.chdir(root)

    th_path = (Path(args.thresholds_json) if Path(args.thresholds_json).is_absolute() else root / args.thresholds_json).resolve()
    if not th_path.is_file():
        raise SystemExit(f"Missing thresholds file: {th_path}")

    with th_path.open(encoding="utf-8") as f:
        th_payload = json.load(f)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if (args.out_dir or "").strip():
        out_dir = Path(args.out_dir.strip()).resolve()
    else:
        out_dir = (root / "artifacts" / "snapshots" / f"{ts}_{args.tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files_manifest: List[Dict[str, Any]] = []

    # Thresholds JSON
    rel_th = Path("artifacts") / "tuned_thresholds_rules_svm.json"
    dst_th = out_dir / rel_th
    dst_th.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(th_path, dst_th)
    files_manifest.append(
        {
            "path": str(rel_th),
            "source": str(th_path),
            "bytes": dst_th.stat().st_size,
            "sha256": _sha256(dst_th),
        }
    )

    # Per-head SVM config from JSON path (absolute in file → store under snapshot/artifacts/)
    hj = (th_payload.get("svm_head_config_json") or "").strip()
    if hj:
        head_src = Path(hj).resolve()
        if head_src.is_file():
            rel_hj = Path("artifacts") / head_src.name
            dst_hj = out_dir / rel_hj
            shutil.copy2(head_src, dst_hj)
            files_manifest.append(
                {
                    "path": str(rel_hj),
                    "source": str(head_src),
                    "bytes": dst_hj.stat().st_size,
                    "sha256": _sha256(dst_hj),
                }
            )

    # Label CSV
    csv_arg = Path(args.csv)
    csv_src = csv_arg if csv_arg.is_absolute() else (root / csv_arg).resolve()
    if not csv_src.is_file():
        raise SystemExit(f"Missing label CSV: {csv_src}")
    rel_csv = Path("labels") / csv_src.name
    dst_csv = out_dir / rel_csv
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_src, dst_csv)
    files_manifest.append(
        {
            "path": str(rel_csv),
            "source": str(csv_src),
            "bytes": dst_csv.stat().st_size,
            "sha256": _sha256(dst_csv),
        }
    )

    # Source tree mirror under snapshot/src/
    for rel in SOURCE_FILES:
        p = root / rel
        if not p.is_file():
            raise SystemExit(f"Missing source file for snapshot: {p}")
        sub = Path("src") / rel
        files_manifest.append(_copy_file(p, out_dir, str(sub)))

    run_cmd = (args.run_command or "").strip() or " ".join(sys.argv)
    (out_dir / "run_command.txt").write_text(run_cmd + "\n", encoding="utf-8")

    # Runnable helper: cd to repo root then exec the saved compare line.
    reproduce = out_dir / "REPRODUCE_COMPARE.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "# Repo root = three levels up from this file (.../artifacts/snapshots/<id>/)\n"
        'REPO="$(cd "$(dirname "$0")/../../.." && pwd)"\n'
        'cd "$REPO"\n'
        f"exec {run_cmd}\n",
        encoding="utf-8",
    )
    reproduce.chmod(reproduce.stat().st_mode | 0o111)

    meta = {
        "created_utc": ts,
        "tag": args.tag,
        "reproduce_script": "REPRODUCE_COMPARE.sh",
        "compare_command": run_cmd,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(root),
        "data_root": str(Path(args.data_root)),
        "data_root_note": "WAV clips are not copied; restore from SEP-28k or your mirror.",
        "thresholds_json_source": str(th_path),
        "git": _git_info(root),
        "files": files_manifest,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Snapshot written to: {out_dir}", flush=True)
    print(f"  files recorded: {len(files_manifest)}", flush=True)


if __name__ == "__main__":
    main()
