#!/usr/bin/env python3
"""
Copy missing SEP-28k clips into data/sep28k/clips/{Show}/{EpId}/.

Priority:
  1) data/sep28k-5sec/**/wavs/*.wav  (5 s)
  2) data/SEP-28k_3sec/{Show}/{Ep}/*.wav (3 s, only if still missing)

Never overwrites an existing file under sep28k/clips.
Skips hidden dirs like .cache.
"""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parent
DEST_ROOT = REPO / "data" / "sep28k" / "clips"
SEC5_ROOT = REPO / "data" / "sep28k-5sec"
SEC3_ROOT = REPO / "data" / "SEP-28k_3sec"


def parse_show_ep_clip(basename: str) -> tuple[str, str, str] | None:
    if not basename.lower().endswith(".wav"):
        return None
    stem = basename[:-4]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    ep, clip = parts[-2], parts[-1]
    show = "_".join(parts[:-2])
    if not show or not ep or not clip:
        return None
    return show, ep, clip


def dest_path(show: str, ep: str, clip_basename: str) -> Path:
    return DEST_ROOT / show / ep / clip_basename


def iter_wavs_skip_cache(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "/.cache" in dirpath.replace("\\", "/") or dirpath.endswith(".cache"):
            continue
        dirnames[:] = [d for d in dirnames if d != ".cache" and not d.startswith(".")]
        for fn in filenames:
            if fn.lower().endswith(".wav"):
                out.append(Path(dirpath) / fn)
    return out


def main() -> None:
    added_5: list[tuple[str, str, str]] = []  # show, ep, basename
    added_3: list[tuple[str, str, str]] = []
    skipped_exists = 0
    skipped_badname = 0

    # --- 5 sec first ---
    for src in iter_wavs_skip_cache(SEC5_ROOT):
        if "wavs" not in src.parts:
            continue
        parsed = parse_show_ep_clip(src.name)
        if parsed is None:
            skipped_badname += 1
            continue
        show, ep, _clip = parsed
        dst = dest_path(show, ep, src.name)
        if dst.is_file():
            skipped_exists += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        added_5.append((show, ep, src.name))

    # --- 3 sec supplement ---
    for src in iter_wavs_skip_cache(SEC3_ROOT):
        parsed = parse_show_ep_clip(src.name)
        if parsed is None:
            skipped_badname += 1
            continue
        show, ep, _clip = parsed
        # Layout is already Show/Ep/file.wav — require consistency
        try:
            rel = src.relative_to(SEC3_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 3:
            continue
        show_d, ep_d, fname = parts[0], parts[1], parts[2]
        if show_d != show or ep_d != ep:
            # Filename wins for show/ep; skip inconsistent tree
            continue
        dst = dest_path(show, ep, fname)
        if dst.is_file():
            skipped_exists += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        added_3.append((show, ep, fname))

    def summarize(rows: list[tuple[str, str, str]], label: str) -> None:
        by_ep: dict[tuple[str, str], int] = defaultdict(int)
        for show, ep, _ in rows:
            by_ep[(show, ep)] += 1
        print(f"\n=== {label}: {len(rows)} file(s) copied ===")
        print(f"Unique podcast+episode pairs: {len(by_ep)}")
        for (show, ep), n in sorted(by_ep.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"  {show}  episode {ep}  ({n} clip(s))")

    summarize(added_5, "Added from sep28k-5sec (5 s)")
    summarize(added_3, "Added from SEP-28k_3sec (3 s)")
    print("\n=== Skipped (already in sep28k/clips) ===")
    print(f"  {skipped_exists} path(s) already existed")
    if skipped_badname:
        print(f"  {skipped_badname} file(s) skipped (unparseable name)")


if __name__ == "__main__":
    main()
