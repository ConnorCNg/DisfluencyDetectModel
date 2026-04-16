from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple


def cache_filename_for_path(abs_path: str) -> str:
    h = hashlib.sha256(os.path.abspath(abs_path).encode("utf-8")).hexdigest()
    return f"{h}.json"


def read_cache(cache_dir: str, abs_path: str) -> Optional[Dict[str, Any]]:
    fp = os.path.join(cache_dir, cache_filename_for_path(abs_path))
    if not os.path.isfile(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def write_cache(
    cache_dir: str,
    abs_path: str,
    logits: List[float],
    words: Optional[List[Tuple[str, float, float]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, cache_filename_for_path(abs_path))
    payload: Dict[str, Any] = {
        "version": 1,
        "path": abs_path,
        "logits": [float(x) for x in logits],
    }
    if words is not None:
        payload["words"] = [
            {"text": t, "start": float(a), "end": float(b)} for t, a, b in words
        ]
    if meta:
        payload["meta"] = meta
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0)
    return fp
