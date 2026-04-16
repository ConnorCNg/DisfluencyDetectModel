from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from zhang_full.cache import read_cache
from zhang_full.cascade import compute_rule_logits_from_wav1c


class ZhangFullRuleLogits(nn.Module):
    """
    Full Zhang-style rule logits: acoustic cascade + optional lexical scores.
    If ``cache_dir`` is set and batch ``paths`` hit a JSON cache (from
    ``precompute_zhang_full_cache.py``), loads stored logits; otherwise runs
    ``compute_rule_logits_from_wav1c`` on the waveform on the **same device** as
    the input batch (no ASR — interjection column stays weak unless you
    precomputed with Whisper).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.cache_dir = cache_dir.strip() if cache_dir else None

    def forward(
        self,
        waveform: torch.Tensor,
        paths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        b, _c, _l = waveform.shape
        device = waveform.device
        dtype = waveform.dtype
        out = torch.empty(b, 4, device=device, dtype=torch.float32)
        with torch.no_grad():
            for i in range(b):
                loaded = False
                p = None
                if paths and i < len(paths):
                    p = paths[i]
                if self.cache_dir and p:
                    data = read_cache(self.cache_dir, os.path.abspath(p))
                    if data and isinstance(data.get("logits"), list) and len(
                        data["logits"]
                    ) >= 4:
                        lo = data["logits"][:4]
                        out[i] = torch.tensor(lo, device=device, dtype=torch.float32)
                        loaded = True
                if not loaded:
                    wav1 = waveform[i : i + 1, 0].detach().float().contiguous()
                    logits = compute_rule_logits_from_wav1c(
                        wav1, self.sample_rate, None
                    )
                    out[i] = logits.to(device=device, dtype=torch.float32)
        return out.to(dtype)
