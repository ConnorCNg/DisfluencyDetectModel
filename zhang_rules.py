"""
Heuristic rule logits inspired by Zhang (2025) arXiv:2508.16681 — prolongation
(MFCC frame similarity + speaking-rate-normalized duration), repetition
(quasi-periodicity via normalized RMS autocorrelation), block (extended
low-energy / silence). Interjections are not reliably detectable without
ASR/lexical cues; that column is left at neutral (0 logit).

These are interpretable acoustic proxies, not a full reproduction of the
paper's MFA/DTW cascade; no gradients are used in forward().
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torchaudio

def _score_to_logit(score: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Map [0, 1] confidence to logits for BCEWithLogitsLoss downstream."""
    p = score.clamp(eps, 1.0 - eps)
    return torch.logit(p)


def _rms_envelope(wav: torch.Tensor, sample_rate: int, win_ms: float = 25.0, hop_ms: float = 10.0) -> torch.Tensor:
    """wav: (B, 1, L) -> RMS (B, T_frames)."""
    win = max(1, int(sample_rate * win_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    b, _, l = wav.shape
    x = wav[:, 0, :]
    if l < win:
        return x.abs().mean(dim=-1, keepdim=True)
    n = 1 + (l - win) // hop
    u = x.unfold(1, win, hop)[:, :n, :]
    return u.pow(2).mean(dim=-1).sqrt()


def _longest_run_seconds(mask: torch.Tensor, hop_sec: float) -> torch.Tensor:
    """mask: (B, T) bool; returns longest True run length in seconds per batch."""
    b, t = mask.shape
    out = torch.zeros(b, device=mask.device, dtype=torch.float32)
    for i in range(b):
        m = mask[i]
        if not m.any():
            continue
        cur = 0
        best = 0
        for j in range(t):
            if m[j]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out[i] = float(best) * hop_sec
    return out


def _estimate_syllable_rate(rms: torch.Tensor, sample_rate: int, hop_ms: float) -> torch.Tensor:
    """Rough syllable nuclei proxy: voiced onset peaks per second, clamped."""
    hop_sec = hop_ms / 1000.0
    thr = rms.median(dim=1, keepdim=True).values * 0.6
    voiced = rms > thr
    dv = voiced[:, 1:] & ~voiced[:, :-1]
    peaks = dv.sum(dim=1).float()
    dur = max(0.25, float(rms.shape[1]) * hop_sec)
    sr_est = peaks / dur
    return sr_est.clamp(2.0, 7.0)


class ZhangStyleRuleLogits(nn.Module):
    """
    Clip-level rule logits (B, 4) for Prolongation, Repetition, Interjection, Block.
    Forward runs under torch.no_grad(); parameters are not trained.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        theta_sim: float = 0.92,
        alpha_dur: float = 1.2,
        silence_block_ms: float = 350.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.theta_sim = theta_sim
        self.alpha_dur = alpha_dur
        self.silence_block_ms = silence_block_ms
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
                "center": True,
            },
        )

    def forward(
        self,
        waveform: torch.Tensor,
        paths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._forward_impl(waveform)

    def _forward_impl(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        dtype = waveform.dtype
        b = waveform.shape[0]
        hop_sec = 160.0 / float(self.sample_rate)

        mfcc = self.mfcc(waveform)
        # torchaudio may return (B, n_mfcc, T) or (B, 1, n_mfcc, T) for (B, 1, time)
        if mfcc.dim() == 4:
            mfcc = mfcc.squeeze(1)
        mf = mfcc.transpose(1, 2).float()  # B, T, n_mfcc
        if mf.shape[1] < 3:
            return torch.zeros(b, 4, device=device, dtype=dtype)

        a = mf[:, :-1, :] - mf[:, :-1, :].mean(dim=-1, keepdim=True)
        b_ = mf[:, 1:, :] - mf[:, 1:, :].mean(dim=-1, keepdim=True)
        den = a.norm(dim=-1) * b_.norm(dim=-1) + 1e-8
        sim = (a * b_).sum(dim=-1) / den
        prolong_mask = sim > self.theta_sim
        sr_est = _estimate_syllable_rate(
            _rms_envelope(waveform, self.sample_rate), self.sample_rate, 10.0
        )
        t_min = self.alpha_dur / sr_est
        max_run_p = _longest_run_seconds(prolong_mask, hop_sec)
        prolong_ratio = max_run_p / t_min.clamp(min=1e-3)
        score_p = torch.clamp((prolong_ratio - 0.85) / 1.5, 0.0, 1.0)

        rms = _rms_envelope(waveform, self.sample_rate)
        rms_n = rms - rms.mean(dim=1, keepdim=True)
        std = rms_n.std(dim=1, keepdim=True).clamp(min=1e-4)
        rms_n = rms_n / std
        t_r = rms_n.shape[1]
        var0 = (rms_n**2).mean(dim=1).clamp(min=1e-4)
        max_acf = torch.zeros(b, device=device, dtype=torch.float32)
        lag_max = min(48, t_r - 2)
        for lag in range(3, lag_max):
            pr = (rms_n[:, :-lag] * rms_n[:, lag:]).mean(dim=1) / var0
            max_acf = torch.maximum(max_acf, pr)
        score_r = torch.clamp((max_acf - 0.25) / 0.55, 0.0, 1.0)

        glob = rms.mean(dim=1, keepdim=True).clamp(min=1e-6)
        silent = rms < (glob * 0.12)
        max_sil = _longest_run_seconds(silent, hop_sec)
        need = self.silence_block_ms / 1000.0
        score_blk = torch.clamp((max_sil - need * 0.5) / (need + 1e-6), 0.0, 1.0)

        # No reliable acoustic-only interjection cue without ASR; weak prior "absent".
        logit_i = torch.full((b,), -2.0, device=device, dtype=torch.float32)

        logits = torch.stack(
            [
                _score_to_logit(score_p).to(dtype),
                _score_to_logit(score_r).to(dtype),
                logit_i.to(dtype),
                _score_to_logit(score_blk).to(dtype),
            ],
            dim=1,
        )
        return logits
