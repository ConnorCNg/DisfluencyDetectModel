"""
Acoustic + optional lexical cascade for clip-level rule logits (4 heads).

Order of cues (coarse Zhang-style): extended silence / block, sound-level
repetition (DTW on MFCC halves in sliding windows), prolongation (high
frame-to-frame MFCC similarity + duration vs speaking rate), word-level
repetition and interjections when `words` is provided (e.g. from Whisper).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from zhang_full.dtw import dtw_total_normalized
from zhang_full.interjection_lex import score_interjection, score_word_repetition
from zhang_rules import (
    _estimate_syllable_rate,
    _longest_run_seconds,
    _rms_envelope,
    _score_to_logit,
)

# One MFCC transform per (sample_rate, device) so weights stay on GPU/MPS when used.
_MFCC_BY_KEY: dict[tuple[int, str], torchaudio.transforms.MFCC] = {}


def _mfcc_key(device: torch.device) -> str:
    return f"{device.type}:{device.index if device.index is not None else 0}"


def _get_mfcc(sample_rate: int, device: torch.device) -> torchaudio.transforms.MFCC:
    key = (sample_rate, _mfcc_key(device))
    t = _MFCC_BY_KEY.get(key)
    if t is None:
        t = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 40,
                "center": True,
            },
        ).to(device)
        _MFCC_BY_KEY[key] = t
    return t


def _mfcc_frames(
    waveform_1ch: torch.Tensor, sample_rate: int
) -> torch.Tensor:
    """waveform_1ch: (1, L) -> (T, 13) on same device as input."""
    mfcc = _get_mfcc(sample_rate, waveform_1ch.device)
    mf = mfcc(waveform_1ch.unsqueeze(0))
    if mf.dim() == 4:
        mf = mf.squeeze(1)
    mf = mf[0].transpose(0, 1).float()
    return mf


def _longest_run_seconds_device_safe(
    mask: torch.Tensor, hop_sec: float
) -> torch.Tensor:
    """
    _longest_run_seconds uses Python loops over time; on CUDA/MPS that forces
    a sync per step. Run the mask scan on CPU (small T) and return on ``mask.device``.
    """
    dev = mask.device
    return _longest_run_seconds(mask.detach().cpu(), hop_sec).to(dev)


def _acf_repetition_score(rms: torch.Tensor) -> torch.Tensor:
    """(1, T) RMS -> (1,) score in [0,1] — same spirit as zhang_rules."""
    b = rms.shape[0]
    rms_n = rms - rms.mean(dim=1, keepdim=True)
    std = rms_n.std(dim=1, keepdim=True).clamp(min=1e-4)
    rms_n = rms_n / std
    t_r = rms_n.shape[1]
    var0 = (rms_n**2).mean(dim=1).clamp(min=1e-4)
    max_acf = torch.zeros(b, device=rms.device, dtype=torch.float32)
    lag_max = min(48, t_r - 2)
    for lag in range(3, lag_max):
        pr = (rms_n[:, :-lag] * rms_n[:, lag:]).mean(dim=1) / var0
        max_acf = torch.maximum(max_acf, pr)
    return torch.clamp((max_acf - 0.25) / 0.55, 0.0, 1.0)


def _sound_rep_dtw_score(mf: np.ndarray, win: int = 32, hop: int = 6) -> float:
    """mf: (T, 13) numpy. Scan sliding windows; low DTW(first_half, second_half) → repetition."""
    t = mf.shape[0]
    if t < win + 4:
        return 0.0
    best = 1e9
    mid = win // 2
    for s in range(0, t - win, hop):
        seg = mf[s : s + win]
        a = seg[:mid]
        b = seg[mid:]
        if a.shape[0] < 4 or b.shape[0] < 4:
            continue
        d = dtw_total_normalized(a, b)
        if d < best:
            best = d
    if best > 1e8:
        return 0.0
    # Empirical: identical halves ~ small d; distinct halves larger.
    return float(np.clip((0.65 - best) / 0.45, 0.0, 1.0))


def compute_rule_logits_from_wav1c(
    wav: torch.Tensor,
    sample_rate: int,
    words: Optional[List[Tuple[str, float, float]]] = None,
    *,
    theta_sim: float = 0.92,
    alpha_dur: float = 1.2,
    silence_block_ms: float = 350.0,
) -> torch.Tensor:
    """
    Acoustic (+ optional lexical) rule logits on the **waveform device**.

    ``wav`` shape ``(1, L)`` mono float32. Returns ``(4,)`` float32 on ``wav.device``.

    MFCC / RMS / ACF run on-device. DTW (NumPy) uses a small MFCC CPU copy per clip.
    Longest-run-over-frames uses a CPU mask scan to avoid per-timestep GPU sync.
    """
    device = wav.device
    if wav.dim() != 2 or wav.shape[0] != 1:
        raise ValueError("wav must have shape (1, L)")
    if wav.shape[1] < int(sample_rate * 0.15):
        return torch.zeros(4, device=device, dtype=torch.float32)

    mf = _mfcc_frames(wav, sample_rate)
    t_frames = mf.shape[0]
    if t_frames < 4:
        return torch.zeros(4, device=device, dtype=torch.float32)

    hop_sec = 160.0 / float(sample_rate)
    mf_np = mf.detach().float().cpu().numpy()

    # --- prolongation (MFCC cosine run vs syllable-scaled min duration)
    a = mf[:-1] - mf[:-1].mean(dim=-1, keepdim=True)
    b_ = mf[1:] - mf[1:].mean(dim=-1, keepdim=True)
    den = a.norm(dim=-1) * b_.norm(dim=-1) + 1e-8
    sim = (a * b_).sum(dim=-1) / den
    prolong_mask = sim > theta_sim
    prolong_mask_2d = prolong_mask.unsqueeze(0)
    rms = _rms_envelope(wav.unsqueeze(0), sample_rate)
    sr_est = _estimate_syllable_rate(rms, sample_rate, 10.0)
    t_min = alpha_dur / sr_est
    max_run_p = _longest_run_seconds_device_safe(prolong_mask_2d, hop_sec)[0]
    prolong_ratio = max_run_p / t_min.clamp(min=1e-3)
    score_p = torch.clamp((prolong_ratio - 0.85) / 1.5, 0.0, 1.0)

    # --- sound repetition: max(DTW halves, RMS ACF)
    score_dtw = _sound_rep_dtw_score(mf_np)
    score_acf = _acf_repetition_score(rms)[0]
    score_sound = torch.maximum(
        torch.as_tensor(score_dtw, device=device, dtype=torch.float32), score_acf
    )

    # --- block: long low-RMS run vs global floor
    glob = rms.mean(dim=1, keepdim=True).clamp(min=1e-6)
    silent = rms < (glob * 0.12)
    max_sil = _longest_run_seconds_device_safe(silent, hop_sec)[0]
    need = silence_block_ms / 1000.0
    score_blk = torch.clamp((max_sil - need * 0.5) / (need + 1e-6), 0.0, 1.0)

    # --- lexical
    if words:
        score_word = float(score_word_repetition(words))
        score_fill = float(score_interjection(words))
    else:
        score_word = 0.0
        score_fill = 0.0

    sw = torch.as_tensor(score_word, device=device, dtype=torch.float32)
    score_rep = torch.maximum(score_sound, sw)

    logit_p = _score_to_logit(score_p)
    logit_r = _score_to_logit(score_rep)
    logit_b = _score_to_logit(score_blk)
    if words:
        logit_i = _score_to_logit(
            torch.as_tensor(score_fill, device=device, dtype=torch.float32)
        )
    else:
        logit_i = torch.full((), -2.0, device=device, dtype=torch.float32)

    def _sc(t: torch.Tensor) -> torch.Tensor:
        return t.float().reshape(-1)[0]

    return torch.stack([_sc(logit_p), _sc(logit_r), _sc(logit_i), _sc(logit_b)])


def compute_rule_logits_mono(
    mono: np.ndarray,
    sample_rate: int,
    words: Optional[List[Tuple[str, float, float]]] = None,
    *,
    theta_sim: float = 0.92,
    alpha_dur: float = 1.2,
    silence_block_ms: float = 350.0,
) -> np.ndarray:
    """
    mono: float32 (L,) single channel
    words: optional list of (text_segment, start_s, end_s) from ASR
    returns: (4,) float32 logits [Prolongation, Repetition, Interjection, Block]

    CPU-only entry point (NumPy in/out). For GPU/MPS, use
    ``compute_rule_logits_from_wav1c`` from a ``(1, L)`` tensor on-device.
    """
    mono = np.asarray(mono, dtype=np.float32).reshape(-1)
    if mono.size < int(sample_rate * 0.15):
        return np.zeros(4, dtype=np.float32)

    wav = torch.from_numpy(mono.copy()).float().unsqueeze(0)
    logits = compute_rule_logits_from_wav1c(
        wav,
        sample_rate,
        words,
        theta_sim=theta_sim,
        alpha_dur=alpha_dur,
        silence_block_ms=silence_block_ms,
    )
    return logits.detach().cpu().numpy().astype(np.float32)
