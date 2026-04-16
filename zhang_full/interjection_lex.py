from __future__ import annotations

import re
from typing import List, Tuple

# Disfluency fillers / hesitation markers (English). Kept small to limit false positives.
FILLER_TOKENS = frozenset(
    {
        "uh",
        "um",
        "uhh",
        "umm",
        "erm",
        "er",
        "ah",
        "oh",
        "hmm",
        "hm",
        "mm",
        "mhm",
        "uhm",
        "ummm",
        "uhhh",
    }
)


_token_re = re.compile(r"[\w']+", re.UNICODE)


def normalize_word(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"^[^\w']+", "", t)
    t = re.sub(r"[^\w']+$", "", t)
    return t


def tokens_from_segment(text: str) -> List[str]:
    return [normalize_word(m.group(0)) for m in _token_re.finditer(text)]


def score_interjection(
    words: List[Tuple[str, float, float]],
) -> float:
    """Confidence in [0, 1] from word-timed segments."""
    if not words:
        return 0.0
    best = 0.0
    for text, _t0, _t1 in words:
        for tok in tokens_from_segment(text):
            if not tok:
                continue
            if tok in FILLER_TOKENS:
                best = 1.0
                break
            if len(tok) <= 2 and tok in ("uh", "um", "er", "ah", "oh", "hm"):
                best = max(best, 0.85)
        if best >= 1.0:
            break
    return best


def score_word_repetition(
    words: List[Tuple[str, float, float]],
    max_gap_s: float = 0.85,
) -> float:
    """
    Duplicate content words on a coarse timeline: consecutive tokens (within
    or across ASR segments) with the same normalized form and small time gap.
    """
    events: List[Tuple[str, float]] = []
    for text, t0, t1 in words:
        toks = tokens_from_segment(text)
        if not toks:
            continue
        t0f, t1f = float(t0), float(t1)
        span = max(t1f - t0f, 1e-4)
        step = span / max(len(toks), 1)
        for k, tok in enumerate(toks):
            if len(tok) < 2:
                continue
            ts = t0f + (k + 0.5) * step
            events.append((tok, ts))
    best = 0.0
    for i in range(len(events) - 1):
        ta, tsa = events[i]
        tb, tsb = events[i + 1]
        if ta == tb and (tsb - tsa) <= max_gap_s:
            best = max(best, 0.95)
    if len(words) >= 2:
        segs = [
            (tokens_from_segment(text), float(t0), float(t1))
            for text, t0, t1 in words
        ]
        for i in range(len(segs) - 1):
            toks_a, t0a, t1a = segs[i]
            toks_b, t0b, _t1b = segs[i + 1]
            gap = max(0.0, t0b - t1a)
            if gap > max_gap_s:
                continue
            sa = set(t for t in toks_a if len(t) >= 2)
            sb = set(t for t in toks_b if len(t) >= 2)
            inter = sa & sb
            if inter:
                best = max(best, min(1.0, 0.55 + 0.15 * len(inter)))
    return best
