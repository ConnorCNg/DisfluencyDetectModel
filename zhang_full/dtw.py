from __future__ import annotations

import numpy as np


def dtw_total_normalized(a: np.ndarray, b: np.ndarray) -> float:
    """
    Classic DTW total path cost, normalized by (n + m) for scale across sizes.

    a: (n, d), b: (m, d) float
    """
    n, m = int(a.shape[0]), int(b.shape[0])
    if n == 0 or m == 0:
        return 1e6
    inf = 1e30
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dp[i, j] = c + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
    return float(dp[n, m] / max(n + m, 1))
