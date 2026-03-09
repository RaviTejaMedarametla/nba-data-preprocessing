from __future__ import annotations

import numpy as np


def bootstrap_ci(arr: np.ndarray, random_seed: int, n_bootstrap: int = 400) -> dict[str, float]:
    if len(arr) == 0:
        return {'sample_size': 0, 'mean': 0.0, 'std': 0.0, 'median': 0.0, 'p95': 0.0, 'ci95_low': 0.0, 'ci95_high': 0.0}
    rng = np.random.default_rng(random_seed)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_bootstrap)]
    return {
        'sample_size': int(len(arr)),
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=0)),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'ci95_low': float(np.percentile(means, 2.5)),
        'ci95_high': float(np.percentile(means, 97.5)),
    }


def permutation_pvalue(a: np.ndarray, b: np.ndarray, random_seed: int, n_perm: int = 1000) -> float:
    if len(a) == 0 or len(b) == 0:
        return 1.0
    rng = np.random.default_rng(random_seed)
    observed = abs(float(a.mean() - b.mean()))
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(combined)
        if abs(float(shuffled[: len(a)].mean() - shuffled[len(a) :].mean())) >= observed:
            count += 1
    return float((count + 1) / (n_perm + 1))
