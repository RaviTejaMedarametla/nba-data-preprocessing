from __future__ import annotations

from typing import Any, Callable

import numpy as np


def bootstrap_ci(arr: np.ndarray, random_seed: int, n_bootstrap: int = 400) -> dict[str, float]:
    """Return bootstrap confidence interval summary statistics."""
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
    """Estimate p-value via a two-sided permutation test on mean difference."""
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


def run_repeated_benchmark(fn: Callable[[], dict[str, Any]], runs: int = 3) -> dict[str, Any]:
    """Run a benchmark function repeatedly and summarize numeric outputs."""
    results = [fn() for _ in range(runs)]
    latency = np.array([float(r.get('latency_s', 0.0)) for r in results], dtype=float)
    throughput = np.array([float(r.get('throughput_rows_s', 0.0)) for r in results], dtype=float)
    return {
        'runs': results,
        'latency': {
            'mean': float(latency.mean()) if latency.size else 0.0,
            'std': float(latency.std(ddof=0)) if latency.size else 0.0,
        },
        'throughput': {
            'mean': float(throughput.mean()) if throughput.size else 0.0,
            'std': float(throughput.std(ddof=0)) if throughput.size else 0.0,
        },
    }
