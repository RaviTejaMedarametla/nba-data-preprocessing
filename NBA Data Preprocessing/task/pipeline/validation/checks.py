from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    missing_ratio: dict
    outlier_rate: float
    drift_score: float


class DataValidator:
    def quality_report(self, df: pd.DataFrame, outlier_mask: pd.Series) -> ValidationReport:
        missing_ratio = (df.isna().sum() / len(df)).to_dict()
        outlier_rate = float(outlier_mask.mean())
        return ValidationReport(missing_ratio=missing_ratio, outlier_rate=outlier_rate, drift_score=0.0)

    def drift_detection(self, baseline: pd.DataFrame, candidate: pd.DataFrame, numeric_col: str = 'salary') -> float:
        base_series = pd.to_numeric(baseline[numeric_col].astype(str).str.replace('$', '', regex=False), errors='coerce').dropna()
        cand_series = pd.to_numeric(candidate[numeric_col].astype(str).str.replace('$', '', regex=False), errors='coerce').dropna()
        base = base_series.to_numpy(dtype=float)
        cand = cand_series.to_numpy(dtype=float)
        if len(base) == 0 or len(cand) == 0:
            return 0.0

        # lightweight KS-style statistic without scipy
        base_sorted = np.sort(base)
        cand_sorted = np.sort(cand)
        values = np.unique(np.concatenate([base_sorted, cand_sorted]))
        base_cdf = np.searchsorted(base_sorted, values, side='right') / len(base_sorted)
        cand_cdf = np.searchsorted(cand_sorted, values, side='right') / len(cand_sorted)
        return float(np.max(np.abs(base_cdf - cand_cdf)))
