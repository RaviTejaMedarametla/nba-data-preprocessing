from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    missing_ratio: dict
    outlier_rate: float
    drift_score: float
    schema_ok: bool
    schema_issues: list[str]
    feature_drift: dict
    temporal_drift: dict


class DataValidator:
    def quality_report(self, df: pd.DataFrame, outlier_mask: pd.Series) -> ValidationReport:
        missing_ratio = (df.isna().sum() / max(len(df), 1)).to_dict()
        outlier_rate = float(outlier_mask.mean()) if len(outlier_mask) else 0.0
        return ValidationReport(
            missing_ratio=missing_ratio,
            outlier_rate=outlier_rate,
            drift_score=0.0,
            schema_ok=True,
            schema_issues=[],
            feature_drift={},
            temporal_drift={},
        )

    def schema_validation(self, df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, list[str]]:
        issues = []
        missing_cols = [c for c in required_columns if c not in df.columns]
        if missing_cols:
            issues.append(f'missing_columns:{missing_cols}')

        if 'salary' in df.columns and pd.to_numeric(df['salary'].astype(str).str.replace('$', '', regex=False), errors='coerce').isna().all():
            issues.append('salary_column_not_numeric')

        return len(issues) == 0, issues

    def drift_detection(self, baseline: pd.DataFrame, candidate: pd.DataFrame, numeric_col: str = 'salary') -> float:
        base_series = pd.to_numeric(baseline[numeric_col].astype(str).str.replace('$', '', regex=False), errors='coerce').dropna()
        cand_series = pd.to_numeric(candidate[numeric_col].astype(str).str.replace('$', '', regex=False), errors='coerce').dropna()
        base = base_series.to_numpy(dtype=float)
        cand = cand_series.to_numpy(dtype=float)
        if len(base) == 0 or len(cand) == 0:
            return 0.0

        base_sorted = np.sort(base)
        cand_sorted = np.sort(cand)
        values = np.unique(np.concatenate([base_sorted, cand_sorted]))
        base_cdf = np.searchsorted(base_sorted, values, side='right') / len(base_sorted)
        cand_cdf = np.searchsorted(cand_sorted, values, side='right') / len(cand_sorted)
        return float(np.max(np.abs(base_cdf - cand_cdf)))

    def feature_wise_drift(self, baseline: pd.DataFrame, candidate: pd.DataFrame) -> dict:
        result: dict[str, dict[str, float]] = {}
        common_cols = [c for c in baseline.columns if c in candidate.columns]
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(baseline[col]):
                b = pd.to_numeric(baseline[col], errors='coerce').dropna()
                c = pd.to_numeric(candidate[col], errors='coerce').dropna()
                if len(b) and len(c):
                    result[col] = {
                        'type': 'numeric',
                        'ks_like': self.drift_detection(pd.DataFrame({col: b}), pd.DataFrame({col: c}), numeric_col=col),
                        'mean_shift': float(c.mean() - b.mean()),
                    }
            else:
                b = baseline[col].astype(str)
                c = candidate[col].astype(str)
                b_dist = b.value_counts(normalize=True)
                c_dist = c.value_counts(normalize=True)
                cats = sorted(set(b_dist.index) | set(c_dist.index))
                l1 = float(sum(abs(float(b_dist.get(cat, 0.0)) - float(c_dist.get(cat, 0.0))) for cat in cats))
                result[col] = {'type': 'categorical', 'l1_drift': l1}
        return result

    def temporal_drift(self, df: pd.DataFrame, time_col: str = 'version') -> dict:
        if time_col not in df.columns or 'salary' not in df.columns or len(df) < 4:
            return {'status': 'insufficient_data'}

        years = pd.to_numeric(df[time_col].astype(str).str.extract(r'(\d+)$')[0], errors='coerce').fillna(0).astype(int)
        tmp = df.copy()
        tmp['_year'] = np.where(years < 100, years + 2000, years)
        tmp = tmp.sort_values('_year')
        mid = len(tmp) // 2
        early = tmp.iloc[:mid]
        late = tmp.iloc[mid:]
        return {
            'early_vs_late_salary_drift': self.drift_detection(early, late, numeric_col='salary'),
            'early_mean_salary': float(pd.to_numeric(early['salary'].astype(str).str.replace('$', '', regex=False), errors='coerce').mean()),
            'late_mean_salary': float(pd.to_numeric(late['salary'].astype(str).str.replace('$', '', regex=False), errors='coerce').mean()),
        }
