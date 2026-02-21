from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollingState:
    rolling_window: int
    salary_window: deque[float]


class FeatureEngineer:
    def init_rolling_state(self, rolling_window: int = 5) -> RollingState:
        return RollingState(rolling_window=rolling_window, salary_window=deque(maxlen=rolling_window))

    def _add_streaming_rolling_features(self, featured: pd.DataFrame, state: RollingState) -> pd.DataFrame:
        means, stds = [], []
        for salary in featured['salary'].astype(float):
            state.salary_window.append(float(salary))
            vals = np.array(state.salary_window, dtype=float)
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)
        featured['salary_roll_mean'] = means
        featured['salary_roll_std'] = stds
        return featured

    def build_features(self, df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
        featured = df.copy()

        year = featured['version'].astype(str).str.extract(r'(\d+)$')[0].astype(int)
        year = np.where(year < 100, year + 2000, year)

        featured['version_year'] = year
        featured['age'] = featured['version_year'] - featured['b_day'].dt.year
        featured['experience'] = featured['version_year'] - featured['draft_year'].dt.year
        featured['bmi'] = featured['weight'] / (featured['height'] ** 2)

        featured = featured.sort_values('version_year').reset_index(drop=True)
        featured['salary_roll_mean'] = featured['salary'].rolling(window=rolling_window, min_periods=1).mean()
        featured['salary_roll_std'] = featured['salary'].rolling(window=rolling_window, min_periods=1).std().fillna(0.0)

        featured['birth_month'] = featured['b_day'].dt.month
        featured['draft_decade'] = (featured['draft_year'].dt.year // 10) * 10

        z = (featured['salary'] - featured['salary'].mean()) / featured['salary'].std(ddof=0)
        featured['salary_anomaly'] = (z.abs() > 2.5).astype(int)

        return featured

    def build_features_streaming(self, df: pd.DataFrame, state: RollingState) -> pd.DataFrame:
        featured = df.copy()
        year = featured['version'].astype(str).str.extract(r'(\d+)$')[0].astype(int)
        year = np.where(year < 100, year + 2000, year)

        featured['version_year'] = year
        featured['age'] = featured['version_year'] - featured['b_day'].dt.year
        featured['experience'] = featured['version_year'] - featured['draft_year'].dt.year
        featured['bmi'] = featured['weight'] / (featured['height'] ** 2)

        featured = featured.sort_values('version_year').reset_index(drop=True)
        featured = self._add_streaming_rolling_features(featured, state)

        featured['birth_month'] = featured['b_day'].dt.month
        featured['draft_decade'] = (featured['draft_year'].dt.year // 10) * 10
        salary = featured['salary'].astype(float)
        denom = salary.std(ddof=0)
        z = (salary - salary.mean()) / (denom if denom else 1.0)
        featured['salary_anomaly'] = (z.abs() > 2.5).astype(int)
        return featured

    def drop_multicollinearity(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        transformed = df.copy()
        numeric_columns = [c for c in transformed.select_dtypes(include='number').columns if c != 'salary']

        while len(numeric_columns) > 1:
            variable_columns = [c for c in numeric_columns if transformed[c].nunique(dropna=False) > 1]
            if len(variable_columns) <= 1:
                break
            corr_matrix = transformed[variable_columns].corr().fillna(0.0)
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            pairs = upper_triangle.stack()
            if pairs.empty or pairs.abs().max() <= threshold:
                break

            first, second = pairs.abs().idxmax()
            target_corr = transformed[variable_columns].corrwith(transformed['salary']).abs().fillna(0.0)
            drop_col = first if target_corr[first] < target_corr[second] else second
            transformed = transformed.drop(columns=drop_col)
            numeric_columns.remove(drop_col)

        return transformed

    def encode_and_scale(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        y = df['salary'].copy()
        x = df.drop(columns=['salary', 'version', 'b_day', 'draft_year', 'weight', 'height'], errors='ignore')

        for col in list(x.select_dtypes(include='object').columns):
            if x[col].nunique(dropna=False) >= 50:
                x = x.drop(columns=col)

        num_cols = list(x.select_dtypes(include='number').columns)
        num = x[num_cols].astype(float)
        num = num.fillna(num.median())
        num = (num - num.mean()) / num.std(ddof=0).replace(0.0, 1.0)

        cat_cols = list(x.select_dtypes(include='object').columns)
        cat_frames = []
        for col in cat_cols:
            filled = x[col].fillna(f'Unknown_{col}')
            cats = pd.Index(pd.unique(filled))
            one_hot = pd.DataFrame((filled.to_numpy()[:, None] == cats.to_numpy()).astype(int), columns=[f'{col}__{c}' for c in cats.astype(str)], index=x.index)
            cat_frames.append(one_hot)

        cat = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=x.index)
        return pd.concat([num, cat], axis=1), y
