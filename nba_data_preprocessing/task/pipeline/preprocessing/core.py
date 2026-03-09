from __future__ import annotations

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, random_seed: int = 42, missing_strategy: str = 'median'):
        self.random_seed = random_seed
        self.missing_strategy = missing_strategy

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()

        cleaned['b_day'] = pd.to_datetime(cleaned['b_day'], format='%m/%d/%y', errors='coerce')
        cleaned['draft_year'] = pd.to_datetime(cleaned['draft_year'], format='%Y', errors='coerce')
        cleaned['team'] = cleaned['team'].fillna('No Team')
        cleaned['height'] = cleaned['height'].astype(str).str.split(' / ').str[1].astype(float)
        cleaned['weight'] = cleaned['weight'].astype(str).str.split(' / ').str[1].str.replace(' kg.', '', regex=False).astype(float)
        cleaned['salary'] = cleaned['salary'].astype(str).str.replace('$', '', regex=False).astype(float)
        cleaned['country'] = np.where(cleaned['country'] == 'USA', 'USA', 'Not-USA')
        cleaned['draft_round'] = cleaned['draft_round'].replace('Undrafted', '0')

        return self.handle_missing(cleaned)

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        num_cols = list(out.select_dtypes(include='number').columns)
        cat_cols = list(out.select_dtypes(include='object').columns)

        if self.missing_strategy == 'median':
            out[num_cols] = out[num_cols].fillna(out[num_cols].median())
        elif self.missing_strategy == 'mean':
            out[num_cols] = out[num_cols].fillna(out[num_cols].mean())
        else:
            out[num_cols] = out[num_cols].fillna(0)

        for col in cat_cols:
            out[col] = out[col].fillna(f'Unknown_{col}')

        return out

    def detect_outliers_iqr(self, df: pd.DataFrame, multiplier: float = 1.5) -> pd.Series:
        numeric = df.select_dtypes(include='number')
        if numeric.empty:
            return pd.Series([False] * len(df), index=df.index)
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        mask = ((numeric < (q1 - multiplier * iqr)) | (numeric > (q3 + multiplier * iqr))).any(axis=1)
        return mask
