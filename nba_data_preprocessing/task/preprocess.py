from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.feature_engineering.legacy_helpers import feature_data, multicol_data, transform_data
from pipeline.models.linear_helpers import evaluate_linear_regression_cv, train_linear_regression_model
from pipeline.preprocessing.cleaners import clean_dataframe as _clean_dataframe
from pipeline.preprocessing.cleaners import ensure_dataframe as _ensure_dataframe


def clean_data(path: str | pd.DataFrame) -> pd.DataFrame:
    return _clean_dataframe(_ensure_dataframe(path))


class NBAPreprocessingPipeline:
    def __init__(self):
        self._is_fitted = False

    def fit(self, data):
        df = _ensure_dataframe(data)
        cleaned = _clean_dataframe(df)
        featured = feature_data(cleaned)
        filtered = multicol_data(featured)
        x = filtered.drop(columns='salary') if 'salary' in filtered.columns else filtered.copy()
        self.feature_columns_ = [column for column in filtered.columns if column != 'salary']
        self.num_columns_ = list(x.select_dtypes(include='number').columns)
        num_data = x[self.num_columns_].astype(float)
        self.num_medians_ = num_data.median()
        num_data = num_data.fillna(self.num_medians_)
        self.num_means_ = num_data.mean()
        self.num_stds_ = num_data.std(ddof=0).replace(0.0, 1.0)
        self.cat_columns_ = list(x.select_dtypes(include='object').columns)
        self.cat_categories_ = {}
        for column in self.cat_columns_:
            filled = x[column].fillna(f'Unknown_{column}')
            self.cat_categories_[column] = pd.Index(pd.unique(filled))
        self._is_fitted = True
        return self

    def transform(self, data):
        if not self._is_fitted:
            raise ValueError('Pipeline is not fitted. Call fit() before transform().')
        df = _ensure_dataframe(data)
        cleaned = _clean_dataframe(df)
        featured = feature_data(cleaned)
        x = featured.reindex(columns=self.feature_columns_, fill_value=np.nan)
        num_data = x[self.num_columns_].astype(float).fillna(self.num_medians_)
        num_scaled = (num_data - self.num_means_) / self.num_stds_
        cat_frames = []
        for column in self.cat_columns_:
            filled = x[column].fillna(f'Unknown_{column}')
            categories = self.cat_categories_[column]
            encoded = pd.DataFrame((filled.to_numpy()[:, None] == categories.to_numpy()).astype(int), columns=categories.astype(str), index=x.index)
            cat_frames.append(encoded)
        cat_data = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=x.index)
        return pd.concat([num_scaled, cat_data], axis=1)

    def fit_transform(self, data):
        self.fit(data)
        df = _ensure_dataframe(data)
        y = _clean_dataframe(df)['salary'].copy() if 'salary' in df.columns else pd.Series(index=df.index, dtype=float)
        return self.transform(df), y


def create_preprocessing_pipeline():
    return NBAPreprocessingPipeline()
