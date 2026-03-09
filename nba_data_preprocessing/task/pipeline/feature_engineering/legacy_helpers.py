from __future__ import annotations

import numpy as np
import pandas as pd


def feature_data(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    year = featured['version'].str.extract(r'(\d+)$')[0].astype(int)
    year = np.where(year < 100, year + 2000, year)
    version_year = pd.to_datetime(year.astype(str), format='%Y').year
    featured['age'] = version_year - featured['b_day'].dt.year
    featured['experience'] = version_year - featured['draft_year'].dt.year
    featured['bmi'] = featured['weight'] / (featured['height'] ** 2)
    featured = featured.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])
    for column in list(featured.select_dtypes(include='object').columns):
        if featured[column].nunique(dropna=False) >= 50:
            featured = featured.drop(columns=column)
    return featured


def multicol_data(df: pd.DataFrame) -> pd.DataFrame:
    transformed = df.copy()
    threshold = 0.5
    numeric_columns = [column for column in transformed.select_dtypes(include='number').columns if column != 'salary']
    while len(numeric_columns) > 1:
        corr_matrix = transformed[numeric_columns].corr().fillna(0.0)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        candidate_pairs = upper_triangle.stack()
        if candidate_pairs.empty:
            break
        max_corr = candidate_pairs.abs().max()
        if max_corr <= threshold:
            break
        first_column, second_column = candidate_pairs.abs().idxmax()
        target_corr = transformed[numeric_columns].corrwith(transformed['salary']).abs().fillna(0.0)
        first_target_corr = target_corr[first_column]
        second_target_corr = target_corr[second_column]
        drop_column = first_column if first_target_corr < second_target_corr else second_column
        transformed = transformed.drop(columns=drop_column)
        numeric_columns.remove(drop_column)
    return transformed


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df['salary'].copy()
    x = df.drop(columns='salary')
    num_columns = list(x.select_dtypes(include='number').columns)
    num_feat_df = x[num_columns].astype(float)
    num_feat_df = num_feat_df.fillna(num_feat_df.median())
    means = num_feat_df.mean()
    stds = num_feat_df.std(ddof=0).replace(0.0, 1.0)
    num_scaled = (num_feat_df - means) / stds
    cat_columns = list(x.select_dtypes(include='object').columns)
    cat_frames = []
    for column in cat_columns:
        filled_column = x[column].fillna(f'Unknown_{column}')
        categories = pd.Index(pd.unique(filled_column))
        encoded = pd.DataFrame((filled_column.to_numpy()[:, None] == categories.to_numpy()).astype(int), columns=categories.astype(str), index=x.index)
        cat_frames.append(encoded)
    cat_feat_df = pd.concat(cat_frames, axis=1) if cat_frames else pd.DataFrame(index=x.index)
    X = pd.concat([num_scaled, cat_feat_df], axis=1)
    return X, y
