from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_dataframe(data: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.read_csv(data)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned['b_day'] = pd.to_datetime(cleaned['b_day'], format='%m/%d/%y')
    cleaned['draft_year'] = pd.to_datetime(cleaned['draft_year'], format='%Y')
    cleaned['team'] = cleaned['team'].fillna('No Team')
    cleaned['height'] = cleaned['height'].str.split(' / ').str[1].astype(float)
    cleaned['weight'] = cleaned['weight'].str.split(' / ').str[1].str.replace(' kg.', '', regex=False).astype(float)
    cleaned['salary'] = cleaned['salary'].str.replace('$', '', regex=False).astype(float)
    cleaned['country'] = np.where(cleaned['country'] == 'USA', 'USA', 'Not-USA')
    cleaned['draft_round'] = cleaned['draft_round'].replace('Undrafted', '0')
    return cleaned
