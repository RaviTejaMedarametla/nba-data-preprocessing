import pandas as pd

from pipeline.feature_engineering.features import FeatureEngineer
from pipeline.preprocessing.cleaners import clean_dataframe


def test_build_features_columns() -> None:
    df = pd.read_csv('nba_data_preprocessing/data/nba2k-full.csv').head(10)
    cleaned = clean_dataframe(df)
    featured = FeatureEngineer().build_features(cleaned)
    assert 'age' in featured.columns
    assert 'salary_roll_mean' in featured.columns
