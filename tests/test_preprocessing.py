import pandas as pd

from pipeline.preprocessing.cleaners import clean_dataframe


def test_cleaning_numeric_and_country() -> None:
    df = pd.read_csv('nba_data_preprocessing/data/nba2k-full.csv').head(5)
    cleaned = clean_dataframe(df)
    assert cleaned['salary'].dtype.kind in {'f', 'i'}
    assert set(cleaned['country'].unique()).issubset({'USA', 'Not-USA'})
