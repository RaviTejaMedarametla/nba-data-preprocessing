import pandas as pd

from pipeline.feature_engineering.features import FeatureEngineer
from pipeline.models.linear_helpers import evaluate_linear_regression_cv, train_linear_regression_model
from pipeline.preprocessing.cleaners import clean_dataframe


def test_train_and_eval_linear_helpers() -> None:
    df = pd.read_csv('nba_data_preprocessing/data/nba2k-full.csv').head(50)
    cleaned = clean_dataframe(df)
    X, y = FeatureEngineer().encode_and_scale(FeatureEngineer().drop_multicollinearity(FeatureEngineer().build_features(cleaned)))
    model = train_linear_regression_model(X, y)
    metrics = evaluate_linear_regression_cv(X, y, n_splits=3)
    assert 'coefficients' in model
    assert metrics['rmse_mean'] >= 0.0
