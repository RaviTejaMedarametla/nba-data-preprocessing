from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_linear_regression(x_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    x_matrix = np.column_stack([np.ones(len(x_train)), x_train.to_numpy(dtype=float)])
    y_vector = y_train.to_numpy(dtype=float)
    return np.linalg.pinv(x_matrix) @ y_vector


def _predict_linear_regression(x_data: pd.DataFrame, coefficients: np.ndarray) -> np.ndarray:
    x_matrix = np.column_stack([np.ones(len(x_data)), x_data.to_numpy(dtype=float)])
    return x_matrix @ coefficients


def train_linear_regression_model(X: pd.DataFrame, y: pd.Series) -> dict:
    coefficients = _fit_linear_regression(X, y)
    return {'intercept': float(coefficients[0]), 'coefficients': pd.Series(coefficients[1:], index=X.columns, dtype=float)}


def evaluate_linear_regression_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42) -> dict:
    if n_splits < 2:
        raise ValueError('n_splits must be at least 2')
    if len(X) < n_splits:
        raise ValueError('n_splits cannot exceed number of samples')
    indices = np.arange(len(X))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, n_splits)
    fold_metrics = []
    for fold_id in range(n_splits):
        test_idx = fold_indices[fold_id]
        train_idx = np.concatenate([fold_indices[i] for i in range(n_splits) if i != fold_id])
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        coefficients = _fit_linear_regression(x_train, y_train)
        predictions = _predict_linear_regression(x_test, coefficients)
        residuals = y_test.to_numpy(dtype=float) - predictions
        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(residuals)))
        y_true = y_test.to_numpy(dtype=float)
        denominator = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residuals**2)) / denominator if denominator != 0 else 0.0
        fold_metrics.append({'fold': fold_id + 1, 'rmse': rmse, 'mae': mae, 'r2': r2})
    metrics_df = pd.DataFrame(fold_metrics)
    return {
        'rmse_mean': float(metrics_df['rmse'].mean()),
        'rmse_std': float(metrics_df['rmse'].std(ddof=0)),
        'mae_mean': float(metrics_df['mae'].mean()),
        'mae_std': float(metrics_df['mae'].std(ddof=0)),
        'r2_mean': float(metrics_df['r2'].mean()),
        'r2_std': float(metrics_df['r2'].std(ddof=0)),
        'fold_metrics': metrics_df,
    }
