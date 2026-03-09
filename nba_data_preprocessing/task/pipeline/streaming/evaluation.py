from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    X = X.fillna(0.0)
    if len(X) < 5:
        return {'rmse': 0.0, 'r2': 0.0, 'training_time_s': 0.0}
    split = int(len(X) * 0.8)
    x_train, x_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression()
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return {'rmse': float(np.sqrt(mean_squared_error(y_test, pred))), 'r2': float(r2_score(y_test, pred)), 'training_time_s': float(time.perf_counter() - t0)}
