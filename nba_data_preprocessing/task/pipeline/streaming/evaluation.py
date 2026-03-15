from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Train a lightweight linear model and report accuracy/latency metrics."""
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
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
        'r2': float(r2_score(y_test, pred)),
        'training_time_s': float(time.perf_counter() - t0),
    }


def evaluate_detection_latency(detection_timestamps: Iterable[float], event_timestamps: Iterable[float]) -> dict[str, float]:
    """Compute anomaly detection latency metrics.

    Each event is paired with the first detection that occurs at or after the event.
    """
    detections = sorted(float(v) for v in detection_timestamps)
    events = sorted(float(v) for v in event_timestamps)
    if not detections or not events:
        return {'count': 0.0, 'mean_latency_s': 0.0, 'p95_latency_s': 0.0, 'max_latency_s': 0.0}

    latencies: list[float] = []
    det_idx = 0
    for event in events:
        while det_idx < len(detections) and detections[det_idx] < event:
            det_idx += 1
        if det_idx < len(detections):
            latencies.append(max(0.0, detections[det_idx] - event))

    if not latencies:
        return {'count': 0.0, 'mean_latency_s': 0.0, 'p95_latency_s': 0.0, 'max_latency_s': 0.0}

    arr = np.asarray(latencies, dtype=float)
    return {
        'count': float(arr.size),
        'mean_latency_s': float(arr.mean()),
        'p95_latency_s': float(np.percentile(arr, 95)),
        'max_latency_s': float(arr.max()),
    }
