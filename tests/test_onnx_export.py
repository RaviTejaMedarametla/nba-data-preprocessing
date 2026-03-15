import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

skl2onnx = pytest.importorskip('skl2onnx')


def test_onnx_export_smoke() -> None:
    X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    model = LogisticRegression(random_state=42).fit(X, y)
    onx = skl2onnx.to_onnx(model, X[:1])
    assert onx is not None
