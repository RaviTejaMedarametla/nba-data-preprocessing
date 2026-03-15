from pipeline.streaming.evaluation import evaluate_detection_latency


def test_evaluate_detection_latency_basic() -> None:
    result = evaluate_detection_latency([1.2, 2.5, 3.5], [1.0, 2.0, 3.0])
    assert result['count'] == 3.0
    assert result['mean_latency_s'] >= 0.0
