from pipeline.streaming.statistics import run_repeated_benchmark


def test_run_repeated_benchmark() -> None:
    out = run_repeated_benchmark(lambda: {'latency_s': 1.0, 'throughput_rows_s': 10.0}, runs=3)
    assert out['latency']['mean'] == 1.0
    assert out['throughput']['mean'] == 10.0
