from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.streaming import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Research-grade NBA preprocessing pipeline runner')
    parser.add_argument('--input', required=True, help='Path to raw CSV dataset')
    parser.add_argument('--output-dir', default='artifacts', help='Directory for reports and benchmark artifacts')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--max-memory-mb', type=int, default=1024)
    parser.add_argument('--max-compute-units', type=float, default=1.0)
    parser.add_argument('--benchmark-runs', type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory_mb,
        max_compute_units=args.max_compute_units,
        benchmark_runs=args.benchmark_runs,
        output_dir=Path(args.output_dir),
    )
    runner = PipelineRunner(config)
    report = runner.run_all(args.input)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
