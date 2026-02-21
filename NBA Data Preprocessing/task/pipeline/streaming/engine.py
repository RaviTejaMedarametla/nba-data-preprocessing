from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict
from math import sqrt
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from pipeline.config import PipelineConfig
from pipeline.feature_engineering import FeatureEngineer
from pipeline.ingestion import DataIngestor
from pipeline.preprocessing import Preprocessor
from pipeline.validation import DataValidator

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ingestor = DataIngestor(config.random_seed)
        self.preprocessor = Preprocessor(config.random_seed)
        self.engineer = FeatureEngineer()
        self.validator = DataValidator()

    def _hardware_adjusted_sizes(
        self,
        rows: int,
        chunk_size: int | None = None,
        batch_size: int | None = None,
        max_memory_mb: int | None = None,
        max_compute_units: float | None = None,
    ) -> tuple[int, int]:
        chunk_base = self.config.chunk_size if chunk_size is None else chunk_size
        batch_base = self.config.batch_size if batch_size is None else batch_size
        memory_cap = self.config.max_memory_mb if max_memory_mb is None else max_memory_mb
        compute_cap = self.config.max_compute_units if max_compute_units is None else max_compute_units

        memory_factor = max(0.1, min(1.0, memory_cap / 1024))
        compute_factor = max(0.1, min(1.0, compute_cap))
        scale = memory_factor * compute_factor
        adjusted_batch = max(16, int(batch_base * scale))
        adjusted_chunk = max(16, int(chunk_base * scale))
        return min(adjusted_batch, rows), min(adjusted_chunk, rows)

    def _process_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        cleaned = self.preprocessor.clean(df)
        featured = self.engineer.build_features(cleaned)
        filtered = self.engineer.drop_multicollinearity(featured)
        X, y = self.engineer.encode_and_scale(filtered)
        return X, y

    def run_batch(self, source: str | Path | pd.DataFrame) -> dict:
        df = self.ingestor.load(source)
        tracemalloc.start()
        t0 = time.perf_counter()
        X, y = self._process_df(df)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        throughput = len(df) / max(elapsed, 1e-9)
        model_metrics = self._evaluate_model(X, y)
        return {
            'mode': 'batch',
            'rows': len(df),
            'latency_s': elapsed,
            'throughput_rows_s': throughput,
            'peak_memory_mb': peak / (1024 * 1024),
            'energy_estimate_j': elapsed * 45.0,
            'model': model_metrics,
        }

    def run_streaming(
        self,
        source: str | Path | pd.DataFrame,
        chunk_size: int | None = None,
        max_memory_mb: int | None = None,
        max_compute_units: float | None = None,
    ) -> dict:
        full_df = self.ingestor.load(source)
        batch_size, adjusted_chunk_size = self._hardware_adjusted_sizes(
            len(full_df),
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb,
            max_compute_units=max_compute_units,
        )

        all_X, all_y = [], []
        chunk_metrics = []
        tracemalloc.start()
        stream_start = time.perf_counter()

        for i, chunk in enumerate(self.ingestor.stream_chunks(full_df, adjusted_chunk_size), start=1):
            t0 = time.perf_counter()
            X_chunk, y_chunk = self._process_df(chunk)
            elapsed = time.perf_counter() - t0
            all_X.append(X_chunk)
            all_y.append(y_chunk)
            chunk_metrics.append(
                {
                    'chunk_id': i,
                    'rows': len(chunk),
                    'latency_s': elapsed,
                    'throughput_rows_s': len(chunk) / max(elapsed, 1e-9),
                    'batch_size': batch_size,
                    'chunk_size': adjusted_chunk_size,
                }
            )

        total_elapsed = time.perf_counter() - stream_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        X = pd.concat(all_X, axis=0).fillna(0.0)
        y = pd.concat(all_y, axis=0)
        model_metrics = self._evaluate_model(X, y)

        return {
            'mode': 'streaming',
            'rows': len(full_df),
            'latency_s': total_elapsed,
            'throughput_rows_s': len(full_df) / max(total_elapsed, 1e-9),
            'peak_memory_mb': peak / (1024 * 1024),
            'energy_estimate_j': total_elapsed * 30.0,
            'chunk_metrics': chunk_metrics,
            'model': model_metrics,
        }

    def benchmark(self, source: str | Path | pd.DataFrame) -> dict:
        df = self.ingestor.load(source)
        runs = []
        for _ in range(self.config.benchmark_runs):
            runs.append({'batch': self.run_batch(df), 'streaming': self.run_streaming(df)})

        batch_latencies = np.array([r['batch']['latency_s'] for r in runs], dtype=float)
        stream_latencies = np.array([r['streaming']['latency_s'] for r in runs], dtype=float)
        batch_tp = np.array([r['batch']['throughput_rows_s'] for r in runs], dtype=float)
        stream_tp = np.array([r['streaming']['throughput_rows_s'] for r in runs], dtype=float)

        def summary(arr: np.ndarray) -> dict:
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            ci95 = float(1.96 * std / sqrt(len(arr)))
            return {'mean': mean, 'std': std, 'ci95': ci95}

        sizes = [min(len(df), s) for s in (64, 128, 256, len(df))]
        latency_vs_size = []
        throughput_vs_memory = []
        for size in sizes:
            sample = df.iloc[:size]
            b = self.run_batch(sample)
            latency_vs_size.append({'rows': size, 'latency_s': b['latency_s']})
            throughput_vs_memory.append({'peak_memory_mb': b['peak_memory_mb'], 'throughput_rows_s': b['throughput_rows_s']})

        return {
            'runs': runs,
            'latency_batch': summary(batch_latencies),
            'latency_streaming': summary(stream_latencies),
            'throughput_batch': summary(batch_tp),
            'throughput_streaming': summary(stream_tp),
            'latency_vs_data_size': latency_vs_size,
            'throughput_vs_memory': throughput_vs_memory,
            'resource_vs_accuracy': [
                {
                    'mode': r['batch']['mode'],
                    'peak_memory_mb': r['batch']['peak_memory_mb'],
                    'r2': r['batch']['model']['r2'],
                }
                for r in runs
            ]
            + [
                {
                    'mode': r['streaming']['mode'],
                    'peak_memory_mb': r['streaming']['peak_memory_mb'],
                    'r2': r['streaming']['model']['r2'],
                }
                for r in runs
            ],
        }

    def run_constraint_experiment(self, source: str | Path | pd.DataFrame) -> dict:
        df = self.ingestor.load(source)
        rows = len(df)

        chunk_sizes = sorted(set([max(16, min(rows, s)) for s in [64, self.config.chunk_size]]))
        memory_limits = sorted(set([256, self.config.max_memory_mb]))
        compute_limits = sorted(set([0.5, self.config.max_compute_units]))

        experiment_rows = []
        for chunk in chunk_sizes:
            for memory in memory_limits:
                for compute in compute_limits:
                    run = self.run_streaming(df, chunk_size=chunk, max_memory_mb=memory, max_compute_units=compute)
                    experiment_rows.append(
                        {
                            'chunk_size': int(chunk),
                            'memory_limit_mb': int(memory),
                            'compute_limit': float(compute),
                            'preprocessing_latency_s': float(run['latency_s']),
                            'peak_memory_mb': float(run['peak_memory_mb']),
                            'training_time_s': float(run['model']['training_time_s']),
                            'model_accuracy_r2': float(run['model']['r2']),
                            'model_rmse': float(run['model']['rmse']),
                        }
                    )

        results_df = pd.DataFrame(experiment_rows).sort_values(
            ['chunk_size', 'memory_limit_mb', 'compute_limit']
        )

        return {
            'records': results_df.to_dict(orient='records'),
            'summary': {
                'best_accuracy_r2': float(results_df['model_accuracy_r2'].max()),
                'lowest_latency_s': float(results_df['preprocessing_latency_s'].min()),
                'lowest_training_time_s': float(results_df['training_time_s'].min()),
                'max_peak_memory_mb': float(results_df['peak_memory_mb'].max()),
            },
        }

    def run_all(self, source: str | Path | pd.DataFrame) -> dict:
        self.config.ensure_output_dirs()
        df = self.ingestor.load(source)
        fp = self.ingestor.fingerprint(df)

        batch_report = self.run_batch(df)
        streaming_report = self.run_streaming(df)
        benchmark = self.benchmark(df)
        constraint_experiment = self.run_constraint_experiment(df)

        outlier_mask = self.preprocessor.detect_outliers_iqr(df.select_dtypes(include='number'))
        quality = self.validator.quality_report(df, outlier_mask)
        drift_score = self.validator.drift_detection(df, df.sample(frac=1.0, random_state=self.config.random_seed))

        report = {
            'dataset_fingerprint': asdict(fp),
            'batch': batch_report,
            'streaming': streaming_report,
            'benchmark': benchmark,
            'constraint_experiment': constraint_experiment,
            'quality': asdict(quality) | {'drift_score': drift_score},
        }

        self._write_artifacts(report)
        return report

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X = X.fillna(0.0)
        if len(X) < 5:
            return {'rmse': 0.0, 'r2': 0.0, 'training_time_s': 0.0}
        split = int(len(X) * 0.8)
        x_train, x_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = LinearRegression()
        train_start = time.perf_counter()
        model.fit(x_train, y_train)
        training_time = time.perf_counter() - train_start
        pred = model.predict(x_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        return {'rmse': rmse, 'r2': r2, 'training_time_s': float(training_time)}

    def _write_artifacts(self, report: dict) -> None:
        out = self.config.output_dir
        with (out / 'reports' / 'pipeline_report.json').open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        pd.DataFrame(report['streaming']['chunk_metrics']).to_csv(out / 'benchmarks' / 'streaming_chunks.csv', index=False)
        pd.DataFrame(report['benchmark']['latency_vs_data_size']).to_csv(out / 'benchmarks' / 'latency_vs_data_size.csv', index=False)
        pd.DataFrame(report['benchmark']['throughput_vs_memory']).to_csv(out / 'benchmarks' / 'throughput_vs_memory.csv', index=False)
        pd.DataFrame(report['benchmark']['resource_vs_accuracy']).to_csv(out / 'benchmarks' / 'resource_vs_accuracy.csv', index=False)

        experiment_df = pd.DataFrame(report['constraint_experiment']['records'])
        experiment_df.to_csv(out / 'benchmarks' / 'constraint_experiment.csv', index=False)

        with (out / 'reports' / 'constraint_experiment_log.jsonl').open('w', encoding='utf-8') as f:
            for row in report['constraint_experiment']['records']:
                f.write(json.dumps(row) + '\n')

        self._plot_experiment_results(experiment_df, out / 'benchmarks')

    def _plot_experiment_results(self, experiment_df: pd.DataFrame, benchmark_dir: Path) -> None:
        plt.figure(figsize=(8, 5))
        plt.scatter(
            experiment_df['preprocessing_latency_s'],
            experiment_df['model_accuracy_r2'],
            c=experiment_df['compute_limit'],
            cmap='viridis',
        )
        plt.colorbar(label='Compute constraint')
        plt.xlabel('Preprocessing latency (s)')
        plt.ylabel('Model accuracy (R²)')
        plt.title('Latency vs Accuracy')
        plt.tight_layout()
        plt.savefig(benchmark_dir / 'latency_vs_accuracy.png', dpi=160)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(
            experiment_df['peak_memory_mb'],
            experiment_df['model_accuracy_r2'],
            c=experiment_df['memory_limit_mb'],
            cmap='plasma',
        )
        plt.colorbar(label='Memory limit (MB)')
        plt.xlabel('Peak memory (MB)')
        plt.ylabel('Model accuracy (R²)')
        plt.title('Memory vs Accuracy')
        plt.tight_layout()
        plt.savefig(benchmark_dir / 'memory_vs_accuracy.png', dpi=160)
        plt.close()
