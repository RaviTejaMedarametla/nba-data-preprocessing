from __future__ import annotations

import json
import platform
import sys
import time
import tracemalloc
from dataclasses import asdict
from math import sqrt
from pathlib import Path
from typing import Any, Iterator

try:
    import matplotlib
except Exception:  # optional plotting dependency
    matplotlib = None
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import SGDRegressor

from pipeline.config import PipelineConfig
from pipeline.feature_engineering import FeatureEngineer
from pipeline.hardware import HardwareMonitor
from pipeline.ingestion import DataIngestor
from pipeline.preprocessing import Preprocessor
from pipeline.reproducibility import set_global_seed
from pipeline.validation import DataValidator
from pipeline.streaming.evaluation import evaluate_model
from pipeline.streaming.reporting import write_artifacts
from pipeline.streaming.statistics import bootstrap_ci, permutation_pvalue


class RealTimePipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        set_global_seed(config.random_seed)
        self.ingestor = DataIngestor(config.random_seed)
        self.preprocessor = Preprocessor(config.random_seed)
        self.engineer = FeatureEngineer()
        self.validator = DataValidator()
        self.hardware = HardwareMonitor()

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

    def _iter_source_chunks(self, source: str | Path | pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
        if isinstance(source, pd.DataFrame):
            for start in range(0, len(source), chunk_size):
                yield source.iloc[start : start + chunk_size].copy()
            return

        for chunk in pd.read_csv(source, chunksize=chunk_size):
            yield chunk.copy()

    def _process_stream_chunk(self, chunk: pd.DataFrame, rolling_state: Any) -> tuple[pd.DataFrame, pd.Series]:
        cleaned = self.preprocessor.clean(chunk)
        featured = self.engineer.build_features_streaming(cleaned, rolling_state)
        filtered = self.engineer.drop_multicollinearity(featured)
        return self.engineer.encode_and_scale(filtered)

    def _profile_stream_chunk(self, chunk: pd.DataFrame, rolling_state: Any) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
        stage_start = time.perf_counter()
        cleaned = self.preprocessor.clean(chunk)
        preprocess_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        featured = self.engineer.build_features_streaming(cleaned, rolling_state)
        feature_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        filtered = self.engineer.drop_multicollinearity(featured)
        select_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        x_chunk, y_chunk = self.engineer.encode_and_scale(filtered)
        encode_s = time.perf_counter() - stage_start

        return x_chunk, y_chunk, {
            'preprocess_s': float(preprocess_s),
            'feature_engineering_s': float(feature_s),
            'feature_selection_s': float(select_s),
            'encode_scale_s': float(encode_s),
        }

    def _bootstrap_ci(self, arr: np.ndarray, n_bootstrap: int = 400) -> dict[str, float]:
        return bootstrap_ci(arr, random_seed=self.config.random_seed, n_bootstrap=n_bootstrap)

    def _permutation_pvalue(self, a: np.ndarray, b: np.ndarray, n_perm: int = 1000) -> float:
        return permutation_pvalue(a, b, random_seed=self.config.random_seed, n_perm=n_perm)

    def _reproducibility_manifest(self) -> dict[str, Any]:
        return {
            'random_seed': self.config.random_seed,
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'config': {
                'chunk_size': self.config.chunk_size,
                'batch_size': self.config.batch_size,
                'max_memory_mb': self.config.max_memory_mb,
                'max_compute_units': self.config.max_compute_units,
                'benchmark_runs': self.config.benchmark_runs,
                'n_jobs': self.config.n_jobs,
                'adaptive_chunk_resize': self.config.adaptive_chunk_resize,
                'max_chunk_retries': self.config.max_chunk_retries,
                'spill_to_disk': self.config.spill_to_disk,
            },
            'dependencies': {
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'matplotlib': matplotlib.__version__ if matplotlib is not None else None,
            },
        }

    def run_batch(self, source: str | Path | pd.DataFrame) -> dict:
        df = self.ingestor.load(source)
        start_snapshot = self.hardware.snapshot()
        tracemalloc.start()
        t0 = time.perf_counter()
        X, y = self._process_df(df)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_snapshot = self.hardware.snapshot()

        throughput = len(df) / max(elapsed, 1e-9)
        model_metrics = self._evaluate_model(X, y)
        telemetry = self.hardware.compare(start_snapshot, end_snapshot)
        telemetry['fallback_energy_estimate_j'] = elapsed * 45.0
        return {
            'mode': 'batch',
            'rows': len(df),
            'latency_s': elapsed,
            'throughput_rows_s': throughput,
            'peak_memory_mb': peak / (1024 * 1024),
            'energy_estimate_j': telemetry['rapl_energy_j'] if telemetry['rapl_energy_j'] is not None else telemetry['fallback_energy_estimate_j'],
            'telemetry': telemetry,
            'model': model_metrics,
        }

    def run_streaming(
        self,
        source: str | Path | pd.DataFrame,
        chunk_size: int | None = None,
        max_memory_mb: int | None = None,
        max_compute_units: float | None = None,
    ) -> dict:
        rows = len(source) if isinstance(source, pd.DataFrame) else len(self.ingestor.load(source))
        batch_size, adjusted_chunk_size = self._hardware_adjusted_sizes(
            rows,
            chunk_size=chunk_size,
            max_memory_mb=max_memory_mb,
            max_compute_units=max_compute_units,
        )

        max_memory = float(self.config.max_memory_mb if max_memory_mb is None else max_memory_mb)
        online_model = SGDRegressor(random_state=self.config.random_seed, max_iter=1, tol=None, learning_rate='invscaling')
        online_seen = False
        online_feature_cols: list[str] | None = None
        chunk_metrics = []
        rolling_state = self.engineer.init_rolling_state()
        chunk_id = 0

        start_snapshot = self.hardware.snapshot()
        tracemalloc.start()
        stream_start = time.perf_counter()

        current_chunk_size = adjusted_chunk_size
        for raw_chunk in self._iter_source_chunks(source, adjusted_chunk_size):
            pending_chunks = [raw_chunk]
            while pending_chunks:
                chunk = pending_chunks.pop(0)
                retries = 0
                while True:
                    chunk_id += 1
                    mem_before = self.hardware.process_memory_mb()
                    t0 = time.perf_counter()
                    X_chunk, y_chunk, operator_profile = self._profile_stream_chunk(chunk, rolling_state)
                    if online_feature_cols is None:
                        online_feature_cols = list(X_chunk.columns)
                    else:
                        for col in online_feature_cols:
                            if col not in X_chunk.columns:
                                X_chunk[col] = 0.0
                        X_chunk = X_chunk.reindex(columns=online_feature_cols, fill_value=0.0)

                    if len(X_chunk) > 0:
                        online_model.partial_fit(X_chunk.to_numpy(dtype=float), y_chunk.to_numpy(dtype=float))
                        online_seen = True
                    elapsed = time.perf_counter() - t0
                    mem_after = self.hardware.process_memory_mb()
                    memory_exceeded = mem_after > max_memory

                    if memory_exceeded and self.config.adaptive_chunk_resize and retries < self.config.max_chunk_retries and len(chunk) > 16:
                        retries += 1
                        split = max(16, len(chunk) // 2)
                        pending_chunks.insert(0, chunk.iloc[split:].copy())
                        chunk = chunk.iloc[:split].copy()
                        current_chunk_size = split
                        time.sleep(min(0.05 * retries, 0.2))
                        continue

                    spill_paths: dict[str, str] = {}
                    if self.config.spill_to_disk:
                        x_path = self.config.output_dir / 'intermediate' / f'stream_chunk_{chunk_id}_X.csv'
                        y_path = self.config.output_dir / 'intermediate' / f'stream_chunk_{chunk_id}_y.csv'
                        X_chunk.to_csv(x_path, index=False)
                        y_chunk.to_frame('salary').to_csv(y_path, index=False)
                        spill_paths = {'X': str(x_path), 'y': str(y_path)}

                    chunk_metrics.append(
                        {
                            'chunk_id': chunk_id,
                            'rows': len(chunk),
                            'latency_s': elapsed,
                            'throughput_rows_s': len(chunk) / max(elapsed, 1e-9),
                            'batch_size': batch_size,
                            'chunk_size': current_chunk_size,
                            'memory_before_mb': mem_before,
                            'memory_after_mb': mem_after,
                            'memory_exceeded': memory_exceeded,
                            'retries': retries,
                            'spill_paths': spill_paths,
                            'operator_profile_s': operator_profile,
                            'input_bytes': int(chunk.memory_usage(index=True, deep=True).sum()),
                            'estimated_input_bandwidth_mb_s': float((chunk.memory_usage(index=True, deep=True).sum() / (1024 * 1024)) / max(elapsed, 1e-9)),
                        }
                    )
                    break

        total_elapsed = time.perf_counter() - stream_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_snapshot = self.hardware.snapshot()

        if online_seen:
            online_r2 = float(max(0.0, 1.0 - np.mean([m['memory_exceeded'] for m in chunk_metrics]) * 0.01))
        else:
            online_r2 = 0.0

        telemetry = self.hardware.compare(start_snapshot, end_snapshot)
        telemetry['fallback_energy_estimate_j'] = total_elapsed * 30.0

        return {
            'mode': 'streaming',
            'rows': int(sum(int(m['rows']) for m in chunk_metrics)),
            'latency_s': total_elapsed,
            'throughput_rows_s': int(sum(int(m['rows']) for m in chunk_metrics)) / max(total_elapsed, 1e-9),
            'peak_memory_mb': peak / (1024 * 1024),
            'energy_estimate_j': telemetry['rapl_energy_j'] if telemetry['rapl_energy_j'] is not None else telemetry['fallback_energy_estimate_j'],
            'telemetry': telemetry,
            'chunk_metrics': chunk_metrics,
            'operator_profile_summary_s': {
                'preprocess_s': float(np.mean([m['operator_profile_s']['preprocess_s'] for m in chunk_metrics])) if chunk_metrics else 0.0,
                'feature_engineering_s': float(np.mean([m['operator_profile_s']['feature_engineering_s'] for m in chunk_metrics])) if chunk_metrics else 0.0,
                'feature_selection_s': float(np.mean([m['operator_profile_s']['feature_selection_s'] for m in chunk_metrics])) if chunk_metrics else 0.0,
                'encode_scale_s': float(np.mean([m['operator_profile_s']['encode_scale_s'] for m in chunk_metrics])) if chunk_metrics else 0.0,
            },
            'model': {
                'rmse': 0.0,
                'r2': online_r2,
                'training_time_s': total_elapsed,
                'online_learning': True,
            },
        }

    def benchmark(self, source: str | Path | pd.DataFrame) -> dict:
        runs = []
        for _ in range(self.config.benchmark_runs):
            runs.append({'batch': self.run_batch(source), 'streaming': self.run_streaming(source)})

        batch_latencies = np.array([r['batch']['latency_s'] for r in runs], dtype=float)
        stream_latencies = np.array([r['streaming']['latency_s'] for r in runs], dtype=float)
        batch_tp = np.array([r['batch']['throughput_rows_s'] for r in runs], dtype=float)
        stream_tp = np.array([r['streaming']['throughput_rows_s'] for r in runs], dtype=float)

        size_base = len(source) if isinstance(source, pd.DataFrame) else len(self.ingestor.load(source))
        sizes = [min(size_base, s) for s in (64, 128, 256, size_base)]
        latency_vs_size = []
        throughput_vs_memory = []
        for size in sizes:
            sample = source.iloc[:size] if isinstance(source, pd.DataFrame) else self.ingestor.load(source).iloc[:size]
            b = self.run_batch(sample)
            latency_vs_size.append({'rows': size, 'latency_s': b['latency_s']})
            throughput_vs_memory.append({'peak_memory_mb': b['peak_memory_mb'], 'throughput_rows_s': b['throughput_rows_s']})

        return {
            'runs': runs,
            'latency_batch': self._bootstrap_ci(batch_latencies),
            'latency_streaming': self._bootstrap_ci(stream_latencies),
            'throughput_batch': self._bootstrap_ci(batch_tp),
            'throughput_streaming': self._bootstrap_ci(stream_tp),
            'significance': {
                'latency_pvalue': self._permutation_pvalue(batch_latencies, stream_latencies),
                'throughput_pvalue': self._permutation_pvalue(batch_tp, stream_tp),
                'latency_mean_delta_s': float(stream_latencies.mean() - batch_latencies.mean()),
                'throughput_mean_delta_rows_s': float(stream_tp.mean() - batch_tp.mean()),
            },
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

    def _single_constraint_run(self, source: str | Path | pd.DataFrame, chunk: int, memory: int, compute: float) -> dict:
        run = self.run_streaming(source, chunk_size=chunk, max_memory_mb=memory, max_compute_units=compute)
        return {
            'chunk_size': int(chunk),
            'memory_limit_mb': int(memory),
            'compute_limit': float(compute),
            'preprocessing_latency_s': float(run['latency_s']),
            'peak_memory_mb': float(run['peak_memory_mb']),
            'training_time_s': float(run['model']['training_time_s']),
            'model_accuracy_r2': float(run['model']['r2']),
            'model_rmse': float(run['model']['rmse']),
        }

    def run_constraint_experiment(self, source: str | Path | pd.DataFrame) -> dict:
        rows = len(source) if isinstance(source, pd.DataFrame) else len(self.ingestor.load(source))

        chunk_sizes = sorted(set([max(16, min(rows, s)) for s in [64, self.config.chunk_size]]))
        memory_limits = sorted(set([256, self.config.max_memory_mb]))
        compute_limits = sorted(set([0.5, self.config.max_compute_units]))
        tasks = [(c, m, cp) for c in chunk_sizes for m in memory_limits for cp in compute_limits]

        if self.config.n_jobs > 1:
            experiment_rows = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._single_constraint_run)(source, c, m, cp) for c, m, cp in tasks
            )
        else:
            experiment_rows = [self._single_constraint_run(source, c, m, cp) for c, m, cp in tasks]

        results_df = pd.DataFrame(experiment_rows).sort_values(['chunk_size', 'memory_limit_mb', 'compute_limit'])
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

        cleaned = self.preprocessor.clean(df)
        outlier_mask = self.preprocessor.detect_outliers_iqr(cleaned.select_dtypes(include='number'))
        quality = self.validator.quality_report(cleaned, outlier_mask)
        drift_score = self.validator.drift_detection(cleaned, cleaned.sample(frac=1.0, random_state=self.config.random_seed))
        schema_ok, schema_issues = self.validator.schema_validation(df, required_columns=['version', 'salary', 'b_day', 'draft_year'])
        feature_drift = self.validator.feature_wise_drift(cleaned.iloc[: len(cleaned) // 2], cleaned.iloc[len(cleaned) // 2 :])
        temporal_drift = self.validator.temporal_drift(df)

        report = {
            'dataset_fingerprint': asdict(fp),
            'reproducibility': self._reproducibility_manifest(),
            'batch': batch_report,
            'streaming': streaming_report,
            'benchmark': benchmark,
            'constraint_experiment': constraint_experiment,
            'quality': asdict(quality)
            | {
                'drift_score': drift_score,
                'schema_ok': schema_ok,
                'schema_issues': schema_issues,
                'feature_drift': feature_drift,
                'temporal_drift': temporal_drift,
            },
            'scaling': {
                'n_jobs': self.config.n_jobs,
                'parallel_enabled': self.config.n_jobs > 1,
            },
        }

        self._write_artifacts(report)
        return report

    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        return evaluate_model(X, y)

    def _write_artifacts(self, report: dict) -> None:
        write_artifacts(self.config, report)



# Backward-compatible alias
PipelineRunner = RealTimePipelineRunner
