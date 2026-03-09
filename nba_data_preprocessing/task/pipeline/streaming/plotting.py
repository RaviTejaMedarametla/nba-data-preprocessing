from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_experiment_results(experiment_df: pd.DataFrame, benchmark_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError('Plotting dependencies are not installed. Install requirements-plot.txt to enable chart artifacts.') from exc

    plt.figure(figsize=(8, 5))
    plt.scatter(experiment_df['preprocessing_latency_s'], experiment_df['model_accuracy_r2'], c=experiment_df['compute_limit'], cmap='viridis')
    plt.colorbar(label='Compute constraint')
    plt.xlabel('Preprocessing latency (s)')
    plt.ylabel('Model accuracy (R²)')
    plt.title('Latency vs Accuracy')
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'latency_vs_accuracy.png', dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(experiment_df['peak_memory_mb'], experiment_df['model_accuracy_r2'], c=experiment_df['memory_limit_mb'], cmap='plasma')
    plt.colorbar(label='Memory limit (MB)')
    plt.xlabel('Peak memory (MB)')
    plt.ylabel('Model accuracy (R²)')
    plt.title('Memory vs Accuracy')
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'memory_vs_accuracy.png', dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(experiment_df['peak_memory_mb'], experiment_df['preprocessing_latency_s'], c=experiment_df['model_accuracy_r2'], cmap='coolwarm')
    plt.colorbar(label='Model accuracy (R²)')
    plt.xlabel('Peak memory (MB)')
    plt.ylabel('Latency (s)')
    plt.title('Latency vs Memory vs Accuracy')
    plt.tight_layout()
    plt.savefig(benchmark_dir / 'latency_memory_accuracy.png', dpi=160)
    plt.close()
