from __future__ import annotations

import json

import pandas as pd

from .plotting import plot_experiment_results


def write_artifacts(config, report: dict) -> None:
    out = config.output_dir
    with (out / 'reports' / 'pipeline_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    with (out / 'metadata' / 'run_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(report['reproducibility'], f, indent=2)
    pd.DataFrame(report['streaming']['chunk_metrics']).to_csv(out / 'benchmarks' / 'streaming_chunks.csv', index=False)
    pd.DataFrame([{'chunk_id': row['chunk_id'], **row['operator_profile_s'], 'estimated_input_bandwidth_mb_s': row['estimated_input_bandwidth_mb_s'], 'input_bytes': row['input_bytes']} for row in report['streaming']['chunk_metrics']]).to_csv(out / 'profiles' / 'operator_profile.csv', index=False)
    with (out / 'reports' / 'streaming_chunks.jsonl').open('w', encoding='utf-8') as f:
        for row in report['streaming']['chunk_metrics']:
            f.write(json.dumps(row) + '\n')
    pd.DataFrame(report['benchmark']['latency_vs_data_size']).to_csv(out / 'benchmarks' / 'latency_vs_data_size.csv', index=False)
    pd.DataFrame(report['benchmark']['throughput_vs_memory']).to_csv(out / 'benchmarks' / 'throughput_vs_memory.csv', index=False)
    pd.DataFrame(report['benchmark']['resource_vs_accuracy']).to_csv(out / 'benchmarks' / 'resource_vs_accuracy.csv', index=False)
    pd.DataFrame([report['benchmark']['significance']]).to_csv(out / 'benchmarks' / 'significance_tests.csv', index=False)
    experiment_df = pd.DataFrame(report['constraint_experiment']['records'])
    experiment_df.to_csv(out / 'benchmarks' / 'constraint_experiment.csv', index=False)
    with (out / 'reports' / 'constraint_experiment_log.jsonl').open('w', encoding='utf-8') as f:
        for row in report['constraint_experiment']['records']:
            f.write(json.dumps(row) + '\n')
    try:
        plot_experiment_results(experiment_df, out / 'benchmarks')
    except RuntimeError as exc:
        with (out / 'reports' / 'plotting_warning.txt').open('w', encoding='utf-8') as f:
            f.write(str(exc) + '\n')
