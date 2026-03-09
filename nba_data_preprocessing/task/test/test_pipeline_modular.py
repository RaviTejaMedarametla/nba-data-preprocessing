import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from pipeline.config import PipelineConfig
from pipeline.hardware.monitor import HardwareMonitor, TelemetrySnapshot
from pipeline.streaming import PipelineRunner
from pipeline.validation import DataValidator
from preprocess import clean_data, feature_data, multicol_data, transform_data


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = Path(__file__).resolve().parents[2] / 'data' / 'nba2k-full.csv'
        cls.output_dir = Path('artifacts_test')
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def test_legacy_contract_still_works(self):
        X, y = transform_data(multicol_data(feature_data(clean_data(self.data_path))))
        self.assertEqual(X.shape, (439, 46))
        self.assertEqual(y.shape, (439,))

    def test_modular_runner_generates_reports(self):
        config = PipelineConfig(output_dir=self.output_dir, benchmark_runs=2, chunk_size=100, batch_size=150, spill_to_disk=True)
        runner = PipelineRunner(config)
        report = runner.run_all(self.data_path)

        self.assertIn('batch', report)
        self.assertIn('streaming', report)
        self.assertIn('constraint_experiment', report)
        self.assertGreater(report['batch']['throughput_rows_s'], 0)
        self.assertGreaterEqual(report['quality']['drift_score'], 0)
        self.assertIn('schema_ok', report['quality'])

        first_record = report['constraint_experiment']['records'][0]
        self.assertIn('model_accuracy_r2', first_record)
        self.assertIn('training_time_s', first_record)
        self.assertIn('preprocessing_latency_s', first_record)

        benchmark_dir = self.output_dir / 'benchmarks'
        reports_dir = self.output_dir / 'reports'
        plot_paths = [
            benchmark_dir / 'latency_vs_accuracy.png',
            benchmark_dir / 'memory_vs_accuracy.png',
            benchmark_dir / 'latency_memory_accuracy.png',
        ]
        if all(path.exists() for path in plot_paths):
            for path in plot_paths:
                self.assertTrue(path.exists())
        else:
            self.assertTrue((reports_dir / 'plotting_warning.txt').exists())

        self.assertTrue((reports_dir / 'streaming_chunks.jsonl').exists())

    def test_deterministic_pipeline_report(self):
        dir_a = Path('artifacts_deterministic_a')
        dir_b = Path('artifacts_deterministic_b')
        for d in (dir_a, dir_b):
            if d.exists():
                shutil.rmtree(d)

        cfg_a = PipelineConfig(output_dir=dir_a, benchmark_runs=1, random_seed=42)
        cfg_b = PipelineConfig(output_dir=dir_b, benchmark_runs=1, random_seed=42)
        rep_a = PipelineRunner(cfg_a).run_all(self.data_path)
        rep_b = PipelineRunner(cfg_b).run_all(self.data_path)

        self.assertEqual(rep_a['dataset_fingerprint']['sha256'], rep_b['dataset_fingerprint']['sha256'])
        self.assertEqual(rep_a['scaling']['parallel_enabled'], rep_b['scaling']['parallel_enabled'])
        self.assertEqual(rep_a['quality']['schema_ok'], rep_b['quality']['schema_ok'])

        json_a = json.loads((dir_a / 'reports' / 'pipeline_report.json').read_text(encoding='utf-8'))
        json_b = json.loads((dir_b / 'reports' / 'pipeline_report.json').read_text(encoding='utf-8'))
        self.assertEqual(json_a['dataset_fingerprint'], json_b['dataset_fingerprint'])

    def test_schema_validation_negative_paths(self):
        validator = DataValidator()
        malformed = pd.DataFrame({'salary': ['abc', 'oops'], 'b_day': ['01/01/90', '02/02/90']})
        ok, issues = validator.schema_validation(malformed, required_columns=['version', 'salary', 'b_day', 'draft_year'])
        self.assertFalse(ok)
        self.assertTrue(any('missing_columns' in issue for issue in issues))
        self.assertIn('salary_column_not_numeric', issues)

    def test_extreme_missingness_is_handled(self):
        missing_df = pd.DataFrame(
            {
                'version': ['NBA2K20', 'NBA2K20'],
                'b_day': ['01/01/90', '02/02/90'],
                'draft_year': ['2010', '2011'],
                'team': [None, None],
                'height': ['6-0 / 1.83', None],
                'weight': ['200 lbs. / 90 kg.', None],
                'salary': ['$1000000', None],
                'country': ['USA', None],
                'draft_round': [None, 'Undrafted'],
            }
        )
        cleaned = clean_data(missing_df)
        self.assertLess(cleaned.isna().sum().sum(), len(cleaned.columns))

    def test_invalid_config_values_raise(self):
        invalid_cases = [
            {'chunk_size': 0},
            {'chunk_size': -1},
            {'batch_size': 0},
            {'batch_size': -1},
            {'max_memory_mb': 0},
            {'max_memory_mb': -1},
            {'max_compute_units': 0},
            {'max_compute_units': -0.1},
            {'benchmark_runs': 0},
            {'benchmark_runs': -1},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PipelineConfig(**kwargs)

    def test_unknown_config_keys_raise_type_error(self):
        with self.assertRaises(TypeError):
            PipelineConfig(**{'unknown_option': 123})

    def test_hardware_monitor_fallback_without_psutil(self):
        monitor = HardwareMonitor()
        monitor._psutil = None
        snap = monitor.snapshot()
        self.assertEqual(snap.cpu_percent, 0.0)
        telemetry = monitor.compare(
            TelemetrySnapshot(cpu_percent=0.0, process_memory_mb=0.0, system_memory_percent=0.0, energy_uj=None),
            snap,
        )
        self.assertIn('rapl_energy_j', telemetry)

    def test_streaming_empty_or_corrupt_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_csv = Path(tmpdir) / 'empty_input.csv'
            empty_csv.write_text('', encoding='utf-8')
            runner = PipelineRunner(PipelineConfig(output_dir=Path(tmpdir) / 'artifacts_empty', benchmark_runs=1))
            with self.assertRaises(EmptyDataError):
                runner.run_streaming(empty_csv)


if __name__ == '__main__':
    unittest.main()
