import json
import shutil
import unittest
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.streaming import PipelineRunner
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

        self.assertTrue((self.output_dir / 'benchmarks' / 'latency_vs_accuracy.png').exists())
        self.assertTrue((self.output_dir / 'benchmarks' / 'memory_vs_accuracy.png').exists())
        self.assertTrue((self.output_dir / 'benchmarks' / 'latency_memory_accuracy.png').exists())
        self.assertTrue((self.output_dir / 'reports' / 'streaming_chunks.jsonl').exists())

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


if __name__ == '__main__':
    unittest.main()
