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

    def test_legacy_contract_still_works(self):
        X, y = transform_data(multicol_data(feature_data(clean_data(self.data_path))))
        self.assertEqual(X.shape, (439, 46))
        self.assertEqual(y.shape, (439,))

    def test_modular_runner_generates_reports(self):
        config = PipelineConfig(output_dir=self.output_dir, benchmark_runs=2, chunk_size=100, batch_size=150)
        runner = PipelineRunner(config)
        report = runner.run_all(self.data_path)

        self.assertIn('batch', report)
        self.assertIn('streaming', report)
        self.assertIn('constraint_experiment', report)
        self.assertGreater(report['batch']['throughput_rows_s'], 0)
        self.assertGreaterEqual(report['quality']['drift_score'], 0)

        first_record = report['constraint_experiment']['records'][0]
        self.assertIn('model_accuracy_r2', first_record)
        self.assertIn('training_time_s', first_record)
        self.assertIn('preprocessing_latency_s', first_record)

        self.assertTrue((self.output_dir / 'benchmarks' / 'latency_vs_accuracy.png').exists())
        self.assertTrue((self.output_dir / 'benchmarks' / 'memory_vs_accuracy.png').exists())


if __name__ == '__main__':
    unittest.main()
