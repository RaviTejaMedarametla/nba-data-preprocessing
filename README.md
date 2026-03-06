# NBA Data Preprocessing

## Overview
This repository implements a deterministic NBA data preprocessing and feature-engineering pipeline for salary modeling workflows. It supports both full-dataset batch execution and constrained streaming execution while preserving reproducibility and consistent experiment outputs.

This project is part of a broader AI systems engineering portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.

## System Architecture
The runtime is organized as a staged pipeline:

1. **Ingestion**: load CSV data and compute dataset fingerprints.
2. **Preprocessing**: normalize fields, handle missing values, and apply outlier logic.
3. **Feature Engineering**: derive temporal and rolling features and remove high-correlation inputs.
4. **Validation**: run schema and drift checks.
5. **Evaluation and Reporting**: train/evaluate models and export reports and benchmarks.

Architecture diagram source: `docs/architecture.mmd`.

## Features
- Deterministic pipeline execution with configurable random seeds.
- Batch and streaming processing modes with shared schema assumptions.
- Constraint-aware execution controls for memory and compute budgets.
- Reproducibility artifacts including reports, benchmark CSV files, and operator profiles.
- Backward-compatible CLI behavior through `run_pipeline.py` and `preprocess.py`.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run with a configuration template:

```bash
cd "NBA Data Preprocessing/task"
python run_pipeline.py \
  --input ../data/nba2k-full.csv \
  --config-template ../../configs/pipeline.edge.template.json
```

CLI arguments can be used to override template values.

## Reproducibility
- Set `random_seed` explicitly in configuration or CLI.
- Keep `benchmark_runs` constant for comparable results.
- Use separate `output_dir` locations for independent runs.
- Validate changes with the existing test suite:

```bash
cd "NBA Data Preprocessing/task"
python -m unittest discover -s test -p 'test_*.py'
```

## Related Projects
This repository is part of a larger AI systems engineering portfolio:

- `neural-network-systems`
- `digit-classification-benchmark`
- `edge-ai-model-optimization`
- `hospital-analytics-pipeline`
- `nba-data-engineering`
- `ai-systems-ml-platform`

> Naming recommendation: for portfolio consistency, consider renaming this repository from `nba-data-preprocessing` to `nba-data-engineering` manually in GitHub repository settings.
