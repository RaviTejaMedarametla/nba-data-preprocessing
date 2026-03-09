# nba_data_preprocessing

Deterministic data preprocessing and feature-engineering pipeline for NBA salary modeling experiments under reproducible and resource-constrained execution settings.

## Overview
This repository implements a reproducible data engineering workflow for NBA player salary modeling. The system ingests tabular player data, applies standardized preprocessing transformations, constructs derived features, validates schema and data quality assumptions, and generates experiment artifacts for downstream modeling and benchmarking.

The project addresses a common challenge in machine learning systems: obtaining consistent, comparable results across runs and across deployment constraints. In practical settings, preprocessing behavior can drift when input distributions shift or when hardware limits require changes in execution mode. This repository emphasizes deterministic processing, configuration-driven experimentation, and constraint-aware execution so that model development and evaluation remain traceable and reproducible.

## Project Motivation
The primary motivation is to support rigorous experimentation in applied machine learning pipelines where data preparation quality directly affects model performance. The repository focuses on:

- deterministic experiment setup for repeatable preprocessing and evaluation,
- execution patterns that can be profiled under memory or compute constraints,
- reproducible artifact generation for benchmarking and auditability,
- engineering practices that align with hardware-aware ML development and production-oriented workflows.

## System Architecture
The pipeline is organized as modular stages:

- **Data Pipeline**  
  Loads NBA CSV datasets, computes dataset fingerprints, and establishes deterministic run context.

- **Preprocessing**  
  Applies field normalization, missing-value handling, and outlier-aware transformations to produce stable cleaned data.

- **Feature Engineering**  
  Builds derived and rolling statistics features and removes highly correlated inputs to improve downstream signal quality.

- **Hardware-Aware Evaluation**  
  Executes benchmark and profiling steps under configurable constraints, supporting both full-dataset and constrained execution modes.

- **Inference / Deployment Readiness**  
  Exports structured outputs, reports, and metadata that can be consumed by downstream training, evaluation, or production integration workflows.

Architecture diagram source: `docs/architecture.mmd`.

## Repository Structure
- `nba_data_preprocessing/task/`  
  Core pipeline entry points (`run_pipeline.py`, `preprocess.py`) and modular pipeline components.

- `nba_data_preprocessing/task/pipeline/`  
  Configuration, reproducibility, and pipeline-stage implementation modules.

- `nba_data_preprocessing/task/test/`  
  Unit tests for modular pipeline behavior and regression checks.

- `nba_data_preprocessing/data/`  
  Input NBA datasets (full and cleaned CSV variants) used by the workflow.

- `configs/`  
  Execution templates for different operating profiles (e.g., server and edge-oriented settings).

- `docs/`  
  Supporting technical documentation, including architecture and hardware profiling notes.

- `.github/workflows/`  
  Continuous integration workflows for automated repository checks.

## Features
- Deterministic preprocessing and experiment behavior via controlled seeds and run metadata.
- Configuration-driven execution with template-based operating profiles.
- Batch and constrained processing modes for resource-aware experimentation.
- Built-in reporting, benchmark outputs, and operator-level profiling artifacts.
- CLI-driven pipeline orchestration for repeatable and scriptable runs.
- Plot generation is optional; install `requirements-plot.txt` to enable benchmark image outputs (e.g., `latency_vs_accuracy.png`).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional plotting/report visuals (benchmark PNGs)
pip install -r requirements-plot.txt
```

## Usage
Run the end-to-end pipeline with a configuration template:

```bash
cd "nba_data_preprocessing/task"
python run_pipeline.py \
  --input ../data/nba2k-full.csv \
  --config-template ../../configs/pipeline.edge.template.json
```

Run preprocessing directly:

```bash
cd "nba_data_preprocessing/task"
python preprocess.py --input ../data/nba2k-full.csv
```

## Reproducibility
Experiments can be reproduced by combining fixed configuration templates with deterministic execution controls:

- define and preserve configuration values in `configs/*.template.json`,
- set deterministic seeds (for example via `random_seed`) and keep benchmark settings constant across runs,
- store each run in separate output directories to preserve experiment artifacts,
- validate consistency using the repository test suite.

Example validation command:

```bash
cd "nba_data_preprocessing/task"
python -m unittest discover -s test -p 'test_*.py'
```

## Related Projects
This repository is part of a broader portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.

Related repositories:

- `neural-network-from-scratch`
- `classification-of-handwritten-digits1`
- `edge-ai-hardware-optimization`
- `data-analysis-for-hospitals`
- `nba-data-preprocessing`
- `Data-Science-AI-Portfolio`

## Future Work
Potential extensions include:

- deployment-oriented validation on embedded or edge-class devices,
- expanded preprocessing and model-compression experiment interfaces,
- richer benchmarking frameworks for latency, memory, and throughput analysis,
- tighter integration of artifact tracking for end-to-end experiment lineage.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
