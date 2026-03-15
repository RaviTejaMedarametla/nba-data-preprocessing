#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python nba_data_preprocessing/task/run_pipeline.py --input nba_data_preprocessing/data/nba2k-full.csv --output-dir artifacts/run
python nba_data_preprocessing/task/run_pipeline.py --input nba_data_preprocessing/data/nba2k-full.csv --output-dir artifacts/manifest --benchmark-runs 2
python nba_data_preprocessing/task/run_pipeline.py --input nba_data_preprocessing/data/nba2k-full.csv --output-dir artifacts/ablation --chunk-size 64 --max-memory-mb 512

pytest
