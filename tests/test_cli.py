from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_smoke(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        'nba_data_preprocessing/task/run_pipeline.py',
        '--input',
        'nba_data_preprocessing/data/nba2k-full.csv',
        '--output-dir',
        str(tmp_path / 'artifacts'),
        '--benchmark-runs',
        '1',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert '"batch"' in result.stdout
