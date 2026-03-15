from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / 'nba_data_preprocessing' / 'task'
if str(TASK_PATH) not in sys.path:
    sys.path.insert(0, str(TASK_PATH))
