from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Centralized configuration for deterministic, research-grade runs."""

    random_seed: int = 42
    chunk_size: int = 128
    batch_size: int = 256
    n_jobs: int = 1
    max_memory_mb: int = 1024
    max_compute_units: float = 1.0
    benchmark_runs: int = 5
    adaptive_chunk_resize: bool = True
    max_chunk_retries: int = 3
    spill_to_disk: bool = False
    output_dir: Path = field(default_factory=lambda: Path('artifacts'))

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'intermediate').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'benchmarks').mkdir(parents=True, exist_ok=True)
