from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Centralized configuration for deterministic pipeline runs."""

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

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if self.max_memory_mb <= 0:
            raise ValueError('max_memory_mb must be positive')
        if self.max_compute_units <= 0:
            raise ValueError('max_compute_units must be positive')
        if self.benchmark_runs <= 0:
            raise ValueError('benchmark_runs must be positive')

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for dirname in ('intermediate', 'reports', 'benchmarks', 'metadata', 'profiles'):
            (self.output_dir / dirname).mkdir(parents=True, exist_ok=True)
