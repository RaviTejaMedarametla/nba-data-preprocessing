from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TelemetrySnapshot:
    cpu_percent: float
    process_memory_mb: float
    system_memory_percent: float
    energy_uj: float | None


class HardwareMonitor:
    """Optional hardware telemetry with fallback-safe behavior."""

    def __init__(self) -> None:
        self._psutil = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:
            self._psutil = None

        self._rapl_path = self._discover_rapl_path()

    def _discover_rapl_path(self) -> Path | None:
        base = Path('/sys/class/powercap')
        if not base.exists():
            return None
        for cand in base.glob('intel-rapl*/energy_uj'):
            if cand.is_file():
                return cand
        return None

    def _read_rapl_energy_uj(self) -> float | None:
        if self._rapl_path is None:
            return None
        try:
            return float(self._rapl_path.read_text(encoding='utf-8').strip())
        except Exception:
            return None

    def snapshot(self) -> TelemetrySnapshot:
        if self._psutil is None:
            return TelemetrySnapshot(cpu_percent=0.0, process_memory_mb=0.0, system_memory_percent=0.0, energy_uj=self._read_rapl_energy_uj())

        proc = self._psutil.Process()
        process_memory_mb = proc.memory_info().rss / (1024 * 1024)
        return TelemetrySnapshot(
            cpu_percent=float(self._psutil.cpu_percent(interval=None)),
            process_memory_mb=float(process_memory_mb),
            system_memory_percent=float(self._psutil.virtual_memory().percent),
            energy_uj=self._read_rapl_energy_uj(),
        )

    def process_memory_mb(self) -> float:
        snap = self.snapshot()
        return snap.process_memory_mb

    def compare(self, start: TelemetrySnapshot, end: TelemetrySnapshot) -> dict[str, Any]:
        energy_j = None
        if start.energy_uj is not None and end.energy_uj is not None and end.energy_uj >= start.energy_uj:
            energy_j = (end.energy_uj - start.energy_uj) / 1_000_000.0
        return {
            'cpu_percent_start': start.cpu_percent,
            'cpu_percent_end': end.cpu_percent,
            'process_memory_start_mb': start.process_memory_mb,
            'process_memory_end_mb': end.process_memory_mb,
            'system_memory_percent_start': start.system_memory_percent,
            'system_memory_percent_end': end.system_memory_percent,
            'rapl_energy_j': energy_j,
        }
