from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class DatasetFingerprint:
    path: str
    sha256: str
    rows: int
    columns: int


class DataIngestor:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def load(self, source: str | Path | pd.DataFrame) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(source)

    def stream_chunks(self, source: str | Path | pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
            for start in range(0, len(df), chunk_size):
                yield df.iloc[start : start + chunk_size].copy()
            return

        for chunk in pd.read_csv(source, chunksize=chunk_size):
            yield chunk.copy()

    def fingerprint(self, source: str | Path | pd.DataFrame) -> DatasetFingerprint:
        df = self.load(source)
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        digest = hashlib.sha256(csv_bytes).hexdigest()
        path = str(source) if not isinstance(source, pd.DataFrame) else '<in-memory>'
        return DatasetFingerprint(path=path, sha256=digest, rows=len(df), columns=len(df.columns))
