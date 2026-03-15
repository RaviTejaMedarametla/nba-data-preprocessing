from pathlib import Path

from pipeline.ingestion.loader import DataIngestor


def test_loading_and_fingerprint() -> None:
    ingestor = DataIngestor(random_seed=42)
    data = ingestor.load(Path('nba_data_preprocessing/data/nba2k-full.csv'))
    fp = ingestor.fingerprint(Path('nba_data_preprocessing/data/nba2k-full.csv'))
    assert len(data) > 0
    assert fp.rows == len(data)
    assert len(fp.sha256) == 64
