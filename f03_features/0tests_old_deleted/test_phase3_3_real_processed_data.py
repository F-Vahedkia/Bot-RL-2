# RUN:
# python -m pytest -q f03_features/0tests/test_phase3_3_real_processed_data.py

from pathlib import Path
import pandas as pd

from f03_features.feature_B_bootstrap import build_feature_system


DATA_FILE = Path(
    "f02_data/processed/XAUUSD/H1.parquet"
)


def test_real_processed_file_exists():

    assert DATA_FILE.exists()


def test_real_processed_can_load():

    df = pd.read_parquet(DATA_FILE)

    assert len(df) > 0
    assert len(df.columns) > 0


def test_engine_against_real_processed_data():

    system = build_feature_system()
    engine = system.get_engine()

    df = pd.read_parquet(DATA_FILE)

    specs = [
        "sma(10)",
        "ema(10)",
    ]

    result = engine.execute(
        df=df,
        specs=specs,
        mode="train",
    )

    assert isinstance(result, pd.DataFrame)

