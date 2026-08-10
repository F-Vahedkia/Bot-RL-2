# RUN:
# python -m pytest -q f03_features/0tests/test_phase3_4_real_feature_generation.py

from pathlib import Path
import pandas as pd

from f03_features.feature_B_bootstrap import build_feature_system


DATA_FILE = Path(
    "f02_data/processed/XAUUSD/H1.parquet"
)


def test_real_feature_columns_created():

    system = build_feature_system()
    engine = system.get_engine()

    df = pd.read_parquet(DATA_FILE)

    before_cols = set(df.columns)

    result = engine.execute(
        df=df,
        specs=["sma(10)", "ema(20)"],
        mode="train",
    )

    after_cols = set(result.columns)

    new_cols = after_cols - before_cols

    assert len(new_cols) > 0


def test_result_not_identical_to_input():

    system = build_feature_system()
    engine = system.get_engine()

    df = pd.read_parquet(DATA_FILE)

    result = engine.execute(
        df=df,
        specs=["sma(10)"],
        mode="train",
    )

    assert result.shape[1] > df.shape[1]


def test_feature_column_contains_values():

    system = build_feature_system()
    engine = system.get_engine()

    df = pd.read_parquet(DATA_FILE)

    result = engine.execute(
        df=df,
        specs=["sma(10)"],
        mode="train",
    )

    new_cols = list(set(result.columns) - set(df.columns))

    assert len(new_cols) > 0

    col = new_cols[0]

    assert result[col].notna().sum() > 0

