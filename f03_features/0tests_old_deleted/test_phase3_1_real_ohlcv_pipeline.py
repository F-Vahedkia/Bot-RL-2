# Run: python -m pytest -q f03_features/0tests/test_phase3_1_real_ohlcv_pipeline.py

import pandas as pd
from f03_features.feature_B_bootstrap import build_feature_system
from f03_features.feature_B_graph import FeatureGraph
from f03_features.observation_B_builder import ObservationBuilder


# ============================================================
# LOAD REALISTIC DATA (simulated path structure)
# ============================================================
def _load_mock_ohlcv():

    # NOTE: in Phase 3 this will be replaced by:
    # f02_data/processed/symbol/TF/*.parquet

    return pd.DataFrame({
        "open":   range(1, 200),
        "high":   range(1, 200),
        "low":    range(1, 200),
        "close":  range(1, 200),
        "volume": range(100, 299),
    })


def _make_specs():
    return [
        "sma(10)",
        "ema(10)",
        "rsi(14)"
    ]


# ============================================================
# TEST 1: full pipeline on OHLCV
# ============================================================
def test_phase3_full_pipeline():

    system = build_feature_system()
    engine = system.get_engine()

    df = _load_mock_ohlcv()
    specs = _make_specs()

    feature_df = engine.execute(df, specs, mode="train")

    graph = FeatureGraph(specs)
    builder = ObservationBuilder(system.get_config())

    obs = builder.build(feature_df, graph)

    assert isinstance(obs, pd.DataFrame)
    assert len(obs) > 0


# ============================================================
# TEST 2: no NaN explosion
# ============================================================
def test_phase3_no_nan_explosion():

    system = build_feature_system()
    engine = system.get_engine()

    df = _load_mock_ohlcv()
    specs = _make_specs()

    feature_df = engine.execute(df, specs, mode="train")

    assert feature_df.isna().sum().sum() < len(feature_df) * 0.5

