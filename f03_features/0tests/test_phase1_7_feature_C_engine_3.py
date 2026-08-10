# VSCode run: python -m pytest -q f03_features/0tests/test_phase1_7_feature_C_engine_3.py

import pandas as pd
from f03_features.OLD_2.feature_C_engine_3 import FeatureEngine

# ---------------------------------------------------------
def test_engine_empty_specs_returns_df():
    engine = FeatureEngine()

    df = pd.DataFrame({"close": [1, 2, 3]})
    out = engine.execute(df, specs=[], mode="train")

    assert len(out) == 3
    assert out.equals(df)

# ---------------------------------------------------------
def test_engine_invalid_spec_is_ignored():
    engine = FeatureEngine()

    df = pd.DataFrame({"close": [1, 2, 3]})

    out = engine.execute(df, specs=["invalid_spec"], mode="train")

    assert len(out) == 3

# ---------------------------------------------------------
def test_engine_runs_without_crash():
    engine = FeatureEngine()

    df = pd.DataFrame({"close": [1, 2, 3, 4]})

    specs = ["sma(10)"]  # depends on registry

    out = engine.execute(df, specs=specs, mode="train")

    assert out is not None
    assert len(out) == 4

# ---------------------------------------------------------
def test_live_mode_does_not_crash():
    engine = FeatureEngine()

    df = pd.DataFrame({
        "open": [1,2,3],
        "high": [2,3,4],
        "low": [0.5,1.5,2.5],
        "close": [1.2,2.2,3.2],
        "volume": [100,200,300],
    })

    out = engine.execute(df, specs=[], mode="live")

    assert len(out) == 3
