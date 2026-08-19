# VSCode run: python -m pytest -q f03_features/0tests/test_phase1_8_observation_B_builder.py

import pandas as pd

from f03_features.observation_B_builder import ObservationBuilder
from f03_features.feature_B_graph import FeatureGraph


def _make_graph():
    return FeatureGraph([
        "sma(10)",
        "ema(10)",
        "rsi(14)",
    ])


def test_builder_basic_column_extraction():
    cfg = {"env": {}, "features": {}}
    builder = ObservationBuilder(cfg)

    df = pd.DataFrame({
        "sma(10)": [1, 2, 3],
        "ema(10)": [4, 5, 6],
        "rsi(14)": [7, 8, 9],
    })

    graph = _make_graph()

    out = builder.build(df, graph)

    assert out.shape[0] == 3
    assert set(out.columns) == {"sma(10)", "ema(10)", "rsi(14)"}


def test_builder_whitelist_filter():
    cfg = {
        "features": {
            "observation": {
                "features_whitelist": ["sma"]
            }
        },
    }
    builder = ObservationBuilder(cfg)

    df = pd.DataFrame({
        "sma(10)": [1, 2, 3],
        "ema(10)": [4, 5, 6],
    })

    graph = _make_graph()

    out = builder.build(df, graph)

    assert "sma(10)" in out.columns
    assert "ema(10)" not in out.columns


def test_builder_blacklist_filter():
    cfg = {
        "features": {
            "observation": {
                "features_blacklist": ["ema*"],
            }
        }
    }
    builder = ObservationBuilder(cfg)

    df = pd.DataFrame({
        "sma(10)": [1, 2, 3],
        "ema(10)": [4, 5, 6],
    })

    graph = _make_graph()

    out = builder.build(df, graph)

    assert "ema(10)" not in out.columns


def test_builder_shift_prevents_leakage():
    cfg = {
        "env": {},
        "features": {
            "observation": {
                "shift_features_by": 1,
                "drop_na_head": True,
            },
        },
    }
    builder = ObservationBuilder(cfg)

    df = pd.DataFrame({
        "sma(10)": [1.0, 2.0, 3.0],
    })

    graph = _make_graph()

    out = builder.build(df, graph)

    # shift + dropna => length reduces
    assert len(out) == 2


def test_builder_numpy_output():
    cfg = {"env": {}, "features": {}}
    builder = ObservationBuilder(cfg)

    df = pd.DataFrame({
        "sma(10)": [1, 2, 3],
    })

    graph = _make_graph()

    arr = builder.build_numpy(df, graph)

    assert arr.shape[1] == 1
    