# VSCode run: python -m pytest -q f03_features/0tests/test_feature_B_config_integration.py

import pandas as pd
from f03_features.feature_B_bootstrap import build_feature_system


def test_bootstrap_loads_config_and_engine():
    system = build_feature_system()

    assert system.get_engine() is not None
    assert system.get_registry() is not None
    assert system.get_cache() is not None
    assert system.get_store() is not None


def test_engine_runs_with_real_system():
    system = build_feature_system()
    engine = system.get_engine()

    df = pd.DataFrame({
        "close": [1, 2, 3],
        "open": [1, 1, 1],
        "high": [2, 2, 2],
        "low": [0.5, 0.5, 0.5],
        "volume": [100, 100, 100],
    })

    out = engine.execute(df, specs=["sma(3)"], mode="train")

    assert len(out) == 3

   