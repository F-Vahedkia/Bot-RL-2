# VSCode run: python -m pytest -q f03_features/0tests/test_phase1_5_feature_B_bootstrap.py

import pandas as pd
from f03_features.feature_B_bootstrap import build_feature_system


def test_bootstrap_builds_all_layers():
    system = build_feature_system()

    assert system is not None
    assert system.get_engine() is not None
    assert system.get_store() is not None
    assert system.get_registry() is not None
    assert system.get_cache() is not None


def test_engine_basic_wiring():
    system = build_feature_system()
    engine = system.get_engine()

    df = pd.DataFrame({"close": [1, 2, 3]})

    # minimal smoke test (no spec dependency)
    out = engine.execute(df, specs=[], mode="train")

    assert out is not None
    assert len(out) == 3


def test_config_loaded():
    system = build_feature_system()
    cfg = system.get_config()

    assert isinstance(cfg, dict)
    