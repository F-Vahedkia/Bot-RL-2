# Run: python -m pytest -q f03_features/0tests/test_phase2_1_config_integration.py

import pandas as pd
import numpy as np

from f03_features.feature_B_bootstrap import build_feature_system
from f03_features.feature_B_graph import FeatureGraph
from f03_features.observation_B_builder import ObservationBuilder


# ============================================================
# MOCK CONFIG (minimal but real pipeline compatible)
# ============================================================
def _make_config():

    return {
        "execution": {
            "mode": "train"
        },
        "env": {
        },
        "features": {
            "observation": {
                "shift_features_by": 0,
                "drop_na_head": True,
                "features_whitelist": [],
                "features_blacklist": []
            }
        }
    }


# ============================================================
# MOCK DATA (OHLCV-like minimal)
# ============================================================
def _make_df():
    return pd.DataFrame({
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low":  [1, 2, 3, 4, 5],
        "close":[1, 2, 3, 4, 5],
        "volume":[10, 11, 12, 13, 14],
    })


# ============================================================
# MOCK FEATURE SPECS
# ============================================================
def _make_specs():
    return [
        "sma(10)",
        "ema(10)",
        "rsi(14)"
    ]


# ============================================================
# TEST 1: full system bootstrap
# ============================================================
def test_phase2_bootstrap():

    system = build_feature_system()

    assert system.get_engine() is not None
    assert system.get_store() is not None
    assert system.get_registry() is not None
    assert system.get_cache() is not None


# ============================================================
# TEST 2: engine + registry execution
# ============================================================
def test_phase2_engine_execution():

    system = build_feature_system()
    engine = system.get_engine()

    df = _make_df()
    specs = _make_specs()

    out = engine.execute(df, specs, mode="train")

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df)


# ============================================================
# TEST 3: graph + observation pipeline
# ============================================================
def test_phase2_observation_pipeline():

    system = build_feature_system()
    engine = system.get_engine()

    df = _make_df()
    specs = _make_specs()

    feature_df = engine.execute(df, specs, mode="train")

    graph = FeatureGraph(specs)

    builder = ObservationBuilder(_make_config())
    obs = builder.build(feature_df, graph)

    assert isinstance(obs, pd.DataFrame)
    assert len(obs) <= len(df)


# ============================================================
# TEST 4: deterministic run (no cache instability)
# ============================================================
def test_phase2_determinism():

    system = build_feature_system()
    engine = system.get_engine()

    df = _make_df()
    specs = _make_specs()

    out1 = engine.execute(df.copy(), specs, mode="train")
    out2 = engine.execute(df.copy(), specs, mode="train")

    pd.testing.assert_frame_equal(out1, out2)


# ============================================================
# TEST 5: cache exists + no crash
# ============================================================
def test_phase2_cache_consistency():

    system = build_feature_system()
    engine = system.get_engine()

    df = _make_df()
    specs = _make_specs()

    for _ in range(3):
        out = engine.execute(df.copy(), specs, mode="train")

    assert isinstance(out, pd.DataFrame)



#################################################
"""
🧠 این تست‌ها چه چیزی را واقعاً validate می‌کنند؟
1. 🟢 bootstrap integrity
engine exists
store exists
registry exists
2. 🟡 execution chain correctness
engine → registry → batch indicators
no crash on real dataframe
3. 🔵 observation integration
engine output → graph → builder
alignment بدون KeyError
4. ⚙️ determinism
دو اجرای پشت سر هم باید identical باشند
5. 🧊 cache stability
اجرای چندباره بدون corruption
⚠️ نکته مهم (خیلی مهم)

این تست‌ها عمداً:

real market data نمی‌زنند
data_handler را وارد نمی‌کنند
IO ندارند

📌 چون این مرحله فقط:

integration correctness of system core

🚀 خروجی مورد انتظار

اگر سیستم شما درست باشد:

5 passed
"""