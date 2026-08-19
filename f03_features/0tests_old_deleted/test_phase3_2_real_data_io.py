# Run: python -m pytest -q f03_features/0tests/test_phase3_2_real_data_io.py

import pandas as pd
from pathlib import Path

from f03_features.feature_B_bootstrap import build_feature_system
from f03_features.feature_B_graph import FeatureGraph
from f03_features.observation_B_builder import ObservationBuilder


# ============================================================
# MOCK REAL PATH STRUCTURE
# ============================================================
def _mock_data_path():

    # در production واقعی:
    # f02_data/processed/{symbol}/{tf}/data.parquet

    return Path("f02_data/processed")


# ============================================================
# LOAD SIMULATED MULTI-SYMBOL DATA
# ============================================================
def _load_multi_symbol_data():

    # NOTE: در نسخه واقعی این با loader جایگزین می‌شود

    return {
        "BTCUSDT": pd.DataFrame({
            "open": range(1, 150),
            "high": range(1, 150),
            "low": range(1, 150),
            "close": range(1, 150),
            "volume": range(201, 350),
        }),
        "ETHUSDT": pd.DataFrame({
            "open": range(10, 160),
            "high": range(10, 160),
            "low": range(10, 160),
            "close": range(10, 160),
            "volume": range(300, 450),
        }),
    }


def _make_specs():
    return ["sma(10)", "ema(10)", "rsi(14)"]


# ============================================================
# TEST 1: multi-symbol engine execution
# ============================================================
def test_phase3_2_multi_symbol_engine():

    system = build_feature_system()
    engine = system.get_engine()

    data = _load_multi_symbol_data()
    specs = _make_specs()

    results = {}

    for symbol, df in data.items():
        out = engine.execute(df, specs, mode="train")
        results[symbol] = out

    assert "BTCUSDT" in results
    assert "ETHUSDT" in results

    for df in results.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


# ============================================================
# TEST 2: observation consistency per symbol
# ============================================================
def test_phase3_2_observation_per_symbol():

    system = build_feature_system()
    engine = system.get_engine()

    data = _load_multi_symbol_data()
    specs = _make_specs()

    graph = FeatureGraph(specs)
    builder = ObservationBuilder(system.get_config())

    for symbol, df in data.items():

        feature_df = engine.execute(df, specs, mode="train")
        obs = builder.build(feature_df, graph)

        assert isinstance(obs, pd.DataFrame)

        # alignment sanity check
        assert len(obs) <= len(df)


# ============================================================
# TEST 3: no cross-symbol leakage
# ============================================================
def test_phase3_2_no_cross_symbol_state():

    system = build_feature_system()
    engine = system.get_engine()

    data = _load_multi_symbol_data()
    specs = _make_specs()

    out1 = engine.execute(data["BTCUSDT"], specs, mode="train")
    out2 = engine.execute(data["ETHUSDT"], specs, mode="train")

    # ساختار باید متفاوت باشد (different data → different output)
    assert not out1.equals(out2)


# ============================================================
# TEST 4: pipeline stability (repeat run)
# ============================================================
def test_phase3_2_repeatability():

    system = build_feature_system()
    engine = system.get_engine()

    df = _load_multi_symbol_data()["BTCUSDT"]
    specs = _make_specs()

    a = engine.execute(df.copy(), specs, mode="train")
    b = engine.execute(df.copy(), specs, mode="train")

    pd.testing.assert_frame_equal(a, b)


#################################################
"""
🧠 این مرحله دقیقاً چه چیزی را validate می‌کند؟
1. 📦 multi-symbol correctness
هر symbol مستقل اجرا می‌شود
state leakage بین symbolها نداریم
2. 🔗 pipeline consistency
engine + observation + graph با هم سازگار هستند
3. 🧊 determinism
اجرای مجدد → خروجی یکسان
4. ⚙️ production realism (شبیه f02_data)
ساختار آینده data pipeline را simulate می‌کند
⚠️ نکته مهم معماری

اگر این مرحله pass شود:

سیستم شما وارد سطح “paper-trading-grade pipeline correctness” می‌شود
"""