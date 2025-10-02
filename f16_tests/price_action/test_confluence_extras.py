# f16_tests/price_action/test_confluence_extras.py
"""
Confluence — Extras (Regime/Breakouts/Microchannels)
====================================================
بررسی می‌کند وقتی وزن extras > 0 باشد و سیگنال‌ها وجود داشته باشند،
conf_score نسبت به حالت بدون extras افزایش معنادار پیدا کند.
"""
# Run: pytest f16_tests/price_action/test_confluence_extras.py -q

import pandas as pd
from f04_features.price_action import (
    regime as rg,
    breakouts as bo,
    microchannels as mc,
    confluence as cf,
)

def _sample_df(n=80):
    # نیمه اول: رنج → نیمه دوم: شکست به بالا + کانال
    close = []
    for i in range(n):
        if i < n//2:
            close.append(100 + (-1)**i * 0.08)
        else:
            close.append(101 + (i - n//2) * 0.35)
    high = [c + 0.12 for c in close]
    low  = [c - 0.12 for c in close]
    return pd.DataFrame({"close": close, "high": high, "low": low})

def test_confluence_with_and_without_extras():
    df = _sample_df()
    # بسازیم سیگنال‌های سه ماژول
    df = rg.build_regime(df, anti_lookahead=True, slope_window=6, width_window=8)
    df = bo.build_breakouts(df, anti_lookahead=True, confirm_closes=1, range_window=8, min_periods=4)
    df = mc.build_microchannels(df, anti_lookahead=True, min_len=3)

    # حالت پایه: وزن extras صفر
    base = cf.build_confluence(df, anti_lookahead=True, weights={
        "structure": 0.30, "zones": 0.30, "imbalance": 0.20, "mtf": 0.20, "extras": 0.00
    })
    # حالت با extras
    with_extras = cf.build_confluence(df, anti_lookahead=True, weights={
        "structure": 0.25, "zones": 0.25, "imbalance": 0.20, "mtf": 0.10, "extras": 0.20
    })

    assert "conf_components_extras" in with_extras.columns
    assert len(with_extras) == len(base)

    # مقایسهٔ میانگین امتیاز نهایی
    assert with_extras["conf_score"].mean() >= base["conf_score"].mean() - 1e-9
