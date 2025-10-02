# f16_tests/price_action/test_breakouts.py
"""
Unit Tests — Price Action: Breakouts
====================================
تست‌های ماژول build_breakouts:
- تولید ستون‌های اصلی با طول صحیح
- رفتار anti_lookahead روی فلگ‌ها
- سناریوی سادهٔ شکست رو به بالا/پایین و Retest/Fail
"""
# Run: pytest f16_tests/price_action/test_breakouts.py -q

import pandas as pd
from f04_features.price_action import breakouts as bo


def _sample_range_then_break(n=80):
    # 0..39: رنج باریک؛ 40..79: بریک‌اوت به بالا
    close = []
    for i in range(n):
        if i < n//2:
            close.append(100 + (-1)**i * 0.1)  # رنج کوچک
        else:
            close.append(101.0 + (i - n//2) * 0.3)  # شکست رو به بالا و ادامه
    high = [c + 0.15 for c in close]
    low  = [c - 0.15 for c in close]
    return pd.DataFrame({"close": close, "high": high, "low": low})


def test_breakouts_columns_and_lengths():
    df = _sample_range_then_break()
    out = bo.build_breakouts(df, anti_lookahead=True)

    cols = [
        "range_upper", "range_lower",
        "breakout_up", "breakout_down",
        "retest_up", "retest_down",
        "fail_break_up", "fail_break_down",
    ]
    for c in cols:
        assert c in out.columns
        assert len(out[c]) == len(df)


def test_breakouts_antilookahead_flags_shift():
    df = _sample_range_then_break()
    out0 = bo.build_breakouts(df, anti_lookahead=False)
    out1 = bo.build_breakouts(df, anti_lookahead=True)

    for c in ("breakout_up", "breakout_down", "retest_up", "retest_down", "fail_break_up", "fail_break_down"):
        assert len(out0[c]) == len(out1[c])
        # اولین ردیف در نسخهٔ شیفت‌شده باید 0/تغییر کرده باشد
        assert out1[c].iloc[0] in (0, 1)  # باینری
        if out0[c].iloc[0] != 0:
            assert out1[c].iloc[0] != out0[c].iloc[0]


def test_breakouts_simple_up_scenario_has_some_signals():
    df = _sample_range_then_break()
    out = bo.build_breakouts(df, anti_lookahead=True, confirm_closes=1)

    # انتظار داریم جایی پس از نیمه راه، شکست رو به بالا رخ داده باشد
    assert out["breakout_up"].sum() >= 1

    # Retest ممکن است بسته به داده رخ دهد؛ اگر رخ داده، باید باینری معتبر باشد
    assert set(out["retest_up"].unique()).issubset({0, 1})
    assert set(out["fail_break_up"].unique()).issubset({0, 1})
