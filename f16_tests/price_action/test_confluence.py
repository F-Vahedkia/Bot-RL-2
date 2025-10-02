# f16_tests/price_action/test_confluence.py
"""
Unit Tests — Confluence Engine (Price Action)
=============================================
تست‌ها با یک دیتافریم ساختگی که شبیه‌ساز خروجی ماژول‌های قبلی است، صحت نمره‌ی
Confluence و رفتار anti_lookahead را کنترل می‌کنند.
"""
# روش اجرای برنامه:
# pytest f16_tests/price_action/test_confluence.py -q

import pandas as pd
from f04_features.price_action import (
    regime as rg,
    breakouts as bo,
    microchannels as mc,
    confluence as cf,
)

def _sample_df(n=80):
    n = 20
    df = pd.DataFrame({
        "close": [100 + i*0.4 for i in range(n)],
        # market_structure
        "swing_type": ["", "HL", "HH", "", "LH", "LL", "HL", "HH", "", "", "LH", "LL", "", "HL", "HH", "", "", "LH", "LL", ""],
        "bos_up":     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "bos_down":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        "choch_up":   [0]*20,
        "choch_down": [0]*20,
        # zones
        "dist_to_sd": [0.8, 0.5, 0.2, 0.3, 1.0, 1.5, 0.7, 0.1, 0.4, 0.6, 0.9, 0.3, 0.25, 0.2, 0.5, 0.8, 0.7, 1.2, 0.9, 0.4],
        "dist_to_ob": [1.1, 0.6, 0.4, 0.2, 0.9, 0.7, 1.4, 0.3, 0.25, 0.2, 0.6, 0.7, 0.8, 0.5, 0.4, 0.9, 0.95, 1.1, 0.8, 0.3],
        # imbalance
        "dist_to_fvg_mid": [0.9, 0.4, 0.2, 0.7, 1.2, 1.5, 0.8, 0.15, 0.5, 1.0, 0.9, 0.2, 0.3, 0.25, 0.45, 0.6, 1.3, 0.95, 0.85, 0.4],
        "sweep_up":   [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0],
        "sweep_down": [0,0,1,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
        # mtf
        "mtf_confluence_score": [0.6,0.55,0.7,0.8,0.4,0.35,0.5,0.9,0.6,0.55,0.45,0.5,0.6,0.65,0.7,0.4,0.35,0.5,0.55,0.6],
        "mtf_conflict": [0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0],
        "mtf_strength": [0.2,0.3,0.4,0.6,0.5,0.3,0.2,0.8,0.6,0.4,0.2,0.2,0.3,0.5,0.6,0.4,0.3,0.2,0.4,0.5],
    })
    return df


def test_confluence_columns_and_ranges():
    base = _sample_df()
    out = cf.build_confluence(base, anti_lookahead=True)

    # ستون‌های مورد انتظار
    for col in (
        "conf_components_structure",
        "conf_components_zones",
        "conf_components_imbalance",
        "conf_components_mtf",
        "conf_score",
        "conf_flag_strong_entry",
        "conf_flag_filter_pass",
    ):
        assert col in out.columns
        assert len(out[col]) == len(base)

    # امتیازها و پرچم‌ها در بازه‌های معتبر
    assert out["conf_score"].between(0.0, 1.0, inclusive="both").all()
    assert set(out["conf_flag_strong_entry"].unique()).issubset({0, 1})
    assert set(out["conf_flag_filter_pass"].unique()).issubset({0, 1})


def test_confluence_antilookahead_shift():
    base = _sample_df()
    out0 = cf.build_confluence(base, anti_lookahead=False)
    out1 = cf.build_confluence(base, anti_lookahead=True)

    # اثر شیفت: ردیف اول در نسخه‌ی anti_lookahead باید NaN/تغییر کرده باشد
    for col in (
        "conf_components_structure",
        "conf_components_zones",
        "conf_components_imbalance",
        "conf_components_mtf",
        "conf_score",
    ):
        if pd.notna(out0[col].iloc[0]):
            assert (pd.isna(out1[col].iloc[0])) or (out0[col].iloc[0] != out1[col].iloc[0])
