# f04_features/indicators/extras_channel.py
# -*- coding: utf-8 -*-
"""کانال‌ها: Donchian, Chaikin Volatility"""
from __future__ import annotations
import pandas as pd
from typing import Dict

def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    upper = high.rolling(n, min_periods=n).max().astype("float32")
    lower = low.rolling(n, min_periods=n).min().astype("float32")
    mid = ((upper + lower)/2.0).astype("float32")
    return upper, mid, lower

def chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10, roc: int = 10):
    ema_range = (high - low).ewm(span=n, adjust=False, min_periods=n).mean()
    return (100.0 * (ema_range - ema_range.shift(roc)) / ema_range.shift(roc)).astype("float32")



def registry() -> Dict[str, callable]:
    
    def make_donchian(df, n: int = 20, **_):
        up, mid, lo = donchian(df["high"], df["low"], n)
        return {f"donch_up_{n}": up, f"donch_mid_{n}": mid, f"donch_lo_{n}": lo}
    
    def make_ch_vol(df, n: int = 10, roc: int = 10, **_):
        return {f"chaikin_vol_{n}_{roc}": chaikin_volatility(df["high"], df["low"], n, roc)}
    return {"donchian": make_donchian, "chaikin_vol": make_ch_vol}