# f04_features/indicators/patterns.py
# # -*- coding: utf-8 -*-
"""الگوهای سادهٔ کندلی (فلگ‌ها)"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Engulfing (ساده و محافظه‌کار)
def engulfing_flags(open_, high, low, close):
    body = close - open_
    prev_body = body.shift(1)
    bull = (prev_body < 0) & (close > open_.shift(1)) & (open_ < close.shift(1))
    bear = (prev_body > 0) & (close < open_.shift(1)) & (open_ > close.shift(1))
    return bull.astype("int8"), bear.astype("int8")

# Doji
def doji_flag(open_, close, thresh: float = 0.1):
    rng = (close - open_).abs()
    atr_like = (rng.rolling(20, min_periods=20).mean()).replace(0, np.nan)
    return ((rng / atr_like) < thresh).astype("int8")

# Pinbar (تقریب)
def pinbar_flags(open_, high, low, close, ratio: float = 2.0):
    body = (close - open_).abs()
    upper_wick = (high - close.where(close > open_, open_))
    lower_wick = (close.where(close < open_, open_) - low)
    bull = (lower_wick > ratio * body) & (upper_wick < body)
    bear = (upper_wick > ratio * body) & (lower_wick < body)
    return bull.astype("int8"), bear.astype("int8")

from typing import Dict

def registry() -> Dict[str, callable]:
    def make_engulf(df, **_):
        b, s = engulfing_flags(df["open"], df["high"], df["low"], df["close"])
        return {"pat_engulf_bull": b, "pat_engulf_bear": s}
    def make_doji(df, thresh: float = 0.1, **_):
        return {f"pat_doji_{thresh}": doji_flag(df["open"], df["close"], thresh)}
    def make_pinbar(df, ratio: float = 2.0, **_):
        b, s = pinbar_flags(df["open"], df["high"], df["low"], df["close"], ratio)
        return {f"pat_pin_bull_{ratio}": b, f"pat_pin_bear_{ratio}": s}
    return {"pat_engulf": make_engulf, "pat_doji": make_doji, "pat_pin": make_pinbar}