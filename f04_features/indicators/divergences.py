# -*- coding: utf-8 -*-
# f04_features/indicators/divergences.py
# Status in (Bot-RL-2): Completed
"""
واگرایی کلاسیک/مخفی روی RSI و MACD - با pivot تاییدشده و shift(+1) برای حذف look-ahead
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from .core import rsi, macd

# pivots ساده: قله/دره تاییدشده با پنجره k
def pivots(series: pd.Series, k: int = 2):
    hi = series.rolling(2*k+1, center=True).apply(lambda x: float(x[k] == x.max()), raw=True).fillna(0).astype("int8")
    lo = series.rolling(2*k+1, center=True).apply(lambda x: float(x[k] == x.min()), raw=True).fillna(0).astype("int8")
    return hi, lo

# واگرایی کلاسیک/مخفی بین قیمت و یک اسیلاتور
# mode: "classic" (price HH & osc LH → bear, price LL & osc HL → bull) یا "hidden"

def divergence_flags(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = "classic"):
    ph, pl = pivots(price, k)
    oh, ol = pivots(osc, k)
    bull = pd.Series(0, index=price.index, dtype="int8")
    bear = pd.Series(0, index=price.index, dtype="int8")

    last_ph = None; last_pl = None; last_oh = None; last_ol = None
    for i in range(len(price)):
        if ph.iloc[i]:
            if last_ph is not None and last_oh is not None:
                # سقف قیمت ↑ اما سقف اسیلاتور ↓ → واگرایی خرسی کلاسیک
                if mode == "classic" and price.iloc[last_ph] < price.iloc[i] and osc.iloc[last_oh] > osc.iloc[i]:
                    bear.iloc[i] = 1
                # سقف قیمت ↓ اما اسیلاتور ↑ → واگرایی خرسی مخفی
                if mode == "hidden" and price.iloc[last_ph] > price.iloc[i] and osc.iloc[last_oh] < osc.iloc[i]:
                    bear.iloc[i] = 1
            last_ph = i
            last_oh = i if oh.iloc[i] else last_oh
        if pl.iloc[i]:
            if last_pl is not None and last_ol is not None:
                # کف قیمت ↓ اما کف اسیلاتور ↑ → واگرایی گاوی کلاسیک
                if mode == "classic" and price.iloc[last_pl] > price.iloc[i] and osc.iloc[last_ol] < osc.iloc[i]:
                    bull.iloc[i] = 1
                # کف قیمت ↑ اما اسیلاتور ↓ → واگرایی گاوی مخفی
                if mode == "hidden" and price.iloc[last_pl] < price.iloc[i] and osc.iloc[last_ol] > osc.iloc[i]:
                    bull.iloc[i] = 1
            last_pl = i
            last_ol = i if ol.iloc[i] else last_ol

    # جلوگیری از استفادهٔ همان کندل: shift(+1)
    return bull.shift(1).fillna(0).astype("int8"), bear.shift(1).fillna(0).astype("int8")


def registry() -> Dict[str, callable]:
    
    def make_div_rsi(df, period: int = 14, k: int = 2, mode: str = "classic", **_):
        osc = rsi(df["close"], period)
        b, s = divergence_flags(df["close"], osc, k=k, mode=mode)
        return {f"div_rsi_bull_{period}_{k}_{mode}": b, f"div_rsi_bear_{period}_{k}_{mode}": s}
    
    def make_div_macd(df, fast: int = 12, slow: int = 26, signal: int = 9, k: int = 2, mode: str = "classic", **_):
        line, _, _ = macd(df["close"], fast, slow, signal)
        b, s = divergence_flags(df["close"], line, k=k, mode=mode)
        return {f"div_macd_bull_{fast}_{slow}_{signal}_{k}_{mode}": b, f"div_macd_bear_{fast}_{slow}_{signal}_{k}_{mode}": s}
    return {"div_rsi": make_div_rsi, "div_macd": make_div_macd}

