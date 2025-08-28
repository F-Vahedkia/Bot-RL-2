# f04_features/indicators/levels.py
# -*- coding: utf-8 -*-
"""Pivotهای کلاسیک، S/R ساده مبتنی بر فراکتال، و فاصله تا سطوح فیبوناچی اخیر"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Pivotهای کلاسیک

def pivots_classic(high, low, close):
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
    r1 = 2*pivot - low.shift(1)
    s1 = 2*pivot - high.shift(1)
    r2 = pivot + (high.shift(1) - low.shift(1))
    s2 = pivot - (high.shift(1) - low.shift(1))
    r3 = high.shift(1) + 2*(pivot - low.shift(1))
    s3 = low.shift(1) - 2*(high.shift(1) - pivot)
    return pivot.astype("float32"), r1.astype("float32"), s1.astype("float32"), r2.astype("float32"), s2.astype("float32"), r3.astype("float32"), s3.astype("float32")

# فراکتال ساده برای S/R (K کندل چپ/راست)

def fractal_points(high, low, k: int = 2):
    # کد اولیه
    #hh = high.rolling(2*k+1, center=True).apply(lambda x: float(np.argmax(x)==k), raw=True).astype("int8")
    #ll = low.rolling(2*k+1, center=True).apply(lambda x: float(np.argmin(x)==k), raw=True).astype("int8")
    # کد ثانویه
    window = 2*k + 1
    hh = (high.rolling(window, center=True, min_periods=window)
              .apply(lambda x: float(np.argmax(x) == k), raw=True)
              .fillna(0.0)                 # ← جلوگیری از NaN
              .astype("int8"))
    ll = (low.rolling(window, center=True, min_periods=window)
              .apply(lambda x: float(np.argmin(x) == k), raw=True)
              .fillna(0.0)                 # ← جلوگیری از NaN
              .astype("int8"))    
    return hh, ll

# فاصله تا نزدیک‌ترین سطح S/R نزدیک

def sr_distance(close, high, low, k: int = 2, lookback: int = 500):
    hh, ll = fractal_points(high, low, k)
    idx = close.index
    res = pd.Series(index=idx, dtype="float32")
    sup = pd.Series(index=idx, dtype="float32")
    for i in range(len(idx)):
        lo = max(0, i - lookback)
        # مقاومت: آخرین high-فرکتال قبل از i
        prev_h = high[hh.astype(bool)].iloc[lo:i]
        prev_l = low[ll.astype(bool)].iloc[lo:i]
        r = (prev_h.iloc[-1] - close.iloc[i]) if len(prev_h) else np.nan
        s = (close.iloc[i] - prev_l.iloc[-1]) if len(prev_l) else np.nan
        res.iloc[i] = np.float32(r) if pd.notna(r) else (np.nan)
        sup.iloc[i] = np.float32(s) if pd.notna(s) else (np.nan)
    return res, sup

# فیبوناچی: سطوح اخیر بین آخرین سوینگ بالا/پایین (فراکتال k)

def fibo_levels(close, high, low, k: int = 2):
    hh, ll = fractal_points(high, low, k)
    # آخرین سوینگ‌ها
    last_high = high[hh.astype(bool)].ffill()
    last_low = low[ll.astype(bool)].ffill()
    diff = (last_high - last_low)
    levels = {
        "fib_236": (last_high - 0.236*diff).astype("float32"),
        "fib_382": (last_high - 0.382*diff).astype("float32"),
        "fib_500": (last_high - 0.5*diff).astype("float32"),
        "fib_618": (last_high - 0.618*diff).astype("float32"),
        "fib_786": (last_high - 0.786*diff).astype("float32"),
    }
    # فاصلهٔ قیمت تا هر سطح
    dists = {f"dist_{k}": (close - v).astype("float32") for k, v in levels.items()}
    levels.update(dists)
    return levels

from typing import Dict

def registry() -> Dict[str, callable]:
    def make_pivots(df, **_):
        p, r1, s1, r2, s2, r3, s3 = pivots_classic(df["high"], df["low"], df["close"])
        return {"pivot": p, "pivot_r1": r1, "pivot_s1": s1, "pivot_r2": r2, "pivot_s2": s2, "pivot_r3": r3, "pivot_s3": s3}
    def make_sr(df, k: int = 2, lookback: int = 500, **_):
        r, s = sr_distance(df["close"], df["high"], df["low"], k, lookback)
        return {f"sr_res_{k}_{lookback}": r, f"sr_sup_{k}_{lookback}": s}
    def make_fibo(df, k: int = 2, **_):
        return fibo_levels(df["close"], df["high"], df["low"], k)
    return {"pivots": make_pivots, "sr": make_sr, "fibo": make_fibo}