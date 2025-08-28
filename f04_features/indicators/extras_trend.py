# f04_features/indicators/extras_trend.py
# -*- coding: utf-8 -*-
"""اندیکاتورهای روندی تکمیلی: Supertrend, ADX/DI/ADXR, Aroon, KAMA/DEMA/TEMA/HMA, Ichimoku"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .core import atr, ema, sma, wma

# Supertrend (classic)
def supertrend(high, low, close, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atrv = atr(high, low, close, n=period).astype("float32")
    hl2 = ((high + low) / 2.0).astype("float32")
    m = np.float32(multiplier)
    upper = (hl2 + m * atrv).astype("float32")
    lower = (hl2 - m * atrv).astype("float32")
    st = pd.Series(index=close.index, dtype="float32")
    dir_up = True
    for i in range(len(close)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]
            dir_up = True
            continue
        prev = st.iloc[i-1]
        if close.iloc[i] > prev:
            st.iloc[i] = np.float32(max(lower.iloc[i], prev))
            dir_up = True
        else:
            st.iloc[i] = np.float32(min(upper.iloc[i], prev))
            dir_up = False
    return st   #.astype("float32")

# ADX/DI
def adx_di(high, low, close, n: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = (pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)).replace(0, np.nan)
    atrn = tr.rolling(n, min_periods=n).mean()
    pdi = (100.0 * pd.Series(plus_dm, index=high.index).rolling(n, min_periods=n).sum() / atrn).astype("float32")
    mdi = (100.0 * pd.Series(minus_dm, index=high.index).rolling(n, min_periods=n).sum() / atrn).astype("float32")
    dx = (100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).astype("float32")
    adx = dx.rolling(n, min_periods=n).mean().astype("float32")
    adxr = ((adx + adx.shift(n)) / 2.0).astype("float32")
    return pdi, mdi, adx, adxr

# Aroon

def aroon(high, low, n: int = 25):
    def _aroon_up(s):
        return (100.0 * s.rolling(n, min_periods=n).apply(lambda x: np.argmax(x[::-1]) / (n-1) * 100.0, raw=True)).astype("float32")
    def _aroon_down(s):
        return (100.0 * s.rolling(n, min_periods=n).apply(lambda x: np.argmin(x[::-1]) / (n-1) * 100.0, raw=True)).astype("float32")
    up = _aroon_up(high)
    down = _aroon_down(low)
    osc = (up - down).astype("float32")
    return up, down, osc

# KAMA/DEMA/TEMA/HMA

def kama(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    ef = (s.diff(n).abs()) / (s.diff().abs().rolling(n, min_periods=n).sum().replace(0, np.nan))
    sc = ((ef * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2)
    out = pd.Series(index=s.index, dtype="float32")
    out.iloc[:n] = s.iloc[:n]
    for i in range(n, len(s)):
        out.iloc[i] = out.iloc[i-1] + sc.iloc[i] * (s.iloc[i] - out.iloc[i-1])
    return out.astype("float32")

def dema(s: pd.Series, n: int = 20) -> pd.Series:
    e = ema(s, n); e2 = ema(e, n)
    return (2*e - e2).astype("float32")

def tema(s: pd.Series, n: int = 20) -> pd.Series:
    e1 = ema(s, n); e2 = ema(e1, n); e3 = ema(e2, n)
    return (3*e1 - 3*e2 + e3).astype("float32")

def hma(s: pd.Series, n: int = 20) -> pd.Series:
    n2 = max(2, n//2)
    w1 = wma(s, n2)
    w2 = wma(s, n)
    return wma(2*w1 - w2, int(np.sqrt(n))).astype("float32")

# Ichimoku (Tenkan/Kijun/SenkouA/B/Chikou)

def ichimoku(high, low, close, tenkan: int = 9, kijun: int = 26, span_b: int = 52):
    conv = ((high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2.0).astype("float32")
    base = ((high.rolling(kijun).max() + low.rolling(kijun).min()) / 2.0).astype("float32")
    span_a = ((conv + base)/2.0).shift(kijun).astype("float32")
    span_bv = ((high.rolling(span_b).max() + low.rolling(span_b).min())/2.0).shift(kijun).astype("float32")
    chikou = close.shift(-kijun).astype("float32")  # صرفاً جهت رسم/تحلیل تاریخی؛ در فیچرها بعداً shift می‌شود
    return conv, base, span_a, span_bv, chikou

# رجیستری
from typing import Dict

def registry() -> Dict[str, callable]:
    from .core import wma  # برای HMA
    def wrap(name: str, s: pd.Series):
        return {name: s.astype("float32")}
    def make_supertrend(df, period: int = 10, multiplier: float = 3.0, **_):
        return wrap(f"supertrend_{period}_{multiplier}", supertrend(df["high"], df["low"], df["close"], period, multiplier))
    def make_adx(df, n: int = 14, **_):
        pdi, mdi, adxv, adxr = adx_di(df["high"], df["low"], df["close"], n)
        return {f"pdi_{n}": pdi, f"mdi_{n}": mdi, f"adx_{n}": adxv, f"adxr_{n}": adxr}
    def make_aroon(df, n: int = 25, **_):
        up, down, osc = aroon(df["high"], df["low"], n)
        return {f"aroon_up_{n}": up, f"aroon_down_{n}": down, f"aroon_osc_{n}": osc}
    def make_kama(df, col: str = "close", n: int = 10, fast: int = 2, slow: int = 30, **_):
        return wrap(f"kama_{col}_{n}_{fast}_{slow}", kama(df[col], n, fast, slow))
    def make_dema(df, col: str = "close", n: int = 20, **_):
        return wrap(f"dema_{col}_{n}", dema(df[col], n))
    def make_tema(df, col: str = "close", n: int = 20, **_):
        return wrap(f"tema_{col}_{n}", tema(df[col], n))
    def make_hma(df, col: str = "close", n: int = 20, **_):
        return wrap(f"hma_{col}_{n}", hma(df[col], n))
    def make_ichimoku(df, tenkan: int = 9, kijun: int = 26, span_b: int = 52, **_):
        conv, base, sa, sb, ch = ichimoku(df["high"], df["low"], df["close"], tenkan, kijun, span_b)
        return {f"ichi_tenkan_{tenkan}": conv, f"ichi_kijun_{kijun}": base, f"ichi_span_a_{tenkan}_{kijun}": sa, f"ichi_span_b_{span_b}": sb, f"ichi_chikou_{kijun}": ch}
    return {
        "supertrend": make_supertrend,
        "adx": make_adx,
        "aroon": make_aroon,
        "kama": make_kama,
        "dema": make_dema,
        "tema": make_tema,
        "hma": make_hma,
        "ichimoku": make_ichimoku,
    }