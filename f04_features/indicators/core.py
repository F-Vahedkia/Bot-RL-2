# f04_features/indicators/core.py
# -*- coding: utf-8 -*-
"""
اندیکاتورهای پایه (Bot-RL-1) – بدون look-ahead
"""
from __future__ import annotations
from typing import Dict, Tuple, Literal
import numpy as np
import pandas as pd
from .utils import true_range

# میانگین‌ها

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=n).mean().astype("float32")

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean().astype("float32")

def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype="float32")
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True).astype("float32")

# اسیلاتورها

'''
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.astype("float32")
'''
# ============================================================
# Canonical RSI (core-level)  — مرجع واحد
# ============================================================
def rsi(
    close: pd.Series,
    length: int = 14,
    method: Literal["ema", "wilders"] = "ema"
) -> pd.Series:
    """
    توضیح آموزشی (فارسی):
      این پیاده‌سازی «نسخهٔ مرجع» RSI در هسته است تا همهٔ ماژول‌ها از همین استفاده کنند.
      - ورودی: سری قیمت بسته‌شدن.
      - طول: پیش‌فرض 14.
      - method: 'ema' (پیش‌فرض) یا 'wilders' (تقریب کلاسیک RSI).
      خروجی: سری RSI با طول برابر ورودی.

    English:
      Canonical RSI implementation for the whole project.
      Returns a pandas Series aligned with `close`.
    """
    if close is None or len(close) == 0:
        return pd.Series(dtype=float)

    c = pd.Series(close).astype(float)
    delta = c.diff()

    if method.lower() == "wilders":
        # Wilder's smoothing (EMA with alpha=1/length)
        gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    else:
        # EMA-like smoothing (پیش‌فرض)
        gain = delta.clip(lower=0).ewm(alpha=1/length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    out = (100 - (100 / (1 + rs))).fillna(50.0)
    return out.rename("RSI")


def roc(close: pd.Series, n: int = 10) -> pd.Series:
    # return (close.pct_change(n) * 100.0).astype("float32")   added 040705
    return (close.pct_change(n, fill_method=None) * 100.0).astype("float32")

# ATR/TR

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().astype("float32")

# MACD

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = (ema_fast - ema_slow).astype("float32")
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean().astype("float32")
    hist = (macd_line - signal_line).astype("float32")
    return macd_line, signal_line, hist

# باندها

def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = sma(close, n)
    sd = close.rolling(n, min_periods=n).std().astype("float32")
    upper = (ma + k * sd).astype("float32")
    lower = (ma - k * sd).astype("float32")
    return ma, upper, lower

def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, m: float = 2.0):
    ema_mid = ema(close, n)
    atr_val = atr(high, low, close, n)
    upper = (ema_mid + m * atr_val).astype("float32")
    lower = (ema_mid - m * atr_val).astype("float32")
    return ema_mid, upper, lower

# استوکاستیک

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d: int = 3):
    lowest = low.rolling(n, min_periods=n).min()
    highest = high.rolling(n, min_periods=n).max()
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = k.astype("float32")
    dline = k.rolling(d, min_periods=d).mean().astype("float32")
    return k, dline

# CCI/MFI/OBV/Williams%R

def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = sma(tp, n)
    md = (tp - sma_tp).abs().rolling(n, min_periods=n).mean()
    c = (tp - sma_tp) / (0.015 * md.replace(0, np.nan))
    return c.astype("float32")

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume.fillna(0.0)
    pos = np.where(tp.diff() > 0, mf, 0.0)
    neg = np.where(tp.diff() < 0, mf, 0.0)
    pos_sum = pd.Series(pos, index=close.index).rolling(n, min_periods=n).sum()
    neg_sum = pd.Series(neg, index=close.index).rolling(n, min_periods=n).sum()
    ratio = pos_sum / neg_sum.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    return out.astype("float32")

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    dir_ = np.sign(close.diff().fillna(0.0))
    return (dir_ * volume.fillna(0.0)).cumsum().astype("float32")

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    highest = high.rolling(n, min_periods=n).max()
    lowest = low.rolling(n, min_periods=n).min()
    wr = -100.0 * (highest - close) / (highest - lowest).replace(0, np.nan)
    return wr.astype("float32")

# هیکن‌آشی و PSAR

def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series):
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = pd.Series(index=open_.index, dtype="float32")
    ha_open.iloc[0] = float(open_.iloc[0])
    for i in range(1, len(open_)):
        # ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0   added 040705
        ha_open.iloc[i] = np.float32((ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0)
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    return ha_open.astype("float32"), ha_high.astype("float32"), ha_low.astype("float32"), ha_close.astype("float32")

def parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series:
    h = high.values; l = low.values; n = len(h)
    if n == 0:
        return pd.Series([], dtype="float32", index=high.index)
    uptrend = True if h[1] > h[0] else False
    ep = h[0] if uptrend else l[0]
    sar = l[0] if uptrend else h[0]
    af = af_start
    out = np.empty(n, dtype="float32"); out[0] = sar
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if uptrend:
            sar = min(sar, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
        if uptrend:
            if l[i] < sar:
                uptrend = False; sar = ep; ep = l[i]; af = af_start
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            if h[i] > sar:
                uptrend = True; sar = ep; ep = h[i]; af = af_start
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
        out[i] = sar
    return pd.Series(out, index=high.index, dtype="float32")

# رجیستریِ core (نام → تابع سازندهٔ map از series)
IndicatorMap = Dict[str, callable]

def registry() -> IndicatorMap:
    def wrap(name: str, s: pd.Series) -> Dict[str, pd.Series]:
        return {name: s.astype("float32")}
    def make_rsi(df: pd.DataFrame, period: int = 14, **_):
        return wrap(f"rsi_{period}", rsi(df["close"], period))
    def make_sma(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return wrap(f"sma_{col}_{period}", sma(df[col], period))
    def make_ema(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return wrap(f"ema_{col}_{period}", ema(df[col], period))
    def make_wma(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return wrap(f"wma_{col}_{period}", wma(df[col], period))
    def make_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **_):
        line, sig, hist = macd(df["close"], fast, slow, signal)
        return {f"macd_{fast}_{slow}_{signal}": line, f"macd_signal_{fast}_{slow}_{signal}": sig, f"macd_hist_{fast}_{slow}_{signal}": hist}
    def make_bbands(df: pd.DataFrame, col: str = "close", period: int = 20, k: float = 2.0, **_):
        mid, up, lo = bollinger(df[col], period, k)
        return {f"bb_mid_{period}_{k}": mid, f"bb_up_{period}_{k}": up, f"bb_lo_{period}_{k}": lo}
    def make_keltner(df: pd.DataFrame, period: int = 20, m: float = 2.0, **_):
        mid, up, lo = keltner(df["high"], df["low"], df["close"], period, m)
        return {f"kelt_mid_{period}_{m}": mid, f"kelt_up_{period}_{m}": up, f"kelt_lo_{period}_{m}": lo}
    def make_stoch(df: pd.DataFrame, n: int = 14, d: int = 3, **_):
        k, dline = stochastic(df["high"], df["low"], df["close"], n, d)
        return {f"stoch_k_{n}_{d}": k, f"stoch_d_{n}_{d}": dline}
    def make_atr(df: pd.DataFrame, n: int = 14, **_):
        return wrap(f"atr_{n}", atr(df["high"], df["low"], df["close"], n))
    def make_tr(df: pd.DataFrame, **_):
        return {"tr": true_range(df["high"], df["low"], df["close"]) }
    def make_cci(df: pd.DataFrame, n: int = 20, **_):
        return wrap(f"cci_{n}", cci(df["high"], df["low"], df["close"], n))
    def make_mfi(df: pd.DataFrame, n: int = 14, **_):
        return wrap(f"mfi_{n}", mfi(df["high"], df["low"], df["close"], df["volume"], n))
    def make_obv(df: pd.DataFrame, **_):
        return wrap("obv", obv(df["close"], df["volume"]))
    def make_roc(df: pd.DataFrame, n: int = 10, **_):
        return wrap(f"roc_{n}", roc(df["close"], n))
    def make_wr(df: pd.DataFrame, n: int = 14, **_):
        return wrap(f"wr_{n}", williams_r(df["high"], df["low"], df["close"], n))
    def make_ha(df: pd.DataFrame, **_):
        o,h,l,c = heikin_ashi(df["open"], df["high"], df["low"], df["close"])
        return {"ha_open": o, "ha_high": h, "ha_low": l, "ha_close": c}
    def make_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2, **_):
        return {f"sar_{af_start}_{af_step}_{af_max}": parabolic_sar(df["high"], df["low"], af_start, af_step, af_max)}
    return {
        "rsi": make_rsi,
        "sma": make_sma,
        "ema": make_ema,
        "wma": make_wma,
        "macd": make_macd,
        "bbands": make_bbands,
        "keltner": make_keltner,
        "stoch": make_stoch,
        "atr": make_atr,
        "tr": make_tr,
        "cci": make_cci,
        "mfi": make_mfi,
        "obv": make_obv,
        "roc": make_roc,
        "wr": make_wr,
        "ha": make_ha,
        "sar": make_sar,
    }