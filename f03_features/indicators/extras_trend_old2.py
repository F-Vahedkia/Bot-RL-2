# f03_features/indicators/extras_trend.py
# Status in (Bot-RL-2): Reviewed at 1405/01/21

"""اندیکاتورهای روندی تکمیلی: Supertrend, ADX/DI/ADXR, Aroon, KAMA/DEMA/TEMA/HMA, Ichimoku
"""
# =============================================================================
# Imports & Logger
# ============================================
from __future__ import annotations
from typing import Dict, Callable
import numpy as np
import pandas as pd
import logging

from .core import atr, ema, sma, wma, rsi

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# ============================================================
# Utility
# ============================================================

def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Safe division avoiding division-by-zero."""
    b_safe = b.replace(0.0, np.nan)
    out = a / (b_safe + eps)
    return out


# ============================================================
# Trend Indicators
# ============================================================

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    """Supertrend indicator."""
    atr_val = atr(high, low, close, n=period)
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val

    trend = pd.Series(index=close.index, dtype=float)
    direction = 1
    trend.iloc[0] = close.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i - 1]:
            direction = 1
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction = -1

        if direction > 0:
            trend.iloc[i] = lower.iloc[i]
        else:
            trend.iloc[i] = upper.iloc[i]
    return trend.astype("float32")


def adx_di(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
):
    """Directional Movement System."""
    up_move = high.diff()
    down_move = -low.diff()

    # --- Old formulas --------------------------
    # plus_dm = pd.Series(
    #     np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
    #     index=high.index)
    # minus_dm = pd.Series(
    #     np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
    #     index=high.index)

    # --- Below is better than above ------------
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = atr(high, low, close, n=1)

    plus_di = 100 * ema(pd.Series(plus_dm), n) / ema(tr, n)
    minus_di = 100 * ema(pd.Series(minus_dm), n) / ema(tr, n)

    dx = 100 * _safe_div((plus_di - minus_di).abs(), (plus_di + minus_di))
    adx_val = ema(dx, n)
    adxr = (adx_val + adx_val.shift(n)) / 2

    return (
        plus_di.astype("float32"),
        minus_di.astype("float32"),
        adx_val.astype("float32"),
        adxr.astype("float32")
    )


def aroon(
    high: pd.Series,
    low: pd.Series,
    n: int = 25,
):
    """Aroon indicator."""
    rolling_high = high.rolling(n)
    rolling_low = low.rolling(n)

    days_since_high = rolling_high.apply(lambda x: n - 1 - np.argmax(x))
    days_since_low = rolling_low.apply(lambda x: n - 1 - np.argmin(x))

    up = 100 * (n - days_since_high) / n
    down = 100 * (n - days_since_low) / n
    osc = up - down
    return (
        up.astype("float32"),
        down.astype("float32"),
        osc.astype("float32")
    )


def kama(
    s: pd.Series,
    n: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    change = s.diff(n).abs()
    volatility = s.diff().abs().rolling(n).sum()

    er = _safe_div(change, volatility)
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=s.index, dtype=float)
    kama.iloc[0] = s.iloc[0]

    for i in range(1, len(s)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (s.iloc[i] - kama.iloc[i - 1])
    return kama.astype("float32")


def dema(s: pd.Series, n: int = 20) -> pd.Series:
    """Double Exponential Moving Average."""
    e1 = ema(s, n)
    e2 = ema(e1, n)
    return (2 * e1 - e2).astype("float32")


def tema(s: pd.Series, n: int = 20) -> pd.Series:
    """Triple Exponential Moving Average."""
    e1 = ema(s, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return (3 * e1 - 3 * e2 + e3).astype("float32")


def hma(s: pd.Series, n: int = 20) -> pd.Series:
    """Hull Moving Average."""
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    w1 = wma(s, half)
    w2 = wma(s, n)
    raw = 2 * w1 - w2
    return wma(raw, sqrt_n).astype("float32")


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
):
    """Ichimoku core lines."""
    tenkan_line = ((high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2).astype("float32")
    kijun_line = ((high.rolling(kijun).max() + low.rolling(kijun).min()) / 2).astype("float32")

    span_a = ((tenkan_line + kijun_line) / 2).astype("float32")
    span_b_line = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2).astype("float32")
    return tenkan_line, kijun_line, span_a, span_b_line


# ============================================================
# Lightweight Trend Features
# ============================================================

def ma_slope(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    method: str = "ema",
    norm: str = "stdev",
) -> pd.Series:
    """Moving Average slope."""
    price = df[price_col]
    if method == "ema":
        ma = ema(price, window)
    else:
        ma = sma(price, window)
    slope = ma.diff()

    if norm == "stdev":
        scale = price.rolling(window).std()
        slope = _safe_div(slope, scale).astype("float32")
    elif norm == "price":
        slope = _safe_div(slope, price.abs()).astype("float32")
    return slope.rename(f"ma_slope_{method}_{window}")


def rsi_zone(
    df: pd.DataFrame,
    price_col: str = "close",
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
) -> pd.DataFrame:
    """RSI zone classification."""
    r = rsi(df[price_col], period)
    over = (r >= overbought).astype(int)
    under = (r <= oversold).astype(int)
    mid = ((r > oversold) & (r < overbought)).astype(int)

    return pd.DataFrame({
        f"rsi_{period}": r.astype("float32"),
        f"rsi_overbought_{period}": over,
        f"rsi_oversold_{period}": under,
        f"rsi_mid_{period}": mid,
    })


# ============================================================
# Registry
# ============================================================

def registry() -> Dict[str, Callable]:
    """Feature registry."""
    def make_supertrend(df, period=10, multiplier=3.0, **_):
        s = supertrend(df.high, df.low, df.close, period, multiplier)
        return {f"supertrend_{period}_{multiplier}": s}

    def make_adx(df, n=14, **_):
        pdi, mdi, adx_val, adxr = adx_di(df.high, df.low, df.close, n)
        return {
            f"pdi_{n}": pdi,
            f"mdi_{n}": mdi,
            f"adx_{n}": adx_val,
            f"adxr_{n}": adxr,
        }

    def make_aroon(df, n=25, **_):
        up, down, osc = aroon(df.high, df.low, n)
        return {
            f"aroon_up_{n}": up,
            f"aroon_down_{n}": down,
            f"aroon_osc_{n}": osc,
        }

    def make_kama(df, col="close", n=10, fast=2, slow=30, **_):
        s = kama(df[col], n, fast, slow)
        return {f"kama_{col}_{n}_{fast}_{slow}": s}

    def make_dema(df, col="close", n=20, **_):
        s = dema(df[col], n)
        return {f"dema_{col}_{n}": s}

    def make_tema(df, col="close", n=20, **_):
        s = tema(df[col], n)
        return {f"tema_{col}_{n}": s}

    def make_hma(df, col="close", n=20, **_):
        s = hma(df[col], n)
        return {f"hma_{col}_{n}": s}

    def make_ichimoku(df, tenkan=9, kijun=26, span_b=52, **_):
        conv, base, sa, sb = ichimoku(df.high, df.low, tenkan, kijun, span_b)
        return {
            f"ichi_tenkan_{tenkan}": conv,
            f"ichi_kijun_{kijun}": base,
            f"ichi_span_a_{tenkan}_{kijun}": sa,
            f"ichi_span_b_{span_b}": sb,
        }

    def make_ma_slope(df, **_):
        s = ma_slope(df)
        return {s.name: s}

    def make_rsi_zone(df, **_):
        out = rsi_zone(df)
        return {c: out[c] for c in out.columns}

    return {
        "supertrend": make_supertrend,
        "adx": make_adx,
        "aroon": make_aroon,
        "kama": make_kama,
        "dema": make_dema,
        "tema": make_tema,
        "hma": make_hma,
        "ichimoku": make_ichimoku,
        "ma_slope": make_ma_slope,
        "rsi_zone": make_rsi_zone,
    }

