# f03_features/indicators/extras_channel_old1.py
# Status in (Bot-RL-2): Reviewed at 1405/01/21

"""Channel and volatility derived features (production-grade)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict

from .core import bollinger, keltner

# =============================================================================
# Utilities
# =============================================================================

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0.0, np.nan)
    return (a / b).astype("float32")

# =============================================================================
# Donchian Channel
# =============================================================================

def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    upper = high.rolling(n, min_periods=n).max()
    lower = low.rolling(n, min_periods=n).min()
    mid = (upper + lower) / 2.0
    return (
        upper.astype("float32"),
        mid.astype("float32"),
        lower.astype("float32"),
    )


def donchian_width(high: pd.Series, low: pd.Series, n: int = 20):
    upper, _, lower = donchian(high, low, n)
    return (upper - lower).astype("float32")


def donchian_position(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20):
    upper, _, lower = donchian(high, low, n)
    width = (upper - lower).replace(0.0, np.nan)
    pos = (close - lower) / width
    return pos.astype("float32")


def donchian_breakout(close: pd.Series, high: pd.Series, low: pd.Series, n: int = 20):
    upper, _, lower = donchian(high, low, n)
    up = (close > upper.shift(1)).astype("float32")
    down = (close < lower.shift(1)).astype("float32")
    return up, down

# =============================================================================
# Chaikin Volatility
# =============================================================================

def chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10, roc: int = 10):

    hl = high - low
    ema_range = hl.ewm(span=n, adjust=False, min_periods=n).mean()
    prev = ema_range.shift(roc).replace(0.0, np.nan)
    cv = 100.0 * (ema_range - prev) / prev
    return cv.astype("float32")

# =============================================================================
# Bollinger derived features
# =============================================================================

def bollinger_position(close: pd.Series, n: int = 20, k: float = 2.0):

    mid, upper, lower = bollinger(close, n=n, k=k)
    width = (upper - lower).replace(0.0, np.nan)
    pos = (close - lower) / width
    return pos.astype("float32")

# =============================================================================
# Keltner derived features
# =============================================================================

def keltner_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, atr_mult: float = 2.0):

    mid, upper, lower = keltner(high, low, close, n=n, m=atr_mult)
    width = (upper - lower).replace(0.0, np.nan)
    pos = (close - lower) / width
    return pos.astype("float32")

# =============================================================================
# Registry
# =============================================================================

def registry() -> Dict[str, callable]:

    def make_donchian(df, n: int = 20, **_):
        up, mid, lo = donchian(df["high"], df["low"], n)
        width = (up - lo).astype("float32")
        pos = donchian_position(df["close"], df["high"], df["low"], n)
        brk_up, brk_dn = donchian_breakout(df["close"], df["high"], df["low"], n)
        return {
            f"donch_up_{n}": up,
            f"donch_mid_{n}": mid,
            f"donch_lo_{n}": lo,
            f"donch_width_{n}": width,
            f"donch_pos_{n}": pos,
            f"donch_brk_up_{n}": brk_up,
            f"donch_brk_dn_{n}": brk_dn,
        }

    def make_ch_vol(df, n: int = 10, roc: int = 10, **_):
        return {
            f"chaikin_vol_{n}_{roc}": chaikin_volatility(
                df["high"], df["low"], n, roc
            )
        }

    def make_boll_pos(df, n: int = 20, k: float = 2.0, **_):
        return {
            f"boll_pos_{n}_{k}": bollinger_position(
                df["close"], n=n, k=k
            )
        }

    def make_kelt_pos(df, n: int = 20, atr_mult: float = 2.0, **_):
        return {
            f"kelt_pos_{n}_{atr_mult}": keltner_position(
                df["high"], df["low"], df["close"], n=n, atr_mult=atr_mult
            )
        }

    return {
        "donchian": make_donchian,
        "chaikin_vol": make_ch_vol,
        "bollinger_position": make_boll_pos,
        "keltner_position": make_kelt_pos,
    }
