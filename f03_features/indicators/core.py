# -*- coding: utf-8 -*-
# f03_features/indicators/core.py
# Status in (Bot-RL-2): Reviewed before 1404/09/25
"""
اندیکاتورهای پایه - بدون look-ahead
"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from typing import Dict, Tuple, Literal
import numpy as np
import pandas as pd
from .utils import true_range

# ============================================================================= 1,2,3
# Averages
# ============================================================================= OK
def sma(s: pd.Series, n: int) -> pd.Series:
    """
    Simple Moving Average (SMA) with proper warm-up.
    - First n-1 rows are NaN
    """
    if s is None or len(s) < n:
        return pd.Series(dtype="float32")
    s_float = s.astype("float32")
    result = s_float.rolling(window=n, min_periods=n).mean()
    return result.astype("float32").rename(f"sma_{n}")

# ---------------------------------------------------------
def ema(s: pd.Series, n: int) -> pd.Series:
    """
    Exponential Moving Average (EMA) with proper warm-up.
    - First n-1 rows are NaN
    """
    if s is None or len(s) < n:
        return pd.Series(dtype="float32")
    s_float = s.astype("float32")
    result = s_float.ewm(span=n, adjust=False, min_periods=n).mean()
    result[:n-1] = np.nan  # ensure proper warm-up
    return result.astype("float32").rename(f"ema_{n}")

# ---------------------------------------------------------
def wma(s: pd.Series, n: int) -> pd.Series:
    """
    Weighted Moving Average (WMA) with proper warm-up.
    - First n-1 rows are NaN
    """
    if s is None or len(s) < n:
        return pd.Series(dtype="float32")
    s_float = s.astype("float32")
    w = np.arange(1, n + 1, dtype="float32")
    result = s_float.rolling(window=n, min_periods=n).apply(lambda x: np.dot(x, w)/w.sum(), raw=True)
    return result.astype("float32").rename(f"wma_{n}")

# ============================================================================= 4,5
# Ocilators
# ============================================================================= OK
# --- Canonical RSI (core-level) — Single Source ----------
def rsi(
    close: pd.Series,
    length: int = 14,
    method: Literal["ema", "wilders"] = "ema"
) -> pd.Series:

    """ Compute RSI (Relative Strength Index) with proper warm-up handling.
    - close: pd.Series of closing prices
    - length: lookback period
    - method: 'ema' (default) or 'wilders'
    Returns pd.Series with NaN for warm-up period.
    """

    if close is None or len(close) < 2:
        return pd.Series(dtype=float, name="RSI")

    c = pd.Series(close).astype(float)
    delta = c.diff()

    # gains and losses
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))

    if method.lower() == "wilders":
        # Wilder's smoothing (alpha = 1/length)
        gain = gain.ewm(alpha=1/length, adjust=False).mean()
        loss = loss.ewm(alpha=1/length, adjust=False).mean()
    else:
        # EMA-like smoothing
        gain = gain.ewm(span=length, adjust=False).mean()
        loss = loss.ewm(span=length, adjust=False).mean()

    # RS
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        rsi_values = 100 - 100 / (1 + rs)

    # skeep initial NaN for warm-up (first 'length' values)
    rsi_values[:length] = np.nan

    return rsi_values.rename("RSI").rename(f"rsi_{length}")

# ---------------------------------------------------------
def roc(close: pd.Series, n: int = 10) -> pd.Series:
    """
    Rate of Change (percentage) over n periods.
    - Returns percent change * 100 as float32.
    - First n rows are NaN (warm-up).
    """
    if close is None or len(close) < n + 1:
        return pd.Series(dtype="float32", name=f"roc_{n}")
    s = pd.Series(close).astype("float64")  # use float64 for numerics, convert later
    out = s.pct_change(periods=n, fill_method=None) * 100.0
    out[:n] = np.nan
    return out.astype("float32").rename(f"roc_{n}")

# ============================================================================= 6
# ATR/TR
# ============================================================================= OK
def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # Use this function for system decisions only
    """
    Average True Range (ATR) using EWM smoothing with alpha = 1/n (Wilder-like).
    - TR is computed per-bar (True Range).
    - ATR uses ewm(alpha=1/n, adjust=False, min_periods=n).
    - First n rows are NaN to ensure proper warm-up.
    """
    # rely on true_range(high, low, close) existing in module scope
    tr = true_range(high, low, close).astype("float64")
    # EWM with alpha=1/n, require min_periods=n
    if len(tr) < n:
        return pd.Series(dtype="float32", name=f"atr_{n}")
    atr_series = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    atr_series[:n] = np.nan  # warm-up
    return atr_series.astype("float32").rename(f"atr_{n}")

# ============================================================================= 7
# MACD
# ============================================================================= OK
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD line, Signal line, and Histogram with proper warm-up handling.
    Returns pd.Series with NaN for initial warm-up rows.
    """

    # EMA fast and slow with min_periods for correct warm-up
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    # MACD line
    macd_line = (ema_fast - ema_slow).astype("float32")
    macd_line[:slow] = np.nan  # ensure first 'slow' rows are NaN

    # Signal line: EMA on MACD line
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean().astype("float32")
    signal_line[:slow + signal - 1] = np.nan  # first value appears after slow + signal -1

    # Histogram: MACD - Signal
    hist = (macd_line - signal_line).astype("float32")
    hist[:slow + signal - 1] = np.nan  # align histogram with signal

    # Rename series
    macd_line = macd_line.rename(f"macd_{fast}_{slow}_{signal}")
    signal_line = signal_line.rename(f"macd_signal_{fast}_{slow}_{signal}")
    hist = hist.rename(f"macd_hist_{fast}_{slow}_{signal}")

    return macd_line, signal_line, hist

# ============================================================================= 8,9
# Bounds
# ============================================================================= OK
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:

    """
    Bollinger Bands (mid, upper, lower).
    - mid: SMA(n) (first n-1 rows NaN)
    - upper/lower: mid ± k * rolling_std(n) (same warm-up as mid)
    - returns (mid, upper, lower) as float32 Series with names.
    """
    if close is None or len(close) < n:
        return pd.Series(dtype="float32"), pd.Series(dtype="float32"), pd.Series(dtype="float32")

    mid = sma(close, n).astype("float64")  # uses sma() which ensures warm-up
    # use population std (ddof=0) for stability; require same min_periods as mid
    sd = close.rolling(window=n, min_periods=n).std(ddof=0)
    # ensure identical warm-up alignment
    sd[:n-1] = np.nan
    upper = (mid + k * sd).astype("float32")
    lower = (mid - k * sd).astype("float32")
    mid = mid.astype("float32").rename(f"bb_mid_{n}_{k}")
    upper = upper.rename(f"bb_up_{n}_{k}")
    lower = lower.rename(f"bb_lo_{n}_{k}")
    return mid, upper, lower

# ---------------------------------------------------------
def keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, m: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channel (mid = EMA(n), upper = mid + m * ATR(n), lower = mid - m*ATR).
    - Ensures warm-up: first n rows NaN (ATR uses n; EMA uses n -> we align to max).
    - Returns (mid, upper, lower) as float32 Series with names.
    """
    if close is None or len(close) < n:
        return pd.Series(dtype="float32"), pd.Series(dtype="float32"), pd.Series(dtype="float32")

    mid = ema(close, n).astype("float64")   # ema ensures warm-up
    atr_val = atr(high, low, close, n).astype("float64")
    warm = max(n, n)  # explicit, but kept for clarity
    mid[:warm] = np.nan
    atr_val[:warm] = np.nan
    upper = (mid + m * atr_val).astype("float32")
    lower = (mid - m * atr_val).astype("float32")
    mid = mid.astype("float32").rename(f"kelt_mid_{n}_{m}")
    upper = upper.rename(f"kelt_up_{n}_{m}")
    lower = lower.rename(f"kelt_lo_{n}_{m}")
    return mid, upper, lower

# ============================================================================= 10
# Stochastic
# ============================================================================= OK
def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d: int = 3
) -> Tuple[pd.Series, pd.Series]:

    """
    Stochastic Oscillator (%K and %D) – professional grade version.
    - %K = 100 * (close - lowest_n) / (highest_n - lowest_n)
    - %D = SMA(%K, d)
    - First n rows of %K are NaN (warm-up)
    - First n + d - 1 rows of %D are NaN (warm-up)
    - Both returned as float32 Series with proper naming.
    """

    length = len(close)
    if length < n:
        return (
            pd.Series(dtype="float32", name=f"stoch_k_{n}_{d}"),
            pd.Series(dtype="float32", name=f"stoch_d_{n}_{d}")
        )

    high_f = high.astype("float64")
    low_f = low.astype("float64")
    close_f = close.astype("float64")

    # Rolling highest / lowest
    lowest = low_f.rolling(window=n, min_periods=n).min()
    highest = high_f.rolling(window=n, min_periods=n).max()

    # %K calculation
    # Prevent divide-by-zero using replace
    k = 100.0 * (close_f - lowest) / (highest - lowest).replace(0, np.nan)
    k[:n] = np.nan                      # correct warm-up
    k = k.astype("float32").rename(f"stoch_k_{n}_{d}")

    # %D calculation (SMA over K)
    dline = k.rolling(window=d, min_periods=d).mean()
    dline[:n + d - 1] = np.nan          # correct warm-up for %D
    dline = dline.astype("float32").rename(f"stoch_d_{n}_{d}")

    return k, dline

# ============================================================================= 11,12,13,14
# CCI/MFI/OBV/Williams%R
# ============================================================================= OK
def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    tp = ((high + low + close) / 3.0).astype("float64")
    ma = sma(tp, n).astype("float64")
    md = (tp - ma).abs().rolling(window=n, min_periods=n).mean()
    c = (tp - ma) / (0.015 * md.replace(0, np.nan))
    c[: n - 1] = np.nan
    return c.astype("float32").rename(f"cci_{n}")

# ---------------------------------------------------------
def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    tp = ((high + low + close) / 3.0).astype("float64")
    vol = volume.fillna(0.0).astype("float64")
    mf = tp * vol
    pos = pd.Series(np.where(tp.diff() > 0, mf, 0.0), index=tp.index).rolling(window=n, min_periods=n).sum()
    neg = pd.Series(np.where(tp.diff() < 0, mf, 0.0), index=tp.index).rolling(window=n, min_periods=n).sum()
    ratio = pos / neg.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    out[:n] = np.nan
    return out.astype("float32").rename(f"mfi_{n}")

# ---------------------------------------------------------
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV)
    - OBV requires at least 1 previous close → warm-up 1 row (NaN)
    - Uses cumsum of volume * direction
    - Fully deterministic, float32 output
    """
    close_f = close.astype("float64")
    vol_f = volume.fillna(0.0).astype("float64")

    # direction of price movement
    direction = np.sign(close_f.diff())
    # ensure the first value is NaN (no previous candle)
    direction.iloc[0] = np.nan

    obv_raw = (direction * vol_f).cumsum()
    obv_raw.iloc[0] = np.nan  # proper warm-up: first row NaN

    return obv_raw.astype("float32").rename("obv")

# ---------------------------------------------------------
def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    Williams %R (WILLR)
    - First n rows NaN (warm-up)
    - Uses (highest_n - close) / (highest_n - lowest_n)
    - Returns float32, deterministic
    """
    high_f = high.astype("float64")
    low_f = low.astype("float64")
    close_f = close.astype("float64")

    highest = high_f.rolling(window=n, min_periods=n).max()
    lowest = low_f.rolling(window=n, min_periods=n).min()

    wr = -100.0 * (highest - close_f) / (highest - lowest).replace(0, np.nan)
    wr[:n] = np.nan  # proper warm-up

    return wr.astype("float32").rename(f"wr_{n}")

# ============================================================================= 15,16
# PSAR & heikin_ashi
# ============================================================================= OK
def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """
    Professional-grade Parabolic SAR (Wilder)
    - First TWO rows are NaN (true warm-up)
    - No fake values, fully deterministic
    - Exact EP/AF reset behavior on trend reversal
    """

    h = high.astype("float64").values
    l = low.astype("float64").values
    n = len(h)

    if n < 3:
        return pd.Series([np.nan] * n, index=high.index, dtype="float32")

    out = np.full(n, np.nan, dtype="float64")

    # --- determine initial trend (Wilder rule)
    # if current close > previous close → uptrend; else downtrend
    uptrend = h[1] > h[0]

    # --- initialize EP & SAR
    if uptrend:
        ep = h[1]          # highest high so far
        sar = l[0]         # SAR starts from previous low
    else:
        ep = l[1]          # lowest low so far
        sar = h[0]         # SAR starts from previous high

    af = af_start

    # warm-up: first two rows NaN (professional rule)
    out[0] = np.nan
    out[1] = np.nan

    # --- main loop from candle #2
    for i in range(2, n):

        # compute next SAR
        sar = sar + af * (ep - sar)

        # clamp SAR into allowed region (to avoid penetration)
        if uptrend:
            sar = min(sar, l[i-1], l[i-2])
        else:
            sar = max(sar, h[i-1], h[i-2])

        # --- Trend logic
        if uptrend:
            # reversal?
            if l[i] < sar:
                # switch to downtrend
                uptrend = False
                sar = ep       # SAR jumps to previous EP
                ep = l[i]      # new EP
                af = af_start  # reset AF
            else:
                # continue uptrend
                if h[i] > ep:
                    ep = h[i]
                    af = min(af + af_step, af_max)
        else:
            # reversal?
            if h[i] > sar:
                # switch to uptrend
                uptrend = True
                sar = ep
                ep = h[i]
                af = af_start
            else:
                # continue downtrend
                if l[i] < ep:
                    ep = l[i]
                    af = min(af + af_step, af_max)

        out[i] = sar

    return pd.Series(out.astype("float32"), index=high.index, name="sar")

# ---------------------------------------------------------
def heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
)-> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Heikin-Ashi OHLC (professional-grade).
    - ha_close = (open+high+low+close)/4
    - ha_open[0] = (open[0] + close[0]) / 2  (standard initialization)
    - ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    - ha_high = max(high, ha_open, ha_close)
    - ha_low  = min(low,  ha_open, ha_close)
    Outputs: (ha_open, ha_high, ha_low, ha_close) as float32 Series with proper names and index.
    """

    # guard clauses
    length = len(close)
    if length == 0:
        return (
            pd.Series(dtype="float32", name="ha_open"),
            pd.Series(dtype="float32", name="ha_high"),
            pd.Series(dtype="float32", name="ha_low"),
            pd.Series(dtype="float32", name="ha_close"),
        )

    # use float64 internally for numerical stability
    o = pd.Series(open_).astype("float64")
    h = pd.Series(high).astype("float64")
    l = pd.Series(low).astype("float64")
    c = pd.Series(close).astype("float64")

    # ha_close vectorized
    ha_close = ((o + h + l + c) / 4.0).rename("ha_close").astype("float64")

    # ha_open iterative (depends on previous ha_open and previous ha_close)
    ha_open = pd.Series(index=o.index, dtype="float64")
    # standard safe initialization: average of first real open and close
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0

    for i in range(1, length):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0

    # highs / lows: elementwise comparison (no look-ahead)
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1).rename("ha_high")
    ha_low  = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1).rename("ha_low")

    # finalize dtypes (float32 for efficiency) and return in canonical order
    return (
        ha_open.astype("float32").rename("ha_open"),
        ha_high.astype("float32"),
        ha_low.astype("float32"),
        ha_close.astype("float32"),
    )

# ============================================================================= 
# Registry
# =============================================================================

# رجیستریِ core (نام → تابع سازندهٔ map از series)
IndicatorMap = Dict[str, callable]

def registry() -> IndicatorMap:
    def wrap(name: str, s: pd.Series) -> Dict[str, pd.Series]:
        return {name: s.astype("float32")}
    #--- 1,2,3 ---
    def make_sma(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return wrap(f"sma_{col}_{period}", sma(df[col], period))
    def make_ema(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return wrap(f"ema_{col}_{period}", ema(df[col], period))
    def make_wma(df: pd.DataFrame, col: str = "close", period: int = 20, **_):
        return wrap(f"wma_{col}_{period}", wma(df[col], period))
    #--- 4,5,6 ---
    def make_rsi(df: pd.DataFrame, period: int = 14, **_):
        return wrap(f"rsi_{period}", rsi(df["close"], period))
    def make_roc(df: pd.DataFrame, n: int = 10, **_):
        return wrap(f"roc_{n}", roc(df["close"], n))
    def make_atr(df: pd.DataFrame, n: int = 14, **_):
        return wrap(f"atr_{n}", atr(df["high"], df["low"], df["close"], n))
    #--- 7,8,9 ---
    def make_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **_):
        line, sig, hist = macd(df["close"], fast, slow, signal)
        return {f"macd_{fast}_{slow}_{signal}": line, f"macd_signal_{fast}_{slow}_{signal}": sig, f"macd_hist_{fast}_{slow}_{signal}": hist}
    def make_bbands(df: pd.DataFrame, col: str = "close", period: int = 20, k: float = 2.0, **_):
        mid, up, lo = bollinger(df[col], period, k)
        return {f"bb_mid_{period}_{k}": mid, f"bb_up_{period}_{k}": up, f"bb_lo_{period}_{k}": lo}
    def make_keltner(df: pd.DataFrame, period: int = 20, m: float = 2.0, **_):
        mid, up, lo = keltner(df["high"], df["low"], df["close"], period, m)
        return {f"kelt_mid_{period}_{m}": mid, f"kelt_up_{period}_{m}": up, f"kelt_lo_{period}_{m}": lo}
    #--- 10,11,12 ---
    def make_stoch(df: pd.DataFrame, n: int = 14, d: int = 3, **_):
        k, dline = stochastic(df["high"], df["low"], df["close"], n, d)
        return {f"stoch_k_{n}_{d}": k, f"stoch_d_{n}_{d}": dline}
    def make_cci(df: pd.DataFrame, n: int = 20, **_):
        return wrap(f"cci_{n}", cci(df["high"], df["low"], df["close"], n))
    def make_mfi(df: pd.DataFrame, n: int = 14, **_):
        return wrap(f"mfi_{n}", mfi(df["high"], df["low"], df["close"], df["volume"], n))
    #--- 13,14,15 ---
    def make_obv(df: pd.DataFrame, **_):
        return wrap("obv", obv(df["close"], df["volume"]))
    def make_wr(df: pd.DataFrame, n: int = 14, **_):
        return wrap(f"wr_{n}", williams_r(df["high"], df["low"], df["close"], n))
    def make_sar(df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2, **_):
        return {f"sar_{af_start}_{af_step}_{af_max}": parabolic_sar(df["high"], df["low"], af_start, af_step, af_max)}
    #--- 16,17 ---
    def make_ha(df: pd.DataFrame, **_):
        o,h,l,c = heikin_ashi(df["open"], df["high"], df["low"], df["close"])
        return {"ha_open": o, "ha_high": h, "ha_low": l, "ha_close": c}
    def make_tr(df: pd.DataFrame, **_):
        return {"tr": true_range(df["high"], df["low"], df["close"]) }

    
    return {
        "sma": make_sma,
        "ema": make_ema,
        "wma": make_wma,

        "rsi": make_rsi,
        "roc": make_roc,
        "atr": make_atr,

        "macd": make_macd,
        "bbands": make_bbands,
        "keltner": make_keltner,

        "stoch": make_stoch,
        "cci": make_cci,
        "mfi": make_mfi,

        "obv": make_obv,
        "wr": make_wr,
        "sar": make_sar,

        "ha": make_ha,
        "tr": make_tr,  # 17: true_range
    }

# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
1  sma             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
2  ema             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
3  wma             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
4  rsi             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
5  roc             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
6  atr             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
7  macd            --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
8  bollinger       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
9  keltner         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
10 stochastic      --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
11 cci             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
12 mfi             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
13 obv             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
14 williams_r      --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
15 parabolic_sar   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
16 heikin_ashi     --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
17 registry        --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
"""