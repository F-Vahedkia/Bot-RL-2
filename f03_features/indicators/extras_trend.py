# f03_features/indicators/extras_trend.py
# Status: PRODUCTION-GRADE / WORLD-CLASS
# Reviewed & hardened
""" Functions in this file:
0. _ensure_series
0. _safe_div

1. supertrend  -> OK
2. adx_di      -> OK
3. aroon       -> 
4. kama        -> 
5. dema        -> 
6. tema        -> 
7. hma         -> 
8. ichimoku    -> 
9. ma_slope    -> 
10.rsi_zone    -> 

11.registry
"""

from __future__ import annotations
from typing import Dict, Literal, Mapping, Callable
import numpy as np
import pandas as pd
from numba import njit
import logging
from datetime import datetime

from f03_features.indicators.core import (
    sma, ema, wma,
    rsi as rsi_core,
    atr as atr_core
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# Internal helpers
# =============================================================================

def _ensure_series(
    x: pd.Series | np.ndarray,
    index: pd.Index,
    name: str | None = None,
    dtype: str | None = "float64",
) -> pd.Series:
    """
    Ensure x is a pandas.Series with a given index and (optionally) dtype.
    This prevents index misalignment / length explosions in Pandas ops.
    """
    if isinstance(x, pd.Series):
        s = x
        if not s.index.equals(index):
            s = s.reindex(index)
    else:
        s = pd.Series(x, index=index)
    if dtype is not None:
        s = s.astype(dtype)
    if name is not None:
        s.name = name
    return s


def _safe_div(
    num: pd.Series,
    den: pd.Series,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Numerically-safe division: num / (den + eps), preserving index.
    """
    den_safe = den.copy()
    den_safe = den_safe.where(den_safe.abs() > eps, np.nan)
    out = num / den_safe
    return out


# =============================================================================
# Supertrend (classic)
# ============================================================================= OK 05/01/24

def supertrend_numpy(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    """
    Supertrend indicator (causal, production safe).
    Returns float64 series aligned with input index.
    """
    h = high.to_numpy(dtype="float64", copy=False)
    l = low.to_numpy(dtype="float64", copy=False)
    c = close.to_numpy(dtype="float64", copy=False)

    atrv = atr_core(high, low, close, n=period).to_numpy(dtype="float64", copy=False)

    hl2 = (h + l) * 0.5
    m = float(multiplier)

    upper = hl2 + m * atrv
    lower = hl2 - m * atrv

    n = len(c)
    st = np.full(n, np.nan, dtype="float64")

    # پیدا کردن اولین ATR معتبر
    start = np.where(~np.isnan(atrv))[0]
    if len(start) == 0:
        return pd.Series(st, index=close.index, dtype="float64")

    i0 = start[0]
    st[i0] = upper[i0]

    for i in range(i0 + 1, n):
        prev = st[i - 1]
        if c[i] > prev:
            st[i] = lower[i] if lower[i] > prev else prev
        else:
            st[i] = upper[i] if upper[i] < prev else prev

    return pd.Series(st, index=close.index, dtype="float64")

@njit(cache=True)
def _supertrend_njit_core(high, low, close, atr, multiplier):

    n = len(close)
    st = np.empty(n, dtype=np.float64)

    for i in range(n):
        st[i] = np.nan

    hl2 = (high + low) * 0.5
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    start = -1
    for i in range(n):
        if not np.isnan(atr[i]):
            start = i
            break

    if start == -1:
        return st

    st[start] = upper[start]

    for i in range(start + 1, n):
        prev = st[i - 1]
        if close[i] > prev:
            if lower[i] > prev:
                st[i] = lower[i]
            else:
                st[i] = prev
        else:
            if upper[i] < prev:
                st[i] = upper[i]
            else:
                st[i] = prev

    return st

def supertrend_njit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:

    atr_series = atr_core(high, low, close, n=period)

    h = high.to_numpy(dtype=np.float64, copy=False)
    l = low.to_numpy(dtype=np.float64, copy=False)
    c = close.to_numpy(dtype=np.float64, copy=False)
    a = atr_series.to_numpy(dtype=np.float64, copy=False)

    st = _supertrend_njit_core(h, l, c, a, np.float64(multiplier))

    return pd.Series(st, index=close.index, dtype=np.float64)

# WRAPPER
def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    
    if high.size < 1_000_000:   # با سعی و خطا برابر با 1600000 شد.
        result = supertrend_numpy(high, low, close, period, multiplier)
    else:
        result = supertrend_njit(high, low, close, period, multiplier)
    
    return result.rename(f"supertrend_{period}_{multiplier}")


# =============================================================================
# ADX / +DI / -DI / ADXR (Welles Wilder)
# ============================================================================= OK 05/01/26

@njit(cache=True)
def _compute_dm_di_adx_adxr_njit(h, l, c, window):
    n = h.size
    eps = 1e-12

    # --- 0) TR ----------------------------------------------------- ok
    # first row is 1, and is not 0
    tr = np.empty(n)
    tr[1] = h[1] - l[1]  # first bar: standard TR definition

    for i in range(2, n):
        h_l = h[i] - l[i]
        h_pc = abs(h[i] - c[i - 1])
        l_pc = abs(l[i] - c[i - 1])
        tr[i] = max(h_l, h_pc, l_pc)

    tr[0] = np.nan

    # --- 0) ATR ---------------------------------------------------- ok
    # first row is 1, and is not 0
    atr = np.zeros(n, np.float64)
    alpha = 1.0 / window

    # warmup: SMA initial
    s = 0.0
    for i in range(1, window + 1 ):
        s += tr[i]
    s /= window
    atr[window] = s

    # recursive Wilder RMA
    for i in range(window + 1, n):
        atr[i] = (tr[i] * alpha) + (atr[i - 1] * (1.0 - alpha))

    # leading values before window = NaN
    for i in range(window):
        atr[i] = np.nan

    # --- 1) DM (Directional Movement) ------------------------------ ok
    plus_dm = np.zeros(n, np.float64)
    minus_dm = np.zeros(n, np.float64)

    for i in range(1, n):
        up = h[i] - h[i - 1]
        down = l[i - 1] - l[i]

        if up > down and up > 0:
            plus_dm[i] = up
        elif down > up and down > 0:
            minus_dm[i] = down
    
    # leading values before window = NaN
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan

    # --- 2) Wilder smoothing --------------------------------------- ok
    # RMA: "Running Moving Average" or "Wilder’s Smoothed Moving Average"
    plus_sm = np.zeros(n, np.float64)
    minus_sm = np.zeros(n, np.float64)
    acc_p = 0.0
    acc_m = 0.0

    for i in range(1, window + 1):
        acc_p += plus_dm[i]
        acc_m += minus_dm[i]

    # Initial value for Wilder
    plus_sm[window] = acc_p / window
    minus_sm[window] = acc_m / window

    for i in range(window + 1, n):
        plus_sm [i] = (1 - 1 / window) * plus_sm [i-1] + plus_dm [i] / window
        minus_sm[i] = (1 - 1 / window) * minus_sm[i-1] + minus_dm[i] / window

    # leading values before window = NaN
    for i in range(window):
        plus_sm[i] = np.nan
        minus_sm[i] = np.nan

    # --- 3) DI values (Directional Indicator) ---------------------- ok
    # --- 4) DX (Directional Index ) -------------------------------- ok
    plus_di = np.zeros(n, np.float64)
    minus_di = np.zeros(n, np.float64)
    dx = np.zeros(n, np.float64)

    for i in range(n): # range(window, n)
        # --- DI ---
        if atr[i] < eps:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
        else:
            plus_di[i] = 100.0 * (plus_sm[i] / atr[i])
            minus_di[i] = 100.0 * (minus_sm[i] / atr[i])

        # --- DX ---
        if (atr[i] < eps) or (plus_di[i]==0 and minus_di[i]==0):
            dx[i] = 0
        else:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    # --- 5) ADX (Average Directional Index) ------------------------ ok
    adx = np.zeros(n, np.float64)

    # Correct initialization – very important
    if 2 * window <= n:
        # مقدار اولیه = میانگین اولین window مقدار از سری DX
        acc = 0.0
        for i in range(window, 2 * window):
            acc += dx[i]

        # Initial value for Wilder
        adx[2 * window - 1] = acc / window

        # ادامه‌ی Wilder smoothing
        for i in range(2 * window, n):
            adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window

    # leading values before window = NaN
    for i in range(2 * window - 1):
        adx[i] = np.nan

    # --- 6) ADXR (Average Directional Movement Index Rating) ------- ok
    adxr = np.zeros(n, np.float64)

    # fill with NaN
    for i in range(n):
        adxr[i] = np.nan

    # ADXR starts at index: 3*window - 1
    start = 3 * window - 1

    if start < n:
        for i in range(start, n):
            adxr[i] = 0.5 * (adx[i] + adx[i - window])

    return (
        plus_di, minus_di, adx, adxr
        # tr, atr,           # 0) TR (for High, Low, Close), ATR (RMA of TR)- in range(1, n)
        # plus_dm, minus_dm, # 1) DM (Directional Movement)                 - in range(1, n)
        # plus_sm, minus_sm, # 2) Wilder smoothing on "plus_dm", "minus_dm" - in range(window, n)
        # plus_di, minus_di, # 3) DI values (Directional Indicator)         - in range(window, n)
        # dx,                # 4) DX (Directional Index )                   - in range(window, n)
        # adx,               # 5) ADX (Average Directional Index)           - in range(2 * window - 1, n)
        # adxr,              # 6) ADXR (Average Directional Index Rating)   - in range(3 * window - 1, n)
    )

def adx_di(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    New production‑grade ADX / +DI / -DI 
    using existing TR + ATR and a single @njit helper.
    from utils.py (no duplicated code).

    Parameters
    ----------
    high, low, close : pd.Series float-like
    window : int (default 14)
    eps : float, numerical stability

    Returns
    -------
    dict[str, pd.Series]
        {
            "plus_di":  pd.Series,
            "minus_di": pd.Series,
            "adx":      pd.Series,
            "adxr":     pd.Series,
        }
    """
    idx = close.index


    # 1) تبدیل ورودی به numpy
    h = high.to_numpy(np.float64)
    l = low.to_numpy(np.float64)
    c = close.to_numpy(np.float64)

    # 2) برای تست این تابع
    # r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11 = _compute_dm_di_adx_adxr_njit(h, l, c, window)
    # result = pd.DataFrame(
    #     np.column_stack([r1,r2,r3,r4, r5,r6,r7,r8,r9,r10,r11]),
    #     index=idx,
    #     dtype="float64",
    #     columns = ["tr", "atr",
    #                "plus_dm", "minus_dm", "plus_sm", "minus_sm",
    #                "plus_di", "minus_di", "dx", "adx", "adxr"]
    # )
    # return pd.DataFrame(result, index= idx, dtype="float64")

    # 3) فراخوانی تابع Numba واحد
    plus_di, minus_di, adx, adxr = _compute_dm_di_adx_adxr_njit(h, l, c, window)

    # 4) خروجی pandas Series
    return (
        pd.Series(plus_di,  index=idx, dtype="float64", name=f"plus_di_{window}"),
        pd.Series(minus_di, index=idx, dtype="float64", name=f"minus_di_{window}"),
        pd.Series(adx,      index=idx, dtype="float64", name=f"adx_{window}"),
        pd.Series(adxr,     index=idx, dtype="float64", name=f"adxr_{window}"),
    )


# =============================================================================
# Aroon
# ============================================================================= OK 05/01/27

""" Aroon Up / Down / Oscillator.

Parameters
----------
high, low : pd.Series
    High/low series with identical index.
n : int, default 25
    Lookback window.

Returns
-------
up : pd.Series (float32)
    Aroon Up (% time since highest high).
down : pd.Series (float32)
    Aroon Down (% time since lowest low).
osc : pd.Series (float32)
    Aroon Oscillator = up - down.
"""

# --- By numpy ----------------------------------
def aroon_numpy( high: pd.Series, low: pd.Series, n: int = 25
) -> tuple[pd.Series, pd.Series, pd.Series]:
  
    H = high.to_numpy()   # dtype=np.float64
    L = low.to_numpy()    # dtype=np.float64
    N = len(H)

    idx_up  = np.full(N, np.nan, dtype=np.float64)
    idx_down = np.full(N, np.nan, dtype=np.float64)

    for i in range(n - 1, N):
        window_high = H[i - n + 1 : i + 1][::-1]
        window_low  = L[i - n + 1 : i + 1][::-1]
        idx_up[i]   = np.argmax(window_high)
        idx_down[i] = np.argmin(window_low)

    den = float(n - 1)

    up   = 100.0 * (1 - idx_up   / den)
    down = 100.0 * (1 - idx_down / den)
    osc  = up - down

    return up, down, osc

# ---- By njit ----------------------------------
@njit(cache=True, fastmath=True)
def _aroon_njit_core(high: np.ndarray, low: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray ,np.ndarray]:
    
    N = len(high)
    idx_up = np.empty(N, dtype=np.float32)
    idx_down = np.empty(N, dtype=np.float32)

    # fill NaN بدنه
    for i in range(N):
        if i < n - 1:
            idx_up[i] = np.nan
            idx_down[i] = np.nan
        else:
            # همان منطق: برعکس پنجره برای قرار دادن کندل جاری در اندیس ۰
            window_high = high[i - n + 1 : i + 1][::-1]
            window_low  = low[i - n + 1 : i + 1][::-1]

            idx_up[i]   = np.argmax(window_high)
            idx_down[i] = np.argmin(window_low)

    den = float(n - 1)
    up   = 100.0 * (1 - idx_up   / den)
    down = 100.0 * (1 - idx_down / den)
    osc  = up - down

    return up, down, osc

def aroon_njit(high: pd.Series, low: pd.Series, n: int = 25
) -> tuple[pd.Series, pd.Series, pd.Series]:

    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    up, down, osc = _aroon_njit_core(h, l, n)
    return up, down, osc

# --- Wrapper -----------------------------------
def aroon(high: pd.Series, low: pd.Series, n: int = 25
) -> tuple[pd.Series, pd.Series, pd.Series]:
    
    # --- شرط کافی بودن طول سری‌ها ---
    if len(high) < n or len(low) < n:
        raise ValueError(
            f"Input series must have length >= window size n={n}. "
            f"Got len(high)={len(high)}, len(low)={len(low)}."
        )
    if len(high) < 200_000:
        up, down, osc = aroon_numpy(high, low, n)
    else:
        up, down, osc = aroon_njit(high, low, n)
    return (
        pd.Series(up,   index=high.index, dtype="float64", name=f"arron_up_{n}"),
        pd.Series(down, index=high.index, dtype="float64", name=f"arron_down_{n}"),
        pd.Series(osc,  index=high.index, dtype="float64", name=f"arron_osc_{n}"),
    )


# =============================================================================
# KAMA / DEMA / TEMA / HMA
# =============================================================================

# --- KAMA (Kaufman Adaptive Moving Average) --------------
def kama_orig(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA), causal implementation.

    Parameters
    ----------
    s : pd.Series
        Input price series.
    n : int, default 10
        Efficiency ratio lookback.
    fast : int, default 2
        Fast EMA equivalent length.
    slow : int, default 30
        Slow EMA equivalent length.

    Returns
    -------
    kama : pd.Series (float64)
        Adaptive moving average.
    """
    N = len(s)
    s = s.astype("float64")

    change = s.diff(n).abs()
    volatility = s.diff().abs().rolling(n, min_periods=n).sum()
    volatility = volatility.replace(0.0, np.nan)

    ef = _safe_div(change, volatility)     # ==> (Efficiency Ratio)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (ef * (fast_sc - slow_sc) + slow_sc) ** 2   # ==> (Smoothing Constant)

    out = pd.Series(index=s.index, dtype="float64")

    if N == 0:
        return out.astype("float64")

    out.iloc[:n] = s.iloc[:n]

    for i in range(n, N):
        prev = out.iloc[i - 1]
        out.iloc[i] = prev + sc.iloc[i] * (s.iloc[i] - prev)   # ==> (KAMA)

    return (
        change    .astype("float64").rename("change"),
        volatility.astype("float64").rename("volat"),
        ef        .astype("float64").rename("ef"),
        sc        .astype("float64").rename("sc"),
        out       .astype("float64").rename("out")
    )
    return out.astype("float64").rename(f"KAMA_{n}_{fast}_{slow}")

def kama_numpy(arr: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    # arr = arr.astype(np.float64)
    idx = arr.index
    arr = arr.to_numpy("float64")
    N = len(arr)

    out = np.full(N, np.nan, dtype=np.float64)
    if N == 0:
        return out

    # --- 1) change = abs( s[i] - s[i-n] ) ------------------------------
    change = np.full(N, np.nan, dtype=np.float64)
    change[n:] = np.abs(arr[n:] - arr[:-n])

    # --- 2) volatility = sum(abs(diff)) over last n --------------------
    # diff = abs(arr[i] - arr[i-1])
    diff = np.abs(arr[1:] - arr[:-1])
    vol = np.full(N, np.nan, dtype=np.float64)

    # rolling sum of size n on diff
    # volatility[i] = sum(diff[i-n+1 : i+1]) → corresponds to KAMA formula
    cumsum = np.zeros(N, dtype=np.float64)
    cumsum[1:] = np.cumsum(diff)

    for i in range(n, N):
        vol[i] = cumsum[i] - cumsum[i - n]

    # جلوگیری از تقسیم صفر
    vol[vol == 0.0] = np.nan

    # --- 3) ER (efficiency ratio) -------------------------------------
    ef = change / vol

    # --- 4) Smoothing Constants ---------------------------------------
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    sc = (ef * (fast_sc - slow_sc) + slow_sc) ** 2

    # --- 5) KAMA recursive --------------------------------------------
    # مقداردهی اولیه: مثل نسخه pandas → حفظ s[i] برای i < n
    out[:n] = arr[:n]

    for i in range(n, N):
        prev = out[i - 1]
        if np.isnan(sc[i]):
            out[i] = prev
        else:
            out[i] = prev + sc[i] * (arr[i] - prev)

    return pd.Series(out, index=idx, dtype=np.float64, name=f"KAMA_{n}_{fast}_{slow}")


@njit
def kama_njit_core(arr, n=10, fast=2, slow=30) -> np.ndarray:
    arr = arr.astype(np.float64)
    N = len(arr)

    out = np.empty(N, dtype=np.float64)
    for i in range(N):
        out[i] = np.nan

    if N == 0:
        return out

    # --- change = abs(s[i] - s[i-n]) ---
    change = np.empty(N, dtype=np.float64)
    for i in range(n):
        change[i] = np.nan
    for i in range(n, N):
        change[i] = abs(arr[i] - arr[i - n])

    # --- diff = abs(arr[i] - arr[i-1]) ---
    diff = np.empty(N - 1, dtype=np.float64)
    for i in range(N - 1):
        diff[i] = abs(arr[i + 1] - arr[i])

    # cumsum for volatility
    cumsum = np.empty(N, dtype=np.float64)
    cumsum[0] = 0.0
    for i in range(1, N):
        cumsum[i] = cumsum[i - 1] + diff[i - 1]

    # --- volatility = rolling sum of diff ---
    vol = np.empty(N, dtype=np.float64)
    for i in range(n):
        vol[i] = np.nan
    for i in range(n, N):
        vol[i] = cumsum[i] - cumsum[i - n]
        if vol[i] == 0:
            vol[i] = np.nan

    # --- ER ---
    ef = np.empty(N, dtype=np.float64)
    for i in range(N):
        ef[i] = change[i] / vol[i] if not np.isnan(vol[i]) else np.nan

    # --- smoothing constants ---
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)

    sc = np.empty(N, dtype=np.float64)
    for i in range(N):
        if np.isnan(ef[i]):
            sc[i] = np.nan
        else:
            sc[i] = (ef[i] * (fast_sc - slow_sc) + slow_sc) ** 2

    # --- output init (match pandas logic) ---
    for i in range(n):
        out[i] = arr[i]

    # --- recursive KAMA ---
    for i in range(n, N):
        prev = out[i - 1]
        if np.isnan(sc[i]):
            out[i] = prev
        else:
            out[i] = prev + sc[i] * (arr[i] - prev)

    return out

def kama_njit(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Production-grade pandas wrapper for KAMA (Numba core).

    Parameters
    ----------
    s : pd.Series
        Input price series (float compatible)
    n : int
        Efficiency ratio window
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period

    Returns
    -------
    pd.Series
        KAMA values with same index
    """

    if not isinstance(s, pd.Series):
        raise TypeError("Input must be a pandas Series")

    if len(s) == 0:
        return pd.Series(dtype=np.float64)

    # تبدیل به numpy بدون copy در صورت امکان
    values = s.to_numpy(dtype=np.float64, copy=False)

    # فراخوانی هسته numba
    kama_values = kama_njit_core(values, n=n, fast=fast, slow=slow)

    # ساخت خروجی pandas
    return pd.Series(
        kama_values, index=s.index, dtype=np.float64, name=f"KAMA_{n}_{fast}_{slow}"
    )

def kama(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    if len(s) < 200_000:
        return kama_numpy(arr=s, n=n, fast=fast, slow=slow)
    return kama_njit(s=s, n=n, fast=fast, slow=slow)

# --- DEMA (Double Exponential Moving Average) ------------
def dema(s: pd.Series, n: int = 20) -> pd.Series:
    """
    Double Exponential Moving Average (DEMA).

    Parameters
    ----------
    s : pd.Series
        Input series.
    n : int, default 20
        EMA length.

    Returns
    -------
    dema : pd.Series (float64)
    """
    s = s.astype("float64")
    e = ema(s, n).astype("float64")
    e2 = ema(e, n).astype("float64")
    out = 2.0 * e - e2
    return out.astype("float64")


# --- TEMA (Triple Exponential Moving Average) ------------
def tema(s: pd.Series, n: int = 20) -> pd.Series:
    """
    Triple Exponential Moving Average (TEMA).

    Parameters
    ----------
    s : pd.Series
        Input series.
    n : int, default 20
        EMA length.

    Returns
    -------
    tema : pd.Series (float64)
    """
    s = s.astype("float64")
    e1 = ema(s, n).astype("float64")
    e2 = ema(e1, n).astype("float64")
    e3 = ema(e2, n).astype("float64")
    out = 3.0 * e1 - 3.0 * e2 + e3
    return out.astype("float64")


# --- HMA (Hull Moving Average) ---------------------------
def hma(s: pd.Series, n: int = 20) -> pd.Series:
    """
    Hull Moving Average (HMA).

    Parameters
    ----------
    s : pd.Series
        Input series.
    n : int, default 20
        HMA length.

    Returns
    -------
    hma : pd.Series (float32)
    """
    s = s.astype("float64")
    n2 = max(2, n // 2)
    w1 = wma(s, n2).astype("float64")
    w2 = wma(s, n).astype("float64")
    diff = 2.0 * w1 - w2
    out = wma(diff, int(np.sqrt(n))).astype("float64")
    return out.astype("float64")


# =============================================================================
# Ichimoku (Tenkan / Kijun / Senkou A / Senkou B)
# NOTE: Chikou is intentionally excluded from the return to avoid look-ahead.
# =============================================================================

def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ichimoku Kinko Hyo core lines, anti look-ahead (no Chikou output).

    Parameters
    ----------
    high, low, close : pd.Series
        Price series with identical index.
    tenkan : int, default 9
        Tenkan-sen period.
    kijun : int, default 26
        Kijun-sen period / Senkou shift.
    span_b : int, default 52
        Senkou Span B period.

    Returns
    -------
    tenkan_sen : pd.Series (float32)
    kijun_sen : pd.Series (float32)
    span_a : pd.Series (float32)
        Senkou Span A, shifted +kijun.
    span_b_line : pd.Series (float32)
        Senkou Span B, shifted +kijun.

    Notes
    -----
    Chikou (lagging span) is not returned to avoid direct look-ahead bias
    in feature pipelines. If needed, it should be derived externally with
    proper shifting consistent with the ML/RL setup.
    """
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("high, low, close must share the same index")

    high = high.astype("float64")
    low = low.astype("float64")
    close = close.astype("float64")

    tenkan_sen = ((high.rolling(tenkan).max() +
                   low.rolling(tenkan).min()) / 2.0)
    kijun_sen = ((high.rolling(kijun).max() +
                  low.rolling(kijun).min()) / 2.0)

    span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    span_b_line = ((high.rolling(span_b).max() +
                    low.rolling(span_b).min()) / 2.0).shift(kijun)

    return (
        tenkan_sen.astype("float32"),
        kijun_sen.astype("float32"),
        span_a.astype("float32"),
        span_b_line.astype("float32"),
    )


# =============================================================================
# MA Slope (normalized)
# =============================================================================

def ma_slope(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    method: Literal["sma", "ema"] = "ema",
    norm: Literal["stdev", "price", "none"] = "stdev",
    eps: float = 1e-12,
) -> pd.Series:
    """
    Normalized moving-average slope, suitable as a trend-strength feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, must contain `price_col`.
    price_col : str, default "close"
        Column to use for price.
    window : int, default 20
        MA and normalization window.
    method : {"sma", "ema"}, default "ema"
        Moving average type.
    norm : {"stdev", "price", "none"}, default "stdev"
        Normalization scheme:
          - "stdev": slope / rolling std(price)
          - "price": slope / |price|
          - "none" : raw slope
    eps : float, default 1e-12
        Numerical epsilon for denominator.

    Returns
    -------
    slope : pd.Series (float32)
        Normalized slope with name f"ma_slope_{method}_{window}".
    """
    if price_col not in df.columns:
        raise ValueError(f"df must contain column '{price_col}'")

    px = df[price_col].astype("float64")

    if method == "ema":
        ma = px.ewm(
            span=window,
            adjust=False,
            min_periods=max(2, window // 2),
        ).mean()
    elif method == "sma":
        ma = px.rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")

    slope = ma.diff()

    if norm == "stdev":
        denom = px.rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).std()
        slope = _safe_div(slope, denom + eps)
    elif norm == "price":
        slope = _safe_div(slope, px.abs() + eps)
    elif norm == "none":
        pass
    else:
        raise ValueError("norm must be 'stdev', 'price', or 'none'")

    slope = slope.astype("float32")
    slope.name = f"ma_slope_{method}_{window}"
    return slope


# =============================================================================
# RSI Zone flags / score
# =============================================================================

def rsi_zone(
    df: pd.DataFrame,
    price_col: str = "close",
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
    mid: float = 50.0,
    band: float = 5.0,
) -> pd.DataFrame:
    """
    RSI zoning with binary flags (and optional score extension).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, must contain `price_col`.
    price_col : str, default "close"
        Column used for RSI computation.
    period : int, default 14
        RSI lookback length.
    overbought : float, default 70.0
        Threshold for overbought region.
    oversold : float, default 30.0
        Threshold for oversold region.
    mid : float, default 50.0
        Midline used for "mid zone".
    band : float, default 5.0
        +/- band around mid for mid-zone flag.

    Returns
    -------
    out : pd.DataFrame
        Columns:
        - rsi_value : float32
        - rsi_is_overbought : bool
        - rsi_is_oversold : bool
        - rsi_mid_zone : bool
    """
    if price_col not in df.columns:
        raise ValueError(f"df must contain column '{price_col}'")

    px = df[price_col].astype("float64")
    rsi = rsi_core(px, length=period).astype("float32")

    is_ob = (rsi >= overbought)
    is_os = (rsi <= oversold)
    is_mid = (rsi.sub(mid).abs() <= band)

    out = pd.DataFrame(
        {
            "rsi_value": rsi.astype("float32"),
            "rsi_is_overbought": is_ob.astype(bool),
            "rsi_is_oversold": is_os.astype(bool),
            "rsi_mid_zone": is_mid.astype(bool),
        },
        index=df.index,
    )
    return out


# =============================================================================
# Registry
# =============================================================================

def registry() -> Mapping[str, Callable[..., Dict[str, pd.Series]]]:
    """
    Registry of extra trend indicators, production-grade.

    All returned Series are:
    - index-aligned with the input df
    - dtype float32 (for numeric features)
    - strictly causal (no look-ahead)
    """

    def _align(s: pd.Series, target_index: pd.Index) -> pd.Series:
        # اگر ایندکس‌ها یکی باشند، reindex هیچ سرباری ندارد و سریع عبور می‌کند.
        if not s.index.equals(target_index):
            s = s.reindex(target_index)
        return s

    def make_supertrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
        **_,
    ) -> Dict[str, pd.Series]:
        st = supertrend(df["high"], df["low"], df["close"], period, multiplier)
        return {f"supertrend_{period}_{multiplier}": _align(st, df.index)}

    def make_adx(
        df: pd.DataFrame,
        n: int = 14,
        **_,
    ) -> Dict[str, pd.Series]:
        pdi, mdi, adxv, adxr = adx_di(df["high"], df["low"], df["close"], n)
        idx = df.index
        return {
            f"pdi_{n}": _align(pdi, idx),
            f"mdi_{n}": _align(mdi, idx),
            f"adx_{n}": _align(adxv, idx),
            f"adxr_{n}": _align(adxr, idx),
        }

    def make_aroon(
        df: pd.DataFrame,
        n: int = 25,
        **_,
    ) -> Dict[str, pd.Series]:
        up, down, osc = aroon(df["high"], df["low"], n)
        idx = df.index
        return {
            f"aroon_up_{n}": _align(up, idx),
            f"aroon_down_{n}": _align(down, idx),
            f"aroon_osc_{n}": _align(osc, idx),
        }

    def make_kama(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 10,
        fast: int = 2,
        slow: int = 30,
        **_,
    ) -> Dict[str, pd.Series]:
        s = kama(df[col], n, fast, slow)
        return {f"kama_{col}_{n}_{fast}_{slow}": _align(s, df.index)}

    def make_dema(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 20,
        **_,
    ) -> Dict[str, pd.Series]:
        s = dema(df[col], n)
        return {f"dema_{col}_{n}": _align(s, df.index)}

    def make_tema(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 20,
        **_,
    ) -> Dict[str, pd.Series]:
        s = tema(df[col], n)
        return {f"tema_{col}_{n}": _align(s, df.index)}

    def make_hma(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 20,
        **_,
    ) -> Dict[str, pd.Series]:
        s = hma(df[col], n)
        return {f"hma_{col}_{n}": _align(s, df.index)}

    def make_ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        span_b: int = 52,
        **_,
    ) -> Dict[str, pd.Series]:
        conv, base, sa, sb = ichimoku(
            df["high"], df["low"], df["close"], tenkan, kijun, span_b,
        )
        idx = df.index
        return {
            f"ichi_tenkan_{tenkan}": _align(conv, idx),
            f"ichi_kijun_{kijun}": _align(base, idx),
            f"ichi_span_a_{tenkan}_{kijun}": _align(sa, idx),
            f"ichi_span_b_{span_b}": _align(sb, idx),
        }

    def make_ma_slope(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        method: Literal["sma", "ema"] = "ema",
        norm: Literal["stdev", "price", "none"] = "stdev",
        **_,
    ) -> Dict[str, pd.Series]:
        s = ma_slope(
            df=df,
            price_col=price_col,
            window=window,
            method=method,
            norm=norm,
        )
        name = s.name if s.name else f"ma_slope_{method}_{window}"
        return {name: _align(s, df.index)}

    def make_rsi_zone(
        df: pd.DataFrame,
        price_col: str = "close",
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        mid: float = 50.0,
        band: float = 5.0,
        **_,
    ) -> Dict[str, pd.Series]:
        rz = rsi_zone(
            df=df,
            price_col=price_col,
            period=period,
            overbought=overbought,
            oversold=oversold,
            mid=mid,
            band=band,
        )
        idx = df.index
        out: Dict[str, pd.Series] = {}
        for col in rz.columns:
            out[col] = _align(rz[col], idx)
        return out

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

# =============================================================================
