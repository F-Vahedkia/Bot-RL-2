# -*- coding: utf-8 -*-
# f03_features/indicators/levels.py
# Status in (Bot-RL-2): Completed

"""Pivot های کلاسیک، S/R ساده مبتنی بر فراکتال، و فاصله تا سطوح فیبوناچی اخیر
"""
#==============================================================================
# Imports & Logger
#==============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from typing import List, Sequence, Dict, Optional, Tuple, Any
from numba import njit
from .core import atr
from .zigzag import zigzag as zig

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from f10_utils.config_loader import ConfigLoader
cfg = ConfigLoader().get_all()

""" --------------------------------------------------------------------------- OK Func1
speed=OK
Pivot های کلاسیک
"""
def pivots_classic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, ...]:
    pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
    r1 = 2*pivot - low.shift(1)
    s1 = 2*pivot - high.shift(1)
    r2 = pivot + (high.shift(1) - low.shift(1))
    s2 = pivot - (high.shift(1) - low.shift(1))
    r3 = high.shift(1) + 2*(pivot - low.shift(1))
    s3 = low.shift(1) - 2*(high.shift(1) - pivot)
    return \
        pivot.astype("float32"), \
        r1.astype("float32"), \
        s1.astype("float32"), \
        r2.astype("float32"), \
        s2.astype("float32"), \
        r3.astype("float32"), \
        s3.astype("float32")

""" --------------------------------------------------------------------------- OK Func2
Speed=OK
with two logical branches for small and large data sets
"""
def fractal_points(high: pd.Series, low: pd.Series, k: int = 2) -> tuple[pd.Series, pd.Series]:
    if len(high) < 200_000:   
        """ --- برای داده‌های کوچک‌تر از 200000، نسخه قدیمی سریع‌تر است
        Compute simple fractals (high/low) using a rolling window [i-k : i+k+1].
        فراکتال ساده برای استفاده در توابع sr_distance, fibo_levels
        قرارداد خروجی:
        - 1.0  → فرکتال تأییدشده
        - 0.0  → قطعاً فرکتال نیست
        - NaN  → هنوز قابل قضاوت نیست (warm-up یا ناحیهٔ look-ahead)
        """
        window = 2*k + 1
        hh = (high.rolling(window, center=True, min_periods=window)
                .apply(lambda x: float(np.argmax(x) == k), raw=True)
            )
        ll = (low.rolling(window, center=True, min_periods=window)
                .apply(lambda x: float(np.argmin(x) == k), raw=True)
            )
        # حذف صریح کندل‌های ابتدایی و انتهایی که هنوز آینده یا گذشته را ندارند
        if k > 0:
            hh.iloc[ :k] = np.nan
            hh.iloc[-k:] = np.nan
            ll.iloc[ :k] = np.nan
            ll.iloc[-k:] = np.nan
        return hh.astype("float32"), ll.astype("float32")
    
    else:
        """ --- برای داده‌های بزرگ‌تر از 200000، نسخه Numba سریع‌تر است
        Speed=FAST (Numba JIT)
        Compute simple fractals (high/low) using a rolling window [i-k : i+k+1].
        Numba JIT optimized for speed.
        
        Parameters
        ----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        k : int, default=2
            Fractal window size (total window = 2k+1)
        
        Returns
        -------
        tuple[pd.Series, pd.Series]
            hh, ll: fractal highs and lows as float32 Series
        """
        high_vals = high.values.astype(np.float32)
        low_vals = low.values.astype(np.float32)
        n = len(high_vals)

        # Preallocate output
        hh_arr = np.full(n, np.nan, dtype=np.float32)
        ll_arr = np.full(n, np.nan, dtype=np.float32)

        # Numba-optimized inner loop
        @njit
        def compute_fractals(h_vals, l_vals, hh_out, ll_out, k, n):
            for i in range(k, n-k):
                h_window = h_vals[i-k:i+k+1]
                l_window = l_vals[i-k:i+k+1]

                hh_out[i] = 1.0 if np.argmax(h_window) == k else 0.0
                ll_out[i] = 1.0 if np.argmin(l_window) == k else 0.0
            return hh_out, ll_out

        hh_arr, ll_arr = compute_fractals(high_vals, low_vals, hh_arr, ll_arr, k, n)

        return pd.Series(hh_arr, index=high.index), pd.Series(ll_arr, index=low.index)

""" --------------------------------------------------------------------------- OK Func3 (Not Used)
Speed=Slow
Logic=OK
فاصله قیمتی تا اخیرترین سطح حمایت/مقاومت
این تابع برای هر قیمت بسته شدن، قیمتهای اخیرترین لگ را بدست می‌آورد
"""
def sr_distance(close: pd.Series,
                high: pd.Series,
                low: pd.Series,
                k: int = 2,
                lookback: int = 500
) -> Tuple[pd.Series, pd.Series]:
    hh, ll = fractal_points(high, low, k)
    idx = close.index
    res = pd.Series(index=idx, dtype="float32")    # resistance
    sup = pd.Series(index=idx, dtype="float32")    # support
    for i in range(len(idx)):
        lo = max(0, i - lookback)               # نگاه به گذشته و ساخت حد پایین بازه مورد نظر
        new_hh = (hh.iloc[lo:i] == 1)           # ساخت فیلتر بولی در بازه موردنظر
        new_ll = (ll.iloc[lo:i] == 1)           # ساخت فیلتر بولی در بازه موردنظر
        prev_h = high.iloc[lo:i][new_hh]        # فراکتال‌های بالا در بازه موردنظر
        prev_l =  low.iloc[lo:i][new_ll]        # فراکتال‌های پایین در بازه موردنظر

        # Guard: ensure close time is after last fractal
        if len(prev_h) and len(prev_l) and \
            idx[i] > prev_h.index[-1] and \
            idx[i] > prev_l.index[-1]:
            r = prev_h.iloc[-1] - close.iloc[i]
            s = close.iloc[i] - prev_l.iloc[-1]
        else:
            r = np.nan
            s = np.nan
        
        res.iloc[i] = np.float32(r) if pd.notna(r) else np.nan # ذخیره در سری خروجی
        sup.iloc[i] = np.float32(s) if pd.notna(s) else np.nan # ذخیره در سری خروجی
    return  pd.Series(res, index=close.index, dtype="float32"), \
            pd.Series(sup, index=close.index, dtype="float32")

""" --------------------------------------------------------------------------- OK Func3
Speed=FAST (Numba JIT)
"""
def sr_distance_numba(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    k: int = 2,
    lookback: int = 500
) -> Tuple[pd.Series, pd.Series]:

    # --- fractals (numba version assumed correct) ---
    hh, ll = fractal_points(high, low, k)

    close_v = close.values.astype(np.float32)
    high_v  = high.values.astype(np.float32)
    low_v   = low.values.astype(np.float32)
    hh_v    = hh.values
    ll_v    = ll.values

    n = len(close_v)
    res_v = np.full(n, np.nan, dtype=np.float32)
    sup_v = np.full(n, np.nan, dtype=np.float32)

    @njit
    def compute_sr(close_v, high_v, low_v, hh_v, ll_v, res_v, sup_v, n, lookback):
        for i in range(n):
            lo = i - lookback
            if lo < 0:
                lo = 0

            last_h = np.nan
            last_l = np.nan

            # scan backward (bounded)
            for j in range(i - 1, lo - 1, -1):
                if hh_v[j] == 1.0:
                    last_h = high_v[j]
                if ll_v[j] == 1.0:
                    last_l = low_v[j]
                if not np.isnan(last_h) and not np.isnan(last_l):
                    res_v[i] = last_h - close_v[i]
                    sup_v[i] = close_v[i] - last_l
                    break

    compute_sr(close_v, high_v, low_v, hh_v, ll_v, res_v, sup_v, n, lookback)

    return  pd.Series(res_v, index=close.index, dtype="float32"), \
            pd.Series(sup_v, index=close.index, dtype="float32")
    
""" --------------------------------------------------------------------------- OK Func4 (Not Used)
speed=SLOW
    تولید سطوح فیبوناچی به صورت دیکشنری {ratio: level}.

    - استفاده از آخرین fractal high,low در بازه lookback
    - تشخیص جهت swing (صعودی یا نزولی)
    - خروجی: dict که کلیدها نسبت‌های فیبو و مقادیر سطح قیمت هستند
"""
def fibo_levels_slow(close: pd.Series,
                high: pd.Series,
                low: pd.Series,
                k: int = 2,
                cfg: Optional[dict] = None,
                lookback: int = 500,
) -> pd.DataFrame:

    # پیش فرض درصدهای فیبوناچی ---------------------------
    if cfg is not None:
        fb = (cfg.get("features") or {}).get("fibonacci") or {}
        ratios: List[float] = list(fb.get("retracement_ratios") or [])
    else:
        ratios = [0.236, 0.382, 0.500, 0.618, 0.786]

    # محاسبه فراکتال‌ها ------------------------------------
    hh, ll = fractal_points(high, low, k)  # Series contain 0 and 1
    cols = [f"fibo_{r}" for r in ratios]
    out = pd.DataFrame(np.nan, index=close.index, columns=cols, dtype="float32")

    for i in range(len(close)):
        lo = max(k, i - lookback)
        new_hh = (hh.iloc[lo:i] == 1)
        new_ll = (ll.iloc[lo:i] == 1)
        h_fractals = high.iloc[lo:i][new_hh]   # آخرین fractal high ها
        l_fractals =  low.iloc[lo:i][new_ll]   # آخرین fractal low ها

        if (i < k) or h_fractals.empty or l_fractals.empty:
            continue

        # قیمت و زمان سقف و کف اخیر
        last_high = h_fractals.iloc[-1]
        last_low = l_fractals.iloc[-1]
        time_high = h_fractals.index[-1]
        time_low = l_fractals.index[-1]
        rng = abs(last_high - last_low)
        
        if rng == 0:
            continue

        # تعیین جهت swing
        if time_low < time_high:     # برای روند صعودی
            for r in ratios:
                out.iat[i, cols.index(f"fibo_{r}")] = float(last_high - r * rng)
        else:                        # برای روند نزولی
            for r in ratios:
                out.iat[i, cols.index(f"fibo_{r}")] = float(last_low + r * rng)
    return out

""" --------------------------------------------------------------------------- OK Func4
speed=OK
    Compute Fibonacci retracement levels based on the most recent completed swing
    defined by fractal highs and lows, optimized for performance using NumPy
    vectorization.

    This function is a high-performance equivalent of `fibo_levels`, producing
    identical numerical results while significantly reducing execution time.
    It preserves the original Fibonacci logic:
      - No Fibonacci levels are produced until a valid swing leg is formed
        (i.e., both a high and a low fractal exist within lookback).
      - Swing direction is determined by the temporal order of the last low
        and last high fractals.
      - Levels are computed per-bar and aligned with the input time index.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed by time.
    high : pd.Series
        High prices indexed by time.
    low : pd.Series
        Low prices indexed by time.
    k : int, default=2
        Fractal window size passed to `fractal_points`.
    cfg : dict, optional
        Configuration dictionary. If provided, Fibonacci retracement ratios
        are read from:
            cfg["features"]["fibonacci"]["retracement_ratios"]
        Otherwise, default ratios are used.
    lookback : int, default=500
        Maximum number of bars to look back when searching for the last
        valid fractal high and low.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by the same index as `close`, containing one column
        per Fibonacci retracement level (e.g. fibo_0.236, fibo_0.382, ...).
        Values are NaN until a valid swing leg is available.

    Notes
    -----
    - This implementation is not fully stateless-vectorized due to the
      inherently stateful nature of swing/leg detection, but it minimizes
      Python-level loops and leverages NumPy for per-ratio computations.
    - Numerical output is guaranteed to match `fibo_levels` for the same inputs.
    - Designed for high-frequency use (e.g. M1 data) in feature pipelines.
"""
def fibo_levels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    k: int = 2,
    cfg: Optional[dict] = None,
    lookback: int = 500,
) -> pd.DataFrame:

    # --- Fibonacci ratios as numpy array ---
    if cfg is not None:
        fb = (cfg.get("features") or {}).get("fibonacci") or {}
        ratios = np.array(list(fb.get("retracement_ratios") or []), dtype=np.float32)
    else:
        ratios = np.array([0.236, 0.382, 0.5, 0.618, 0.786], dtype=np.float32)

    # --- Output columns names ---
    cols = [f"fibo_{r:.3f}" for r in ratios]
    n = len(close)

    # --- fractals ---
    hh, ll = fractal_points(high, low, k)

    # --- output array ---
    out = np.full((n, len(ratios)), np.nan, dtype=np.float32)

    # --- precompute high/low values as arrays for indexing ---
    high_vals = high.values
    low_vals = low.values
    hh_vals = (hh.values==1)
    ll_vals = (ll.values==1)

    for i in range(n):
        lo = max(k, i - lookback)
        h_idx = np.flatnonzero(hh_vals[lo:i])
        l_idx = np.flatnonzero(ll_vals[lo:i])

        if i < k or len(h_idx) == 0 or len(l_idx) == 0:
            continue

        last_high_idx = h_idx[-1] + lo
        last_low_idx = l_idx[-1] + lo

        last_high = high_vals[last_high_idx]
        last_low = low_vals[last_low_idx]
        rng = abs(last_high - last_low)
        if rng == 0:
            continue

        if last_low_idx < last_high_idx:
            out[i, :] = last_high - ratios * rng
        else:
            out[i, :] = last_low + ratios * rng

    # ===== موقتی و فقط برای کنترل برنامه
    df_new = pd.concat([pd.Series(high, name="high"),
                        pd.Series(low, name="low"),
                        pd.Series(close, name="close"),
                        pd.Series(hh, name="hh"),
                        pd.Series(ll, name="ll"),
                        pd.DataFrame(out, index=close.index, columns=cols)
                        ], axis=1)
    col_names = ["high", "low", "close", "hh", "ll"] + cols
    return pd.DataFrame(df_new, index=close.index, columns=col_names, dtype=np.float32)
    # ===== پایان موقتی
    
    # --- convert to DataFrame ---
    # return pd.DataFrame(out, index=close.index, columns=cols, dtype=np.float32)

""" --------------------------------------------------------------------------- OK Func5
"""
def registry() -> Dict[str, callable]:
    
    def make_pivots(df, **_):
        p, r1, s1, r2, s2, r3, s3 = pivots_classic(df["high"], df["low"], df["close"])
        return {"pivot": p,
                "pivot_r1": r1, "pivot_s1": s1,
                "pivot_r2": r2, "pivot_s2": s2,
                "pivot_r3": r3, "pivot_s3": s3
                }
    
    def make_sr(df, k: int = 2, lookback: int = 500, **_):
        r, s = sr_distance_numba(df["close"], df["high"], df["low"], k, lookback)
        return {f"sr_res_{k}_{lookback}": r,
                f"sr_sup_{k}_{lookback}": s
                }
    
    def make_fibo(df, k: int = 2, lookback: int = 500, cfg: Optional[dict] = None, **_):
        out = fibo_levels(df["close"], df["high"], df["low"], k=k, cfg=cfg, lookback=lookback)
        return out.to_dict(orient="series")  # خروجی dict مانند سایر توابع
    
    return {"pivots": make_pivots,
            "sr"    : make_sr,
            "fibo"  : make_fibo
            }

# --- New Added ----------------------------------------------------- 040607
"""
افزودنی‌های Levels برای هم‌افزایی با فیبوناچی و امتیازدهی Confluence.
- round_levels(...): تولید سطوح رُند حول یک لنگر
تابع round_levels عیناً از این فایل به فایل utils منتقل شد
- compute_adr(...): محاسبهٔ ADR روزانه و نگاشت به تایم‌استمپ‌های درون‌روزی
- adr_distance_to_open(...): فاصلهٔ نرمال‌شدهٔ قیمت تا «بازِ روز» با ADR
- sr_overlap_score(...): امتیاز همپوشانی یک قیمت با سطوح S/R (۰..۱)

نکته‌ها:
- ورودی‌ها ایندکس زمانی UTC و مرتب فرض شده‌اند.
- همهٔ توابع افزایشی‌اند و چیزی از API موجود را تغییر نمی‌دهند.
"""

""" --------------------------------------------------------------------------- Func6
ADR (Average Daily Range)
"""
def compute_adr(df: pd.DataFrame, window: int = 14, tz: str = "UTC") -> pd.Series:
    """
    ADR کلاسیک: میانگینِ (High-Low) روزانه روی پنجرهٔ rolling.
    - ابتدا OHLC روزانه را می‌سازد (بر اساس resample('1D'))
    - سپس میانگین rolling از دامنهٔ روزانه را می‌گیرد
    - در پایان سری ADR روزانه را به تایم‌استمپ‌های درون‌روزی ffill می‌کند

    ورودی: df با ستون‌های high/low (و بهتر است close برای resample صحیح)
    خروجی: Series با نام 'ADR_{window}' هم‌تراز با df.index
    """
    if not {"high", "low"}.issubset(df.columns):
        raise ValueError("DF must contain at least: high, low")

    # تبدیل به فریم روزانه
    daily = df[["high", "low"]].copy()
    daily = daily.tz_convert(tz) if (daily.index.tz is not None) else daily.tz_localize(tz)
    daily_ohl = pd.DataFrame({
        "hi": daily["high"].resample("1D", label="left", closed="left").max(),
        "lo": daily["low"].resample("1D", label="left", closed="left").min(),
    }).dropna()

    daily_range = (daily_ohl["hi"] - daily_ohl["lo"]).rename("daily_range")
    adr_daily = daily_range.rolling(window=window, min_periods=max(2, window // 2)).mean()
    adr_daily.name = f"ADR_{window}"

    # نگاشت ADR روزانه به ایندکس درون‌روزی با ffill
    adr_intraday = adr_daily.reindex(df.index, method="ffill")
    return adr_intraday

""" --------------------------------------------------------------------------- Func7
"""
def adr_distance_to_open(df: pd.DataFrame, adr: pd.Series, tz: str = "UTC") -> pd.DataFrame:
    """
    فاصلهٔ قیمت تا «بازِ روز» نرمال‌شده به ADR.
    خروجی ستون‌ها:
      - day_open: بازِ روز (نخستین close هر روز)
      - dist_abs: |price - day_open|
      - dist_pct_of_adr: 100 * dist_abs / ADR
    """
    if "close" not in df.columns:
        raise ValueError("DF must contain 'close' to compute day_open distance")

    px = df["close"].copy()
    px = px.tz_convert(tz) if (px.index.tz is not None) else px.tz_localize(tz)

    # بازِ روز = اولین close هر روز
    day_open_daily = px.resample("1D", label="left", closed="left").first().rename("day_open")
    day_open = day_open_daily.reindex(px.index, method="ffill")

    dist_abs = (px - day_open).abs().rename("dist_abs")
    adr_safe = adr.copy()
    adr_safe.replace(0.0, np.nan, inplace=True)
    dist_pct = (100.0 * dist_abs / adr_safe).rename("dist_pct_of_adr")

    out = pd.concat([day_open, dist_abs, dist_pct], axis=1)
    return out

""" --------------------------------------------------------------------------- Func8
S/R Overlap Score (0..1)
"""
def sr_overlap_score(price: float, sr_levels: Sequence[float], tol_pct: float = 0.05) -> float:
    """
    امتیاز همپوشانی قیمت با سطوح S/R:
      - اگر نزدیک‌ترین سطح در فاصلهٔ tol_pct (نسبت به قیمت) باشد → امتیاز ۰..۱ (هرچه نزدیک‌تر، امتیاز بالاتر)
      - اگر چند سطح داخل tol باشند، یک پاداش کوچک اضافه می‌شود (clip به ۱)

    پارامترها:
      price: قیمتِ ارزیابی
      sr_levels: لیست سطوح S/R
      tol_pct: آستانهٔ نسبی (مثلاً 0.05 یعنی 5%)

    خروجی: نمرهٔ ۰..۱
    """
    if not sr_levels:
        return 0.0

    tol_abs = abs(price) * tol_pct
    diffs = np.array([price - lv for lv in sr_levels], dtype=float)
    abs_diffs = np.abs(diffs)

    j = int(np.argmin(abs_diffs))
    min_dist = float(abs_diffs[j])

    if min_dist > tol_abs or tol_abs == 0.0:
        return 0.0

    # امتیاز پایه: نزدیکی خطی تا ۱
    base = 1.0 - (min_dist / tol_abs)

    # پاداش کوچک بابت تعداد سطوح در محدودهٔ tol
    k = int(np.sum(abs_diffs <= tol_abs))
    bonus = 0.1 * max(0, k - 1)

    score = min(1.0, max(0.0, base + bonus))
    return float(score)

# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                           1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
1  pivots_classic         --  --  --  --  ok  --  --  --  --  --  --
2  fractal_points         --  --  ok  ok  --  --  --  --  --  --  --
3  sr_distance            --  --  --  --  ok  --  --  --  --  --  --
4  fibo_levels            --  --  --  --  ok  --  --  --  --  --  --
5  registry               --  --  --  --  --  --  --  --  --  --  --
6  compute_adr            --  --  --  --  --  --  --  --  --  --  --
7  adr_distance_to_open   --  --  --  --  --  --  --  --  --  --  --
8  sr_overlap_score       --  --  --  --  --  --  --  --  --  --  --
"""


######################################################################################


###########################################################  3
# fibo_levels_slow, fibo_levels

# t3 = datetime.now()
# result1 = fibo_levels_slow(df3["close"], df3["high"], df3["low"], k=5, cfg=cfg, lookback=50)
# t4 = datetime.now()
# result1.to_csv("fibo_levels_slow.csv")
# print(f"Time taken to run 'fibo_levels_slow': {round((t4 - t3).total_seconds(), 1)} seconds")
# # print(result1.head(),"\n")

# t5 = datetime.now()
# result2 = fibo_levels(df3["close"], df3["high"], df3["low"], k=5, cfg=cfg, lookback=50)
# t6 = datetime.now()
# result2.to_csv("fibo_levels.csv")
# print(f"Time taken to run 'fibo_levels     ': {round((t6 - t5).total_seconds(), 1)} seconds")
# # print(result2.head())

# diff = result2.fillna(0) - result1.fillna(0)
# print("Max difference between 'fibo_levels_slow' and 'fibo_levels':", diff.abs().max().max())

###########################################################  2
# --- sr_distance

# t1 = datetime.now()
# res, sup = sr_distance(df3["close"], df3["high"], df3["low"], k=10)
# t2 = datetime.now()
# print(f"Time taken to run 'sr_distance': {round((t2 - t1).total_seconds(), 1):.2f} seconds")
# df1 = pd.DataFrame({"res": res, "sup": sup})
# df1.to_csv("sr_distance.csv")
# print(f"len(res): {len(res)}")

# t3 = datetime.now()
# res, sup = sr_distance_numba(df3["close"], df3["high"], df3["low"], k=10)
# t4 = datetime.now()
# print(f"Time taken to run 'sr_distance_numba': {round((t4 - t3).total_seconds(), 1):.2f} seconds")
# df1 = pd.DataFrame({"res": res, "sup": sup})
# df1.to_csv("sr_distance_numba.csv")
# print(f"len(res): {len(res)}")

###########################################################  1
# --- fractal_points

# t1 = datetime.now()
# hh, ll = fractal_points(df3["high"], df3["low"], k=10)
# t2 = datetime.now()
# print(f"Time taken to run 'fractal_points': {round((t2 - t1).total_seconds(), 1):.2f} seconds")
# df1 = pd.DataFrame({"hh": hh, "ll": ll})
# df1.to_csv("fractal_points.csv")
# print(f"len(hh): {len(hh)}")
# print(f"type(hh)): {type(hh)}")
# print(f"type(ll)): {type(ll)}")

###########################################################
