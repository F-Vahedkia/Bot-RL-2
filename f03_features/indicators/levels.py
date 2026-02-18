# f03_features/indicators/levels.py

#==============================================================================
# Imports & Logger
#==============================================================================
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import njit
from datetime import datetime
from typing import List, Sequence, Dict, Optional, Tuple, Any
from .core import atr
from .zigzag2 import zigzag_mtf_adapter

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
سطوح حمایت و مقاومت ثابت را از روی لِگ‌های زیگزاگ می‌سازد و آن‌ها را روی تمام کندل‌های هر لگ پخش می‌کند.

اگر لگ صعودی باشد → مقدار کف ابتدای لگ = Support
اگر لگ نزولی باشد → مقدار سقف ابتدای لگ = Resistance

این مقدار از ابتدای لگ تا انتهای همان لگ، برای همهٔ کندل‌ها ثابت می‌ماند.
"""
def sr_from_zigzag_legs_orig(
    df: pd.DataFrame,
    *,
    tf: str,
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
) -> pd.DataFrame:

    zz = zigzag_mtf_adapter(
        high=df["high"],
        low=df["low"],
        tf_higher=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
    )

    legs = zz.attrs.get("legs", [])
    idx = df.index
    n = len(df)

    sup = pd.Series(np.nan, index=idx, dtype=np.float32)
    res = pd.Series(np.nan, index=idx, dtype=np.float32)

    for leg in legs:
        s = leg["start_ltf_pos"]
        e = leg["end_ltf_pos"]
        ts = leg["start_ts"]

        if leg["direction"] > 0:
            sup.iloc[s:e] = np.float32(df.at[ts, "low"])
        else:
            res.iloc[s:e] = np.float32(df.at[ts, "high"])

    # --- extend last valid leg to end of dataframe (ONLY ONCE) using last extremum ---
    if legs and extend_last_leg:
        last = legs[-1]
        s = last["end_ltf_pos"]    # اولین کندل بعد از آخرین کندل واقعی لگ آخر
        e = n                      # انتهای دیتافریم

        if last["direction"] < 0:
            sup.iloc[s:e] = np.float32(df["low"].iloc[s])
        else:
            res.iloc[s:e] = np.float32(df["high"].iloc[s])

    return pd.DataFrame(
        {"sr_support": sup, "sr_resistance": res},
        index=idx,
        dtype=np.float32,
    )

def sr_from_zigzag_legs_njit(
    df: pd.DataFrame,
    *,
    tf: str,
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
) -> pd.DataFrame:
    """
    Numba-optimized version of sr_from_zigzag_legs.
    API و خروجی دقیقاً مشابه نسخه اصلی است.
    """
    from numba import njit
    from .zigzag import zigzag_mtf_adapter

    # --- Run zigzag_mtf_adapter ---
    zz = zigzag_mtf_adapter(
        high=df["high"],
        low=df["low"],
        tf_higher=tf,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
    )

    legs = zz.attrs.get("legs", [])
    n = len(df)

    sup_arr = np.full(n, np.nan, dtype=np.float32)
    res_arr = np.full(n, np.nan, dtype=np.float32)

    if not legs:
        return pd.DataFrame({"sr_support": sup_arr, "sr_resistance": res_arr},
                            index=df.index, dtype=np.float32)

    # Prepare leg data for Numba
    leg_array = np.array([
        (leg["start_ltf_pos"], leg["end_ltf_pos"], leg["direction"])
        for leg in legs
        if leg["start_ltf_pos"] < n
    ], dtype=np.int64)

    high_vals = df["high"].values.astype(np.float32)
    low_vals = df["low"].values.astype(np.float32)

    @njit
    def fill_sr(sup_arr, res_arr, leg_array, high_vals, low_vals):
        for i in range(leg_array.shape[0]):
            s = leg_array[i, 0]
            e = leg_array[i, 1]
            direction = leg_array[i, 2]
            if e > len(sup_arr):
                e = len(sup_arr)
            if direction > 0:
                val = low_vals[s]
                for j in range(s, e):
                    sup_arr[j] = val
            else:
                val = high_vals[s]
                for j in range(s, e):
                    res_arr[j] = val
        return sup_arr, res_arr

    sup_arr, res_arr = fill_sr(sup_arr, res_arr, leg_array, high_vals, low_vals)

    # --- extend last valid leg to end of dataframe (ONLY ONCE) using last extremum ---
    if legs and extend_last_leg:
        last_leg = legs[-1]
        s_last = last_leg["end_ltf_pos"]   # اولین کندل بعد از آخرین کندل واقعی لگ آخر
        e_last = n                         # انتهای دیتافریم
        direction_last = last_leg["direction"]

        if s_last < n:
            if direction_last < 0:
                val = low_vals[s_last]
                sup_arr[s_last:e_last] = val
            else:
                val = high_vals[s_last]
                res_arr[s_last:e_last] = val

    return pd.DataFrame({"sr_support": sup_arr, "sr_resistance": res_arr},
                        index=df.index, dtype=np.float32)

def sr_from_zigzag_legs(
    df: pd.DataFrame,
    *,
    tf: str,
    depth: int,
    deviation: float,
    backstep: int,
    extend_last_leg: bool = False,
    _njit_threshold: int = 1_000_000,
) -> pd.DataFrame:
    """
    Smart wrapper:
    - small DF  -> pandas implementation
    - large DF  -> njit implementation
    """
    if len(df) < _njit_threshold:
        return sr_from_zigzag_legs_orig(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
            extend_last_leg=extend_last_leg,
        )
    else:
        return sr_from_zigzag_legs_njit(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
            extend_last_leg=extend_last_leg,
        )

""" --------------------------------------------------------------------------- OK Func3
فاصله نرمال‌شده قیمت پایانی تا سطوح حمایت و مقاومت فعال.
نرمال‌سازی بر اساس ATR (Average True Range) انجام می‌شود.

پارامترها:
-----------
df : pd.DataFrame
    دیتافریم شامل ستون‌های ['high', 'low', 'close'] و ایندکس زمانی
sr : pd.DataFrame
    دیتافریم خروجی sr_from_zigzag_legs شامل ستون‌های
    ['sr_support', 'sr_resistance']
atr_window : int
    طول پنجره برای محاسبه ATR
eps : float
    مقدار کوچک برای جلوگیری از تقسیم بر صفر

خروجی:
-------
pd.DataFrame با ستون‌های:
- dist_to_support_norm
- dist_to_resistance_norm
"""
def sr_distance_from_levels(
    df: pd.DataFrame,
    sr: pd.DataFrame,
    *,
    atr_window: int = 14,
    eps: float = 1e-8,
) -> pd.DataFrame:

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # --- محاسبه ATR ---
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=1).mean().astype(np.float32)

    # --- فاصله نرمال شده ---
    dist_sup = ((close - sr["sr_support"]) / (atr + eps)).astype(np.float32)
    dist_res = ((sr["sr_resistance"] - close) / (atr + eps)).astype(np.float32)

    # --- NaN propagation ---
    dist_sup[sr["sr_support"].isna()] = np.nan
    dist_res[sr["sr_resistance"].isna()] = np.nan

    return pd.DataFrame(
        {
            "dist_to_support_norm": dist_sup,
            "dist_to_resistance_norm": dist_res,
        },
        index=df.index,
        dtype=np.float32,
    )

""" --------------------------------------------------------------------------- OK Func4
برای استفاده داخلی است
"""
def _zigzag_leg_mask_orig(zz: pd.Series) -> pd.Series:
    legs = zz.attrs.get("legs", [])
    if not legs:
        return pd.Series(False, index=zz.index, dtype=bool)

    mask = pd.Series(False, index=zz.index)
    n = len(zz)

    for leg in legs:
        s = leg["start_ltf_pos"]
        e = leg["end_ltf_pos"]
        if s >= n:
            continue
        if e > n:
            e = n
        mask.iloc[s:e] = True
    return mask.astype(bool)

def _zigzag_leg_mask_njit(zz: pd.Series) -> pd.Series:
    """
    Numba-optimized version of zigzag_leg_mask.
    Returns boolean mask True for all indices covered by zigzag legs.
    """
    n = len(zz)
    mask = np.zeros(n, dtype=np.bool_)

    legs = zz.attrs.get("legs", [])
    if not legs:
        return pd.Series(mask, index=zz.index, dtype=bool)

    # Prepare leg data as NumPy array for njit
    leg_array = np.array([(leg["start_ltf_pos"], leg["end_ltf_pos"]) for leg in legs], dtype=np.int64)

    @njit
    def fill_mask(mask_arr, leg_arr):
        for i in range(leg_arr.shape[0]):
            s = leg_arr[i, 0]
            e = leg_arr[i, 1]
            if s >= mask_arr.shape[0]:
                continue
            if e > mask_arr.shape[0]:
                e = mask_arr.shape[0]
            for j in range(s, e):
                mask_arr[j] = True
        return mask_arr

    mask = fill_mask(mask, leg_array)
    return pd.Series(mask, index=zz.index, dtype=bool)

def _zigzag_leg_mask(
    zz: pd.Series,
    _njit_threshold: int = 1_400_000,
) -> pd.Series:
    if len(zz) < _njit_threshold:
        return _zigzag_leg_mask_orig(zz=zz)
    else:
        return _zigzag_leg_mask_njit(zz=zz)

""" --------------------------------------------------------------------------- OK Func5
"""
def fibo_levels_from_legs_orig(
    df: pd.DataFrame,
    zz: pd.Series,
    ratios: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Compute Fibonacci levels based on completed zigzag legs (metadata).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["high", "low", "close"].
    zz : pd.Series
        Output of zigzag_mtf_adapter with attrs["legs"].
    ratios : Sequence[float], optional
        Fibonacci retracement ratios. Default: [0.236, 0.382, 0.5, 0.618, 0.786]

    Returns
    -------
    pd.DataFrame
    Columns: fibo_<ratio> with index=df.index
    """
    if ratios is None:
        from f10_utils.config_loader import load_config
        cfg = load_config()
        feat = cfg.get("features", {})
        fibo = feat.get("fibonacci", {})
        ratios = fibo.get("retracement_ratios", None)
        if ratios is None:
            ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    cols = [f"fibo_{r:.3f}" for r in ratios]
    n = len(df)
    out = np.full((n, len(ratios)), np.nan, dtype=np.float32)
    high_vals = df["high"].values
    low_vals = df["low"].values

    for leg in zz.attrs.get("legs", []):
        s = leg["start_ltf_pos"]
        e = leg["end_ltf_pos"]
        direction = leg["direction"]

        if s >= n:
            continue

        start_price = df.iloc[s  ]["low" ] if direction > 0 else df.iloc[s  ]["high"]
        end_price   = df.iloc[e-1]["high"] if direction > 0 else df.iloc[e-1]["low" ]

        rng = abs(end_price - start_price)
        if rng == 0:
            continue

        if direction > 0:
            # صعودی: fibo زیر سقف
            out[s:e, :] = np.array([end_price - r*rng for r in ratios], dtype=np.float32)
        else:
            # نزولی: fibo بالای کف
            out[s:e, :] = np.array([end_price + r*rng for r in ratios], dtype=np.float32)

    return pd.DataFrame(out, index=df.index, columns=cols, dtype=np.float32)


def fibo_levels_from_legs_njit(
    df: pd.DataFrame,
    zz: pd.Series,
    ratios: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    Fibonacci levels based on zigzag legs (metadata) - Numba-optimized.

    Identical output to fibo_levels_from_legs, but much faster for millions of rows.
    """
    # --- Ratios ---
    if ratios is None:
        from f10_utils.config_loader import load_config
        cfg = load_config()
        feat = cfg.get("features", {})
        fibo = feat.get("fibonacci", {})
        ratios = fibo.get("retracement_ratios", None)
        if ratios is None:
            ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    cols = [f"fibo_{r:.3f}" for r in ratios]
    n = len(df)
    out = np.full((n, len(ratios)), np.nan, dtype=np.float32)
    high_vals = df["high"].values.astype(np.float32)
    low_vals  = df["low" ].values.astype(np.float32)

    legs = zz.attrs.get("legs", [])

    # --- Numba-compatible array version of leg processing ---
    leg_data = []
    for leg in legs:
        s = leg["start_ltf_pos"]
        e = leg["end_ltf_pos"]
        direction = leg["direction"]
        if s >= n or e <= s:
            continue
        start_price = low_vals[s] if direction > 0 else high_vals[s]
        end_price   = high_vals[e-1] if direction > 0 else low_vals[e-1]
        rng = abs(end_price - start_price)
        if rng == 0:
            continue
        leg_data.append((s, e, direction, start_price, end_price, rng))

    # --- Convert to NumPy arrays for njit ---
    if not leg_data:
        return pd.DataFrame(out, index=df.index, columns=cols, dtype=np.float32)

    leg_array = np.array(leg_data, dtype=np.float32)
    
    ratios_arr = np.array(ratios, dtype=np.float32)

    @njit
    def fill_fibo(out_arr, leg_arr, ratios_arr):
        for i in range(leg_arr.shape[0]):
            s = int(leg_arr[i, 0])
            e = int(leg_arr[i, 1])
            direction = leg_arr[i, 2]
            end_price = leg_arr[i, 4]
            rng = leg_arr[i, 5]
            for j in range(len(ratios_arr)):
                if direction > 0:
                    val = end_price - ratios_arr[j]*rng
                else:
                    val = end_price + ratios_arr[j]*rng
                for k in range(s, e):
                    out_arr[k, j] = val
        return out_arr

    out = fill_fibo(out, leg_array, ratios_arr)
    return pd.DataFrame(out, index=df.index, columns=cols, dtype=np.float32)


""" --------------------------------------------------------------------------- =Func6
"""
def registry() -> Dict[str, callable]:
    
    # --- func1 -------------------------------------------
    def make_pivots(df, **_):
        p, r1, s1, r2, s2, r3, s3 = pivots_classic(df["high"], df["low"], df["close"])
        return {"pivot": p,
                "pivot_r1": r1, "pivot_s1": s1,
                "pivot_r2": r2, "pivot_s2": s2,
                "pivot_r3": r3, "pivot_s3": s3
                }
    
    # --- func2 -------------------------------------------
    def make_sr_zigzag(df, tf, depth, deviation, backstep, **_):
        return sr_from_zigzag_legs(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
        ).to_dict(orient="series")

    # --- func3 -------------------------------------------
    def make_sr_distance(df, tf, depth, deviation, backstep, atr_window=14, **_):
        # ابتدا SR levels بساز
        sr = sr_from_zigzag_legs(
            df,
            tf=tf,
            depth=depth,
            deviation=deviation,
            backstep=backstep,
        )
        # سپس فاصله نرمال‌شده تا سطوح
        return sr_distance_from_levels(
            df=df,
            sr=sr,
            atr_window=atr_window
        ).to_dict(orient="series")

    # --- func4 -------------------------------------------

    def make_fibo_from_legs(df, zz, ratios=None, **_):
        return fibo_levels_from_legs_orig(   #########################################
            df,
            zz,
            ratios=ratios
        ).to_dict(orient="series")

    return {
        "pivots": make_pivots,
        "sr": make_sr_zigzag,
        "fibo": make_fibo_from_legs,
        "sr_distance": make_sr_distance
    }
# --- New Added -----------------------------------------------------
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

""" --------------------------------------------------------------------------- Func7
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

""" --------------------------------------------------------------------------- Func8
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

""" --------------------------------------------------------------------------- Func9
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

