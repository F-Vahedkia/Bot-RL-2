# f04_features/indicators/levels.py
# -*- coding: utf-8 -*-
"""Pivot های کلاسیک، S/R ساده مبتنی بر فراکتال، و فاصله تا سطوح فیبوناچی اخیر"""
from __future__ import annotations
from typing import List, Sequence, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

""" --------------------------------------------------------------------------- OK Func1
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
فراکتال ساده برای S/R (K کندل چپ/راست)
"""
def fractal_points(high: pd.Series, low: pd.Series, k: int = 2) -> tuple[pd.Series, pd.Series]:
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


""" --------------------------------------------------------------------------- Func3
فاصله تا نزدیک‌ترین سطح S/R نزدیک
"""
def sr_distance(close: pd.Series,
                high: pd.Series,
                low: pd.Series,
                k: int = 2, lookback: int = 500
) -> Tuple[pd.Series, pd.Series]:
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

""" --------------------------------------------------------------------------- Func4
فیبوناچی: سطوح اخیر بین آخرین سوینگ بالا/پایین (فراکتال k)
"""
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

""" --------------------------------------------------------------------------- Func5
"""
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


# --- New Added ----------------------------------------------------- 060407
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

