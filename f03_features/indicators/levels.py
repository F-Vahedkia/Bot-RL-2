# -*- coding: utf-8 -*-
# f03_features/indicators/levels.py
# Status in (Bot-RL-2): Completed

"""Pivot های کلاسیک، S/R ساده مبتنی بر فراکتال، و فاصله تا سطوح فیبوناچی اخیر
"""
#==============================================================================
# Imports & Logger
#==============================================================================
from __future__ import annotations
from typing import List, Sequence, Dict, Optional, Tuple
from unittest import result
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from f10_utils.config_loader import ConfigLoader
cfg = ConfigLoader().get_all()

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
فراکتال ساده برای استفاده در توابع sr_distance, fibo_levels
قرارداد خروجی:
- 1.0  → فرکتال تأییدشده
- 0.0  → قطعاً فرکتال نیست
- NaN  → هنوز قابل قضاوت نیست (warm-up یا ناحیهٔ look-ahead)
"""
def fractal_points(high: pd.Series, low: pd.Series, k: int = 2) -> tuple[pd.Series, pd.Series]:
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

""" --------------------------------------------------------------------------- OK Func3
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
        lo = max(0, i - lookback)                  # نگاه به گذشته و ساخت حد پایین بازه مورد نظر
        new_hh = hh.iloc[lo:i].astype(bool)        # ساخت فیلتر بولی در بازه موردنظر
        new_ll = ll.iloc[lo:i].astype(bool)        # ساخت فیلتر بولی در بازه موردنظر
        prev_h = high.iloc[lo:i][new_hh]           # فراکتال‌های بالا در بازه موردنظر
        prev_l =  low.iloc[lo:i][new_ll]           # فراکتال‌های پایین در بازه موردنظر

        r = (prev_h.iloc[-1] - close.iloc[i]) if len(prev_h) else np.nan # اختلاف کلوز با آخرین فراکتال بالا
        s = (close.iloc[i] - prev_l.iloc[-1]) if len(prev_l) else np.nan # اختلاف کلوز با آخرین فراکتال پایین
        res.iloc[i] = np.float32(r) if pd.notna(r) else np.nan # ذخیره در سری خروجی
        sup.iloc[i] = np.float32(s) if pd.notna(s) else np.nan # ذخیره در سری خروجی
    return res, sup

""" --------------------------------------------------------------------------- DELETED Func4
فیبوناچی: سطوح اخیر بین آخرین سوینگ بالا/پایین (فراکتال k)
"""
def fibo_levels_old(close: pd.Series, high: pd.Series, low: pd.Series, k: int = 2):
    hh, ll = fractal_points(high, low, k)
    # آخرین سوینگ‌ها
    last_high = high[hh.astype(bool)]
    last_low  =  low[ll.astype(bool)]

    diff = (last_high - last_low)
    levels = {
        "fib_236": (last_high - 0.236*diff).astype("float32"),
        "fib_382": (last_high - 0.382*diff).astype("float32"),
        "fib_500": (last_high - 0.500*diff).astype("float32"),
        "fib_618": (last_high - 0.618*diff).astype("float32"),
        "fib_786": (last_high - 0.786*diff).astype("float32"),
    }
    # فاصلهٔ قیمت تا هر سطح
    dists = {f"dist_{k}": (close - v).astype("float32") for k, v in levels.items()}
    levels.update(dists)
    return levels

""" --------------------------------------------------------------------------- OK Func4
    تولید سطوح فیبوناچی به صورت دیکشنری {ratio: level}.

    - استفاده از آخرین fractal high,low در بازه lookback
    - تشخیص جهت swing (صعودی یا نزولی)
    - خروجی: dict که کلیدها نسبت‌های فیبو و مقادیر سطح قیمت هستند
"""
def fibo_levels(close: pd.Series,
                high: pd.Series,
                low: pd.Series,
                k: int = 2,
                cfg: Optional[dict] = None,
                lookback: int = 500,
) -> dict[str, pd.Series]:

    # پیش فرض درصدهای فیبوناچی ---------------------------
    if cfg is not None:
        fb = (cfg.get("features") or {}).get("fibonacci") or {}
        ratios: List[float] = list(fb.get("retracement_ratios") or [])
    else:
        ratios = [0.236, 0.382, 0.500, 0.618, 0.786]

    # محاسبه فراکتال‌ها ------------------------------------
    hh, ll = fractal_points(high, low, k)  # Series contain 0 and 1
    result = []

    for i in range(0, len(close)):
        lo = max(k, i - lookback)
        h_fractals = high.iloc[lo:i][hh.iloc[lo:i].astype(bool)]   # آخرین fractal high ها
        l_fractals =  low.iloc[lo:i][ll.iloc[lo:i].astype(bool)]   # آخرین fractal low ها

        fibo_dict = {}
        if (i < k) or len(h_fractals)==0 or len(l_fractals)==0:
            for r in ratios:
                fibo_dict[f"fibo_{r}"] = np.nan
            result.append(fibo_dict)
            continue

        # قیمت و زمان سقف و کف اخیر
        last_high = h_fractals.iloc[-1]
        last_low = l_fractals.iloc[-1]
        time_high = h_fractals.index[-1]
        time_low = l_fractals.index[-1]
        rng = abs(last_high - last_low)
        
        if rng == 0:
            for r in ratios:
                fibo_dict[f"fibo_{r}"] = np.nan
            result.append(fibo_dict)
            continue

        # تعیین جهت swing
        if time_low < time_high:     # برای روند صعودی
            for r in ratios:
                level = last_high - r * rng
                fibo_dict[f"fibo_{r}"] = float(level)
        else:                        # برای روند نزولی
            for r in ratios:
                level = last_low + r * rng
                fibo_dict[f"fibo_{r}"] = float(level)
        result.append(fibo_dict)        
    return result

""" --------------------------------------------------------------------------- Func5
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
        r, s = sr_distance(df["close"], df["high"], df["low"], k, lookback)
        return {f"sr_res_{k}_{lookback}": r,
                f"sr_sup_{k}_{lookback}": s
                }
    
    def make_fibo(df, k: int = 2, **_):
        return fibo_levels(df["close"], df["high"], df["low"], k)
    
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

data = [
    ['2010-01-04 00:15:00+00:00',1099.5,1099.95,1098.54,1099.6,325,0],
    ['2010-01-04 00:15:00+00:00',1099.5,1099.95,1098.54,1099.6,325,0],
    ['2010-01-04 00:25:00+00:00',1098.85,1099.55,1098.6,1099.2,252,0],
    ['2010-01-04 00:30:00+00:00',1098.9,1099.25,1098.42,1098.83,300,0],
    ['2010-01-04 00:35:00+00:00',1098.68,1098.9,1096.93,1097.26,383,0],
    ['2010-01-04 00:40:00+00:00',1097.55,1098.2,1097.05,1097.73,334,0],
    ['2010-01-04 00:45:00+00:00',1097.7,1098.2,1097.2,1097.7,269,0],
    ['2010-01-04 00:50:00+00:00',1097.67,1098.35,1097.0,1097.0,305,0],
    ['2010-01-04 00:55:00+00:00',1097.21,1097.21,1095.57,1095.57,293,0],
    ['2010-01-04 01:00:00+00:00',1096.0,1096.35,1093.85,1094.59,401,0],
    ['2010-01-04 01:05:00+00:00',1094.61,1095.0,1093.78,1094.36,380,0],
    ['2010-01-04 01:10:00+00:00',1094.37,1094.7,1093.08,1093.4,364,0],
    ['2010-01-04 01:15:00+00:00',1093.75,1095.15,1093.35,1095.1,376,0],
    ['2010-01-04 01:20:00+00:00',1094.95,1095.75,1094.18,1095.3,331,0],
    ['2010-01-04 01:25:00+00:00',1095.31,1096.45,1095.18,1095.65,313,0],
    ['2010-01-04 01:30:00+00:00',1095.36,1095.6,1094.95,1095.06,188,0],
    ['2010-01-04 01:35:00+00:00',1095.3,1095.55,1094.78,1094.93,209,0],
    ['2010-01-04 01:40:00+00:00',1094.98,1096.15,1094.28,1095.68,324,0],
    ['2010-01-04 01:45:00+00:00',1095.72,1096.9,1095.58,1096.08,297,0],
    ['2010-01-04 01:50:00+00:00',1096.01,1096.3,1094.55,1094.55,284,0],
    ['2010-01-04 01:55:00+00:00',1094.6,1095.6,1094.25,1094.49,298,0],
    ['2010-01-04 02:00:00+00:00',1094.95,1095.45,1094.0,1095.11,320,0],
    ['2010-01-04 02:05:00+00:00',1095.08,1095.75,1094.25,1095.6,300,0],
    ['2010-01-04 02:10:00+00:00',1095.26,1095.95,1094.93,1095.37,281,0],
    ['2010-01-04 02:15:00+00:00',1095.38,1095.7,1094.42,1094.8,265,0],
    ['2010-01-04 02:20:00+00:00',1095.05,1095.6,1094.8,1095.06,231,0],
    ['2010-01-04 02:25:00+00:00',1095.05,1095.65,1094.98,1095.06,204,0],
    ['2010-01-04 02:30:00+00:00',1095.05,1095.45,1094.66,1094.9,177,0],
    ['2010-01-04 02:35:00+00:00',1094.89,1095.25,1094.54,1095.0,276,0],
    ['2010-01-04 02:40:00+00:00',1095.05,1095.45,1094.74,1094.9,290,0],
]


data2 = [
    ['2010-01-04 00:15:00+00:00',49.3,50,49.1,49.7],
    ['2010-01-04 00:15:00+00:00',50.3,51,50.1,50.7],
    ['2010-01-04 00:25:00+00:00',51.3,52,51.1,51.7],
    ['2010-01-04 00:30:00+00:00',52.3,53,52.1,52.7],
    ['2010-01-04 00:35:00+00:00',53.3,54,53.1,53.7],
    ['2010-01-04 00:40:00+00:00',54.3,55,54.1,54.7],
    ['2010-01-04 00:45:00+00:00',55.3,56,55.1,55.7],
    ['2010-01-04 00:50:00+00:00',56.3,57,56.1,56.7],
    ['2010-01-04 00:55:00+00:00',57.3,58,57.1,57.7],
    ['2010-01-04 01:00:00+00:00',56.3,57,56.1,56.7],
    ['2010-01-04 01:05:00+00:00',55.3,56,55.1,55.7],
    ['2010-01-04 01:10:00+00:00',54.3,55,54.1,54.7],
    ['2010-01-04 01:15:00+00:00',53.3,54,53.1,53.7],
    ['2010-01-04 01:20:00+00:00',52.3,53,52.1,52.7],
    ['2010-01-04 01:25:00+00:00',51.3,52,51.1,51.7],
    ['2010-01-04 01:30:00+00:00',50.3,51,50.1,50.7],
    ['2010-01-04 01:35:00+00:00',51.3,52,51.1,51.7],
    ['2010-01-04 01:40:00+00:00',52.3,53,52.1,52.7],
    ['2010-01-04 01:45:00+00:00',53.3,54,53.1,53.7],
    ['2010-01-04 01:50:00+00:00',54.3,55,54.1,54.7],
    ['2010-01-04 01:55:00+00:00',55.3,56,55.1,55.7],
    ['2010-01-04 02:00:00+00:00',54.3,55,54.1,54.7],
    ['2010-01-04 02:05:00+00:00',53.3,54,53.1,53.7],
    ['2010-01-04 02:10:00+00:00',52.3,53,52.1,52.7],
    ['2010-01-04 02:15:00+00:00',51.3,52,51.1,51.7],
    ['2010-01-04 02:20:00+00:00',50.3,51,50.1,50.7],
    ['2010-01-04 02:25:00+00:00',49.3,50,49.1,49.7],
    ['2010-01-04 02:30:00+00:00',50.3,51,50.1,50.7],
    ['2010-01-04 02:35:00+00:00',51.3,52,51.1,51.7],
    ['2010-01-04 02:40:00+00:00',52.3,53,52.1,52.7],
    ['2010-01-04 02:45:00+00:00',53.3,54,53.1,53.7],
    ['2010-01-04 02:50:00+00:00',52.3,53,52.1,52.7],
    ['2010-01-04 02:55:00+00:00',51.3,52,51.1,51.7],
    ['2010-01-04 03:00:00+00:00',50.3,51,50.1,50.7],
    ['2010-01-04 03:05:00+00:00',49.3,50,49.1,49.7],
    ['2010-01-04 03:10:00+00:00',48.3,49,48.1,48.7],
    ['2010-01-04 03:15:00+00:00',47.3,48,47.1,47.7],
    ['2010-01-04 03:20:00+00:00',46.3,47,46.1,46.7],
    ['2010-01-04 03:25:00+00:00',45.3,46,45.1,45.7],
    ['2010-01-04 03:30:00+00:00',44.3,45,44.1,44.7],
    ['2010-01-04 03:35:00+00:00',43.3,44,43.1,43.7],
    ['2010-01-04 03:40:00+00:00',42.3,43,42.1,42.7],
    ['2010-01-04 03:45:00+00:00',41.3,42,41.1,41.7],
    ['2010-01-04 03:50:00+00:00',40.3,41,40.1,40.7],
    ['2010-01-04 03:55:00+00:00',39.3,40,39.1,39.7],
    ['2010-01-04 04:00:00+00:00',40.3,41,40.1,40.7],
    ['2010-01-04 04:05:00+00:00',41.3,42,41.1,41.7],
    ['2010-01-04 04:10:00+00:00',42.3,43,42.1,42.7],
    ['2010-01-04 04:15:00+00:00',43.3,44,43.1,43.7],
    ['2010-01-04 04:20:00+00:00',44.3,45,44.1,44.7],
]

df = pd.DataFrame(data2, columns=["time","open","high","low","close"])
df.set_index(pd.to_datetime(df["time"]), inplace=True)
df.drop(columns=["time"], inplace=True)





result = fibo_levels(df["close"], df["high"], df["low"], k=1, cfg=cfg, lookback=50)
print(pd.DataFrame(result))