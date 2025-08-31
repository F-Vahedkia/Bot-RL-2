# f04_features/indicators/utils.py
# -*- coding: utf-8 -*-
"""
ابزارهای کمکی مشترک برای اندیکاتورها (Bot-RL-1)
- کشف تایم‌فریم‌ها از نام ستون‌ها
- نگهبان NaN/Inf و سبک کردن dtype
- zscore، true_range
"""
from __future__ import annotations
from typing import Sequence, Optional, Dict, Tuple, List, Iterable
import re
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# کشف تایم‌فریم‌ها از روی نام ستون‌ها
@dataclass
class TFView:
    tf: str
    cols: Dict[str, str]  # map: std_name -> df_column_name

_TF_REGEX = re.compile(r"^(?P<tf>[A-Z0-9]+)_(?P<field>open|high|low|close|tick_volume|spread)$", re.IGNORECASE)

def detect_timeframes(df: pd.DataFrame) -> Dict[str, TFView]:
    buckets: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        m = _TF_REGEX.match(col)
        if not m:
            continue
        tf = m.group("tf").upper()
        field = m.group("field").lower()
        buckets.setdefault(tf, {})[field] = col
    return {tf: TFView(tf=tf, cols=mapping) for tf, mapping in buckets.items()}

# برش یک TF با استانداردسازی نام ستون‌ها
def slice_tf(df: pd.DataFrame, view: TFView) -> pd.DataFrame:
    cols = []
    rename_map = {}
    for k_std, c in view.cols.items():
        cols.append(c)
        rename_map[c] = "volume" if k_std == "tick_volume" else k_std
    sdf = df[cols].rename(columns=rename_map).copy()
    return sdf

# نگهبان NaN و dtype سبک
def nan_guard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            # Nullable Int64 را دست‌نخورده می‌گذاریم
            pass
    return df

# z-Score ساده
def zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or window
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    return ((s - mean) / std.replace(0, np.nan)).astype("float32")

# True Range (برای ATR و ...)
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.astype("float32")

# --- New Added ----------------------------------------------------- 040607
"""
افزودنی‌های کم‌خطر برای Utils اندیکاتورها (Bot-RL-2)
- تشخیص سوئینگ‌ها (بدون SciPy) با فیلترهای prominence و ATR
- محاسبهٔ ATR (کلاسیک با SMA)
- زی‌اسکورِ فاصله (zscore_distance)
- فاصله تا نزدیک‌ترین سطح (nearest_level_distance)

نکته‌ها:
- همهٔ ورودی/خروجی‌ها با ایندکس زمانی UTC مرتب فرض شده‌اند.
- پیام‌های لاگ انگلیسی‌اند؛ توضیحات فارسی.
- نام‌گذاری مطابق «اولین سند اصلاح اندیکاتورها».
"""
# =========================
# Additions for swing/metrics, ATR (SMA-based)
# =========================
def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    English: Classic ATR (not Wilder smoothing). Returns a pandas Series aligned with df.index.
    Persian: محاسبهٔ ATR کلاسیک برای داده‌ای که ستون‌های high/low/close دارد.
    """
    # توضیح: انتظار داریم ستون‌های high/low/close موجود باشند.
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        raise ValueError("DF must contain columns: high, low, close")

    # True Range
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR: میانگین متحرک (ساده) — در آینده می‌توان Wilder یا EMA افزود.
    atr = tr.rolling(window=window, min_periods=max(2, window // 2)).mean()
    atr.name = f"ATR_{window}"
    return atr

# =========================
# Swing Detection (H/L)
# =========================
def detect_swings(
    price: pd.Series,
    prominence: Optional[float] = None,
    min_distance: int = 5,
    atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
) -> pd.DataFrame:
    """
    English:
      Local swing detection without SciPy:
      - A point is swing-high if it's the maximum in a ±min_distance window.
      - A point is swing-low  if it's the minimum in a ±min_distance window.
      - Optional 'prominence' and 'atr_mult * ATR' filters to drop weak swings.
    Persian:
      تشخیص قله/کف محلی بدون SciPy:
      - بیشینه/کمینه در پنجرهٔ ±min_distance
      - فیلتر اختیاری بر اساس prominence و همچنین آستانهٔ ATR (atr_mult * ATR)
    """
    # نکته: price باید ایندکس زمانی UTC و مرتب داشته باشد.
    
    
    """
    تشخیص قله/کف محلی بدون SciPy:
      - نقطه swing-high اگر در پنجرهٔ ±min_distance بیشینه باشد.
      - نقطه swing-low  اگر در پنجرهٔ ±min_distance کمینه باشد.
      - فیلتر اختیاری با prominence و همچنین آستانهٔ ATR (atr_mult * ATR).

    ورودی‌ها:
      price: Series اندیس‌گذاری‌شده بر حسب زمان (UTC)
      prominence: حداقل برجستگی نسبت به لبه‌های پنجره (اختیاری)
      min_distance: نصفِ اندازهٔ پنجره به دو طرف
      atr: سری ATR هم‌تراز (اختیاری)
      atr_mult: اگر داده شود، آستانهٔ حذف سوئینگ‌های ضعیف = atr_mult * ATR
      tf: نام تایم‌فریم برای متادیتا (اختیاری)

    خروجی:
      DataFrame با ایندکس زمانی، ستون‌ها: ['price','kind','atr','tf']
        kind ∈ {'H','L'}
    """    
    
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series indexed by time")

    idx = price.index
    n = len(price)
    if n < (2 * min_distance + 1):
        logger.debug("detect_swings: insufficient length (n=%d, min_distance=%d)", n, min_distance)
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    highs: list[Tuple[pd.Timestamp, float]] = []
    lows:  list[Tuple[pd.Timestamp, float]] = []

    # پنجره‌ی لغزان برای اکسترمم محلی
    for i in range(min_distance, n - min_distance):
        p = price.iloc[i]
        left = price.iloc[i - min_distance : i]
        right = price.iloc[i + 1 : i + 1 + min_distance]

        is_high = p >= left.max() and p >= right.max()
        is_low  = p <= left.min() and p <= right.min()

        if not (is_high or is_low):
            continue

        # فیلتر prominence ساده: فاصله از نزدیک‌ترین همسایهٔ طرفین
        if (prominence is not None) and (prominence > 0):
            prom_left = abs(p - left.iloc[-1])
            prom_right = abs(p - right.iloc[0])
            prom_ok = (prom_left >= prominence) and (prom_right >= prominence)
            if not prom_ok:
                continue

        # فیلتر مبتنی بر ATR (اگر دادهٔ ATR و ضریب atr_mult داده شده باشد)
        atr_here = float(atr.iloc[i]) if (atr is not None and pd.notna(atr.iloc[i])) else np.nan
        if (atr is not None) and (atr_mult is not None) and (atr_mult > 0) and not np.isnan(atr_here):
            left_edge = left.iloc[-1]
            right_edge = right.iloc[0]
            local_prom = max(abs(p - left_edge), abs(p - right_edge))
            if local_prom < atr_mult * atr_here:
                continue

        ts = idx[i]
        if is_high:
            highs.append((ts, float(p)))
        if is_low:
            lows.append((ts, float(p)))

    # خروجی یکدست
    rows: List[Dict] = []
    for ts, val in highs:
        rows.append({"ts": ts, "price": val, "kind": "H", "atr": float(atr.loc[ts]) if atr is not None and ts in atr.index else np.nan, "tf": tf})
    for ts, val in lows:
        rows.append({"ts": ts, "price": val, "kind": "L", "atr": float(atr.loc[ts]) if atr is not None and ts in atr.index else np.nan, "tf": tf})

    swings = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    if not swings.empty:
        swings.set_index("ts", inplace=True)
        swings.index = pd.to_datetime(swings.index, utc=True)
    else:
        logger.debug("detect_swings: no swings detected")
    return swings

# =========================
# Z-Score distance
# =========================
def zscore_distance(x: float, mu: float, sigma: float, eps: float = 1e-12) -> float:
    """
    English: Return (x - mu) / sigma with small epsilon for stability.
    Persian: نرمال‌سازی فاصله با زی‌اسکور.
    """
    s = abs(sigma) if sigma is not None else 0.0
    return float((x - mu) / (s + eps))

# =========================
# Nearest level distance
# =========================
def nearest_level_distance(price: float, levels: Sequence[float]) -> Dict[str, float]:
    """
    English: Find nearest level to 'price' and return distances (signed/abs) and the level.
    Persian: نزدیک‌ترین سطح قیمتی به price را برمی‌گرداند.
    """
    if not levels:
        return {"nearest_level": float("nan"), "signed": float("nan"), "abs": float("nan")}
    # محاسبهٔ فاصله‌ها
    diffs = [price - lv for lv in levels]
    j = int(np.argmin([abs(d) for d in diffs]))
    return {
        "nearest_level": float(levels[j]),
        "signed": float(diffs[j]),
        "abs": float(abs(diffs[j])),
    }


# --- New Added ----------------------------------------------------- 040608
def levels_from_recent_legs(
    ohlc_df: pd.DataFrame,
    n_legs: int = 10,
    ratios: Optional[Iterable[float]] = None,
    prominence: Optional[float] = None,
    min_distance: int = 5,
    atr_mult: Optional[float] = 1.0,
) -> pd.DataFrame:
    """
    ساخت سطوح فیبوی رتریسمنت برای «n لگ اخیر» از روی سوئینگ‌های بسته.

    ورودی:
      - ohlc_df: DataFrame با ستون‌های open/high/low/close (ایندکس UTC مرتب)
      - n_legs: تعداد لگ‌های اخیر (پیش‌فرض 10)
      - ratios: نسبت‌های فیبو (اگر None → [0.236,0.382,0.5,0.618,0.786])
      - prominence/min_distance/atr_mult: پارامترهای فیلتر سوئینگ (برای حذف نویز)

    خروجی:
      DataFrame ستون‌ها: ['ratio','price','leg_up','leg_idx']
      - leg_idx: شمارهٔ لگ از انتها (1 = آخرین لگ، 2 = یکی قبل‌تر، ...)
    """
    if ratios is None:
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    # محاسبهٔ ATR برای فیلتر سوئینگ (در صورت نیاز)
    atr = None
    try:
        from .utils import compute_atr as _compute_atr  # اجتناب از import حلقوی
        atr = _compute_atr(ohlc_df, window=14)
    except Exception:
        pass

    close = ohlc_df["close"].astype(float)
    try:
        from .utils import detect_swings as _detect_swings
        swings = _detect_swings(
            close,
            prominence=prominence,
            min_distance=min_distance,
            atr=atr,
            atr_mult=atr_mult,
            tf=None,
        )
    except Exception as ex:
        # اگر detect_swings در دسترس نبود
        return pd.DataFrame(columns=["ratio", "price", "leg_up", "leg_idx"])

    if swings is None or swings.empty or len(swings) < 2:
        return pd.DataFrame(columns=["ratio", "price", "leg_up", "leg_idx"])

    s = swings.sort_index()
    prices = s["price"].astype(float).to_numpy()

    rows: List[dict] = []
    # از آخرین نقطه شروع می‌کنیم: (i-1 → i) یک لگ
    # i: اندیس آخرین سوئینگ، i-1: سوئینگ قبلی
    last_i = len(prices) - 1
    max_legs = min(n_legs, last_i)  # به تعداد جفت‌ها می‌تونیم لگ بسازیم

    for k in range(1, max_legs + 1):
        i = last_i - (k - 1)
        j = i - 1
        if j < 0:
            break
        p1, p2 = prices[j], prices[i]
        leg_up = p2 > p1
        low, high = (p1, p2) if leg_up else (p2, p1)

        rng = high - low
        if rng <= 0:
            continue

        for r in ratios:
            # قیمت رتریسمنتِ لگ (استاندارد)
            price = (high - r * rng) if leg_up else (low + r * rng)
            rows.append({
                "ratio": float(r),
                "price": float(price),
                "leg_up": bool(leg_up),
                "leg_idx": int(k),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["price", "leg_idx"]).reset_index(drop=True)
    return out
