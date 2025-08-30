# -*- coding: utf-8 -*-
# f04_features/fibonacci_3.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Sequence
import logging
import numpy as np
import pandas as pd

from f04_features.indicators.levels import sr_overlap_score

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =========================
# نسبت‌های پیش‌فرض فیبوناچی
# =========================
DEFAULT_RETR_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
DEFAULT_EXT_RATIOS  = (1.272, 1.618, 2.0)

# ------------------------------------------------------------
# کمکی: ساخت سطوح فیبو برای یک لگ (low→high یا high→low)
# ------------------------------------------------------------
def _fib_levels_for_leg(
    low: float,
    high: float, 
    ratios: Sequence[float] = DEFAULT_RETR_RATIOS) -> pd.DataFrame:
    """
    English: Build retracement levels for a leg [low, high].
    Persian: ساخت سطوح رتریسمنت برای یک لگ (low→high یا بالعکس).
    """
    if high == low:
        raise ValueError("Invalid leg: high == low")
    leg_up = high > low
    levels = []
    for r in ratios:
        # اگر لگ صعودی باشد، سطح رتریسمنت = high - r*(high-low)
        # اگر لگ نزولی باشد، سطح رتریسمنت = low  + r*(high-low)
        price = (high - r * (high - low)) if leg_up else (low + r * (high - low))
        levels.append({"ratio": float(r), "price": float(price), "leg_up": bool(leg_up)})
    df = pd.DataFrame(levels)
    return df.sort_values("price").reset_index(drop=True)


# ------------------------------------------------------------
# انتخاب آخرین لگ معتبر از روی swings (آخرین H در کنار L یا برعکس)
# ------------------------------------------------------------
def _last_valid_leg(swings: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, float, float]:
    """
    English: Find the most recent swing leg (H with a preceding L, or L with a preceding H).
    Persian: آخرین لگ معتبر را بر اساس جفت H/L مجاور برمی‌گرداند.
    """
    if swings is None or swings.empty:
        raise ValueError("swings is empty")

    # انتظار داریم ستون‌های: index=time(UTC), columns: ['price','kind' in {'H','L'}]
    s = swings.sort_index()
    kinds = s["kind"].values
    prices = s["price"].values
    times = s.index.values

    # از انتها عقب می‌رویم تا به جفت H..L یا L..H برسیم
    for i in range(len(s) - 1, 0, -1):
        k2 = kinds[i]
        k1 = kinds[i - 1]
        if {k1, k2} == {"H", "L"}:
            t1, p1 = pd.to_datetime(times[i - 1]), float(prices[i - 1])
            t2, p2 = pd.to_datetime(times[i]), float(prices[i])
            # جهت لگ: از t1→t2
            return t1, t2, p1, p2
    # اگر پیدا نشد، از دو آخرین نقطه استفاده می‌کنیم:
    t1, p1 = pd.to_datetime(times[-2]), float(prices[-2])
    t2, p2 = pd.to_datetime(times[-1]), float(prices[-1])
    return t1, t2, p1, p2


# ------------------------------------------------------------
# Golden Zone (نسخهٔ کامل‌تر)
# ------------------------------------------------------------
def golden_zone(
    swings: pd.DataFrame,
    ratios: Tuple[float, float] = (0.382, 0.618),
    extra_ratios: Sequence[float] = DEFAULT_RETR_RATIOS,
) -> pd.DataFrame:
    """
    English:
      Build Golden Zone from the last valid leg. Returns a one-row DataFrame:
      ['zone_low','zone_high','center','width','width_pct','leg_up','from_ts','to_ts','nearest_ratio'].
    Persian:
      ساخت ناحیهٔ گلدن‌زون از آخرین لگ معتبر؛ خروجی یک ردیف با ویژگی‌های ناحیه.
    """
    # توضیح: swings باید ایندکس زمانی UTC داشته باشد.
    t1, t2, p1, p2 = _last_valid_leg(swings)
    leg_up = p2 > p1
    low, high = (p1, p2) if leg_up else (p2, p1)

    # سطوح فیبو برای لگ
    retr = _fib_levels_for_leg(low, high, ratios=extra_ratios)
    r_low, r_high = float(min(ratios)), float(max(ratios))
    # استخراج قیمت‌های محدودهٔ گلدن‌زون
    gz_low  = float(retr.loc[(retr["ratio"] - r_low).abs().idxmin(), "price"])
    gz_high = float(retr.loc[(retr["ratio"] - r_high).abs().idxmin(), "price"])
    zone_low, zone_high = (gz_low, gz_high) if gz_low <= gz_high else (gz_high, gz_low)
    center = 0.5 * (zone_low + zone_high)
    width = zone_high - zone_low
    width_pct = 100.0 * (width / center) if center != 0 else np.nan

    # نزدیک‌ترین نسبت به مرکز (برای گزارش)
    retr["dist"] = (retr["price"] - center).abs()
    nearest_idx = retr["dist"].idxmin()
    nearest_ratio = float(retr.loc[nearest_idx, "ratio"])

    out = pd.DataFrame(
        [{
            "zone_low": zone_low,
            "zone_high": zone_high,
            "center": center,
            "width": width,
            "width_pct": width_pct,
            "leg_up": bool(leg_up),
            "from_ts": pd.to_datetime(t1, utc=True),
            "to_ts": pd.to_datetime(t2, utc=True),
            "nearest_ratio": nearest_ratio,
        }]
    ).set_index("to_ts")
    return out


# ------------------------------------------------------------
# Cluster/Confluence بین چند تایم‌فریم
# ------------------------------------------------------------
def fib_cluster(
    tf_levels: Dict[str, pd.DataFrame],
    tol_pct: float = 0.08,
    prefer_ratio: float = 0.618,
    tf_weights: Optional[Dict[str, float]] = None,
    # --- افزوده‌های وزن‌دهی اختیاری ---
    ma_slope: Optional[pd.Series] = None,         # سری شیب MA (مثلاً از M5/H1)
    rsi_zone_score: Optional[pd.Series] = None,   # سری امتیاز RSI (-1,0,1 → [0..1])
    sr_levels: Optional[Sequence[float]] = None,  # سطوح S/R برای همپوشانی
    ref_time: Optional[pd.Timestamp] = None,      # زمان مرجع برای خواندن سری‌ها
    w_trend: float = 10.0,                        # وزن مؤلفهٔ ترند
    w_rsi: float = 10.0,                          # وزن مؤلفهٔ RSI
    w_sr: float = 10.0,                           # وزن مؤلفهٔ S/R
    sr_tol_pct: float = 0.05,                     # تلورانس نسبی برای همپوشانی S/R
) -> pd.DataFrame:
    """
    خوشه‌بندی سطوح فیبو در چند تایم‌فریم + امتیاز Confluence با وزن‌دهی اختیاری MA/RSI/SR.

    ورودی:
      tf_levels: {TF → DataFrame['ratio','price','leg_up'(bool)]}
      tol_pct: آستانهٔ نزدیکی برای خوشه‌بندی (٪ نسبت به قیمت پایه)
      prefer_ratio: نسبت مرجح (نزدیک‌تر به این نسبت امتیاز بهتر)
      tf_weights: وزن پیش‌فرض هر TF (اگر None → استفاده از نگاشت داخلی)
      ma_slope/rsi_zone_score: سری‌های زمینه‌ای (در صورت ارائه، وزن‌دهی فعال می‌شود)
      sr_levels: لیست سطوح S/R (برای امتیاز همپوشانی قیمت خوشه)
      ref_time: زمان مرجع جهت خواندن مقدار آخر ≤ ref_time از سری‌ها
      w_*: وزن هر مؤلفه در امتیاز نهایی
    خروجی:
      DataFrame مرتب بر اساس 'score' با ستون‌های:
      ['price_mean','price_min','price_max','members','tfs','ratios','score']
    """
    # --- گردآوری سطوح ---
    rows = []
    for tf, df in tf_levels.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            rows.append({
                "tf": tf,
                "ratio": float(r["ratio"]),
                "price": float(r["price"]),
                "leg_up": bool(r.get("leg_up", True)),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "price_mean","price_min","price_max","members","tfs","ratios","score"
        ])

    all_levels = pd.DataFrame(rows).sort_values("price").reset_index(drop=True)

    # --- وزن‌های TF (پیش‌فرض) ---
    if tf_weights is None:
        tf_weights = {"M1": 1, "M5": 1.2, "M15": 1.4, "M30": 1.6, "H1": 1.8, "H4": 2.0, "D1": 2.3, "W1": 2.6, "MN1": 3.0}

    # --- کمکی: مقدار سری در ref_time یا آخرین مقدار ---
    def _at_or_before(s: Optional[pd.Series], ts: Optional[pd.Timestamp]) -> float:
        if s is None or len(s) == 0:
            return np.nan
        s = s.dropna()
        if len(s) == 0:
            return np.nan
        if ts is None:
            return float(s.iloc[-1])
        ts = pd.to_datetime(ts, utc=True)
        s2 = s.loc[:ts]
        if len(s2) == 0:
            return np.nan
        return float(s2.iloc[-1])

    trend_val = _at_or_before(ma_slope, ref_time)            # ممکن است NaN باشد
    rsi_val   = _at_or_before(rsi_zone_score, ref_time)      # -1/0/1 در پیاده‌سازی پیشنهادی

    # نرمال‌سازی سادهٔ مؤلفه‌ها به [0..1]
    trend_score = float(np.tanh(abs(trend_val))) if not np.isnan(trend_val) else 0.0
    rsi_score   = float((rsi_val + 1.0) / 2.0) if not np.isnan(rsi_val) else 0.5  # -1→0, 0→0.5, 1→1

    clusters: List[Dict] = []
    used = np.zeros(len(all_levels), dtype=bool)

    i = 0
    while i < len(all_levels):
        if used[i]:
            i += 1
            continue

        base_price = all_levels.loc[i, "price"]
        tol_abs = base_price * (tol_pct / 100.0)
        members_idx = [i]
        used[i] = True

        j = i + 1
        while j < len(all_levels):
            if used[j]:
                j += 1
                continue
            if abs(all_levels.loc[j, "price"] - base_price) <= tol_abs:
                members_idx.append(j)
                used[j] = True
                j += 1
            else:
                break

        members = all_levels.loc[members_idx].copy()
        price_mean = float(members["price"].mean())
        price_min  = float(members["price"].min())
        price_max  = float(members["price"].max())
        tfs = list(members["tf"].unique())
        ratios = list(sorted(set([float(x) for x in members["ratio"].tolist()])))

        # --- امتیاز پایه (مانند نسخهٔ قبلی) ---
        tf_score = sum(tf_weights.get(tf, 1.0) for tf in tfs)
        ratio_closeness = min([abs(r - prefer_ratio) for r in ratios]) if ratios else 1.0
        ratio_score = 1.0 - min(1.0, ratio_closeness)  # نزدیک‌تر به 0.618 بهتر (۰..۱)
        tightness = price_max - price_min
        tight_score = 1.0 - float(tightness / (base_price * 0.01 + 1e-9))  # نرمال در ۱٪ قیمت
        tight_score = max(0.0, min(1.0, tight_score))

        base_score = 50.0 * (tf_score / (len(tf_levels) + 1e-9)) + 30.0 * ratio_score + 20.0 * tight_score

        # --- مؤلفهٔ SR (در صورت وجود sr_levels) ---
        sr_score = 0.0
        if sr_levels:
            sr_score = sr_overlap_score(price_mean, sr_levels, tol_pct=sr_tol_pct)  # ۰..۱

        # --- امتیاز نهایی با وزن‌دهی اختیاری ---
        score = base_score + (w_trend * trend_score) + (w_rsi * rsi_score) + (w_sr * sr_score)

        clusters.append({
            "price_mean": price_mean,
            "price_min": price_min,
            "price_max": price_max,
            "members": len(members_idx),
            "tfs": tfs,
            "ratios": ratios,
            "score": float(score),
        })
        i += 1

    out = pd.DataFrame(clusters).sort_values(["score","price_mean"], ascending=[False, True]).reset_index(drop=True)
    return out

'''
نحوهٔ مصرف (مثال)
اگر خواستی از طریق Engine صدا بزنی، چون fib_cluster به‌صورت «تجمیعی» است، آن را این‌طور Spec بده
 (بدون DF ورودی؛ Engine تلاش دوم را بدون df فراخوانی می‌کند):

fib_cluster(tf={'H1': df_h1_levels,'H4': df_h4_levels}, 
                tol_pct=0.1,
                ref_time='2025-08-29T08:13:00Z',
                ma_slope=ma_slope_series,
                rsi_zone_score=rsi_series,
                sr_levels=[...] )

(برای اجرای واقعی، معمولاً این فراخوانی را در کد پایتون می‌کنی نه CLI
؛ چون لازم است tf_levels و Series ها را پاس بدهی.)
'''
# ------------------------------------------------------------
# سطوح Extension / تارگت + برآورد RR ساده
# ------------------------------------------------------------
def fib_ext_targets(
    entry_price: float,
    leg_low: float,
    leg_high: float,
    side: str,
    ext_ratios: Sequence[float] = DEFAULT_EXT_RATIOS,
    sl_atr: Optional[float] = None,
    sl_atr_mult: float = 1.5,
) -> pd.DataFrame:
    """
    English:
      Build extension targets (e.g., 1.272/1.618/2.0) and a simple RR estimate.
      'side' in {'long','short'}. If sl_atr supplied, SL distance = sl_atr_mult * sl_atr.
    Persian:
      تولید تارگت‌های Extension و برآورد سادهٔ RR با SL مبتنی بر ATR.
    """
    if side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")
    leg_up = leg_high > leg_low
    rng = (leg_high - leg_low)
    if rng <= 0:
        raise ValueError("Invalid leg (high <= low)")

    # محاسبهٔ تارگت‌ها
    targets: List[Dict] = []
    for r in ext_ratios:
        if side == "long":
            price = leg_high + r * rng
            rr_den = (sl_atr_mult * sl_atr) if sl_atr else abs(entry_price - (leg_low if leg_up else leg_high))
            rr_num = abs(price - entry_price)
        else:
            price = leg_low - r * rng
            rr_den = (sl_atr_mult * sl_atr) if sl_atr else abs(entry_price - (leg_high if leg_up else leg_low))
            rr_num = abs(entry_price - price)
        rr = (rr_num / rr_den) if rr_den and rr_den > 0 else np.nan
        targets.append({"ratio": float(r), "price": float(price), "RR": float(rr)})

    df = pd.DataFrame(targets).sort_values("price").reset_index(drop=True)
    return df

