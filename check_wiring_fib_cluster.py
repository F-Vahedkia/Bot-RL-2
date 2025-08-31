# -*- coding: utf-8 -*-
# check_wiring_fib_cluster.py
"""
Wiring fib_cluster with MA/RSI/SR weighting (Bot-RL-2)
- خواندن دیتاست پردازش‌شده (data_handler)
- ساخت سطوح فیبوناچی از «n لگ اخیر» برای چند TF
- وزن‌دهی با MA-slope / RSI-zone / سطوح رُند (SR)
- چاپ خوشه‌های برتر

نکته‌ها:
- پیام‌ها انگلیسی هستند (مطابق سیاست پروژه)
- کامنت‌ها فارسی
- اجرای پیشنهادی از ریشه‌ی پروژه:
    python -m check_wiring_fib_cluster
"""

from __future__ import annotations

# ------------------------------------------------------------
# تنظیم مسیر ایمپورت‌ها (ریشه‌ی پروژه را به sys.path اضافه می‌کنیم)
# ------------------------------------------------------------
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))  # ریشه همین فایل (وقتی با -m اجرا شود کافی است)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # یک لایه بالاتر (برای اطمینان)

# ------------------------------------------------------------
# ایمپورت‌ها
# ------------------------------------------------------------
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

from f04_features.indicators.fibonacci import fib_cluster, DEFAULT_RETR_RATIOS
from f04_features.indicators.levels import round_levels
from f04_features.indicators.utils import (
    levels_from_recent_legs,  # هِلپر «n لگ اخیر»
    compute_atr,              # برای last_leg_levels
    detect_swings,            # برای last_leg_levels
)

# ------------------------------------------------------------
# پیکربندیِ تست (در صورت نیاز تغییر بدهید)
# ------------------------------------------------------------
DATA_FILE: str = r"f02_data/processed/XAUUSD/M1.parquet"  # دیتاست پردازش‌شده‌ی data_handler
TFS: List[str] = ["H1", "H4"]                              # تایم‌فریم‌های هدف
TAILS: Dict[str, int] = {"H1": 1500, "H4": 1000}           # برش پنجره‌ی اخیر برای هر TF
N_LEGS: int = 10                                           # تعداد لگ‌های اخیر برای هر TF
TOL_PCT: float = 0.20                                      # پنجره‌ی همگرایی خوشه‌ها (٪)
PREFER_RATIO: float = 0.618                                # نسبت مرجح
SR_STEP: float = 10.0                                      # گام سطوح رُند (XAUUSD≈10)
SR_COUNT: int = 25                                         # تعداد سطوح رُند حول قیمت آخر
W_TREND: float = 10.0
W_RSI: float = 10.0
W_SR: float = 10.0
SR_TOL_PCT: float = 0.05

# وزن‌دهی — نام ستون‌های قابل‌قبول (اولین موجود انتخاب می‌شود)
MA_SLOPE_CANDIDATES: List[str] = [
    "__ma_slope@M5", "__ma_slope@H1", "__ma_slope@H4"
]
RSI_SCORE_CANDIDATES: List[str] = [
    "__rsi_zone@H1__rsi_zone_score", "__rsi_zone@H4__rsi_zone_score"
]


# ------------------------------------------------------------
# کمکی‌ها
# ------------------------------------------------------------
def ohlc_view(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """استخراج نمای استاندارد OHLC از ستون‌های prefix‌دارِ همان TF."""
    cols = {}
    for k in ["open", "high", "low", "close", "tick_volume", "spread"]:
        c = f"{tf}_{k}"
        if c in df.columns:
            cols[k] = df[c]
    out = pd.DataFrame(cols).dropna(how="all")
    if out.empty:
        raise ValueError(f"OHLC for TF={tf} not found")
    out.index = pd.to_datetime(out.index, utc=True)
    out.sort_index(inplace=True)
    return out


def last_leg_levels(
    ohlc_df: pd.DataFrame,
    prominence: float | None = None,
    min_distance: int = 5,
    atr_mult: float | None = 1.0,
    ratios: Iterable[float] = DEFAULT_RETR_RATIOS,
) -> pd.DataFrame:
    """ساخت سطوح رتریسمنت برای «آخرین لگ معتبر» از روی سوئینگ‌های بسته."""
    close = ohlc_df["close"].astype(float)
    atr = compute_atr(ohlc_df, window=14)
    swings = detect_swings(
        close, prominence=prominence, min_distance=min_distance,
        atr=atr, atr_mult=atr_mult, tf=None
    )
    if swings is None or swings.empty or len(swings) < 2:
        return pd.DataFrame(columns=["ratio", "price", "leg_up"])
    s = swings.sort_index()
    p1, p2 = float(s["price"].iloc[-2]), float(s["price"].iloc[-1])
    leg_up = p2 > p1
    low, high = (p1, p2) if leg_up else (p2, p1)

    rows = []
    rng = high - low
    for r in ratios:
        # قیمت رتریسمنت استاندارد: در لگ صعودی از بالا رو به پایین، در لگ نزولی بالعکس
        price = (high - r * rng) if leg_up else (low + r * rng)
        rows.append({"ratio": float(r), "price": float(price), "leg_up": bool(leg_up)})
    return pd.DataFrame(rows)


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:
    """اولین ستونی که در دیتافریم موجود است را برمی‌گرداند؛ در غیر این‌صورت None."""
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s is not None and len(s) > 0:
                return s
    return None


def build_tf_levels_recent(
    df: pd.DataFrame,
    tfs: Iterable[str],
    tails: Dict[str, int],
    n_legs: int,
) -> Dict[str, pd.DataFrame]:
    """ساخت سطوح فیبو از n لگ اخیر برای هر TF داده‌شده."""
    out: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        view = ohlc_view(df, tf)
        if tf in tails and tails[tf] and tails[tf] > 0:
            view = view.tail(int(tails[tf]))
        out[tf] = levels_from_recent_legs(view, n_legs=n_legs, min_distance=5, atr_mult=1.0)
    return out


# ------------------------------------------------------------
# اجرای اصلی
# ------------------------------------------------------------
def main() -> int:
    # 1) بارگذاری دیتاست
    df = pd.read_parquet(DATA_FILE)

    # 2) آماده‌سازی نمای OHLC و سطوحِ لگ‌های اخیر
    tf_levels = build_tf_levels_recent(df, TFS, TAILS, N_LEGS)

    # 3) سری‌های وزن‌دهی (اولین موجود از بین کاندیدها انتخاب می‌شود)
    ma_slope_series = pick_first_existing(df, MA_SLOPE_CANDIDATES)
    rsi_score_series = pick_first_existing(df, RSI_SCORE_CANDIDATES)

    # 4) ساخت سطوح رُند (SR) حول آخرین قیمت H1 (در صورت وجود H1؛ وگرنه از اولین TF)
    first_tf = TFS[0]
    h1_view = ohlc_view(df, first_tf)
    last_close = float(h1_view["close"].dropna().iloc[-1])
    sr_levels = round_levels(last_close, step=SR_STEP, n=SR_COUNT)

    # 5) زمان مرجع = آخرین ردیف دیتافریم
    ref_ts = pd.to_datetime(df.index[-1], utc=True)

    # 6) اجرای fib_cluster با وزن‌دهی
    clusters = fib_cluster(
        tf_levels=tf_levels,
        tol_pct=TOL_PCT,
        prefer_ratio=PREFER_RATIO,
        ma_slope=ma_slope_series,
        rsi_zone_score=rsi_score_series,
        sr_levels=sr_levels,
        ref_time=ref_ts,
        w_trend=W_TREND, w_rsi=W_RSI, w_sr=W_SR,
        sr_tol_pct=SR_TOL_PCT,
    )

    # 7) چاپ خروجی
    print("Top clusters (by score):")
    if clusters is None or clusters.empty:
        print("No clusters found.")
    else:
        print(clusters.head(12).to_string(index=False))

    # 7.1) تبدیل خوشه‌ها به «زون/سیگنال قابل‌مصرف» و چاپ
    # (اگر ماژول cluster_to_zones موجود نباشد، با پیام هشدار از این مرحله عبور می‌کنیم)
    try:
        # ایمپورت درون‌تابعی تا در صورت نبود فایل، اجرای بخش خوشه‌ها از کار نیفتد
        from f08_evaluation.cluster_zones import clusters_to_zones

        # قیمت فعلی از نمای TF اول (اینجا first_tf)
        current_price = float(h1_view["close"].dropna().iloc[-1])

        zones = clusters_to_zones(
            clusters,
            current_price=current_price,
            atr_series=None,   # اگر سری ATR داری، اینجا پاس بده (مثلاً df["H1_ATR_14"])
            ref_time=ref_ts,
            sl_atr_mult=1.5,
            tp_atr_mult=2.0,
            top_k=8,
            min_members=1,
            min_score=None,
            proximity_pct=0.20,  # 0.20% around price
        )

        print("\nTrade zones (sorted):")
        if zones is None or zones.empty:
            print("No zones.")
        else:
            print(zones.head(10).to_string(index=False))
    except Exception as ex:
        print(f"[WARN] zones conversion skipped: {ex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
نکات:
last_leg_levels را نگه داشتم (برای مقایسهٔ سریع)، ولی در مسیر اصلی از levels_from_recent_legs استفاده می‌کنیم.
ستون‌های وزن‌دهی با pick_first_existing پیدا می‌شوند؛ اگر مثلاً __ma_slope@M5 نباشد، به @H1 یا @H4 می‌افتد.
پنجره‌های Tail، تعداد لگ‌ها، وزن‌ها، و tol_pct همه پارامتر بالا قابل‌تغییر هستند.
"""
