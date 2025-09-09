# -*- coding: utf-8 -*-
# f15_scripts/check_wiring_fib_cluster.py

"""
Wiring fib_cluster with MA/RSI/SR weighting (Bot-RL-2)
- خواندن دیتاست پردازش‌شده (data_handler)
- ساخت سطوح فیبوناچی از «n لگ اخیر» برای چند TF
- وزن‌دهی با MA-slope / RSI-zone / سطوح رُند (SR)
- چاپ خوشه‌های برتر و (در صورت موجود بودن ماژول) تبدیل به زون

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
from typing import Dict, List, Iterable
import pandas as pd

from f04_features.indicators import fibonacci
from f04_features.indicators.fibonacci import (
    # build_tf_levels_recent,
    fib_cluster,
    last_leg_levels,
    # DEFAULT_RETR_RATIOS,  # در این نسخه استفاده نشده؛ در صورت نیاز آزاد کنید
)
from f04_features.indicators.utils import (
    levels_from_recent_legs,  # هِلپر «n لگ اخیر»
    compute_atr,              # برای last_leg_levels / ساخت سوئینگ
    detect_swings,            # برای last_leg_levels / ساخت سوئینگ
    ohlc_view,                # نمای استاندارد OHLC برای یک TF
    pick_first_existing,      # انتخاب اولین ستون موجود از لیست کاندید
    FiboTestConfig,           # پیکربندی تست متمرکز (بدون وابستگی به scripts)
)

from f10_utils.config_loader import load_config
config = load_config()

# ------------------------------------------------------------
# پیکربندیِ تست (متمرکز در هسته؛ قابل استفاده در این رانر)
# ------------------------------------------------------------
cfg = FiboTestConfig()
data_file = cfg.DATA_FILE         # دیتاست پردازش‌شده‌ی data_handler
tfs = cfg.TFS                     # تایم‌فریم‌های هدف
tails = cfg.TAILS                 # برش پنجره‌ی اخیر برای هر TF
n_legs = cfg.N_LEGS               # تعداد لگ‌های اخیر برای هر TF


""" برای هر TF، ویوی OHLC می‌سازد، «n لگ اخیر» را به سطح فیبو تبدیل و دیکشنری """
# این فایل فقط در check_wiring_fib_cluster.py استفاده میشود.
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
    print("[RUN] load dataset ...")
    df = pd.read_parquet(data_file)

    # 2) آماده‌سازی نمای OHLC و سطوحِ لگ‌های اخیر
    print("[RUN] build tf-levels from recent legs ...")
    tf_levels = build_tf_levels_recent(df, tfs, tails, n_legs)  # (فعلاً نگه می‌داریم اگر جای دیگر نیاز شد)

    # 3) سری‌های وزن‌دهی (در این نسخه اختیاری/خاموش؛ اگر ستون‌های آماده داری، از pick_first_existing استفاده کن)
    # ma_slope_series = pick_first_existing(df, MA_SLOPE_CANDIDATES)   # در صورت نیاز آزاد کنید
    # rsi_score_series = pick_first_existing(df, RSI_SCORE_CANDIDATES) # در صورت نیاز آزاد کنید

    # 4) ساخت سطوح رُند (SR) حول آخرین قیمت TF اول
    first_tf = tfs[0]
    h1_view = ohlc_view(df, first_tf)
    last_close = float(h1_view["close"].dropna().iloc[-1])
    sr_levels = cfg.sr_levels(ref_price=last_close)  # ← جایگزین round_levels(...)

    # 5) زمان مرجع = آخرین ردیف دیتافریم
    ref_ts = pd.to_datetime(df.index[-1], utc=True)

    # 6) ساخت سوئینگ‌ها و لِگ‌ها برای هر TF (خوراک ورودی fib_cluster_cfg)
    swings_by_tf: Dict[str, pd.DataFrame] = {}
    for tf in tfs:
        print(f"[RUN] build swings: start {tf}")
        view = ohlc_view(df, tf)
        atr = compute_atr(view, window=14)
        swings = detect_swings(
            view["close"],
            prominence=None,
            min_distance=5,
            atr=atr,
            atr_mult=1.0,
            tf=tf
        )
        print(f"[RUN] build swings: done {tf} -> {len(swings) if swings is not None else 0}")

        # نگاشت سوئینگ‌ها به لِگ‌ها با اسکیمای مورد انتظار
        if swings is None or swings.empty or len(swings) < 2:
            swings_by_tf[tf] = pd.DataFrame(columns=["bar","time","idx_low","idx_high","low","high"])
        else:
            sw = swings.copy().sort_index()
            sw2 = sw.reset_index()

            # اطمینان از ستون time
            if "time" not in sw2.columns:
                for cand in ("index", "level_0", "datetime", "ts"):
                    if cand in sw2.columns:
                        sw2 = sw2.rename(columns={cand: "time"})
                        break
                else:
                    sw2["time"] = pd.to_datetime(sw2.index, utc=True)
            sw2["time"] = pd.to_datetime(sw2["time"], utc=True, errors="coerce")

            # نگاشت timestamp → نزدیک‌ترین ایندکس در نمای OHLC همان TF
            def _idx_of(ts):
                pos = view.index.get_indexer([pd.to_datetime(ts, utc=True)], method="nearest")[0]
                return int(pos)

            rows: List[dict] = []
            for i in range(1, len(sw2)):
                k1, p1, t1 = str(sw2.loc[i-1,"kind"]).upper(), float(sw2.loc[i-1,"price"]), sw2.loc[i-1,"time"]
                k2, p2, t2 = str(sw2.loc[i  ,"kind"]).upper(), float(sw2.loc[i  ,"price"]), sw2.loc[i  ,"time"]
                if {"H","L"} == {k1, k2}:  # فقط جفت‌های معتبر H/L یا L/H
                    lo, hi = (p1, p2) if p1 < p2 else (p2, p1)
                    i_low  = _idx_of(t1 if p1 <= p2 else t2)
                    i_high = _idx_of(t2 if p2 >= p1 else t1)
                    rows.append({
                        "bar": i,
                        "time": pd.to_datetime(t2, utc=True),
                        "idx_low": i_low,
                        "idx_high": i_high,
                        "low": float(lo),
                        "high": float(hi),
                    })
            legs_df = pd.DataFrame(rows)
            swings_by_tf[tf] = legs_df if not legs_df.empty else pd.DataFrame(columns=["bar","time","idx_low","idx_high","low","high"])

    # 7) ساخت دیکشنری نمای OHLC برای هر TF
    views_by_tf = {tf: ohlc_view(df, tf) for tf in tfs}

    # 8) خوشه‌بندی فیبو با پارامترهای متمرکز (cfg.to_cluster_kwargs)
    print("[RUN] run fib_cluster_cfg ...")
    cluster_kwargs = cfg.to_cluster_kwargs()  # tol_pct, prefer_ratio, w_trend, w_rsi, w_sr, sr_tol_pct
    clusters = fibonacci.fib_cluster_cfg(
        df_by_tf=views_by_tf,          # ← نمای OHLC هر TF
        swings_by_tf=swings_by_tf,     # ← لِگ‌های ساخته‌شده برای هر TF
        global_cfg=config,             # ← کانفیگ اصلی پروژه
        symbol="XAUUSD",               # ← نماد نمونه؛ در عمل از منبع جاری بخوان
        #sr_levels=sr_levels,           # ← سطوح رُند اطراف قیمت
        #ref_time=ref_ts,               # ← زمان مرجع
        #ma_slope=running_ma_slope,   # اگر سری آماده داری، پاس بده
        #rsi_zone_score=running_rsi,  # اگر سری آماده داری، پاس بده
        #**cluster_kwargs,              # ← پارامترهای هم‌گرا شده از cfg
    )

    # 9) چاپ خروجی
    print("Top clusters (by score):")
    if clusters is None or clusters.empty:
        print("No clusters found.")
    else:
        print(clusters.head(12).to_string(index=False))

    # 10) تبدیل خوشه‌ها به «زون/سیگنال قابل‌مصرف» و چاپ (اختیاری در صورت وجود ماژول)
    try:
        from f08_evaluation.cluster_zones import clusters_to_zones

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
- last_leg_levels را نگه داشتیم (برای مقایسهٔ سریع)، ولی در مسیر اصلی از levels_from_recent_legs / build_tf_levels_recent استفاده می‌کنیم.
- اگر ستون‌های وزن‌دهی آماده داری، با pick_first_existing آن‌ها را پیدا و به fib_cluster_cfg پاس بده (در این نسخه برای سادگی خاموش است).
- پنجره‌های Tail، تعداد لگ‌ها و پارامترهای کانفلوئنس/تلورانس از cfg (FiboTestConfig) خوانده می‌شود تا «بلوک پیکربندی تست» پراکنده نباشد.
"""
