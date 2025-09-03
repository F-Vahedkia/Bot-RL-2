# -*- coding: utf-8 -*-
# f04_features/fibonacci_3.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Sequence
import logging
import numpy as np
import pandas as pd

from f04_features.indicators.levels import sr_overlap_score
from f10_utils.config_loader import ConfigLoader

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



# === [PATCH] f04_features/indicators/fibonacci.py :: Config-aware wrappers === pach_1
# -*- coding: utf-8 -*-
# توجه: این بخش را در ابتدای فایل (پس از importهای موجود) اضافه کنید یا نزدیک بخش ثابت‌ها.
# هدف: افزودن خوانش اختیاری پارامترها از config.yaml «بدون تغییر» منطق توابع موجود.
#  - هیچ رفتار قبلی را نمی‌شکنیم؛ فقط توابع wrapper جدید می‌سازیم که از کانفیگ بخوانند.
#  - اگر کلیدهای کانفیگ موجود نباشند، به مقادیر پیش‌فرض فعلی برمی‌گردیم.
#  - پیام‌های اجرایی (log) انگلیسی هستند؛ توضیحات فارسی.

# ──────────────────────────────────────────────────────────────────────────────
# راهنمای ساختار کلیدهای کانفیگ (اختیاری):
# features:
#   fibonacci:
#     retracement_ratios: [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
#     extension_ratios:   [1.272, 1.618, 2.0, 2.618, 3.618]
#     sl_atr_mult:        1.5
#     golden_zone:
#       ratios: [0.382, 0.618]
#     cluster:
#       tol_pct:       0.08
#       prefer_ratio:  0.618
#       tf_weights:    {"M15":0.5, "H1":1.0, "H4":1.5, "D1":2.0}
#       w_trend:       10.0
#       w_rsi:         10.0
#       w_sr:          10.0
#       sr_tol_pct:    0.05
#
# نکته: اگر هر کلیدی موجود نباشد، به پیش‌فرض‌های فعلی ماژول برمی‌گردیم.
# ──────────────────────────────────────────────────────────────────────────────

def _deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """دسترسی امن به مسیرهای تو در توی دیکشنری با dot-notation؛ در نبود کلید، مقدار پیش‌فرض برمی‌گردد."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_fibo_cfg() -> Dict[str, Any]:
    """
    بارگذاری کل کانفیگ و استخراج زیربخش فیبوناچی.
    اگر لودر/فایل موجود نباشد یا خطایی رخ دهد، دیکشنری خالی بازمی‌گردد.
    """
    try:
        cfg_all: Dict[str, Any] = ConfigLoader().get_all()
    except Exception as e:
        # پیام انگلیسی برای log
        logger.debug("Failed to load config for Fibonacci wrappers: %s", e)
        return {}
    fib = _deep_get(cfg_all, "features.fibonacci", {}) or {}
    if not isinstance(fib, dict):
        fib = {}
    return fib


# ----------------------------------------------------------------------
# Wrapper 1: golden_zone_cfg  → خواندن پارامترها از config و فراخوانی golden_zone
# ----------------------------------------------------------------------
def golden_zone_cfg(
    swings,  # pd.DataFrame
    ratios: Optional[Tuple[float, float]] = None,
    extra_ratios: Optional[Sequence[float]] = None,
):
    """
    نسخهٔ کانفیگ‌محور برای golden_zone:
      - اگر ratios/extra_ratios داده نشود، از config.yaml خوانده می‌شود.
      - در صورت نبود کلیدها در کانفیگ، از پیش‌فرض‌های اصلی ماژول استفاده می‌گردد.

    ورودی‌ها:
        swings: دیتافریم سوئینگ‌ها
        ratios: نسبت‌های بازگشتی اصلی زون طلایی (اختیاری؛ اگر None → از کانفیگ/پیش‌فرض)
        extra_ratios: سایر نسبت‌های رتریسمنت برای محاسبات تکمیلی (اختیاری)

    خروجی:
        pd.DataFrame مطابق خروجی تابع اصلی golden_zone
    """
    fib_cfg = _load_fibo_cfg()

    # نسبت‌های زون طلایی: اولویت با آرگومان تابع → سپس کانفیگ → سپس پیش‌فرض فعلی
    gz_ratios = ratios
    if gz_ratios is None:
        cfg_gz = fib_cfg.get("golden_zone", {}) if isinstance(fib_cfg, dict) else {}
        if isinstance(cfg_gz, dict):
            cfg_r = cfg_gz.get("ratios")
            if isinstance(cfg_r, (list, tuple)) and len(cfg_r) == 2:
                gz_ratios = (float(cfg_r[0]), float(cfg_r[1]))

    if gz_ratios is None:
        # fallback به پیش‌فرض تابع اصلی (0.382, 0.618)
        gz_ratios = (0.382, 0.618)

    # نسبت‌های رتریسمنت تکمیلی
    extra = extra_ratios
    if extra is None:
        cfg_extra = fib_cfg.get("retracement_ratios")
        if isinstance(cfg_extra, (list, tuple)) and cfg_extra:
            extra = tuple(float(x) for x in cfg_extra)

    if extra is None:
        # fallback به ثابت‌های فعلی ماژول (DEFAULT_RETR_RATIOS)
        try:
            extra = DEFAULT_RETR_RATIOS  # noqa: F821  (در همین فایل تعریف شده)
        except NameError:
            extra = (0.236, 0.382, 0.5, 0.618, 0.786)

    logger.info("Golden zone (config-driven) → ratios=%s, extra=%s", gz_ratios, extra)
    return golden_zone(swings=swings, ratios=gz_ratios, extra_ratios=extra)  # noqa: F821


# ----------------------------------------------------------------------
# Wrapper 2: fib_cluster_cfg  → خواندن پارامترها از config و فراخوانی fib_cluster
# ----------------------------------------------------------------------
# نسخهٔ به‌روزِ رَپر: پارامترهای اختیاری ref_price/adr_value/atr_value اضافه شد
def fib_cluster_cfg(
    tf_levels: Dict[str, "pd.DataFrame"],  # type: ignore[name-defined]
    tol_pct: Optional[float] = None,
    prefer_ratio: Optional[float] = None,
    tf_weights: Optional[Dict[str, float]] = None,
    ma_slope: Optional["pd.Series"] = None,      # type: ignore[name-defined]
    rsi_zone_score: Optional["pd.Series"] = None, # type: ignore[name-defined]
    sr_levels: Optional[Sequence[float]] = None,
    ref_time: Optional["pd.Timestamp"] = None,    # type: ignore[name-defined]
    w_trend: Optional[float] = None,
    w_rsi: Optional[float] = None,
    w_sr: Optional[float] = None,
    sr_tol_pct: Optional[float] = None,

    # ⬇️ ورودی‌های اختیاری برای حالت Adaptive (تغییر جدید)
    ref_price: Optional[float] = None,  # قیمت مرجع (اگر None باشد، تلاش می‌کنیم از tf_levels حدس بزنیم)
    adr_value: Optional[float] = None,  # مقدار ADR (مثلاً پیپ/نقطه)
    atr_value: Optional[float] = None,  # مقدار ATR (مثلاً پیپ/نقطه)
):
    """
    نسخهٔ کانفیگ‌محور برای fib_cluster (به‌روزرسانی‌شده با Adaptive tol_pct):
      - اگر tol_pct آرگومان داشته باشیم، همان استفاده می‌شود (اولویت اول).
      - اگر tol_pct آرگومان None باشد:
          * ابتدا مقدار ثابت از کانفیگ خوانده می‌شود (cluster.tol_pct).
          * اگر Adaptive Tol در کانفیگ فعال باشد، و ورودی کافی داشته باشیم،
            tol_pct بر اساس ADR یا ATR محاسبه و جایگزین می‌شود.
      - سایر پارامترها مانند گذشته از آرگومان → کانفیگ → پیش‌فرض پر می‌شوند.
    """
    fib_cfg = _load_fibo_cfg()
    cluster_cfg = fib_cfg.get("cluster", {}) if isinstance(fib_cfg, dict) else {}

    # 1) مقدار پایهٔ tol_pct (ثابت) از آرگومان یا کانفیگ
    tol = tol_pct if tol_pct is not None else float(cluster_cfg.get("tol_pct", 0.08))

    # 2) سایر پارامترها (بدون تغییر)
    pr  = prefer_ratio if prefer_ratio is not None else float(cluster_cfg.get("prefer_ratio", 0.618))
    tfw = tf_weights  if tf_weights  is not None else cluster_cfg.get("tf_weights")
    wtr = w_trend     if w_trend     is not None else float(cluster_cfg.get("w_trend", 10.0))
    wr  = w_rsi       if w_rsi       is not None else float(cluster_cfg.get("w_rsi", 10.0))
    wsr = w_sr        if w_sr        is not None else float(cluster_cfg.get("w_sr", 10.0))
    srt = sr_tol_pct  if sr_tol_pct  is not None else float(cluster_cfg.get("sr_tol_pct", 0.05))

    # 3) اگر Adaptive Tol فعال است، تلاش برای محاسبه tol_pct انطباقی
    adapt_cfg = cluster_cfg.get("adaptive_tol", {}) if isinstance(cluster_cfg, dict) else {}
    if isinstance(adapt_cfg, dict) and bool(adapt_cfg.get("enabled", False)) and tol_pct is None:
        mode = str(adapt_cfg.get("mode", "ADR")).upper()   # ADR | ATR
        k = float(adapt_cfg.get("k", 1.0))                 # ضریب مقیاس
        min_pct = float(adapt_cfg.get("min_pct", 0.02))    # حداقل درصد
        max_pct = float(adapt_cfg.get("max_pct", 0.15))    # حداکثر درصد

        # ref_price را اگر داده نشده، سعی می‌کنیم حدس بزنیم (ممکن است نتیجه ندهد)
        rp = ref_price if ref_price is not None else _infer_ref_price_from_tf_levels(tf_levels, ref_time)

        # انتخاب منبع نوسان
        vol_value = None
        if mode == "ADR":
            vol_value = adr_value
        elif mode == "ATR":
            vol_value = atr_value

        # محاسبه tol انطباقی در صورت داشتن ورودی کافی
        tol_adaptive = _compute_adaptive_tol_pct(rp, vol_value, k=k, min_pct=min_pct, max_pct=max_pct)
        if tol_adaptive is not None:
            logger.info(
                "Adaptive tol_pct enabled → mode=%s k=%.3f ref=%.5f vol=%.5f → tol_pct=%.5f",
                mode, k, float(rp), float(vol_value), tol_adaptive
            )
            tol = tol_adaptive
        else:
            logger.info(
                "Adaptive tol_pct enabled but insufficient inputs (mode=%s). Falling back to fixed tol_pct=%.5f",
                mode, tol
            )

    logger.info(
        "Fibo cluster (config-driven) → tol=%.5f prefer=%.3f tfw=%s w_trend=%.2f w_rsi=%.2f w_sr=%.2f sr_tol=%.3f",
        tol, pr, str(tfw), wtr, wr, wsr, srt
    )

    return fib_cluster(  # noqa: F821
        tf_levels=tf_levels,
        tol_pct=tol,
        prefer_ratio=pr,
        tf_weights=tfw,
        ma_slope=ma_slope,
        rsi_zone_score=rsi_zone_score,
        sr_levels=sr_levels,
        ref_time=ref_time,
        w_trend=wtr,
        w_rsi=wr,
        w_sr=wsr,
        sr_tol_pct=srt,
    )


# ----------------------------------------------------------------------
# Wrapper 3: fib_ext_targets_cfg  → خواندن پارامترها از config و فراخوانی fib_ext_targets
# ----------------------------------------------------------------------
def fib_ext_targets_cfg(
    entry_price: float,
    leg_low: float,
    leg_high: float,
    side: str,
    ext_ratios: Optional[Sequence[float]] = None,
    sl_atr: Optional[float] = None,
    sl_atr_mult: Optional[float] = None,
):
    """
    نسخهٔ کانفیگ‌محور برای fib_ext_targets:
      - اگر ext_ratios/sl_atr_mult داده نشود، از کانفیگ خوانده می‌شود.
      - در نبود کلیدها، به پیش‌فرض‌های فعلی تابع اصلی بازمی‌گردیم.
    """
    fib_cfg = _load_fibo_cfg()

    exts = ext_ratios
    if exts is None:
        cfg_exts = fib_cfg.get("extension_ratios")
        if isinstance(cfg_exts, (list, tuple)) and cfg_exts:
            exts = tuple(float(x) for x in cfg_exts)

    if exts is None:
        try:
            exts = DEFAULT_EXT_RATIOS  # noqa: F821
        except NameError:
            exts = (1.272, 1.618, 2.0)

    sl_mult = sl_atr_mult
    if sl_mult is None:
        cfg_slm = fib_cfg.get("sl_atr_mult")
        if isinstance(cfg_slm, (int, float)):
            sl_mult = float(cfg_slm)

    if sl_mult is None:
        sl_mult = 1.5  # پیش‌فرض فعلی تابع اصلی

    logger.info("Fibo ext targets (config-driven) → ext=%s sl_atr_mult=%.3f", str(exts), sl_mult)

    return fib_ext_targets(  # noqa: F821
        entry_price=entry_price,
        leg_low=leg_low,
        leg_high=leg_high,
        side=side,
        ext_ratios=exts,
        sl_atr=sl_atr,
        sl_atr_mult=sl_mult,
    )

# ────────────────────────────────────────────────────────────────────────────── Pach_2
# Adaptive tol_pct for fib_cluster (Option 1)
# - اگر در کانفیگ فعال باشد و ورودی لازم را داشته باشیم، tol_pct بر اساس ADR یا ATR محاسبه می‌شود.
# - اگر ورودی کافی نبود، به مقدار ثابت (config/پیش‌فرض) برمی‌گردیم.
# - هیچ تغییری در تابع اصلی fib_cluster نمی‌دهیم؛ فقط رَپر fib_cluster_cfg را هوشمند می‌کنیم.
# ──────────────────────────────────────────────────────────────────────────────

def _infer_ref_price_from_tf_levels(
    tf_levels: Dict[str, "pd.DataFrame"],  # type: ignore[name-defined]
    ref_time: Optional["pd.Timestamp"] = None  # type: ignore[name-defined]
) -> Optional[float]:
    """
    تلاش ابتدایی برای حدس ref_price از روی tf_levels:
      - tf_levels معمولاً خروجی سطوح است، نه قیمت. بنابراین اغلب ref_price از اینجا قابل‌استخراج نیست.
      - این تابع برای سازگاری آینده گذاشته شده؛ اگر دیتافریم‌ها ستونی مانند 'close' داشته باشند، از آخرین مقدار استفاده می‌کند.
      - در غیر این صورت None برمی‌گرداند و رَپر از مقدار ثابت tol_pct استفاده می‌کند.

    توجه: این حدس‌زدن اختیاری است و اگر به نتیجه نرسیم، رفتار قبلی حفظ می‌شود.
    """
    try:
        for _, df in tf_levels.items():
            if hasattr(df, "columns"):
                cols = [c.lower() for c in df.columns]
                if "close" in cols:
                    # اگر ref_time داریم، نزدیک‌ترین مقدار قبل یا در ref_time را برداشت می‌کنیم
                    if hasattr(df, "index") and df.index.size > 0 and ref_time is not None:
                        # فیلتر تا ref_time
                        subset = df.loc[df.index <= ref_time]
                        if subset.shape[0] > 0:
                            return float(subset["close"].iloc[-1])
                    # در غیر این صورت آخرین مقدار
                    return float(df["close"].iloc[-1])
    except Exception as e:
        logger.debug("Could not infer ref price from tf_levels: %s", e)
    return None


def _compute_adaptive_tol_pct(
    ref_price: Optional[float],
    vol_value: Optional[float],
    k: float,
    min_pct: float,
    max_pct: float
) -> Optional[float]:
    """
    محاسبه tol_pct انطباقی:
      tol_pct = clip( (vol_value / ref_price) * k , [min_pct, max_pct] )

    اگر هرکدام از ref_price یا vol_value نداشتیم، None برمی‌گردانیم.
    """
    if ref_price is None or vol_value is None:
        return None
    try:
        raw = (float(vol_value) / float(ref_price)) * float(k)
        # محدودسازی در بازهٔ [min_pct, max_pct]
        return max(min(raw, max_pct), min_pct)
    except Exception as e:
        logger.debug("Adaptive tol_pct computation failed: %s", e)
        return None

