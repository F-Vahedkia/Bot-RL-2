# f04_features/pipelines/fibo_pipeline.py
# -*- coding: utf-8 -*-
"""
Pipeline فیبوناچی (قابل‌مصرف در هستهٔ ربات)
- ساخت سطوح فیبو per-TF (multi-leg → long)
- محاسبه ADR/MA-slope/SR
- اجرای خوشه‌ها (config-driven) و تقویت امتیاز با MA/SR
- یک API سطح‌بالا برای گرفتن خروجی نهایی DataFrame

قوانین پروژه:
- پیام‌های ترمینال/لاگ: انگلیسی
- توضیحات/کامنت‌ها: فارسی
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import pandas as pd
import numpy as np

# توجه: ماژول‌های هسته (همین‌جا مصرف می‌شوند)
import f04_features.indicators.levels as lv
from f04_features.indicators.fibonacci import fib_cluster_cfg, _deep_get
from f04_features.indicators.extras_trend import func_ma_slope
from f10_utils.config_loader import ConfigLoader  # :contentReference[oaicite:0]{index=0}

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[FIBO-PIPE] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# ابزار کمکی عمومی
# -----------------------------------------------------------------------------

def _ratios_map_from_cfg(cfg_all: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    استخراج نقشهٔ ستون→نسبت بر اساس levels.fibo_levels فعلی.
    - در پروژهٔ فعلی ستون‌ها fib_236..fib_786 هستند (levels.py). :contentReference[oaicite:1]{index=1}
    - اگر در آینده نسبت جدیدی به کانفیگ اضافه شد (مثلاً 0.886) اینجا می‌توانیم map را گسترش دهیم.
    """
    # در وضعیت فعلی تابع levels.fibo_levels ستون‌های ثابت می‌سازد:
    base = [( "fib_236", 0.236),
            ( "fib_382", 0.382),
            ( "fib_500", 0.500),
            ( "fib_618", 0.618),
            ( "fib_786", 0.786)]
    # اگر config نسبت‌های رتریس اضافه داشت اما levels.py ستونش را نمی‌سازد، نادیده گرفته می‌شود (سازگار).
    # می‌توان در آینده levels.fibo_levels را همگام کرد.
    return base


# -----------------------------------------------------------------------------
# ۱) ساخت سطوح فیبو per-TF (multi-leg → long)
# -----------------------------------------------------------------------------
def build_fibo_levels_per_tf_multi(
    df: pd.DataFrame,
    k: int,
    max_legs_per_tf: int = 3,
    cfg_all: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    ورودی:
      df: دیتافریم OHLCV تایم‌فریم
      k: پارامتر فراکتال/سوئینگ برای levels.fibo_levels
      max_legs_per_tf: تعداد n مقدار اخیرِ غیر-NaN از هر نسبت (multi-leg)
      cfg_all: برای همخوانی با کانفیگ (مثلاً استخراج نسبت‌ها)

    خروجی:
      DataFrame با ستون‌های ['price','ratio'] (چند ردیف برای هر نسبت)

    توضیح:
      - از lv.fibo_levels استفاده می‌کنیم (در پروژه دیکشنریِ ستون‌ها می‌دهد). :contentReference[oaicite:2]{index=2}
      - سپس چند مقدار آخر هر ستون را به فرمت long تبدیل می‌کنیم.
    """
    if df is None or df.empty or not {"close","high","low"}.issubset(df.columns):
        return None

    out = lv.fibo_levels(close=df["close"], high=df["high"], low=df["low"], k=int(k))
    if isinstance(out, dict):
        levels_df = pd.DataFrame(out)
    elif hasattr(out, "empty"):
        levels_df = out
    else:
        logger.info("Unexpected fibo_levels output type: %s", type(out))
        return None

    if levels_df is None or levels_df.empty:
        return None

    # wide → long (آخرین n مقدار معتبر هر نسبت)
    ratios_map = _ratios_map_from_cfg(cfg_all or {})
    rows: List[Dict[str, float]] = []
    for col, r in ratios_map:
        if col not in levels_df.columns:
            continue
        s = pd.to_numeric(levels_df[col], errors="coerce").dropna()
        if s.empty:
            continue
        tail = s.tail(int(max_legs_per_tf))
        rows += [{"price": float(v), "ratio": float(r)} for v in tail.tolist()]

    return pd.DataFrame(rows, columns=["price", "ratio"]) if rows else None


# -----------------------------------------------------------------------------
# ۲) ADR / MA-slope / SR
# -----------------------------------------------------------------------------
def compute_adr_value(df: pd.DataFrame, window: int, tz: str) -> Optional[float]:
    """
    محاسبه ADR از lv.compute_adr و برگرداندن مقدار آخر (برای نرمال‌سازی tol/Trend).
    """
    try:
        adr_series = lv.compute_adr(df=df, window=int(window), tz=str(tz))  # :contentReference[oaicite:3]{index=3}
        if adr_series is None or len(adr_series) == 0:
            return None
        return float(adr_series.iloc[-1])
    except Exception as e:
        logger.info("ADR computation failed: %s", e)
        return None

'''
def compute_ma_slope_series(df: pd.DataFrame, window: int) -> Optional[pd.Series]:
    """
    محاسبه شیب MA ساده:
      - SMA(window) از close
      - سپس شیب sma.diff()
    خروجی Series هم‌اندیس با df (نام: 'ma_slope').
    """
    try:
        if "close" not in df.columns or df["close"].empty:
            return None
        sma = df["close"].rolling(int(window)).mean()
        slope = sma.diff()
        slope.name = "ma_slope"
        return slope
    except Exception as e:
        logger.info("MA slope computation failed: %s", e)
        return None
'''

def extract_sr_levels_from_fractals(
    df: pd.DataFrame, k: int = 2, lookback: int = 1500, max_levels: int = 30
) -> Optional[List[float]]:
    """
    استخراج سطوح SR از روی فراکتال‌ها؛ اگر چیزی برنگشت از پیوت‌های کلاسیک fallback می‌گیریم.
    """
    try:
        seg = df.iloc[-int(lookback):] if len(df) > lookback else df
        if not {"high","low"}.issubset(seg.columns):
            return None

        fp = lv.fractal_points(high=seg["high"], low=seg["low"], k=int(k))  # :contentReference[oaicite:4]{index=4}
        sr_vals: List[float] = []

        # حالت‌های رایج: dict/DataFrame/tuple of Series (نسخه فعلی tuple برمی‌گرداند)
        if isinstance(fp, dict):
            for key in ("highs", "lows", "H", "L"):
                if key in fp:
                    vals = pd.Series(fp[key])
                    tail = pd.to_numeric(vals, errors="coerce").dropna().tail(int(max_levels))
                    sr_vals.extend([float(x) for x in tail])
        elif hasattr(fp, "columns"):
            for key in [c for c in fp.columns if "high" in c.lower() or "low" in c.lower()]:
                tail = pd.to_numeric(fp[key], errors="coerce").dropna().tail(int(max_levels))
                sr_vals.extend([float(x) for x in tail])

        # Fallback: پیوت‌های کلاسیک — توجه اینکه در پروژه tuple از Series بازمی‌گرداند. :contentReference[oaicite:5]{index=5}
        sr_vals = sorted(set(sr_vals))
        if not sr_vals:
            piv = lv.pivots_classic(high=seg["high"], low=seg["low"], close=seg["close"])
            if isinstance(piv, tuple):
                try:
                    cand = [float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1])
                            for s in piv if hasattr(s, "iloc") and not pd.to_numeric(s, errors="coerce").dropna().empty]
                except Exception:
                    cand = []
                sr_vals = sorted(set(cand))
            elif isinstance(piv, dict):
                cand = [piv.get(k) for k in ("P","S1","S2","S3","R1","R2","R3")]
                cand = [float(x) for x in cand if x is not None]
                sr_vals = sorted(set(cand))

        return sr_vals if sr_vals else None

    except Exception as e:
        logger.info("SR extraction failed: %s", e)
        return None


# -----------------------------------------------------------------------------
# ۳) تقویت امتیاز خوشه با MA و SR (بدون دست‌زدن به هسته)
# -----------------------------------------------------------------------------
def enhance_cluster_scores(
    cluster_df: pd.DataFrame,
    ma_slope: Optional[pd.Series],
    sr_levels: Optional[List[float]],
    cfg_all: dict,
    ref_time: Optional[pd.Timestamp],
    adr_value: Optional[float],
) -> pd.DataFrame:
    """
    ترکیب امتیاز پایهٔ خروجی خوشه با مؤلفه‌های Trend و SR (وزن‌ها از کانفیگ).
    """
    if cluster_df is None or cluster_df.empty:
        return cluster_df.copy()

    out = cluster_df.copy()
    w_trend = float(_deep_get(cfg_all, "features.fibonacci.cluster.w_trend", 10.0))
    w_sr    = float(_deep_get(cfg_all, "features.fibonacci.cluster.w_sr", 10.0))
    sr_tol  = float(_deep_get(cfg_all, "features.fibonacci.cluster.sr_tol_pct", 0.05))

    # Trend scalar از MA slope
    trend_scalar = 0.0
    if ma_slope is not None and isinstance(ma_slope, pd.Series) and len(ma_slope) > 0:
        try:
            if ref_time is not None and ref_time in ma_slope.index:
                slope_val = float(ma_slope.loc[ref_time])
            elif ref_time is not None:
                slope_val = float(ma_slope.loc[:ref_time].iloc[-1])
            else:
                slope_val = float(ma_slope.iloc[-1])
            trend_scalar = float(np.tanh((slope_val / adr_value) if (adr_value and adr_value > 0) else slope_val))
        except Exception:
            trend_scalar = 0.0

    out["trend_score"] = trend_scalar  # اسکالر

    # SR overlap per-row
    def _sr_score(row):
        if not sr_levels:
            return 0.0
        price = float(row.get("price_mean", np.nan))
        if not np.isfinite(price):
            return 0.0
        try:
            return float(lv.sr_overlap_score(price=price, sr_levels=sr_levels, tol_pct=sr_tol))  # :contentReference[oaicite:6]{index=6}
        except Exception:
            return 0.0

    out["sr_score"] = out.apply(_sr_score, axis=1)
    if "score" not in out.columns:
        out["score"] = 0.0

    out["score"] = out["score"] + w_trend * out["trend_score"] + w_sr * out["sr_score"]
    logger.info("Enhanced scores applied (w_trend=%.2f, w_sr=%.2f, sr_tol=%.3f).", w_trend, w_sr, sr_tol)
    return out


# -----------------------------------------------------------------------------
# ۴) API سطح‌بالا: اجرای کامل خوشه بر اساس Config
# -----------------------------------------------------------------------------
@dataclass
class FiboRunResult:
    """خروجی سطح‌بالای خوشهٔ فیبوناچی برای مصرف مستقیم در ربات/RL."""
    symbol: str
    base_tf: str
    ref_time: Optional[pd.Timestamp]
    ref_price: Optional[float]
    clusters: pd.DataFrame

def run_fibo_cluster(
    symbol: str,
    tf_dfs: Dict[str, pd.DataFrame],
    base_tf: str,
    k_fractal: int = 2,
    max_legs_per_tf: int = 3,
    tz: str = "UTC",
) -> FiboRunResult:
    """
    اجرای سرراست خوشهٔ فیبوناچی:
      - ساخت سطوح فیبو برای هر TF
      - گرفتن ref_time/ref_price از TF پایه
      - محاسبه ADR/MA/SR
      - صدا زدن خوشه (config-driven) و تقویت امتیاز
    """
    cfg_all = ConfigLoader().get_all()  # :contentReference[oaicite:7]{index=7}

    # ۱) build tf_levels (long)
    tf_levels: Dict[str, pd.DataFrame] = {}
    for tf, df in tf_dfs.items():
        lv_df = build_fibo_levels_per_tf_multi(
            df=df, k=int(k_fractal),
            max_legs_per_tf=int(_deep_get(cfg_all, "features.fibonacci.leg_selection.max_legs_per_tf", max_legs_per_tf)),
            cfg_all=cfg_all,
        )
        if lv_df is not None and not lv_df.empty:
            tf_levels[tf] = lv_df

    if base_tf not in tf_dfs:
        raise ValueError(f"Base TF '{base_tf}' not in provided dataframes.")

    base_df = tf_dfs[base_tf]
    if base_df is None or base_df.empty:
        raise ValueError(f"Base TF '{base_tf}' dataframe is empty.")

    # ۲) مرجع زمانی/قیمت
    ref_time = base_df.index[-1] if len(base_df.index) > 0 else None
    ref_price = float(base_df["close"].iloc[-1]) if "close" in base_df.columns and len(base_df) else None

    # ۳) ADR / MA / SR
    adr_window = int(_deep_get(cfg_all, "features.levels.adr.window", 14))
    adr_value = compute_adr_value(base_df, window=adr_window, tz=str(tz))

    ma_window = int(_deep_get(cfg_all, "features.levels.ma.window", 50))
    # ma_slope = compute_ma_slope_series(base_df, window=ma_window)
    ma_slope = func_ma_slope(base_df, 
                            price_col = "close",
                            window = ma_window,
                            method = "sma",
                            norm = "none")

    sr_k = int(_deep_get(cfg_all, "features.levels.sr.k", 2))
    sr_lookback = int(_deep_get(cfg_all, "features.levels.sr.lookback", 1500))
    sr_max = int(_deep_get(cfg_all, "features.levels.sr.max_levels", 30))
    sr_levels = extract_sr_levels_from_fractals(base_df, k=sr_k, lookback=sr_lookback, max_levels=sr_max)

    if ma_slope is None:
        logger.info("MA slope not available; clustering will run without ma_slope.")
    else:
        logger.info("MA slope available with window=%d.", ma_window)

    if not sr_levels:
        logger.info("SR levels not available; clustering will run without sr_levels.")
    else:
        logger.info("SR levels extracted: %d levels.", len(sr_levels))

    # ۴) خوشه (با رَپر کانفیگ‌محور موجود در fibonacci.py) :contentReference[oaicite:8]{index=8}
    #    این رَپر tol_pct/weights/... را از config.yaml می‌خواند.
    cluster_df = fib_cluster_cfg(
        tf_levels=tf_levels,
        ref_time=ref_time,
        # Adaptive tol (اگر در رَپر/کانفیگ فعال است)
        ref_price=ref_price,
        adr_value=adr_value,
        atr_value=None,
        # Confluence hooks
        ma_slope=ma_slope,
        sr_levels=sr_levels,
        # rsi_zone_score را فعلاً پاس نمی‌دهیم (گام بعد)
    )

    # ۵) تقویت نمره با MA/SR (اختیاری و ایمن)
    cluster_df = enhance_cluster_scores(
        cluster_df=cluster_df,
        ma_slope=ma_slope,
        sr_levels=sr_levels,
        cfg_all=cfg_all,
        ref_time=ref_time,
        adr_value=adr_value,
    )

    return FiboRunResult(
        symbol=symbol,
        base_tf=base_tf,
        ref_time=ref_time,
        ref_price=ref_price,
        clusters=cluster_df if cluster_df is not None else pd.DataFrame(),
    )
