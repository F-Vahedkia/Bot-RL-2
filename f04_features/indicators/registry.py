# f04_features/indicators/registry.py
# -*- coding: utf-8 -*-
r"""
رجیستری یکپارچه: core + extras + volume + patterns + levels + divergences
فرمان اجرای برنامه از ریشه:
python -m f04_features.indicators --list
"""

from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import logging
import numpy as np
import pandas as pd

# --- From f04_features/indicators ----------------------------------
from .fibonacci import (golden_zone, fib_cluster, fib_ext_targets,
                        levels_from_legs, select_legs_from_swings)
from .fibo_pipeline import run_fibo_cluster
from .extras_trend import ma_slope, rsi_zone
from .core import rsi as rsi_core, ema as ema_core
from .levels import compute_adr, adr_distance_to_open, sr_overlap_score
from .utils import round_levels, compute_atr, nearest_level_distance
from f10_utils.config_loader import ConfigLoader  # از f01_config/config.yaml می‌خواند
_loader = ConfigLoader()                          # به‌طور پیش‌فرض f01_config/config.yaml را لود می‌کند
from f10_utils.config_ops import _deep_get

# --- Advanced Support/Resistance -----------------------------------
from f04_features.indicators.sr_advanced import (
    make_fvg,           # FVG detector (advanced S/R)
    make_sd,            # Supply/Demand
    make_ob,            # Order Block
    make_liq_sweep,     # Liquidity Sweep
    make_breaker_flip,  # Breaker / Flip Zone
    make_sr_fusion,     #
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Functions -----------------------------------------------------

# --- SR config injection (common/component/overrides by (Symbol, TF)) -------- start

# Deep-get بدون وابستگی بیرونی
'''
def _deep_get(d, path, default=None):
    cur = d
    for key in str(path).split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
'''
# استنتاج اختیاری Symbol/TF (اگر در کانفیگ باشد)
def _infer_symbol_tf(cfg_all, default_sym=None, default_tf=None):
    # کلیدهای رایج؛ اگر نبودند، overrides اعمال نمی‌شود (بدون خطا)
    sym_keys = ["data.symbol", "dataset.symbol", "active.symbol", "symbol"]
    tf_keys  = ["data.base_timeframe", "dataset.base_timeframe", "active.timeframe", "timeframe", "base_timeframe"]
    sym = next(( _deep_get(cfg_all, k) for k in sym_keys if _deep_get(cfg_all, k) is not None), default_sym)
    tf  = next(( _deep_get(cfg_all, k) for k in tf_keys  if _deep_get(cfg_all, k) is not None), default_tf)
    return sym, tf

def _merge_sr_kwargs(name: str, cfg: dict, df: pd.DataFrame) -> dict:
    cfg_all = _loader.get_all()
    # 1) مشترک (فقط کلیدهای عمومی)
    base = _deep_get(cfg_all, "features.support_resistance.sr_advanced.common", {}) or {}
    # 2) پیش‌فرض‌های سطح کامپوننت (fvg/supply_demand/...)
    comp = _deep_get(cfg_all, f"features.support_resistance.sr_advanced.{name}", {}) or {}
    # 3) پروفایل اختیاری (Symbol/TF) اگر تعریف شده باشد
    sym, tf = _infer_symbol_tf(cfg_all)
    over = {}
    if sym and tf:
        over = _deep_get(cfg_all, f"features.support_resistance.sr_advanced.overrides.{sym}.{tf}.{name}", {}) or {}
    # 4) ادغام نهایی: پیش‌فرض کد ← base ← component ← overrides ← پارامترهای صریح کاربر
    return {**base, **comp, **over, **(cfg or {})}

# --- SR config injection ----------------------------------------------------- end

def _fibo_features_full_adapter(*, symbol, tf_dfs, base_tf, atr_len: int = 14, **_) -> Dict[str, pd.Series]:
    """
    خروجی per-bar هم‌تراز با base_tf از خوشه‌های فیبوناچی:
    - فاصله‌ها (abs/signed/level)
    - نرمال‌سازی فاصله با ATR
    - نسبت فیبو و پرچم extension
    - نرمال‌سازی امتیاز خوشه (score به [0,1])
    - عبور دادن sr_score و trend_score (اگر در کلستر موجود باشند)
    """
    # 1) اجرای پایپلاین فیبو و دریافت کلسترها
    res = run_fibo_cluster(symbol=symbol, tf_dfs=tf_dfs, base_tf=str(base_tf))
    clusters = res.clusters if (res is not None and isinstance(res.clusters, pd.DataFrame)) else pd.DataFrame()

    # 2) سری‌های پایه از TF مبنا
    base = tf_dfs[str(base_tf)]
    close = base["close"]
    high  = base.get("high", None)
    low   = base.get("low", None)

    # 3) اگر کلستری نیست، ستون‌ها را NaN برگردان
    if clusters is None or clusters.empty:
        nan = close.astype("float32") * np.nan
        return {
            f"fibc_nearest_abs@{base_tf}": nan,
            f"fibc_nearest_signed@{base_tf}": nan,
            f"fibc_nearest_level@{base_tf}": nan,
            f"fibc_nearest_abs_atr@{base_tf}": nan,
            f"fibc_nearest_ratio@{base_tf}": nan,
            f"fibc_is_extension@{base_tf}": nan,
            f"fibc_score_norm@{base_tf}": nan,
            f"fibc_sr_score@{base_tf}": nan,
            f"fibc_trend_score@{base_tf}": nan,
        }

    # 4) لیست سطوح خوشه (price_mean اگر موجود؛ در غیراینصورت price)
    level_col = "price_mean" if "price_mean" in clusters.columns else ("price" if "price" in clusters.columns else None)
    if level_col is None:
        nan = close.astype("float32") * np.nan
        return {
            f"fibc_nearest_abs@{base_tf}": nan,
            f"fibc_nearest_signed@{base_tf}": nan,
            f"fibc_nearest_level@{base_tf}": nan,
            f"fibc_nearest_abs_atr@{base_tf}": nan,
            f"fibc_nearest_ratio@{base_tf}": nan,
            f"fibc_is_extension@{base_tf}": nan,
            f"fibc_score_norm@{base_tf}": nan,
            f"fibc_sr_score@{base_tf}": nan,
            f"fibc_trend_score@{base_tf}": nan,
        }
    levels = clusters[level_col].astype("float32").tolist()
    #logger.info("FIBO clusters columns: %s", list(clusters.columns))   # temp_logger

    # 5) nearest level metrics per-bar
    def _nearest_tuple(p: float):
        d = nearest_level_distance(float(p), levels)
        # اندیس نزدیک‌ترین کلستر
        j = int(np.argmin([abs(float(p) - float(lv)) for lv in levels])) if levels else -1
        if j < 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        row = clusters.iloc[j]
        
        # compute ratio from 'ratio' or nearest-to-1.0 from 'ratios'
        score = float(row.get("score", np.nan))
        if pd.notna(row.get("ratio", np.nan)):
            r = float(row["ratio"])
        else:
            rats = row.get("ratios", None)
            r = (float(min(rats, key=lambda x: abs(float(x) - 1.0)))
                if isinstance(rats, (list, tuple)) and len(rats) else float("nan"))
        is_ext = 1.0 if (np.isfinite(r) and r > 1.0) else 0.0    # is_extension
        srsc = float(row.get("sr_score", np.nan))
        trsc = float(row.get("trend_score", np.nan))
        tolp = float(row.get("tol_pct", np.nan))
        prfd = float(row.get("prefer_ratio_dist", np.nan))

        return (float(d["abs"]), float(d["signed"]), float(d["nearest_level"]),
                r, score, is_ext, srsc, trsc, tolp, prfd)

    vals = close.apply(_nearest_tuple)
    abs_ser    = vals.apply(lambda t: t[0]).astype("float32")
    signed_ser = vals.apply(lambda t: t[1]).astype("float32")
    level_ser  = vals.apply(lambda t: t[2]).astype("float32")
    ratio_ser  = vals.apply(lambda t: t[3]).astype("float32")
    score_raw  = vals.apply(lambda t: t[4]).astype("float32")
    is_ext_ser = vals.apply(lambda t: t[5]).astype("float32")
    srsc_ser   = vals.apply(lambda t: t[6]).astype("float32")
    trsc_ser   = vals.apply(lambda t: t[7]).astype("float32")
    tolpct_ser = vals.apply(lambda t: t[8]).astype("float32")
    prf_ser    = vals.apply(lambda t: t[9]).astype("float32")


    #logger.info("sr_score/trend_score NaN ratio: %.3f / %.3f", 
    #            float(srsc_ser.isna().mean()), float(trsc_ser.isna().mean()))   # temp_logger

    # 6) نرمال‌سازی score به [0,1] بر اساس کلسترهای موجود
    if "score" in clusters.columns and clusters["score"].notna().any():
        sc = clusters["score"].astype("float32")
        mn, mx = float(np.nanmin(sc)), float(np.nanmax(sc))
        scale = (mx - mn) if np.isfinite(mx - mn) and (mx - mn) > 0 else np.nan
        def _norm(v):
            return (float(v) - mn) / scale if np.isfinite(scale) else np.nan
        score_norm = score_raw.apply(_norm).astype("float32")
    else:
        score_norm = score_raw * np.nan

    # 7) ATR-based normalization (اختیاری، اگر high/low موجود باشد)
    if (high is not None) and (low is not None):
        atrv = compute_atr(base[["high","low","close"]], window=int(atr_len))
        abs_atr = (abs_ser / atrv.replace(0, np.nan)).astype("float32")
    else:
        abs_atr = abs_ser * np.nan

    # 8) عبور دادن sr_score / trend_score در صورت وجود (ثابت بر کل سری، بر اساس نزدیک‌ترین کلستر هر بار)
    # توجه: چون برای هر بار «نزدیک‌ترین کلستر» متفاوت می‌شود، استفادهٔ مستقیم از clusters برای این دو ستون قابل اتکا نیست،
    # مگر این‌که این مقادیر را در همان حلقه استخراج کنیم. (در صورت نیاز بعداً اضافه می‌کنیم.)
    
    conf_sr_ser = (srsc_ser > 0).astype("float32")  # 1=confluence present, 0=absent
    ratio_dev1 = (ratio_ser - 1.0).abs().astype("float32")
    valid_mask = (
        abs_ser.notna()
        & signed_ser.notna()
        & level_ser.notna()
        & ratio_ser.notna()
        & score_norm.notna()
    ).astype("float32")

    out = {
        f"fibc_nearest_abs@{base_tf}": abs_ser,
        f"fibc_nearest_signed@{base_tf}": signed_ser,
        f"fibc_nearest_level@{base_tf}": level_ser,
        f"fibc_nearest_abs_atr@{base_tf}": abs_atr,
        f"fibc_nearest_ratio@{base_tf}": ratio_ser,
        f"fibc_is_extension@{base_tf}": is_ext_ser,
        f"fibc_score_norm@{base_tf}": score_norm,
        f"fibc_sr_score@{base_tf}": srsc_ser,
        f"fibc_trend_score@{base_tf}": trsc_ser,
        f"fibc_tol_pct@{base_tf}": tolpct_ser,
        f"fibc_prefer_ratio_dist@{base_tf}": prf_ser,
        f"fibc_sr_confluence@{base_tf}": conf_sr_ser,
        f"fibc_ratio_dev1@{base_tf}": ratio_dev1,
        f"fibc_valid@{base_tf}": valid_mask,
    }
    return out


def _rsi_adapter(ohlc, n: int = 14, **_) -> Dict[str, pd.Series]:
    return {f"rsi_{n}": rsi_core(ohlc["close"], n)}


def _ema_adapter(ohlc, col: str = "close", n: int = 20, **_) -> Dict[str, pd.Series]:
    return {f"ema_{col}_{n}": ema_core(ohlc[col], n)}


Registry = Dict[str, Callable]
def build_registry() -> Registry:
    reg: Registry = {}
    # core
    from .core import registry as core_reg
    reg.update(core_reg())
    # extras
    from .extras_trend import registry as trend_reg
    reg.update(trend_reg())
    from .extras_channel import registry as ch_reg
    reg.update(ch_reg())
    # volume
    from .volume import registry as vol_reg
    reg.update(vol_reg())
    # patterns
    from .patterns import registry as pat_reg
    reg.update(pat_reg())
    # levels
    from .levels import registry as lvl_reg
    reg.update(lvl_reg())
    # divergences
    from .divergences import registry as div_reg
    reg.update(div_reg())
    return reg


"""
افزودنی‌های رجیستری (Bot-RL-2)
- ADV_INDICATOR_REGISTRY: ثبت اندیکاتورهای جدید (فیبو/ترند)
- get_indicator_v2 / list_all_indicators_v2: بدون شکستن APIهای قبلی

افزودنی‌های رجیستری (Bot-RL-2) — نسخهٔ گسترش‌یافته
رجیستری جدید (advanced) — افزایشی
"""
ADV_INDICATOR_REGISTRY: Dict[str, Callable[..., Any]] = {
    # فیبوناچی ------------------------------------------------------
    "golden_zone": golden_zone,
    "fib_cluster": fib_cluster,
    "fib_ext_targets": fib_ext_targets,
    "fibo_features_full": _fibo_features_full_adapter,

    # فیبو — هِلپرهای پیشرفته/آماده برای استفادهٔ مستقیم ------------
    "levels_from_legs": levels_from_legs,
    "select_legs_from_swings": select_legs_from_swings,

    # ترندی/ممنتوم سبک
    "ma_slope": ma_slope,
    "rsi_zone": rsi_zone,

    # هِلپرهای Levels (برای استفادهٔ مستقیم در صورت نیاز) -----------
    #"round_levels": round_levels,                 # خروجی: list[float]
    #"compute_adr": compute_adr,                   # خروجی: Series ADR
    #"adr_distance_to_open": adr_distance_to_open, # خروجی: DataFrame
    #"sr_overlap_score": sr_overlap_score,         # خروجی: float (برای مصرف مستقیم در کد بهتر است؛ در Spec هم ممکن است ثابت برگردانیم)

    "rsi": _rsi_adapter,
    "ema": _ema_adapter,

    # Advanced Support/Resistance -----------------------------------
    "fvg": make_fvg,
    "supply_demand": make_sd,
    "order_block": make_ob,
    "liq_sweep": make_liq_sweep,
    "breaker_flip": make_breaker_flip,
    "sr_fusion": make_sr_fusion,
}
# wrap S/R indicators to inject merged config (common/component/overrides) ---- start
for _name in ("fvg", "supply_demand", "order_block", "liq_sweep", "breaker_flip", "sr_fusion"):
    _fn = ADV_INDICATOR_REGISTRY[_name]
    def _wrap(fn, name):
        def runner(df: pd.DataFrame, **cfg):
            merged = _merge_sr_kwargs(name, cfg, df)
            return fn(df, **merged)  # sr_advanced.make_* امضای **cfg دارد
        return runner
    ADV_INDICATOR_REGISTRY[_name] = _wrap(_fn, _name)
# --- wrap S/R indicators ----------------------------------------------------- end

def get_indicator_v2(name: str) -> Optional[Callable[..., Any]]:
    key = str(name).strip()
    if key in ADV_INDICATOR_REGISTRY:
        return ADV_INDICATOR_REGISTRY[key]
    try:
        fn = (globals().get("INDICATOR_REGISTRY") or {}).get(key)
        if fn is not None:
            return fn
    except Exception:
        pass
    logger.warning("Indicator not found in v2 registries: %s", name)
    return None

def list_all_indicators_v2(include_legacy: bool = True) -> Dict[str, str]:
    out: Dict[str, str] = {k: "advanced" for k in ADV_INDICATOR_REGISTRY.keys()}
    if include_legacy:
        try:
            legacy = globals().get("INDICATOR_REGISTRY") or {}
            for k in legacy.keys():
                out.setdefault(k, "legacy")
        except Exception:
            pass
    return out

'''
نکات کوتاه:

Engine: برای fib_cluster اگر DF پاس بدهی، به‌خاطر mismatch یک‌بار TypeError می‌خورد
 و مسیر fallback (بدون DF) فعال می‌شود—این همان طراحی افزایشی قبلی ماست.

CLI: هِلپرهایی مثل compute_adr و adr_distance_to_open را می‌توان جداگانه روی دیتاست اجرا کرد
 و ستون‌هایشان را به خروجی افزود.
'''

def _sr_cfg(name, cfg):
    cfg_all = _loader.get_all()
    base = _deep_get(cfg_all,"features.support_resistance.sr_advanced.common", {}) or {}
    comp = _deep_get(cfg_all,f"features.support_resistance.sr_advanced.{name}", {}) or {}
    return {**base, **comp, **(cfg or {})}

for _name in ("fvg","supply_demand","order_block","breaker_flip","liq_sweep","sr_fusion"):
    _fn = ADV_INDICATOR_REGISTRY[_name]
    ADV_INDICATOR_REGISTRY[_name] = (lambda f, n: (lambda df, **cfg: f(df, **_sr_cfg(n, cfg))))(_fn, _name)
