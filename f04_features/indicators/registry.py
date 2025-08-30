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

# توجه: مسیرها را با ساختار جدید هماهنگ کرده‌ایم (داخل indicators/)
from f04_features.indicators.fibonacci import golden_zone, fib_cluster, fib_ext_targets
from f04_features.indicators.extras_trend import ma_slope, rsi_zone
# (در صورت نیاز بعداً: از levels/utils هم import می‌کنیم)
# from f04_features.indicators.levels import round_levels, compute_adr, adr_distance_to_open, sr_overlap_score
from f04_features.indicators.levels import round_levels, compute_adr, adr_distance_to_open, sr_overlap_score

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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

# --- New Added ----------------------------------------------------- 040607
# -*- coding: utf-8 -*-
"""
افزودنی‌های رجیستری (Bot-RL-2)
- ADV_INDICATOR_REGISTRY: ثبت اندیکاتورهای جدید (فیبو/ترند)
- get_indicator_v2 / list_all_indicators_v2: بدون شکستن APIهای قبلی
"""
# -----------------------------
# رجیستریِ افزوده (حلقهٔ جدید)
# -----------------------------
ADV_INDICATOR_REGISTRY: Dict[str, Callable[..., Any]] = {
    # فیبوناچی (ویژگی‌های سطح بالا)
    "golden_zone": golden_zone,
    "fib_cluster": fib_cluster,
    "fib_ext_targets": fib_ext_targets,

    # ترندی/ممنتوم سبک (برای وزن‌دهی Confluence)
    "ma_slope": ma_slope,
    "rsi_zone": rsi_zone,
    # در آینده (اختیاری) می‌توان موارد levels/utils را هم به شکل اندیکاتور ثبت کرد.
}


def get_indicator_v2(name: str) -> Optional[Callable[..., Any]]:
    """
    دریافت تابع اندیکاتور از رجیستری جدید؛
    اگر یافت نشد، تلاش می‌کنیم از رجیستری قبلی (در صورت وجود) بخوانیم.
    """
    key = str(name).strip()
    if key in ADV_INDICATOR_REGISTRY:
        return ADV_INDICATOR_REGISTRY[key]

    # تلاش برای رجیستری‌های قدیمی (اگر در فایل حاضر موجود باشد)
    try:
        # فرض: نام رجیستری قدیمی INDICATOR_REGISTRY بوده
        fn = (globals().get("INDICATOR_REGISTRY") or {}).get(key)
        if fn is not None:
            return fn
    except Exception:
        pass

    logger.warning("Indicator not found in v2 registries: %s", name)
    return None


def list_all_indicators_v2(include_legacy: bool = True) -> Dict[str, str]:
    """
    لیست همهٔ اندیکاتورها (کلید → منبع)، بدون شکستن رجیستری قبلی.
    """
    out: Dict[str, str] = {}
    for k in ADV_INDICATOR_REGISTRY.keys():
        out[k] = "advanced"
    if include_legacy:
        try:
            legacy = globals().get("INDICATOR_REGISTRY") or {}
            for k in legacy.keys():
                out.setdefault(k, "legacy")
        except Exception:
            pass
    return out


# --- New Added ----------------------------------------------------- 040608
"""
افزودنی‌های رجیستری (Bot-RL-2) — نسخهٔ گسترش‌یافته
"""
# رجیستری جدید (advanced) — افزایشی
ADV_INDICATOR_REGISTRY: Dict[str, Callable[..., Any]] = {
    # فیبوناچی
    "golden_zone": golden_zone,
    "fib_cluster": fib_cluster,
    "fib_ext_targets": fib_ext_targets,

    # ترندی/ممنتوم سبک
    "ma_slope": ma_slope,
    "rsi_zone": rsi_zone,

    # هِلپرهای Levels (برای استفادهٔ مستقیم در صورت نیاز)
    "round_levels": round_levels,                 # خروجی: list[float]
    "compute_adr": compute_adr,                   # خروجی: Series ADR
    "adr_distance_to_open": adr_distance_to_open, # خروجی: DataFrame
    "sr_overlap_score": sr_overlap_score,         # خروجی: float (برای مصرف مستقیم در کد بهتر است؛ در Spec هم ممکن است ثابت برگردانیم)
}

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