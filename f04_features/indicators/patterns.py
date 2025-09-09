# f04_features/indicators/patterns.py
# # -*- coding: utf-8 -*-
"""الگوهای سادهٔ کندلی (فلگ‌ها)"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Engulfing (ساده و محافظه‌کار)
def engulfing_flags(open_, high, low, close):
    body = close - open_
    prev_body = body.shift(1)
    bull = (prev_body < 0) & (close > open_.shift(1)) & (open_ < close.shift(1))
    bear = (prev_body > 0) & (close < open_.shift(1)) & (open_ > close.shift(1))
    return bull.astype("int8"), bear.astype("int8")

# Doji
def doji_flag(open_, close, thresh: float = 0.1):
    rng = (close - open_).abs()
    atr_like = (rng.rolling(20, min_periods=20).mean()).replace(0, np.nan)
    return ((rng / atr_like) < thresh).astype("int8")

# Pinbar (تقریب)
def pinbar_flags(open_, high, low, close, ratio: float = 2.0):
    body = (close - open_).abs()
    upper_wick = (high - close.where(close > open_, open_))
    lower_wick = (close.where(close < open_, open_) - low)
    bull = (lower_wick > ratio * body) & (upper_wick < body)
    bear = (upper_wick > ratio * body) & (lower_wick < body)
    return bull.astype("int8"), bear.astype("int8")


def registry() -> Dict[str, callable]:
    def make_engulf(df, **_):
        b, s = engulfing_flags(df["open"], df["high"], df["low"], df["close"])
        return {"pat_engulf_bull": b, "pat_engulf_bear": s}
    def make_doji(df, thresh: float = 0.1, **_):
        return {f"pat_doji_{thresh}": doji_flag(df["open"], df["close"], thresh)}
    def make_pinbar(df, ratio: float = 2.0, **_):
        b, s = pinbar_flags(df["open"], df["high"], df["low"], df["close"], ratio)
        return {f"pat_pin_bull_{ratio}": b, f"pat_pin_bear_{ratio}": s}
    return {"pat_engulf": make_engulf, "pat_doji": make_doji, "pat_pin": make_pinbar}
'''
#============================================================================== Added 1
# در انتهای fibonacci.py (یا patterns.py، در صورت ترجیح شما)
def detect_ab_equal_cd_old(swings: pd.DataFrame,
                       ratio_tol: float = 0.05) -> Optional[Dict[str, Any]]:
    """
    تشخیص بسیار سادهٔ AB=CD:
    - نیازمند چهار نقطهٔ متوالی A-B-C-D
    - طول AB و CD با تلورانس نسبی نزدیک باشند.
    خروجی: مختصات نقاط و نسبت خطا. (برای شروع مینیمال)
    """
    if len(swings) < 4:
        return None
    s = swings.sort_values("bar").reset_index(drop=True)
    A, B, C, D = s.iloc[-4], s.iloc[-3], s.iloc[-2], s.iloc[-1]
    ab = abs(float(B["price"]) - float(A["price"]))
    cd = abs(float(D["price"]) - float(C["price"]))
    if ab <= 0:
        return None
    err = abs(cd/ab - 1.0)
    if err <= ratio_tol:
        return {
            "pattern": "AB=CD",
            "points": {"A": A.to_dict(), "B": B.to_dict(), "C": C.to_dict(), "D": D.to_dict()},
            "error": float(err),
        }
    return None
'''
#============================================================================== Added 2

def detect_ab_equal_cd(swings: pd.DataFrame, ratio_tol: float = 0.05) -> Optional[Dict[str, Any]]:
    """
    English:
      Detect a simple AB=CD pattern from the last four consecutive swing points.
      If 'bar' column is missing, it will be created based on time order.
    Persian:
      تشخیص سادهٔ الگوی AB=CD از چهار نقطهٔ متوالی آخر سوئینگ.
      اگر ستون 'bar' وجود نداشته باشد، با ترتیب زمانی ساخته می‌شود.

    Args:
        swings: DataFrame با ایندکس زمانی UTC و ستون‌های حداقل ['price','kind'] (kind ∈ {'H','L'})
        ratio_tol: تلورانس نسبی برای برابری طول AB و CD (پیش‌فرض 0.05 = 5%)

    Returns:
        dict شامل اطلاعات الگو در صورت موفقیت؛ در غیر این صورت None.
    """
    if swings is None or len(swings) < 4:
        return None

    # نرمال‌سازی حداقلی اسکیمای ورودی
    s = swings.copy()
    # اطمینان از وجود ستون‌های پایه
    if "price" not in s.columns or "kind" not in s.columns:
        logger.warning("[ABCD] swings lacks required columns ['price','kind']")
        return None

    # مرتب‌سازی زمانی و ساخت bar در صورت نبود
    s = s.sort_index()
    if "bar" not in s.columns:
        s = s.reset_index().rename(columns={"index": "time"})  # اگر نام دیگری باشد، pandas خودش index می‌سازد
        if "time" not in s.columns:
            # fallback: بساز از ایندکس فعلی
            s["time"] = pd.to_datetime(s.index, utc=True)
        s["time"] = pd.to_datetime(s["time"], utc=True, errors="coerce")
        s["bar"] = np.arange(len(s), dtype=int)
        s = s.set_index("bar")
    else:
        s = s.sort_values("bar").reset_index(drop=False).set_index("bar")

    # باید حداقل 4 نقطه داشته باشیم
    if len(s) < 4:
        return None

    # آخرین چهار نقطه (A,B,C,D)
    A = s.iloc[-4]
    B = s.iloc[-3]
    C = s.iloc[-2]
    D = s.iloc[-1]

    # طول‌ها
    ab = abs(float(B["price"]) - float(A["price"]))
    cd = abs(float(D["price"]) - float(C["price"]))
    if ab <= 0:
        return None

    err = abs(cd / ab - 1.0)
    if err <= float(ratio_tol):
        out = {
            "pattern": "AB=CD",
            "error": float(err),
            "points": {
                "A": {"price": float(A["price"]), "kind": str(A["kind"])},
                "B": {"price": float(B["price"]), "kind": str(B["kind"])},
                "C": {"price": float(C["price"]), "kind": str(C["kind"])},
                "D": {"price": float(D["price"]), "kind": str(D["kind"])},
            },
        }
        logger.info("[ABCD] pattern detected with error=%.4f", err)
        return out

    logger.info("[ABCD] no pattern within tolerance (err=%.4f, tol=%.4f)", err, ratio_tol)
    return None

def abc_projection_adapter_from_abcd(abcd: dict):
    if not abcd: 
        return None
    A,B,C,D = (abcd["points"][k]["price"] for k in ("A","B","C","D"))
    length_AB = abs(B - A)
    proj_ratio = (abs(D - C) / length_AB) if length_AB else None
    return {"A":A,"B":B,"C":C,"D_real":D,"length_AB":length_AB,"proj_ratio":proj_ratio,"err_pct":abcd.get("error")}
