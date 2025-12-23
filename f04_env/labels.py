# f04_env/labels.py  (لیبل‌ساز بدون لوک‌اِهد، اختیاری)
# -*- coding: utf-8 -*-
"""
لیبل‌سازهای ساده (اختیاری) برای پیش‌گرم یا تحلیل داده
- بدون look-ahead: همه‌ی خروجی‌ها با shift(+1) به گذشته منتقل می‌شوند.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal
import numpy as np
import pandas as pd

LabelMode = Literal["sign", "logret", "triple_barrier"]

@dataclass
class LabelConfig:
    mode: LabelMode = "sign"   # sign | logret | triple_barrier
    horizon: int = 1           # افق پیش‌بینی به تعداد کندل پایه
    tp_sigma: float = 1.5      # فقط triple_barrier
    sl_sigma: float = 1.0      # فقط triple_barrier
    use_logret: bool = True    # برای triple_barrier معیار شکستن

# ----------------------------
# ابزارهای کمکی داخلی
# ----------------------------
def _logret(close: pd.Series) -> pd.Series:
    lr = np.log(close / close.shift(1))
    return lr.fillna(0.0).astype("float32")

def _rolling_vol(s: pd.Series, n: int = 100) -> pd.Series:
    return s.rolling(n, min_periods=max(20, n//5)).std().fillna(method="bfill").astype("float32")

# ----------------------------
# لیبل‌ها
# ----------------------------
def make_labels(df: pd.DataFrame, cfg: LabelConfig, price_col: str = "M1_close") -> pd.Series:
    """
    تولید لیبل سری‌زمانی مطابق cfg.
    - df: دیتافریم پردازش‌شده (aggregate + indicators)
    - price_col: نام ستون قیمت پایه (با قرارداد پروژه: <TF>_close یا <TF>__close)
    خروجی همیشه با shift(+1) برگردانده می‌شود تا از look-ahead جلوگیری شود.
    """
    # پیدا کردن ستون close به‌صورت مقاوم
    if price_col not in df.columns:
        # تلاش دوم: نسخه‌ی با دابل‌زیرخط
        alt = price_col.replace("_", "__", 1)
        if alt in df.columns:
            price_col = alt
        else:
            # fallback: اولین ستونی که به _close ختم شود
            cands = [c for c in df.columns if c.lower().endswith("_close")]
            if not cands:
                raise KeyError("close column not found for labeling")
            price_col = cands[0]

    close = df[price_col].astype("float32")

    if cfg.mode == "sign":
        # علامت بازده افق horizon؛ سپس shift(+1)
        ret = close.pct_change(cfg.horizon).fillna(0.0)
        y = np.sign(ret).replace(0, 0).astype("int8").shift(1).fillna(0).astype("int8")
        return y

    if cfg.mode == "logret":
        lr = _logret(close).rolling(cfg.horizon, min_periods=cfg.horizon).sum().astype("float32")
        return lr.shift(1).fillna(0.0).astype("float32")

    # triple_barrier: آستانه‌ها با سیگما * نوسان اخیر
    lr = _logret(close) if cfg.use_logret else close.pct_change().fillna(0.0)
    vol = _rolling_vol(lr)
    up = cfg.tp_sigma * vol
    dn = cfg.sl_sigma * vol

    # برای هر i بررسی می‌کنیم که در پنجره‌ی آینده تا horizon کدام barrier اول لمس می‌شود
    n = len(close)
    out = np.zeros(n, dtype="int8")
    for i in range(n):
        j = min(n-1, i + cfg.horizon)
        if j <= i:  # ابتدای سری
            continue
        # تغییر تجمعی در بازه (logret cumulative)
        move = lr.iloc[i+1: j+1].cumsum()
        # لمس TP؟
        tp_hit = (move >= up.iloc[i]).any()
        # لمس SL؟
        sl_hit = (move <= -dn.iloc[i]).any()
        if tp_hit and not sl_hit:
            out[i] = 1
        elif sl_hit and not tp_hit:
            out[i] = -1
        else:
            out[i] = 0

    # shift(+1) برای حذف نگاه به آینده
    return pd.Series(out, index=df.index, dtype="int8").shift(1).fillna(0).astype("int8")
