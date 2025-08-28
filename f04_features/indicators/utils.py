# f04_features/indicators/utils.py
# -*- coding: utf-8 -*-
"""
ابزارهای کمکی مشترک برای اندیکاتورها (Bot-RL-1)
- کشف تایم‌فریم‌ها از نام ستون‌ها
- نگهبان NaN/Inf و سبک کردن dtype
- zscore، true_range
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd

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