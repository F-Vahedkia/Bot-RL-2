# f04_features/indicators/engine.py
# -*- coding: utf-8 -*-
r"""IndicatorEngine: اجرای Specها روی دیتای MTF و تولید DataFrame ویژگی‌ها"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

from f04_features.indicators.parser import parse_spec_v2, ParsedSpec
from f04_features.indicators.registry import get_indicator_v2
from .utils import detect_timeframes, slice_tf, nan_guard
from .parser import parse_spec

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass
class EngineConfig:
    specs: List[str]                 # مثل: ["rsi(14)@M1", "ema(close,20)@M5", ...]
    timeframes: Optional[List[str]] = None  # اگر None، همهٔ TFهای موجود
    shift_all: int = 1               # شیفت عمومی جهت جلوگیری از look-ahead
    drop_na_head: bool = True        # حذف NaNهای ابتدای سری‌ها

class IndicatorEngine:
    def __init__(self, cfg: EngineConfig, registry: Dict[str, Any]):
        self.cfg = cfg
        self.registry = registry      # نگاشت نام → تابع سازندهٔ Series map

    def _available_tfs(self, df: pd.DataFrame) -> List[str]:
        views = detect_timeframes(df)
        if self.cfg.timeframes:
            return [tf for tf in self.cfg.timeframes if tf in views]
        return list(views.keys())

    def _apply_one(self, base_df: pd.DataFrame, tf: str, spec_str: str) -> Dict[str, pd.Series]:
        """
        اجرای یک Spec روی یک تایم‌فریم مشخص.
        - اگر آرگومان اول Spec نامِ ستون باشد (open/high/low/close/volume)،
        فقط وقتی پاس می‌دهیم که تابع رجیستری پارامتر «col» داشته باشد؛
        در این صورت «col» را به شکلِ positional قبل از بقیهٔ آرگومان‌ها می‌فرستیم
        تا با عدد بعدی (مثلاً period) تداخل «multiple values for 'col'» پیش نیاید.
        - اگر تابع «col» نداشت، آن توکن ستونی را نادیده می‌گیریم (تابع خودش از df[...] می‌خواند).
        - خروجی‌ها با پیشوند TF برگردانده می‌شوند.
        """
        import inspect  # فقط برای تشخیص پارامترهای تابع (سبک و بی‌هزینه)

        spec = parse_spec(spec_str)
        if spec.name not in self.registry:
            raise ValueError(f"Unknown indicator: {spec.name} (spec={spec_str})")

        views = detect_timeframes(base_df)
        if tf not in views:
            raise KeyError(f"Timeframe {tf} not found in DataFrame. (spec={spec_str})")

        # برشِ دیتای مربوط به TF و استانداردسازی نام ستون‌ها به open/high/low/close/volume
        sdf = slice_tf(base_df, views[tf])

        fn = self.registry[spec.name]
        fn_params = list(inspect.signature(fn).parameters.keys())

        # آماده‌سازی آرگومان‌ها از روی Spec
        args = list(spec.args)  # کپی ایمن
        col_token = None
        if args and isinstance(args[0], str) and args[0].lower() in {"open", "high", "low", "close", "volume"}:
            col_token = args.pop(0).lower()

        # استراتژی پاس‌دادن آرگومان‌ها:
        # 1) اگر تابع پارامتر «col» دارد و کاربر ستون داده، col را positional قبل از بقیه args می‌فرستیم.
        # 2) اگر «col» در امضای تابع نیست، col_token را نادیده می‌گیریم (تابع خودش از df[...] استفاده می‌کند).
        if col_token is not None and "col" in fn_params:
            out_map: Dict[str, pd.Series] = fn(sdf, col_token, *args)
        else:
            out_map: Dict[str, pd.Series] = fn(sdf, *args)

        # پیشونددهی نام ستون‌های خروجی با TF مقصد
        prefixed = {f"{tf}__{k}": v for k, v in out_map.items()}
        return prefixed


    def apply_all(self, df: pd.DataFrame, base_tf_fallback: Optional[str] = None) -> pd.DataFrame:
        tfs_all = self._available_tfs(df)
        if not tfs_all:
            raise ValueError("No timeframe columns detected.")
        base_tf = base_tf_fallback or tfs_all[0]
        features: Dict[str, pd.Series] = {}
        for spec_str in self.cfg.specs:
            tf = parse_spec(spec_str).tf or base_tf
            vals = self._apply_one(df, tf, spec_str)
            features.update(vals)
        feat_df = pd.DataFrame(features, index=df.index).sort_index()
        if self.cfg.shift_all and self.cfg.shift_all > 0:
            feat_df = feat_df.shift(self.cfg.shift_all)
        feat_df = nan_guard(feat_df)
        if self.cfg.drop_na_head:
            feat_df = feat_df.dropna(how="all")
        return feat_df
    
# --- New Added ----------------------------------------------------- 040607
# -*- coding: utf-8 -*-
"""
Engine افزایشی (Bot-RL-2)
- اجرای Specهای v2 (بدون دست‌زدن به موتور قدیمی)
- تکیه بر parse_spec_v2 و get_indicator_v2
- بدون لوک‌اِهد: فقط روی کندل‌های بسته، merge_asof به صورت backward

فرض‌ها:
- دیتای ورودی خروجیِ data_handler است (ستون‌ها با پیشوند TF: مثل M1_close, H1_close, ...)
- ایندکس زمانی UTC و مرتب است.
"""

# -----------------------------
# کمکی‌ها: کشف و استخراج نمای OHLC یک TF
# -----------------------------
_OHLC_KEYS = ("open", "high", "low", "close", "tick_volume", "spread")

def _available_timeframes(df: pd.DataFrame) -> List[str]:
    """
    کشف TFها از روی پیشوند ستون‌ها (مانند 'M1_close', 'H1_open', ...).
    """
    tfs = set()
    for c in df.columns:
        if "_" in c:
            tf, key = c.split("_", 1)
            if key in _OHLC_KEYS:
                tfs.add(tf)
    return sorted(tfs)

def _get_ohlc_view(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    استخراج نمای استانداردِ OHLC برای TF خواسته‌شده از روی ستون‌های پیشونددار.
    خروجی: DataFrame با ستون‌های ['open','high','low','close','tick_volume','spread']
    (هر کدام که موجود باشد)
    """
    cols = {}
    for k in _OHLC_KEYS:
        col = f"{tf}_{k}"
        if col in df.columns:
            cols[k] = df[col]
    if not cols:
        raise KeyError(f"OHLC view for TF='{tf}' not found in columns")
    out = pd.DataFrame(cols).copy()
    out.index = pd.to_datetime(df.index, utc=True)
    out.sort_index(inplace=True)
    return out


# -----------------------------
# کمکی: ایمن‌سازی ایندکس و merge-asof (backward)
# -----------------------------
def _ensure_utc_index(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    if isinstance(x.index, pd.DatetimeIndex):
        if x.index.tz is None:
            x.index = x.index.tz_localize("UTC")
        else:
            x.index = x.index.tz_convert("UTC")
    return x

def _merge_features_on_base(base: pd.DataFrame, feat: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    ادغام ویژگی‌ها بر بستر base با merge_asof backward (بدون لوک‌اِهد).
    اگر index دقیقاً هم‌تراز باشد، join ساده انجام می‌شود.
    """
    base = _ensure_utc_index(base.copy())
    feat = _ensure_utc_index(feat.copy())

    if isinstance(feat, pd.Series):
        feat = feat.to_frame()

    # اگر ایندکس‌ها هم‌ترازند، join مستقیم
    if base.index.equals(feat.index):
        return base.join(feat)

    # در غیر اینصورت merge_asof
    b = base.reset_index().rename(columns={"index": "time"})
    f = feat.reset_index().rename(columns={"index": "time"})
    merged = pd.merge_asof(
        b.sort_values("time"),
        f.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged.set_index("time", inplace=True)
    merged.index = pd.to_datetime(merged.index, utc=True)
    return merged


# -----------------------------
# اجرای یک Spec (v2)
# -----------------------------
def _apply_spec_v2(df_in: pd.DataFrame, spec_str: str, base_tf: str) -> pd.DataFrame:
    """
    اجرای یک Spec:
      - پارس Spec (name/args/kwargs/@TF)
      - واکشی تابع از رجیستری جدید
      - ساخت ورودی مناسب (نمای OHLC همان TF)
      - ادغام نتیجه بر بستر df_in (بدون لوک‌اِهد)

    نام‌گذاری خروجی:
      - اگر تابع Series برگرداند → نام ستون: __{name}@{tf}
      - اگر DataFrame برگرداند → برای هر ستون: __{name}@{tf}__{col}
    """
    ps: ParsedSpec = parse_spec_v2(spec_str)
    tf = (ps.timeframe or base_tf).upper()
    name = ps.name

    fn = get_indicator_v2(name)
    if fn is None:
        logger.warning("Unknown indicator in v2 registry: %s", name)
        return df_in

    # ورودیِ تابع اندیکاتور:
    #  - اگر انتظار OHLC دارد، نمای OHLC را بدهیم (اکثر فیبو/ترندها به close یا OHLC نیاز دارند)
    #  - برخی توابع ممکن است DataFrame کامل بخواهند؛ قرارداد ما این است که
    #    اول OHLC همان TF را می‌سازیم و به fn می‌دهیم؛
    #    اگر fn خودش چیز دیگری خواست، از kwargs/args استفاده می‌کند.
    try:
        ohlc = _get_ohlc_view(df_in, tf)
    except Exception as e:
        logger.error("OHLC view for TF '%s' not available: %s", tf, e)
        return df_in

    # اجرای تابع اندیکاتور
    try:
        result = fn(ohlc, *ps.args, **ps.kwargs)  # قرارداد: df اول و سپس args/kwargs
    except TypeError:
        # اگر امضای تابع مطابق نبود، یک باره بدون df امتحان کنیم (برخی توابع ممکن است df نخواهند)
        result = fn(*ps.args, **ps.kwargs)

    if result is None:
        return df_in

    # نرمال‌سازی خروجی به DataFrame و پیشوندگذاری
    if isinstance(result, pd.Series):
        df_feat = result.rename(f"__{name}@{tf}").to_frame()
    elif isinstance(result, pd.DataFrame):
        df_feat = result.copy()
        # ستون‌های بدون نام را اسم‌گذاری کنیم
        new_cols = {}
        for c in df_feat.columns:
            safe_c = str(c) if (c is not None and str(c) != "") else "val"
            new_cols[c] = f"__{name}@{tf}__{safe_c}"
        df_feat.rename(columns=new_cols, inplace=True)
    else:
        # اگر خروجی اسکالر/لیست بود، آن را به ستون ثابت تبدیل کنیم (feature ثابت)
        ser = pd.Series(np.nan, index=df_in.index, name=f"__{name}@{tf}")
        try:
            ser.iloc[:] = float(result)  # اگر اسکالر عددی باشد
        except Exception:
            ser.iloc[:] = str(result)
        df_feat = ser.to_frame()

    # ادغام بدون لوک‌اِهد
    merged = _merge_features_on_base(df_in, df_feat)
    return merged


def run_specs_v2(df: pd.DataFrame, specs: Iterable[str], base_tf: str) -> pd.DataFrame:
    """
    اجرای چند Spec به‌ترتیب داده‌شده و ساخت دیتاست ویژگی‌ها روی df.
    - base_tf: شبکهٔ زمانی پایه (برای resolve @TF)
    - specs: رشته‌های Spec مانند 'golden_zone(0.382,0.618)@H1'
    """
    out = _ensure_utc_index(df.copy()).sort_index()
    base_tf = base_tf.upper()

    known_tfs = set(_available_timeframes(out))
    if base_tf not in known_tfs:
        logger.warning("Base TF '%s' not found among columns: %s", base_tf, sorted(known_tfs))

    for s in specs:
        try:
            out = _apply_spec_v2(out, s, base_tf=base_tf)
        except Exception as ex:
            logger.exception("Failed to apply spec '%s': %s", s, ex)

    # پاکسازی نهایی
    out = out[~out.index.duplicated(keep="last")]
    return out
