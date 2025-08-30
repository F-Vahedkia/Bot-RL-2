# f04_features/indicators/engine.py
# -*- coding: utf-8 -*-
r"""IndicatorEngine: اجرای Specها روی دیتای MTF و تولید DataFrame ویژگی‌ها"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Any
import logging
import numpy as np
import pandas as pd

# پارسر/رجیستری نسخهٔ جدید (v2)
from f04_features.indicators.parser import parse_spec_v2, ParsedSpec
from f04_features.indicators.registry import get_indicator_v2

# ابزارهای موتور قدیمی (برای سازگاری عقب‌رو)
from .utils import detect_timeframes, slice_tf, nan_guard
from .parser import parse_spec

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# موتور قدیمی (سازگاری کامل؛ بدون تغییر مخرب)
# =============================================================================

@dataclass
class EngineConfig:
    # لیست Specها، مثل: ["rsi(14)@M1", "ema(close,20)@M5", ...]
    specs: List[str]
    # اگر None → همهٔ TFهای موجود از روی ستون‌ها تشخیص داده می‌شود
    timeframes: Optional[List[str]] = None
    # شیفت عمومی برای جلوگیری از لوک‌اِهد
    shift_all: int = 1
    # حذف NaN‌های ابتدای سری‌ها پس از شیفت
    drop_na_head: bool = True


class IndicatorEngine:
    """
    موتور نسل اول که با رجیستری قدیمی کار می‌کرد.
    نگه‌داری می‌شود تا مصرف‌کننده‌های قدیمی نشکنند.
    """
    def __init__(self, cfg: EngineConfig, registry: Dict[str, Any]):
        self.cfg = cfg
        self.registry = registry  # نگاشت: نام اندیکاتور → تابع سازندهٔ Series map

    def _available_tfs(self, df: pd.DataFrame) -> List[str]:
        views = detect_timeframes(df)
        if self.cfg.timeframes:
            return [tf for tf in self.cfg.timeframes if tf in views]
        return list(views.keys())

    def _apply_one(self, base_df: pd.DataFrame, tf: str, spec_str: str) -> Dict[str, pd.Series]:
        """
        اجرای یک Spec روی یک TF مشخص (نسخهٔ قدیمی).
        - اگر آرگومان اول Spec نامِ ستون باشد (open/high/low/close/volume)،
          فقط اگر تابع رجیستری پارامتر «col» داشته باشد به‌صورت positional پاس می‌دهیم.
        - خروجی‌ها با پیشوند TF برگردانده می‌شوند.
        """
        import inspect  # فقط برای تشخیص پارامترهای تابع

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


# =============================================================================
# موتور جدید (v2) — بهینه‌سازی‌شده
#   - کشِ نمای OHLC برای هر TF
#   - گروه‌بندی Specها بر اساس TF و «یک Merge در هر TF»
#   - سبک‌سازی خروجی (nan_guard و حذف ستون‌های کاملاً تهی)
# =============================================================================

# کلیدهای استاندارد OHLC (مطابق دادهٔ data_handler)
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
    خروجی: DataFrame با ستون‌های ['open','high','low','close','tick_volume','spread'] (هر کدام که موجود باشد)
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


def _ensure_utc_index(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """ایندکس را UTC و مرتب نگه می‌داریم."""
    if isinstance(x.index, pd.DatetimeIndex):
        if x.index.tz is None:
            x.index = x.index.tz_localize("UTC")
        else:
            x.index = x.index.tz_convert("UTC")
    return x


def _merge_on_base(base: pd.DataFrame, feat: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    English: Backward merge_asof if indices are not exactly aligned.
    Persian: اگر ایندکس‌ها دقیقاً هم‌تراز نباشند، merge_asof(backward) وگرنه join ساده.
    """
    base = _ensure_utc_index(base.copy())
    feat = _ensure_utc_index(feat.copy())

    if isinstance(feat, pd.Series):
        feat = feat.to_frame()

    if base.index.equals(feat.index):
        return base.join(feat, how="left")

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


def _compute_feature_df_v2(
    df_in: pd.DataFrame,
    ps: ParsedSpec,
    base_tf: str,
    view_cache: Dict[str, pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    محاسبهٔ «فقط» DataFrame فیچر برای یک Spec:
      - نمای OHLC همان TF را از کش می‌خوانیم/می‌سازیم
      - تابع اندیکاتور رجیستری v2 را اجرا می‌کنیم
      - خروجی را به DataFrame با پیشوند __{name}@{TF}[__col] تبدیل می‌کنیم
      - هیچ ادغامی اینجا انجام نمی‌شود (مرحلهٔ بعدی)
    """
    tf = (ps.timeframe or base_tf).upper()
    fn = get_indicator_v2(ps.name)
    if fn is None:
        logger.warning("Unknown indicator in v2 registry: %s", ps.name)
        return None

    # کش نمای OHLC
    ohlc = view_cache.get(tf)
    if ohlc is None:
        try:
            ohlc = _get_ohlc_view(df_in, tf)
        except Exception as e:
            logger.error("OHLC view for TF '%s' not available: %s", tf, e)
            return None
        view_cache[tf] = ohlc

    # اجرای تابع اندیکاتور (قرارداد: df اول و سپس args/kwargs)
    try:
        result = fn(ohlc, *ps.args, **ps.kwargs)
    except TypeError:
        # برخی توابع ممکن است df نخواهند
        result = fn(*ps.args, **ps.kwargs)

    if result is None:
        return None

    # نرمال‌سازی خروجی به DataFrame با نام‌گذاری استاندارد
    if isinstance(result, pd.Series):
        df_feat = result.rename(f"__{ps.name}@{tf}").to_frame()
    elif isinstance(result, pd.DataFrame):
        df_feat = result.copy()
        renamed: Dict[str, str] = {}
        for c in df_feat.columns:
            safe = str(c) if (c is not None and str(c) != "") else "val"
            renamed[c] = f"__{ps.name}@{tf}__{safe}"
        df_feat.rename(columns=renamed, inplace=True)
    else:
        # اسکالر/لیست → ستون ثابت روی ایندکس ورودی
        ser = pd.Series(np.nan, index=df_in.index, name=f"__{ps.name}@{tf}")
        try:
            ser.iloc[:] = float(result)
        except Exception:
            ser.iloc[:] = str(result)
        df_feat = ser.to_frame()

    return df_feat


def run_specs_v2(df: pd.DataFrame, specs: Iterable[str], base_tf: str) -> pd.DataFrame:
    """
    اجرای چند Spec (نسخهٔ بهینه‌شده):
      - Specها را بر اساس TF گروه‌بندی می‌کنیم
      - برای هر TF: همهٔ فیچرهای آن TF را محاسبه و «یک‌بار» ادغام می‌کنیم
      - کش نمای OHLC برای کاهش هزینهٔ slicing
      - سبک‌سازی خروجی با nan_guard و حذف ستون‌های کاملاً تهی
    """
    out = _ensure_utc_index(df.copy()).sort_index()
    base_tf = base_tf.upper()

    known_tfs = set(_available_timeframes(out))
    if base_tf not in known_tfs:
        logger.warning("Base TF '%s' not found among columns: %s", base_tf, sorted(known_tfs))

    # پارس همهٔ Specها و گروه‌بندی بر اساس TF مقصد
    parsed: List[ParsedSpec] = [parse_spec_v2(s) for s in specs]
    groups: Dict[str, List[ParsedSpec]] = {}
    for ps in parsed:
        tf = (ps.timeframe or base_tf).upper()
        groups.setdefault(tf, []).append(ps)

    # کش نمای OHLC برای هر TF
    view_cache: Dict[str, pd.DataFrame] = {}

    merged = out
    for tf, ps_list in groups.items():
        feat_cols: List[pd.DataFrame] = []
        for ps in ps_list:
            try:
                df_feat = _compute_feature_df_v2(merged, ps, base_tf=base_tf, view_cache=view_cache)
                if df_feat is not None and not df_feat.empty:
                    feat_cols.append(df_feat)
            except Exception as ex:
                logger.exception("Failed to compute feature for spec '%s': %s", ps.raw, ex)

        if not feat_cols:
            continue

        # افقی‌سازی فیچرهای همان TF
        tf_feats = pd.concat(feat_cols, axis=1)
        # سبک‌سازی: تبدیل نوع/پر کردن gaps سبک و حذف ستون‌های کاملاً تهی
        tf_feats = nan_guard(tf_feats).dropna(axis=1, how="all")

        # ادغام: اگر ایندکس‌ها هم‌ترازند join، وگرنه merge_asof backward
        if merged.index.equals(tf_feats.index):
            merged = merged.join(tf_feats, how="left")
        else:
            merged = _merge_on_base(merged, tf_feats)

    # یکتا‌سازی ایندکس و برگرداندن خروجی
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged
