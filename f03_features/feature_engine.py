# -*- coding: utf-8 -*-
# f03_features/indicators/engine.py
# Status in (Bot-RL-2): Completed
r"""IndicatorEngine: اجرای Specc ها روی دیتای MTF و تولید DataFrame ویژگی‌ها"""

# =============================================================================
# Imports & Logger
# =============================================================================
from __future__ import annotations
from typing import Iterable, List, Dict, Optional
import numpy as np
import pandas as pd
import re
import logging

# پارسر/رجیستری نسخهٔ جدید
from .indicators.parser import parse_spec, ParsedSpec
from .feature_registry import get_indicator

# ابزارهای عمومی موتور
from .indicators.utils import nan_guard, get_ohlc_view

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# دامنهٔ مجاز برای asof-merge (اختیاری): اگر لازم شد قابل تنظیم است
_ASOF_TOL = pd.Timedelta("3min")
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


def _ensure_utc_index(x: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """ایندکس را UTC و مرتب نگه می‌داریم."""
    if isinstance(x.index, pd.DatetimeIndex):
        if x.index.tz is None:
            x.index = x.index.tz_localize("UTC")
        else:
            x.index = x.index.tz_convert("UTC")
    return x


# ---------------------------------------------------------------------------
# Resample helpers for building OHLC views when only base OHLC exists
# English log strings (runtime); Persian comments (educational)
# ---------------------------------------------------------------------------
def _extract_base_ohlc_from_plain_df(df: pd.DataFrame) -> pd.DataFrame:
    """اگر ستون‌های خام OHLC (open/high/low/close[/volume]) موجود است همان‌ها را برگردان."""
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    if not cols:
        raise ValueError("Base OHLC columns not found in input frame")
    return df[cols].copy()


def _tf_to_rule(tf: str) -> str:
    """
    نگاشت TF پروژه به فرکانس پانداس:
      Mx → xT (minute), Hx → xH, Dx → xD, Wx → xW
    توجه: فقط برای مقیاس‌های دقیقه/ساعت/روز/هفته استفاده می‌شود.
    """
    tf = str(tf).upper().strip()
    m = re.fullmatch(r"([MHDW])(\d+)", tf)
    if not m:
        raise ValueError(f"Unsupported timeframe format: {tf}")
    unit, n = m.group(1), int(m.group(2))
    if unit == "M":  # minutes
        return f"{n}min"
    if unit == "H":  # hours
        return f"{n}h"
    if unit == "D":  # days
        return f"{n}D"
    if unit == "W":  # weeks
        return f"{n}W"
    raise ValueError(f"Unsupported timeframe unit: {unit}")


def _resample_ohlc_from_base(base: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    از نمای پایه (دقیقه‌ای) به TF بالاتر Resample می‌کند.
    Open=first, High=max, Low=min, Close=last, Volume=sum (اگر موجود باشد).
    """
    rule = _tf_to_rule(target_tf)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in base.columns:
        agg["volume"] = "sum"
    # label/closed='right' برای سازگاری با backfill/asof در ادغام‌ها
    out = base.resample(rule, label="right", closed="right").agg(agg)
    # پاکسازی اندیس/ناقص‌ها و UTC
    out = out.dropna(how="all")
    return _ensure_utc_index(out)


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
    
    # تعیین tolerance پویا: یک عرض کندلِ دیتای base (مثلاً M1≈60s)
    _tol = b["time"].sort_values().diff().median()
    if pd.isna(_tol) or _tol <= pd.Timedelta(0):
        _tol = pd.Timedelta("60s")  # fallback امن برای دیتای کم یا یکنواخت‌نشده        
    merged = pd.merge_asof(
        b.sort_values("time"),
        f.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
        tolerance=_tol,   # ← اعمال محدودیت فاصلهٔ زمانی
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
    fn = get_indicator(ps.name)
    if fn is None:
        logger.warning("Unknown indicator in v2 registry: %s", ps.name)
        return None

    # ------------------------------------------------------------------------ added: 040706 start
    # کش نمای OHLC
    ohlc = view_cache.get(tf)
    if ohlc is None:
        try:
            # تلاش اول: از util (اگر خودش قادر به ساخت/اسلایس باشد)
            ohlc = get_ohlc_view(df_in, tf)
        except Exception as e:
            # Log hygiene: primary view نبود؛ به‌جای ERROR، INFO لاگ کن و fallback را امتحان کن
            logger.info("Primary OHLC view not available for TF '%s': %s. Trying resample fallback ...", tf, e)
            try:
                # مرحله دوم: از ورودی نمای پایه را استخراج کن
                base_view = view_cache.get(base_tf)
                if base_view is None:
                    try:
                        base_view = get_ohlc_view(df_in, base_tf)
                    except Exception:
                        base_view = _extract_base_ohlc_from_plain_df(df_in)

                # اگر خود TF درخواست‌شده همان base است → همان را بده
                if str(tf) == str(base_tf):
                    ohlc = base_view.copy()
                else:
                    # وگرنه از base به TF مقصد Resample کن
                    ohlc = _resample_ohlc_from_base(base_view, tf)

                # اگر به اینجا رسیدیم یعنی fallback موفق بوده است
                logger.info("Built OHLC view for TF '%s' via resample fallback from base '%s'.", tf, base_tf)

            except Exception as e2:
                # فقط اگر fallback هم شکست بخورد، ERROR لاگ کن
                logger.error("Failed to build OHLC@%s from base '%s': %s", tf, base_tf, e2)
                return None
        view_cache[tf] = ohlc

    # ------------------------------------------------------------------------ added: 040706 end
    # اگر اندیکاتور به tf_dfs/symbol نیاز دارد (مثل fibo_features_full)
    if ps.name in {"fibo_features_full", "fibo_run"}:
        # ساخت map نمای OHLC همهٔ TFهای موجود
        #tf_dfs = {t: (view_cache.get(t) or get_ohlc_view(df_in, t)) for t in _available_timeframes(df_in)}
        tf_dfs = {t: (view_cache[t] if t in view_cache 
                      else view_cache.setdefault(t, get_ohlc_view(df_in, t))) 
                      for t in _available_timeframes(df_in)}

        for t, v in tf_dfs.items():
            view_cache.setdefault(t, v)
        # تلاش برای استخراج symbol از ستون‌ها/متادیتا (fallback: "SYMBOL")
        symbol = getattr(df_in, "attrs", {}).get("symbol", "SYMBOL")
        result = get_indicator(ps.name)(symbol=symbol, tf_dfs=tf_dfs, base_tf=(ps.timeframe or base_tf))
        return _ensure_utc_index(pd.DataFrame(result))

    # اجرای تابع اندیکاتور (قرارداد: df اول و سپس args/kwargs)
    try:
        result = fn(ohlc, *ps.args, **ps.kwargs)
    except TypeError:
        # برخی توابع ممکن است df نخواهند
        result = fn(*ps.args, **ps.kwargs)

    if result is None:
        return None

    # نرمال‌سازی خروجی به DataFrame با نام‌گذاری استاندارد (dict/Series/DataFrame/scalar-safe)
    if isinstance(result, dict):
        df_feat = pd.DataFrame(result)
        renamed: Dict[str, str] = {}
        for c in df_feat.columns:
            safe = str(c) if (c is not None and str(c) != "") else "val"
            renamed[c] = f"__{ps.name}@{tf}__{safe}"
        df_feat.rename(columns=renamed, inplace=True)

    elif isinstance(result, pd.Series):
        df_feat = result.rename(f"__{ps.name}@{tf}__val").to_frame()

    elif isinstance(result, pd.DataFrame):
        df_feat = result.copy()
        renamed: Dict[str, str] = {}
        for c in df_feat.columns:
            safe = str(c) if (c is not None and str(c) != "") else "val"
            renamed[c] = f"__{ps.name}@{tf}__{safe}"
        df_feat.rename(columns=renamed, inplace=True)

    else:
        # اسکالر عددی → سری float؛ سایر انواع → سری object (بدون هشدار dtype)
        if np.isscalar(result) and isinstance(result, (int, float, np.integer, np.floating)):
            ser = pd.Series(result, index=df_in.index, name=f"__{ps.name}@{tf}__val", dtype="float32")
        else:
            ser = pd.Series(str(result), index=df_in.index, name=f"__{ps.name}@{tf}__val", dtype="object")
        df_feat = ser.to_frame()

    return _ensure_utc_index(df_feat)


def run_specs_v2(df: pd.DataFrame, specs: Iterable[str], base_tf: str) -> pd.DataFrame:
    # de-duplicate specs while preserving order (prevents duplicate columns later)
    # معنی: حذف اسپکهای تکراری در ابتدای اجرای همین تابع
    specs = list(dict.fromkeys(specs))
    
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
    parsed: List[ParsedSpec] = [parse_spec(s) for s in specs]
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

        # ===== normalize feature outputs (Series/dict → DataFrame) =====
        # تبدیل خروجی‌های تابع اندیکاتور به DataFrame یکنواخت برای concat
        norm_cols = []
        for f in feat_cols:
            # اگر یک map از نام‌ستون→Series باشد، آن را به DataFrame تبدیل می‌کنیم
            if isinstance(f, dict):
                norm_cols.append(pd.DataFrame(f))
            # اگر یک Series منفرد باشد، به DataFrame یک‌ستونه تبدیل می‌کنیم
            elif isinstance(f, pd.Series):
                norm_cols.append(f.to_frame())
            else:
                norm_cols.append(f)
        feat_cols = norm_cols
        # ===== end normalize =====

        # افقی‌سازی فیچرهای همان TF
        tf_feats = pd.concat(feat_cols, axis=1)

        # سبک‌سازی: تبدیل نوع/پر کردن gaps سبک و حذف ستون‌های کاملاً تهی
        tf_feats = nan_guard(tf_feats).dropna(axis=1, how="all")

        # تبدیل نوع ستون‌های عددی به float32 برای پایداری و کاهش حافظه
        _num_cols = tf_feats.select_dtypes(include=[np.number]).columns
        tf_feats[_num_cols] = tf_feats[_num_cols].astype("float32")

        # ادغام: اگر ایندکس‌ها هم‌ترازند join، وگرنه merge_asof backward
        if merged.index.equals(tf_feats.index):
            # drop overlapping columns to avoid pandas join collision --- start
            overlap = tf_feats.columns.intersection(merged.columns)
            if len(overlap) > 0:
                logger.info("FeatureEngine: dropping %d overlapping cols before join: first=%s",
                        len(overlap), overlap[:1].tolist())
                tf_feats = tf_feats.drop(columns=list(overlap))
            # --- drop overlapping ... --- end
            merged = merged.join(tf_feats, how="left")
        else:
            merged = _merge_on_base(merged, tf_feats)

    # یکتاسازی نام ستون‌ها (حذف ستون‌های تکراری با حفظ اولین رخداد)
    dup = merged.columns[merged.columns.duplicated()]
    if len(dup) > 0:
        # runtime log: English
        logging.getLogger(__name__).warning(
            "FeatureEngine: dropping %d duplicated cols: first=%s",
            len(dup), dup[:1].tolist()
        )
        merged = merged.loc[:, ~merged.columns.duplicated(keep="first")]

    # یکتا‌سازی ایندکس و برگرداندن خروجی
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


# ===============================
# PATCH: Representation application
# Applies binary / ordinal / continuous transformation in engine only
# ===============================
def apply_representation(feature_df: pd.DataFrame, representation_config: dict) -> pd.DataFrame:
    """
    Transform feature dataframe according to representation configuration.
    
    :param feature_df: dataframe with raw features from indicators / price_action
    :param representation_config: dict with keys:
        - 'default': default representation ('binary', 'continuous', etc.)
        - 'mapping': dict mapping feature names to representation
    :return: dataframe with features transformed as specified
    """
    df = feature_df.copy()
    default = representation_config.get("default", "binary")
    mapping = representation_config.get("mapping", {})

    for col in df.columns:
        rep = mapping.get(col, default)
        if rep == "binary":
            df[col] = (df[col] != 0).astype(int)
        elif rep.startswith("ordinal"):
            # parse number of levels
            try:
                n_levels = int(rep.replace("ordinal", ""))
            except:
                n_levels = 3
            df[col] = pd.qcut(df[col], q=n_levels, labels=False, duplicates='drop')
        elif rep == "continuous":
            pass  # leave as-is
        else:
            raise ValueError(f"Unknown representation '{rep}' for feature '{col}'")
    return df

# ===============================
# Example integration inside feature_engine.py
# Suppose feature_df = engine collects raw features
# representation_config = loaded from config
# feature_df = apply_representation(feature_df, representation_config)
# ===============================

# ============================================================================= Added: َ040705
# --- Generic entry points for integration test (single robust wrapper) ---
def _entry(df: pd.DataFrame, specs: Iterable[str]) -> pd.DataFrame:
    try:
        out = run_specs_v2(df, specs, base_tf="M1")
    except Exception:
        logger.exception("Engine entry failed; fallback to passthrough.")
        out = df.copy()
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, index=df.index)
    out = out.reindex(df.index).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return out.ffill().bfill()


apply = _entry
run = _entry
execute = _entry

# =============================================================================