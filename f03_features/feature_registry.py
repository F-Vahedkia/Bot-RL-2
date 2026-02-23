# -*- coding: utf-8 -*-
# f03_features/feature_registry.py
# Status in (Bot-RL-2): Completed

r"""
رجیستری یکپارچه: core + extras + volume + patterns + levels + divergences
فرمان اجرای برنامه از ریشه:
python -m f03_features.indicators --list
"""
# =============================================================================
# Imports(1) & Logger
# =============================================================================
from __future__ import annotations
from typing import Any, Callable, Dict, Literal, Optional
import logging
import numpy as np
import pandas as pd

from f03_features.indicators.zigzag import zigzag_mtf_adapter
from f10_utils.config_ops import _deep_get
from f10_utils.config_loader import ConfigLoader  # از f01_config/config.yaml می‌خواند
_loader = ConfigLoader()                          # به‌طور پیش‌فرض f01_config/config.yaml را لود می‌کند
_CONFIG_CACHE = _loader.get_all()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# SR config injection (common/component/overrides by (Symbol, TF))
# =============================================================================
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


# =============================================================================
# ADV. Adapters
# =============================================================================
# نکتهٔ مهم قرارداد:
#  هر آداپتر فقط Dict[str, pd.Series] برمی‌گرداند که با ایندکس df ورودی هم‌تراز است؛
# موتور v2 خودش نام نهایی را به‌شکل __{name}@{TF}__{key} می‌سازد

# --- fibo_features_full ------------------------------------------------------ Func.1
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
        d = _nearest_level_distance(float(p), levels)
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

# --- rsi --------------------------------------------------------------------- Func.2
def _rsi_adapter(ohlc, n: int = 14, **_) -> Dict[str, pd.Series]:
    return {f"rsi_{n}": rsi_core(ohlc["close"], n)}

# --- ema --------------------------------------------------------------------- Func.3
def _ema_adapter(ohlc, col: str = "close", n: int = 20, **_) -> Dict[str, pd.Series]:
    return {f"ema_{col}_{n}": ema_core(ohlc[col], n)}

# --- adr --------------------------------------------------------------------- Func.4 (edited 041204) 
# def _adv_adr(df, window: int = 14, tz: str = "UTC", **_) -> Dict[str, pd.Series]:
def _adv_adr(df, **cfg) -> Dict[str, pd.Series]:
    # --- read config ----------------- start
    conf_all = _CONFIG_CACHE
    params = _deep_get(conf_all, "features.adr", {}) or {}

    window = int(params.get("window", params.get("w", 14)))
    tz     = params.get("tz", "UTC")
    # --- read config ----------------- end

    """
    آداپتر ADR:
      - ورودی: df با high/low (و ایندکس UTC)
      - خروجی: {'adr_<window>': Series}
    contract: Dict[str, Series] هم‌تراز با df.index
    """
    s = _compute_adr(df, window=int(window), tz=tz)
    # اطمینان از Series و dtype
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=df.index)
    s = s.astype("float32")
    key = f"adr_{int(window)}"
    return {key: s}

# --- adr_distance_to_open ---------------------------------------------------- Func.5 (edited 041204)
# def _adv_adr_distance_to_open(df, window: int = 14, tz: str = "UTC", **_) -> Dict[str, pd.Series]:
def _adv_adr_distance_to_open(df, **cfg) -> Dict[str, pd.Series]:
    # --- read config ----------------- start
    conf_all = _CONFIG_CACHE
    params = _deep_get(conf_all, "features.adr_distance_to_open", {}) or {}

    window = int(params.get("window", 14))
    tz     = params.get("tz", "UTC")
    # --- read config ----------------- end

    """
    فاصله تا open روز (نرمال‌شده با ADR):
    - اگر خروجی levels ستون 'dist_pct' نداشته باشد، اینجا با dist_abs/adr محاسبه می‌کنیم.
    - نام ستون‌های ورودی را با چند alias چک می‌کنیم.
    خروجی: { 'adr_day_open_<w>', 'adr_dist_abs_<w>', 'adr_dist_pct_<w>' }
    """
    w = int(window)
    adr = _compute_adr(df, window=w, tz=tz)
    # اطمینان از Series
    if not isinstance(adr, pd.Series):
        adr = pd.Series(adr, index=df.index)

    out = _adr_distance_to_open(df, adr=adr, tz=tz)
    # همگن‌سازی: DataFrame و نام ستون‌ها
    if isinstance(out, pd.Series):
        out = out.to_frame("dist_abs")

    # aliasها
    aliases = {
        "day_open": ["day_open", "open_day", "open_d", "open"],
        "dist_abs": ["dist_abs", "distance_abs", "dist"],
        "dist_pct": ["dist_pct_of_adr", "dist_pct", "distance_pct_of_adr", "distance_pct"],
    }

    def pick(name_group: list[str]) -> Optional[pd.Series]:
        for nm in name_group:
            if nm in out.columns:
                return out[nm]
        return None

    s_open = pick(aliases["day_open"])
    s_abs  = pick(aliases["dist_abs"])
    s_pct  = pick(aliases["dist_pct"])

    # اگر بعضی ستون‌ها نبودند، خودمان بسازیم:
    if s_open is None:
        # day_open را از df بازسازی می‌کنیم: نزدیک‌ترین open روز به سمت عقب
        # (فرض بر این‌که df دقیقه‌ای یا هم‌ترازِ broadcast است)
        # اگر در levels قبلاً درست تولید شده، این شاخه اجرا نمی‌شود.
        s_open = df["open"].copy()

    if s_abs is None:
        # |close - day_open|
        s_abs = (df["close"].astype(float) - s_open.astype(float)).abs()

    if s_pct is None:
        # dist_abs / adr  (وقتی adr>0)
        safe_adr = adr.replace(0, np.nan)
        s_pct = (s_abs.astype(float) / safe_adr.astype(float))

    # dtype و نام‌گذاری
    s_open = s_open.astype("float32")
    s_abs  = s_abs.astype("float32")
    s_pct  = s_pct.astype("float32")

    return {
        f"adr_day_open_{w}": s_open,
        f"adr_dist_abs_{w}": s_abs,
        f"adr_dist_pct_{w}": s_pct,
    }

# --- sr_overlap_score -------------------------------------------------------- Func.6 (edited 041204)
# def _adv_sr_overlap_score(df, anchor: float, step: float, n: int = 10, tol_pct: float = 0.05, **_) -> Dict[str, pd.Series]:
def _adv_sr_overlap_score(df, **cfg) -> Dict[str, pd.Series]:
    # --- read config ----------------- start
    conf_all = _CONFIG_CACHE
    params = _deep_get(conf_all, "features.sr_overlap_score", {}) or {}

    anchor  = float(params.get("anchor", params.get("a", 0.0)))
    step    = float(params.get("step", 10))
    n       = int(params.get("n", 5))
    tol_pct = float(params.get("tolerance_pct", params.get("tol_pct", 0.05)))
    # --- read config ----------------- end

    """
    امتیاز همپوشانی قیمت (close) با سطوح S/R «رُند» ساخته شده از round_levels(anchor, step, n).
    - برای هر بارِ close: score = sr_overlap_score(close_t, levels, tol_pct)
    - خروجی: {'sr_overlap_score_<step>_<n>_<tolbp>bp': Series}
      * tolbp = tol_pct * 10000 به واحد basis point برای نام امن
    """
    levels = _round_levels(anchor=anchor, step=step, n=int(n))
    # محاسبهٔ سری امتیاز
    close = df["close"].astype(float)
    vals = [float(_sr_overlap_score(float(px), levels, tol_pct=float(tol_pct))) for px in close]
    s = pd.Series(vals, index=df.index, name="sr_overlap_score").astype("float32")
    # نام ایمن (بدون %/ممیز)
    tolbp = int(round(float(tol_pct) * 10000))
    key = f"sr_overlap_score_{str(step).replace('.','_')}_{int(n)}_{tolbp}bp"
    return {key: s}

# --- round_levels (nearest distance) ----------------------------------------- Func.7
def _adv_round_levels(df, anchor: float, step: float, n: int = 10, **_) -> Dict[str, pd.Series]:
    """
    آداپتر round_levels: برای هر close فاصله تا نزدیک‌ترین سطح رُند.
    - levels = round_levels(anchor, step, n)
    - nearest_level_distance(price_t, levels) → {'nearest_level','signed','abs'}
    خروجی: سه سری هم‌نام با suffix:
       {'rl_nearest_<step>_<n>', 'rl_signed_<step>_<n>', 'rl_abs_<step>_<n>'}
    """
    levels = _round_levels(anchor=anchor, step=step, n=int(n))
    close = df["close"].astype(float)
    nearest_list = []
    signed_list = []
    abs_list = []
    for px in close:
        d = _nearest_level_distance(float(px), levels)
        nearest_list.append(d["nearest_level"])
        signed_list.append(d["signed"])
        abs_list.append(d["abs"])
    s_nearest = pd.Series(nearest_list, index=df.index, name="rl_nearest").astype("float32")
    s_signed  = pd.Series(signed_list,  index=df.index, name="rl_signed").astype("float32")
    s_abs     = pd.Series(abs_list,     index=df.index, name="rl_abs").astype("float32")
    tag = f"{str(step).replace('.','_')}_{int(n)}"
    return {
        f"rl_nearest_{tag}": s_nearest,
        f"rl_signed_{tag}":  s_signed,
        f"rl_abs_{tag}":     s_abs,
    }



# =============================================================================
# Build first part of registry by simple indicators registry summation
# =============================================================================
def build_registry() -> Dict[str, Callable[..., Any]]:
    reg: Dict[str, Callable[..., Any]] = {}
    from .indicators.core           import registry as core_reg;   reg.update(core_reg())
    from .indicators.divergences    import registry as div_reg;    reg.update(div_reg())
    from .indicators.extras_channel import registry as ch_reg;     reg.update(ch_reg())
    from .indicators.extras_trend   import registry as trend_reg;  reg.update(trend_reg())
    from .indicators.levels         import registry as lvl_reg;    reg.update(lvl_reg())
    from .indicators.patterns       import registry as pat_reg;    reg.update(pat_reg())
    from .indicators.volume         import registry as vol_reg;    reg.update(vol_reg())
    return reg



# =============================================================================
# Constructing Simple Registry in ((Module Space))
# =============================================================================
REGISTRY: Dict[str, Callable[..., Any]] = build_registry()


# =============================================================================
# Imports(2) from f03_features/indicators
# =============================================================================
from .indicators.core import rsi as rsi_core, ema as ema_core   # Used in: _..._adapter()
from .indicators.extras_trend import ma_slope, rsi_zone         # Used in: _ADV
from .indicators.fibo_pipeline import run_fibo_cluster          # Used in: _fibo_features_full_adapter()
from .indicators.fibonacci import (
    golden_zone,                                     # Used in: _ADV
    fib_cluster,                                     # Used in: _ADV
    fib_ext_targets,                                 # Used in: _ADV
    levels_from_legs,                                # Used in: _ADV
    select_legs_from_swings)                         # Used in: _ADV
from .indicators.levels import (
    compute_adr as _compute_adr,                     # Used in: _adv_adr, _adv_adr_distance_to_open
    adr_distance_to_open as _adr_distance_to_open,   # Used ih: _adv_adr_distance_to_open
    sr_overlap_score as _sr_overlap_score)           # Used in: _adv_sr_overlap_score
# --- Advanced Support/Resistance ---
from .indicators.sr_advanced import (
    make_fvg,          # FVG detector (advanced S/R) # Used in: _ADV
    make_sd,           # Supply/Demand               # Used in: _ADV
    make_ob,           # Order Block                 # Used in: _ADV
    make_liq_sweep,    # Liquidity Sweep             # Used in: _ADV
    make_breaker_flip, # Breaker/Flip Zone           # Used in: _ADV
    make_sr_fusion,    #                             # Used in: _ADV
)
from .indicators.utils import (
    round_levels as _round_levels,                     # Used in: _adv_sr_overlap_score, _adv_round_levels
    nearest_level_distance as _nearest_level_distance, # Used in: _fibo_features_full_adapter, _adv_round_levels
    compute_atr)                                       # Used in: _fibo_features_full_adapter


# =============================================================================
# Constructing Advanced Registry in ((Module Space))
# =============================================================================
_ADV: Dict[str, Callable[..., Any]] = {
    # اندیکاتورهای پایه (برای اطمینان از وجود در رجیستری) -----------
    "rsi": _rsi_adapter,      # <== Adapters <== core.py
    "ema": _ema_adapter,      # <== Adapters <== core.py

    # ترندی/ممنتوم سبک
    "ma_slope": ma_slope,     # <== extras_trend.py
    "rsi_zone": rsi_zone,     # <== extras_trend.py

    # فیبوناچی ------------------------------------------------------
    "fibo_features_full": _fibo_features_full_adapter,   # <== Function of this module
    "golden_zone": golden_zone,                          # <== fibonacci.py                       
    "fib_cluster": fib_cluster,                          # <== fibonacci.py
    "fib_ext_targets": fib_ext_targets,                  # <== fibonacci.py
    # فیبو — هِلپرهای پیشرفته/آماده برای استفادهٔ مستقیم --
    "levels_from_legs": levels_from_legs,                # <== fibonacci.py
    "select_legs_from_swings": select_legs_from_swings,  # <== fibonacci.py

    # هلپرهای سطوح/ابزارهای کمکی ------------------------------------
    "adr": _adv_adr,                                     # <== fibonacci.py 
    "adr_distance_to_open": _adv_adr_distance_to_open,   # <== fibonacci.py 
    "sr_overlap_score": _adv_sr_overlap_score,           # <== fibonacci.py 
    "round_levels": _adv_round_levels,                   # <== fibonacci.py 

    # Advanced Support/Resistance -----------------------------------
    "fvg": make_fvg,                     # <== sr_advanced
    "supply_demand": make_sd,            # <== sr_advanced
    "order_block": make_ob,              # <== sr_advanced
    "liq_sweep": make_liq_sweep,         # <== sr_advanced
    "breaker_flip": make_breaker_flip,   # <== sr_advanced
    "sr_fusion": make_sr_fusion,         # <== sr_advanced
    "zigzag_mtf": zigzag_mtf_adapter,    # <== zigzag.py
}

# =============================================================================
# wrap S/R indicators to inject merged config (common/component/overrides)
# =============================================================================
for _name in ("fvg", "supply_demand", "order_block", "liq_sweep", "breaker_flip", "sr_fusion"):
    _fn = _ADV[_name]
    def _wrap(fn, name):
        def runner(df: pd.DataFrame, **cfg):
            merged = _merge_sr_kwargs(name, cfg, df)
            return fn(df, **merged)  # sr_advanced.make_* امضای **cfg دارد
        return runner
    _ADV[_name] = _wrap(_fn, _name)

# =============================================================================
# افزودن اندیکاتورهای پیشرفته به رجیستری ساده اولیه
# =============================================================================
REGISTRY.update(_ADV)  

# =============================================================================
# افزودن رجیستری پرایس اکشن به رجیستری اندیکاتورهای ساده و پیشرفته
# =============================================================================
from f03_features.price_action.registry_adapter import register_price_action_to_indicators_registry
register_price_action_to_indicators_registry(REGISTRY)

# ---------------------------------------------------------------------------

def get_indicator(name: str) -> Optional[Callable[..., Any]]:
    """
    [Bot-RL-2] Unified lookup from the single REGISTRY (no legacy/advanced labels).
    """
    key = str(name).strip()
    fn = REGISTRY.get(key)
    if fn is None:
        logger.warning("Indicator not found in unified registry: %s", name)
    return fn


def list_all_indicators(include_legacy: bool = True) -> Dict[str, str]:
    """
    [Bot-RL-2] Report from unified REGISTRY (no legacy/advanced labels).
    """
    return {k: "unified" for k in REGISTRY.keys()}

'''
نکات کوتاه:
Engine: برای fib_cluster اگر DF پاس بدهی، به‌خاطر mismatch یک‌بار TypeError می‌خورد
 و مسیر fallback (بدون DF) فعال می‌شود—این همان طراحی افزایشی قبلی ماست.

CLI:
 هِلپرهایی مثل compute_adr و adr_distance_to_open را می‌توان جداگانه روی دیتاست اجرا کرد
 و ستون‌هایشان را به خروجی افزود.
'''
# =============================================================================
# Explicit public API
# =============================================================================
__all__ = [
    "REGISTRY",
    "get_indicator",
    "list_all_indicators",
]
