# f03_features/feature_registry.py
# Status in (Bot-RL-2): 

r"""
رجیستری یکپارچه: core + extras + volume + patterns + levels + divergences
فرمان اجرای برنامه از ریشه:
python -m f03_features.indicators --list
"""
# =============================================================================
# Imports(1) & Logger
# =============================================================================
from __future__ import annotations
from typing import Any, Callable, Dict, List, Literal, Optional, Set
from dataclasses import dataclass, field
# import numpy as np
import pandas as pd
import logging

from f03_features.indicators.levels import merge_sr_kwargs
from f10_utils.config_operations import _deep_get
# from f10_utils.config_loader import ConfigLoader  # از f01_config/config.yaml می‌خواند
# _loader = ConfigLoader()                          # به‌طور پیش‌فرض f01_config/config.yaml را لود می‌کند
# _CONFIG_CACHE = _loader.get_all()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# IndicatorSpec — metadata هر اندیکاتور در رجیستری
# ============================================================================= by Claude 05/02/24

@dataclass
class IndicatorSpec:
    name: str                          # نام یکتا در رجیستری
    fn: Callable                       # تابع محاسباتی
    modes: Set[str] = field(           # mode های پشتیبانی‌شده
        default_factory=lambda: {"train", "backtest", "live"}
    )
    needs_tf_map: bool = False         # آیا به tf_dfs (همه TFها) نیاز دارد؟ (مثل fibo)
    is_stateful: bool = False          # آیا state بین کندل‌ها نگه می‌دارد؟
    required_cols: list = field(       # ستون‌های ورودی لازم
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    def supports(self, mode: str) -> bool:
        """
        آیا این اندیکاتور مُد داده‌شده را پشتیبانی می‌کند؟
        و به بیان دیگر در این مُد قابل اجرا است؟
        """
        return mode in self.modes

    def __call__(self, *args, **kwargs):
        """
        IndicatorSpec مستقیماً قابل فراخوانی است (مثل callable ساده).
        در واقع، این متد امکان استفاده مستقیم از شیء به صورت تابع را فراهم میسازد
        """
        return self.fn(*args, **kwargs)



# =============================================================================
# ADV. Adapters
# =============================================================================
# نکتهٔ مهم قرارداد:
#  هر آداپتر فقط Dict[str, pd.Series] برمی‌گرداند که با ایندکس df ورودی هم‌تراز است؛
# موتور v2 خودش نام نهایی را به‌شکل __{name}@{TF}__{key} می‌سازد

from f03_features.indicators.fibo_pipeline import _fibo_features_full_adapter


# =============================================================================
# Build first part of registry by simple indicators registry summation
# =============================================================================
def build_registry() -> Dict[str, "IndicatorSpec"]:
    """
    Construct base indicator registry from all submodules.
    
    Returns:
        dict: name → IndicatorSpec
    """
    reg: Dict[str, "IndicatorSpec"] = {}
    from ..indicators.core           import registry as core_reg;   reg.update(core_reg())
    from ..indicators.divergences    import registry as div_reg;    reg.update(div_reg())
    from ..indicators.extras_channel import registry as ch_reg;     reg.update(ch_reg())
    from ..indicators.extras_trend   import registry as trend_reg;  reg.update(trend_reg())
    from ..indicators.levels         import registry as lvl_reg;    reg.update(lvl_reg())
    from ..indicators.patterns       import registry as pat_reg;    reg.update(pat_reg())
    from ..indicators.volume         import registry as vol_reg;    reg.update(vol_reg())
    
    # هر callable ساده را به IndicatorSpec تبدیل می‌کنیم
    wrapped = {}
    for name, fn in reg.items():
        if isinstance(fn, IndicatorSpec):
            wrapped[name] = fn
        else:
            wrapped[name] = IndicatorSpec(
                name=name,
                fn=fn,
                modes={"train", "backtest", "live"},
            )
    return wrapped


# =============================================================================
# Constructing Simple Registry in ((Module Space))
# =============================================================================
REGISTRY: Dict[str, "IndicatorSpec"] = build_registry()


# =============================================================================
# Imports(2) from f03_features/indicators
# =============================================================================
# from .indicators.core import rsi as rsi_core, ema as ema_core   # Used in: _..._adapter()
from ..indicators.extras_trend import (
    ma_slope_multistep as ma_slope,                  # Used in: _ADV
    rsi_zone)                                        # Used in: _ADV
# from .indicators.fibo_pipeline import run_fibo_cluster          # Used in: _fibo_features_full_adapter()
from ..indicators.fibonacci import (
    golden_zone,                                     # Used in: _ADV
    fib_cluster,                                     # Used in: _ADV
    fib_ext_targets,                                 # Used in: _ADV
    levels_from_legs,                                # Used in: _ADV
    select_legs_from_swings)                         # Used in: _ADV
from ..indicators.levels import (
    _adv_adr,
    _adv_adr_distance_to_open,
    _adv_sr_overlap_score,
    _adv_round_levels,
    # compute_adr as _compute_adr,                     # Used in: _adv_adr, _adv_adr_distance_to_open
    # adr_distance_to_open as _adr_distance_to_open,   # Used ih: _adv_adr_distance_to_open
    # sr_overlap_score as _sr_overlap_score            # Used in: _adv_sr_overlap_score
)
# --- Advanced Support/Resistance ---
from ..indicators.sr_advanced import (
    make_fvg,          # FVG detector (advanced S/R) # Used in: _ADV
    make_sd,           # Supply/Demand               # Used in: _ADV
    make_ob,           # Order Block                 # Used in: _ADV
    make_liq_sweep,    # Liquidity Sweep             # Used in: _ADV
    make_breaker_flip, # Breaker/Flip Zone           # Used in: _ADV
    make_sr_fusion,    #                             # Used in: _ADV
)
# from .indicators.utils import (
#     round_levels as _round_levels,                     # Used in: _adv_sr_overlap_score, _adv_round_levels
#     nearest_level_distance as _nearest_level_distance, # Used in: _fibo_features_full_adapter, _adv_round_levels
#     compute_atr)                                       # Used in: _fibo_features_full_adapter

from f03_features.indicators.zigzag import zigzag_mtf_adapter


# =============================================================================
# Constructing Advanced Registry in ((Module Space))
# =============================================================================
_ADV: Dict[str, "IndicatorSpec"] = {
    # اندیکاتورهای پایه -----------------------------------------------
    # "rsi": IndicatorSpec(name="rsi", fn=_rsi_adapter),
    # "ema": IndicatorSpec(name="ema", fn=_ema_adapter),

    # ترندی/ممنتوم سبک ------------------------------------------------
    "ma_slope": IndicatorSpec(name="ma_slope", fn=ma_slope),
    "rsi_zone": IndicatorSpec(name="rsi_zone", fn=rsi_zone),

    # فیبوناچی --------------------------------------------------------
    "fibo_features_full": IndicatorSpec(
        name="fibo_features_full",
        fn=_fibo_features_full_adapter,
        needs_tf_map=True,
        modes={"train", "backtest"},   # در live هنوز stateful نیست
    ),
    "golden_zone": IndicatorSpec(name="golden_zone", fn=golden_zone, needs_tf_map=True),
    "fib_cluster": IndicatorSpec(name="fib_cluster", fn=fib_cluster, needs_tf_map=True),
    "fib_ext_targets": IndicatorSpec(name="fib_ext_targets", fn=fib_ext_targets, needs_tf_map=True),
    "levels_from_legs": IndicatorSpec(name="levels_from_legs", fn=levels_from_legs),
    "select_legs_from_swings": IndicatorSpec(name="select_legs_from_swings", fn=select_legs_from_swings),

    # سطوح/ابزارهای کمکی ---------------------------------------------
    "adr": IndicatorSpec(name="adr", fn=_adv_adr),
    "adr_distance_to_open": IndicatorSpec(name="adr_distance_to_open", fn=_adv_adr_distance_to_open),
    "sr_overlap_score": IndicatorSpec(name="sr_overlap_score", fn=_adv_sr_overlap_score),
    "round_levels": IndicatorSpec(name="round_levels", fn=_adv_round_levels),

    # Advanced Support/Resistance -------------------------------------
    "fvg": IndicatorSpec(name="fvg", fn=make_fvg),
    "supply_demand": IndicatorSpec(name="supply_demand", fn=make_sd),
    "order_block": IndicatorSpec(name="order_block", fn=make_ob),
    "liq_sweep": IndicatorSpec(name="liq_sweep", fn=make_liq_sweep),
    "breaker_flip": IndicatorSpec(name="breaker_flip", fn=make_breaker_flip),
    "sr_fusion": IndicatorSpec(name="sr_fusion", fn=make_sr_fusion),
    "zigzag_mtf": IndicatorSpec(
        name="zigzag_mtf",
        fn=zigzag_mtf_adapter,
        needs_tf_map=True,
        modes={"train", "backtest"},
    ),
}
# =============================================================================
# wrap S/R indicators to inject merged config (common/component/overrides)
# =============================================================================
for _name in ("fvg", "supply_demand", "order_block", "liq_sweep", "breaker_flip", "sr_fusion"):
    _spec = _ADV[_name]
    # -------------------------------------------
    def _wrap(fn, name):
        """
        Wrap S/R indicator with runtime config merging.
        
        Merges config from:
        - global config
        - component config  
        - explicit kwargs
        
        Returns callable that accepts (df, **cfg).
        """
        def runner(df: pd.DataFrame, **cfg):
            merged = merge_sr_kwargs(name, cfg, df)
            return fn(df, **merged)
        return runner
    # -------------------------------------------
    _ADV[_name] = IndicatorSpec(
        name=_name,
        fn=_wrap(_spec.fn, _name),
        modes=_spec.modes,
        needs_tf_map=_spec.needs_tf_map,
        required_cols=_spec.required_cols,
    )


# =============================================================================
# افزودن اندیکاتورهای پیشرفته به رجیستری ساده اولیه
# =============================================================================
REGISTRY.update(_ADV)  


# =============================================================================
# افزودن رجیستری پرایس اکشن به رجیستری اندیکاتورهای ساده و پیشرفته
# =============================================================================
from f03_features.price_action.registry_adapter import register_price_action_to_indicators_registry
register_price_action_to_indicators_registry(REGISTRY, spec_class=IndicatorSpec)


# ---------------------------------------------------------------------------

def get_indicator(name: str) -> Optional["IndicatorSpec"]:
    """
    Unified lookup from the single REGISTRY.
    خروجی IndicatorSpec است که هم callable است هم دارای metadata.
    """
    print("==== get_indicator ==== start ====")
    key = str(name).strip()
    spec = REGISTRY.get(key)
    if spec is None:
        print("==== get_indicator ==== 1 ====")
        logger.warning("Indicator not found in unified registry: %s", name)
    return spec


def list_all_indicators(include_legacy: bool = True) -> Dict[str, str]:
    """Report from unified REGISTRY."""
    return {k: "unified" for k in REGISTRY.keys()}


def list_indicators(mode: str = None) -> list:
    """
    لیست اندیکاتورهایی که mode داده‌شده را support می‌کنند.
    اگر mode=None باشد، همه اندیکاتورها برگردانده می‌شوند.
    مثال: list_indicators("live") → فقط اندیکاتورهای قابل اجرا در live
    """
    if mode is None:
        return list(REGISTRY.keys())
    return [
        name for name, spec in REGISTRY.items()
        if isinstance(spec, IndicatorSpec) and spec.supports(mode)
    ]


'''
نکات کوتاه:
Engine: برای fib_cluster اگر DF پاس بدهی، به‌خاطر mismatch یک‌بار TypeError می‌خورد
 و مسیر fallback (بدون DF) فعال می‌شود—این همان طراحی افزایشی قبلی ماست.
'''
# =============================================================================
# Explicit public API
# =============================================================================
__all__ = [
    "REGISTRY",
    "get_indicator",
    "list_all_indicators",
    "list_indicators",
]
