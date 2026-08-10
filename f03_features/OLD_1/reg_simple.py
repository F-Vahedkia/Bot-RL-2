# f03_features/indicators_new/feature_B_registry.py

from dataclasses import dataclass, field
from typing import Callable, Set, Optional, Dict, Union

# Layer 2 imports (stateful classes for live/backtest)
from f03_features.indicators_new.indicators_B_class import (
    SMA, WMA, EMA, ROC, RSI, TrueRange, ATR, MACD, BollingerBands,
    KeltnerChannel, Stochastic, CCI, MFI, OBV, WilliamsR, ParabolicSAR,
    HeikinAshi, Supertrend, Aroon, DEMA, TEMA, KAMA, HMA
)
# Layer 3 imports (batch functions for train/backtest)
from f03_features.indicators_new.indicators_B_batch import (
    sma_batch_df, wma_batch_df, ema_batch_df, roc_batch_df, rsi_batch_df, truerange_batch_df,
    atr_batch_df, macd_batch_df, bollinger_batch_df, keltner_batch_df,  stochastic_batch_df,
    cci_batch_df, mfi_batch_df, obv_batch_df, williamsr_batch_df, parabolicsar_batch_df,
    heikinashi_batch_df, supertrend_batch_df, aroon_batch_df, dema_batch_df, tema_batch_df,
    kama_batch_df, hma_batch_df,
)
TRAIN_MODES       = {"train", "optimize"}
INCREMENTAL_MODES = {"live", "paper", "shadow", "backtest", "replay", "eval"}
HYBRID_MODES      = {"warmup"}
BATCH_MODES = TRAIN_MODES
ALL_MODES = TRAIN_MODES | INCREMENTAL_MODES | HYBRID_MODES

@dataclass
class IndicatorSpec:
    name         : str
    fn           : Callable
    is_stateful  : bool
    modes        : Set[str] = field(default_factory=lambda: ALL_MODES.copy())
    required_cols: list[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    needs_tf_map : bool = False
    description  : str = ""

    def supports(self, mode: str) -> bool:
        return mode in self.modes
    
    def is_batch_mode(self, mode: str) -> bool:
        """آیا این mode نیاز به batch function دارد؟"""
        return mode in BATCH_MODES
    
    def is_incremental_mode(self, mode: str) -> bool:
        """آیا این mode نیاز به stateful class دارد؟"""
        return mode in INCREMENTAL_MODES
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

# ============================================================================
# Layer 3: Batch Functions (Train/Backtest)
# ============================================================================
_BATCH_INDICATORS = {
    "sma": IndicatorSpec(
        name="sma",
        fn=sma_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Simple Moving Average (batch)"
    ),
    "wma": IndicatorSpec(
        name="wma",
        fn=wma_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Weighted Moving Average (batch)"
    ),
}

# ============================================================================
# Layer 2: Stateful Incremental Indicators
# Used for:
# live / paper / shadow / backtest / replay / eval
# ============================================================================
_LIVE_INDICATORS = {
    "sma": IndicatorSpec(
        name="sma",
        fn=SMA,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Simple Moving Average (stateful)"
    ),
    "wma": IndicatorSpec(
        name="wma",
        fn=WMA,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Weighted Moving Average (stateful)"
    ),
}

# =============================================================================
# Validate Registry
# =============================================================================
def validate_registry() -> None:
    shared = set(_BATCH_INDICATORS) & set(_LIVE_INDICATORS)
    for name in shared:
        batch_spec = _BATCH_INDICATORS[name]
        live_spec  = _LIVE_INDICATORS[name]

        if batch_spec.name != live_spec.name:
            raise ValueError(f"name mismatch for indicator '{name}'")

        if batch_spec.required_cols != live_spec.required_cols:
            raise ValueError(f"required_cols mismatch for indicator '{name}'")

        if batch_spec.is_stateful is not False:
            raise ValueError(f"batch indicator '{name}' must be non-stateful")

        if live_spec.is_stateful is not True:
            raise ValueError(f"live indicator '{name}' must be stateful")

validate_registry()

# =============================================================================
# Registry Builder
# =============================================================================
def build_registry() -> Dict[str, Dict[str, IndicatorSpec]]:
    names = sorted(set(_BATCH_INDICATORS) | set(_LIVE_INDICATORS))
    registry: Dict[str, Dict[str, IndicatorSpec]] = {}
    for name in names:
        entry: Dict[str, IndicatorSpec] = {}
        if name in _BATCH_INDICATORS:
            entry["batch"] = _BATCH_INDICATORS[name]
        if name in _LIVE_INDICATORS:
            entry["live"] = _LIVE_INDICATORS[name]
        registry[name] = entry

    return registry

REGISTRY = build_registry()

## =============================================================================
# Public API
# =============================================================================
def get_indicator(name: str, mode: Optional[str] = None
) -> Optional[Union[IndicatorSpec, Dict[str, IndicatorSpec]]]:
    entry = REGISTRY.get(name)

    if entry is None:
        return None
    if mode is None:
        raise ValueError("mode must be explicitly specified")
    
    if mode not in ALL_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(ALL_MODES)}")
    
    if mode in TRAIN_MODES:
        spec = entry.get("batch")
        return spec if spec and spec.supports(mode) else None
    elif mode in INCREMENTAL_MODES:
        spec = entry.get("live")
        return spec if spec and spec.supports(mode) else None
    elif mode in HYBRID_MODES:
        batch_spec = entry.get("batch")
        live_spec  = entry.get("live")
        if batch_spec is None or live_spec is None:
            return None
        return {"batch": batch_spec, "live": live_spec}
    return None

def list_indicators(mode: Optional[str] = None) -> list[str]:
    if mode is None:
        return sorted(REGISTRY.keys())

    if mode not in ALL_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {sorted(ALL_MODES)}")

    if mode in INCREMENTAL_MODES:
        return [
            name for name, entry in REGISTRY.items()
            if "live" in entry and entry["live"].supports(mode)
        ]
    elif mode in TRAIN_MODES:
        return [
            name for name, entry in REGISTRY.items()
            if "batch" in entry and entry["batch"].supports(mode)
        ]
    elif mode in HYBRID_MODES:
        return sorted(REGISTRY.keys())

    return []

def get_required_columns(name: str, mode: str) -> list[str]:
    spec = get_indicator(name, mode)
    if spec is None:
        return []
    if isinstance(spec, dict):
        batch_spec = spec.get("batch")
        live_spec  = spec.get("live")
        if batch_spec is None or live_spec is None:
            return []
        return live_spec.required_cols
    return spec.required_cols

def is_stateful(name: str, mode: str) -> bool:
    spec = get_indicator(name, mode)
    if spec is None:
        return False
    if isinstance(spec, dict):
        return True
    return spec.is_stateful
