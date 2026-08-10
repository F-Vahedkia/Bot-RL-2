# f03_features/indicators_new/feature_B_registry_1.py
# Run: python -m f03_features.feature_B_registry_1
"""
Feature Registry - Single Source of Truth
==========================================
Unified registry for all features/indicators across train, backtest, and live modes.

==========================================
# در engine:
spec = get_indicator("sma", mode="live")  # → SMA class
spec = get_indicator("sma", mode="train")  # → sma_batch_df function

# لیست indicators:
live_indicators = list_indicators(mode="live")
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Union

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
BATCH_MODES = TRAIN_MODES
ALL_MODES = TRAIN_MODES | INCREMENTAL_MODES


@dataclass
class IndicatorSpec:
    name         : str
    fn           : Callable
    is_stateful  : bool
    
    warmup       : Optional[int] = None
    modes        : Set[str] = field(default_factory=lambda: ALL_MODES.copy())
    required_cols: list[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    needs_tf_map : bool = False
    output_names : Optional[List[str]] = None
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
    "ema": IndicatorSpec(
        name="ema",
        fn=ema_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Exponential Moving Average (batch)"
    ),
    "roc": IndicatorSpec(
        name="roc",
        fn=roc_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Rate of Change (batch)"
    ),
    "rsi": IndicatorSpec(
        name="rsi",
        fn=rsi_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Relative Strength Index (batch)"
    ),
    "true_range": IndicatorSpec(
        name="true_range",
        fn=truerange_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="True Range (batch)"
    ),
    "atr": IndicatorSpec(
        name="atr",
        fn=atr_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Average True Range (batch)"
    ),
    "macd": IndicatorSpec(
        name="macd",
        fn=macd_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="MACD (batch)"
    ),
    "bollinger_bands": IndicatorSpec(
        name="bollinger_bands",
        fn=bollinger_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Bollinger Bands (batch)"
    ),
    "keltner_channel": IndicatorSpec(
        name="keltner_channel",
        fn=keltner_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Keltner Channel (batch)"
    ),
    "stochastic": IndicatorSpec(
        name="stochastic",
        fn=stochastic_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Stochastic Oscillator (batch)"
    ),
    "cci": IndicatorSpec(
        name="cci",
        fn=cci_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Commodity Channel Index (batch)"
    ),
    "mfi": IndicatorSpec(
        name="mfi",
        fn=mfi_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close", "volume"],
        description="Money Flow Index (batch)"
    ),
    "obv": IndicatorSpec(
        name="obv",
        fn=obv_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close", "volume"],
        description="On-Balance Volume (batch)"
    ),
    "williams_r": IndicatorSpec(
        name="williams_r",
        fn=williamsr_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Williams %R (batch)"
    ),
    "parabolic_sar": IndicatorSpec(
        name="parabolic_sar",
        fn=parabolicsar_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Parabolic SAR (batch)"
    ),
    "heikin_ashi": IndicatorSpec(
        name="heikin_ashi",
        fn=heikinashi_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["open", "high", "low", "close"],
        description="Heikin Ashi (batch)"
    ),
    "supertrend": IndicatorSpec(
        name="supertrend",
        fn=supertrend_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low", "close"],
        description="Supertrend (batch)"
    ),
    "aroon": IndicatorSpec(
        name="aroon",
        fn=aroon_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["high", "low"],
        description="Aroon Oscillator (batch)"
    ),
    "dema": IndicatorSpec(
        name="dema",
        fn=dema_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Double Exponential Moving Average (batch)"
    ),
    "tema": IndicatorSpec(
        name="tema",
        fn=tema_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Triple Exponential Moving Average (batch)"
    ),
    "kama": IndicatorSpec(
        name="kama",
        fn=kama_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Kaufman Adaptive Moving Average (batch)"
    ),
    "hma": IndicatorSpec(
        name="hma",
        fn=hma_batch_df,
        modes=TRAIN_MODES.copy(),
        is_stateful=False,
        required_cols=["close"],
        description="Hull Moving Average (batch)"
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
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Simple Moving Average (stateful)"
    ),
    "wma": IndicatorSpec(
        name="wma",
        fn=WMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Weighted Moving Average (stateful)"
    ),
    "ema": IndicatorSpec(
        name="ema",
        fn=EMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Exponential Moving Average (stateful)"
    ),
    "roc": IndicatorSpec(
        name="roc",
        fn=ROC,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Rate of Change (stateful)"
    ),
    "rsi": IndicatorSpec(
        name="rsi",
        fn=RSI,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Relative Strength Index (stateful)"
    ),
    "true_range": IndicatorSpec(
        name="true_range",
        fn=TrueRange,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        description="True Range (stateful)"
    ),
    "atr": IndicatorSpec(
        name="atr",
        fn=ATR,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        description="Average True Range (stateful)"
    ),
    "macd": IndicatorSpec(
        name="macd",
        fn=MACD,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        output_names=["macd", "macd_signal", "macd_hist"],
        description="MACD (stateful)"
    ),
    "bollinger_bands": IndicatorSpec(
        name="bollinger_bands",
        fn=BollingerBands,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        output_names=["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"],
        description="Bollinger Bands (stateful)"
    ),
    "keltner_channel": IndicatorSpec(
        name="keltner_channel",
        fn=KeltnerChannel,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        output_names=["kc_upper", "kc_middle", "kc_lower", "kc_width", "kc_percent"],
        description="Keltner Channel (stateful)"
    ),
    "stochastic": IndicatorSpec(
        name="stochastic",
        fn=Stochastic,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        output_names=["stoch_k", "stoch_d"],
        description="Stochastic Oscillator (stateful)"
    ),
    "cci": IndicatorSpec(
        name="cci",
        fn=CCI,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        description="Commodity Channel Index (stateful)"
    ),
    "mfi": IndicatorSpec(
        name="mfi",
        fn=MFI,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close", "volume"],
        description="Money Flow Index (stateful)"
    ),
    "obv": IndicatorSpec(
        name="obv",
        fn=OBV,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close", "volume"],
        description="On-Balance Volume (stateful)"
    ),
    "williams_r": IndicatorSpec(
        name="williams_r",
        fn=WilliamsR,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        description="Williams %R (stateful)"
    ),
    "parabolic_sar": IndicatorSpec(
        name="parabolic_sar",
        fn=ParabolicSAR,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        description="Parabolic SAR (stateful)"
    ),
    "heikin_ashi": IndicatorSpec(
        name="heikin_ashi",
        fn=HeikinAshi,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["open", "high", "low", "close"],
        output_names=["ha_open", "ha_high", "ha_low", "ha_close"],
        description="Heikin Ashi (stateful)"
    ),
    "supertrend": IndicatorSpec(
        name="supertrend",
        fn=Supertrend,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low", "close"],
        output_names=["supertrend", "st_direction"],
        description="Supertrend (stateful)"
    ),
    "aroon": IndicatorSpec(
        name="aroon",
        fn=Aroon,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["high", "low"],
        output_names=["aroon_up", "aroon_down", "aroon_oscillator"],
        description="Aroon Oscillator (stateful)"
    ),
    "dema": IndicatorSpec(
        name="dema",
        fn=DEMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Double Exponential Moving Average (stateful)"
    ),
    "tema": IndicatorSpec(
        name="tema",
        fn=TEMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Triple Exponential Moving Average (stateful)"
    ),
    "kama": IndicatorSpec(
        name="kama",
        fn=KAMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Kaufman Adaptive Moving Average (stateful)"
    ),
    "hma": IndicatorSpec(
        name="hma",
        fn=HMA,
        warmup=1,
        modes=INCREMENTAL_MODES.copy(),
        is_stateful=True,
        required_cols=["close"],
        description="Hull Moving Average (stateful)"
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

# Global registry instance
REGISTRY = build_registry()

# =============================================================================
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
    # elif mode in HYBRID_MODES:
    #     batch_spec = entry.get("batch")
    #     live_spec  = entry.get("live")
    #     if batch_spec is None or live_spec is None:
    #         return None
    #     return {"batch": batch_spec, "live": live_spec}
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
    # elif mode in HYBRID_MODES:
    #     return sorted(REGISTRY.keys())

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

# ema = REGISTRY["ema"]
# ema_live = ema["live"]
# print(ema_live, "\n")
# ema_batch = ema["batch"]
# print(ema_batch)

""" CONTRACT TEXT FOR feature_B_registry.py:

A) PUBLIC API MAP

**Class**

* `IndicatorSpec`

  * Methods:

    * `supports(mode: str) -> bool`
    * `is_batch_mode(mode: str) -> bool`
    * `is_incremental_mode(mode: str) -> bool`
    * `__call__(*args, **kwargs)`

**Functions**

* `validate_registry() -> None`

  * Validates consistency between batch/live indicator definitions
  * Raises `ValueError` on mismatch conditions

* `build_registry() -> Dict[str, Dict[str, IndicatorSpec]]`

  * Constructs unified registry with keys: `"batch"`, `"live"`

* `get_indicator(name: str, mode: Optional[str]) -> Optional[IndicatorSpec] | Dict[str, IndicatorSpec]`

  * Resolves indicator spec by mode (train vs incremental)
  * Enforces explicit mode requirement

* `list_indicators(mode: Optional[str]) -> list[str]`

  * Returns indicator names filtered by mode

* `get_required_columns(name: str, mode: str) -> list[str]`

  * Returns required OHLCV columns for indicator

* `is_stateful(name: str, mode: str) -> bool`

  * Returns whether indicator is stateful in given mode

**Entry Point / Module Behavior**

* On import:

  * `validate_registry()` is executed immediately
  * `REGISTRY = build_registry()` is created
  * Debug print executes:

    * `"===== REGISTRY DUMP ====="`
    * prints all indicators, batch/live function names and required cols

---

B) DEPENDENCY CONTRACTS

**Standard Library**

* `dataclasses.dataclass`, `dataclasses.field`
* `typing: Callable, Set, Optional, Dict, Union`

**Internal Imports (Stateful / Live Layer)**

* `f03_features.indicators_new.indicators_B_class`

  * SMA, WMA, EMA, ROC, RSI, TrueRange, ATR, MACD, BollingerBands,
    KeltnerChannel, Stochastic, CCI, MFI, OBV, WilliamsR,
    ParabolicSAR, HeikinAshi, Supertrend, Aroon, DEMA, TEMA, KAMA, HMA

**Internal Imports (Batch / Train Layer)**

* `f03_features.indicators_new.indicators_B_batch`

  * sma_batch_df, wma_batch_df, ema_batch_df, roc_batch_df, rsi_batch_df,
    truerange_batch_df, atr_batch_df, macd_batch_df, bollinger_batch_df,
    keltner_batch_df, stochastic_batch_df, cci_batch_df, mfi_batch_df,
    obv_batch_df, williamsr_batch_df, parabolicsar_batch_df,
    heikinashi_batch_df, supertrend_batch_df, aroon_batch_df,
    dema_batch_df, tema_batch_df, kama_batch_df, hma_batch_df

**Internal Dependency Contracts**

* Batch indicators MUST be stateless functions
* Live indicators MUST be stateful classes
* Both layers MUST share identical:

  * `name`
  * `required_cols`
* Registry validation enforces consistency and raises `ValueError` on mismatch
* No external API calls or IO dependencies

---

C) DATA CONTRACT

**Core Schema**

`IndicatorSpec`

* `name: str`
* `fn: Callable`
* `is_stateful: bool`
* `modes: Set[str]`
* `required_cols: list[str]`
* `needs_tf_map: bool`
* `description: str`

**Mode Sets**

* `TRAIN_MODES = {"train", "optimize"}`
* `INCREMENTAL_MODES = {"live", "paper", "shadow", "backtest", "replay", "eval"}`
* `BATCH_MODES = TRAIN_MODES`
* `ALL_MODES = TRAIN_MODES ∪ INCREMENTAL_MODES`

**Registry Structure**

* `REGISTRY: Dict[str, Dict[str, IndicatorSpec]]`

  * Key: indicator name (e.g., `"sma"`)
  * Value:

    * `"batch": IndicatorSpec` (train-mode function)
    * `"live": IndicatorSpec` (stateful class)

**Function I/O Contracts**

* `get_indicator`

  * Input: `(name: str, mode: str | None)`
  * Output:

    * `IndicatorSpec` OR `None`
    * (hybrid dict return is defined but disabled/commented)

* `list_indicators`

  * Input: `mode: str | None`
  * Output: `list[str]`

* `get_required_columns`

  * Input: `(name: str, mode: str)`
  * Output: `list[str]`
  * If unresolved → `[]`

* `is_stateful`

  * Input: `(name: str, mode: str)`
  * Output: `bool`

**Data Constraints**

* `required_cols` defines strict OHLCV schema dependency per indicator
* Mode must be explicitly provided in `get_indicator`
* Invalid mode → `ValueError`
* Missing indicator → `None`
* Registry must pass validation before use (batch/live symmetry enforced)

**Error Contracts**

* `ValueError`:

  * missing mode
  * invalid mode
  * registry mismatch (name/cols/statefulness inconsistencies)

"""

