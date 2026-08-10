"""
indicators_batch_adapter.py

Adapter layer to bridge indicators_batch functions (signature: fn(df, **kwargs))
with feature_engine's expected signature (fn(symbol, tf_dfs, base_tf, **kwargs)).
"""

from typing import Callable, Dict, Any
import pandas as pd
from indicators_A_batch import (
    sma_batch, wma_batch, ema_batch, roc_batch, rsi_batch,
    true_range_batch, atr_batch, macd_batch, bollinger_batch, keltner_batch,
    stochastic_batch, cci_batch, mfi_batch, obv_batch, williamsr_batch,
    parabolic_sar_batch, heikin_ashi_batch, supertrend_batch, aroon_batch,
    dema_batch, tema_batch, hma_batch
)


def _batch_adapter(batch_fn: Callable) -> Callable:
    """
    Generic adapter: wraps a batch function to match feature_engine signature.
    
    Batch function signature:
        fn(df: pd.DataFrame, **kwargs) -> pd.Series | pd.DataFrame
    
    Feature engine signature:
        fn(symbol: str, tf_dfs: Dict[str, pd.DataFrame], base_tf: str, **kwargs) -> pd.Series | pd.DataFrame
    
    The adapter extracts df from tf_dfs[base_tf] and forwards kwargs.
    """
    def wrapper(symbol: str, tf_dfs: Dict[str, pd.DataFrame], base_tf: str, **kwargs) -> pd.Series | pd.DataFrame:
        df = tf_dfs[base_tf]
        return batch_fn(df, **kwargs)
    
    # Preserve original function metadata
    wrapper.__name__ = batch_fn.__name__
    wrapper.__doc__ = batch_fn.__doc__
    
    return wrapper


# Wrap all 22 batch functions
sma_adapted = _batch_adapter(sma_batch)
wma_adapted = _batch_adapter(wma_batch)
ema_adapted = _batch_adapter(ema_batch)
roc_adapted = _batch_adapter(roc_batch)
rsi_adapted = _batch_adapter(rsi_batch)
true_range_adapted = _batch_adapter(true_range_batch)
atr_adapted = _batch_adapter(atr_batch)
macd_adapted = _batch_adapter(macd_batch)
bollinger_adapted = _batch_adapter(bollinger_batch)
keltner_adapted = _batch_adapter(keltner_batch)
stochastic_adapted = _batch_adapter(stochastic_batch)
cci_adapted = _batch_adapter(cci_batch)
mfi_adapted = _batch_adapter(mfi_batch)
obv_adapted = _batch_adapter(obv_batch)
williamsr_adapted = _batch_adapter(williamsr_batch)
parabolic_sar_adapted = _batch_adapter(parabolic_sar_batch)
heikin_ashi_adapted = _batch_adapter(heikin_ashi_batch)
supertrend_adapted = _batch_adapter(supertrend_batch)
aroon_adapted = _batch_adapter(aroon_batch)
dema_adapted = _batch_adapter(dema_batch)
tema_adapted = _batch_adapter(tema_batch)
hma_adapted = _batch_adapter(hma_batch)


# Registry for feature_engine integration
BATCH_INDICATOR_REGISTRY = {
    "sma": sma_adapted,
    "wma": wma_adapted,
    "ema": ema_adapted,
    "roc": roc_adapted,
    "rsi": rsi_adapted,
    "true_range": true_range_adapted,
    "atr": atr_adapted,
    "macd": macd_adapted,
    "bollinger": bollinger_adapted,
    "keltner": keltner_adapted,
    "stochastic": stochastic_adapted,
    "cci": cci_adapted,
    "mfi": mfi_adapted,
    "obv": obv_adapted,
    "williamsr": williamsr_adapted,
    "parabolic_sar": parabolic_sar_adapted,
    "heikin_ashi": heikin_ashi_adapted,
    "supertrend": supertrend_adapted,
    "aroon": aroon_adapted,
    "dema": dema_adapted,
    "tema": tema_adapted,
    "hma": hma_adapted,
}


__all__ = [
    "_batch_adapter",
    "BATCH_INDICATOR_REGISTRY",
    # Adapted functions
    "sma_adapted", "wma_adapted", "ema_adapted", "roc_adapted", "rsi_adapted",
    "true_range_adapted", "atr_adapted", "macd_adapted", "bollinger_adapted", "keltner_adapted",
    "stochastic_adapted", "cci_adapted", "mfi_adapted", "obv_adapted", "williamsr_adapted",
    "parabolic_sar_adapted", "heikin_ashi_adapted", "supertrend_adapted", "aroon_adapted",
    "dema_adapted", "tema_adapted", "hma_adapted",
]
