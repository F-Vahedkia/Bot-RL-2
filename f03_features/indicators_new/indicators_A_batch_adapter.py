"""
indicators_batch_adapter.py

Adapter layer to bridge indicators_batch functions (signature: fn(df, **kwargs))
with feature_engine's expected signature (fn(symbol, tf_dfs, base_tf, **kwargs)).
"""

from typing import Callable, Dict
import pandas as pd
import indicators_A_batch


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


# Auto-wrap all functions from indicators_batch.__all__
BATCH_INDICATOR_REGISTRY = {}

for func_name in indicators_A_batch.__all__:
    original_fn = getattr(indicators_A_batch, func_name)
    
    # Extract indicator name (remove '_batch' suffix)
    indicator_name = func_name.replace('_batch', '')
    
    # Wrap and register
    adapted_fn = _batch_adapter(original_fn)
    BATCH_INDICATOR_REGISTRY[indicator_name] = adapted_fn
    
    # Also expose as module-level variable with '_adapted' suffix
    globals()[f"{indicator_name}_adapted"] = adapted_fn


# Build __all__ dynamically
__all__ = [
    "_batch_adapter",
    "BATCH_INDICATOR_REGISTRY",
] + [f"{name}_adapted" for name in BATCH_INDICATOR_REGISTRY.keys()]
