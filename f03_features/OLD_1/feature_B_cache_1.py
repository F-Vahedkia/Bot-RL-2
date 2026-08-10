# f03_features/caching/feature_B_cache.py
# Status: Production-ready feature cache layer (DSL-aware)

"""
Engine → مسئول execution logic
Cache → مسئول semantic identity
Resolve layer → مسئول schema mapping
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Optional
import pandas as pd
import hashlib
import pickle

# =============================================================================
# Cache Key Builder
# =============================================================================
def _hash_obj(obj: Any) -> str:
    """
    Stable hash for complex objects (args/kwargs/DSL).
    """
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()

def build_cache_key(
    df: pd.DataFrame,
    spec_str: str,
    mode: str,
    extra: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build deterministic cache key for feature computation.

    Key depends on:
    - DSL string
    - mode (train/live/etc.)
    - dataframe shape + columns + last timestamp
    - optional extra metadata
    """

    df_signature = (
        str(df.shape),
        tuple(df.columns),
        str(df.index[-1]) if len(df.index) > 0 else "empty"
    )

    payload = {
        "dsl": spec_str,
        "mode": mode,
        "df_signature": df_signature,
        "extra": extra or {}
    }

    return _hash_obj(payload)

# =============================================================================
# Feature Cache
# =============================================================================
class FeatureCache:
    """
    In-memory feature cache.

    Designed for:
    - RL training loops
    - backtesting
    - live inference acceleration
    """
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._store: Dict[str, Any] = {}
        self._order: list[str] = []

    # -------------------------------------------------------------------------
    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key, None)

    # -------------------------------------------------------------------------
    def set(self, key: str, value: Any) -> None:
        """
        Insert into cache with naive LRU eviction.
        """
        if key in self._store:
            return

        if len(self._order) >= self.max_size:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)

        self._store[key] = value
        self._order.append(key)

    # -------------------------------------------------------------------------
    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    # -------------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "size": len(self._store),
            "max_size": self.max_size
        }

# =============================================================================
# Global cache instance (shared across engine)
# =============================================================================
GLOBAL_FEATURE_CACHE = FeatureCache()

# =============================================================================
# Cached execution wrapper
# =============================================================================
def cached_compute(
    engine,
    df: pd.DataFrame,
    spec_str: str,
    mode: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
    cache: FeatureCache = GLOBAL_FEATURE_CACHE
):
    """
    Wrapper around FeatureEngine.compute with caching.
    Flow:
        key → cache lookup → miss → resolve-aware compute → store
    """
    # -------------------------------------------------------------
    # 1) build base cache key (DSL-level)
    # -------------------------------------------------------------
    key = build_cache_key(df, spec_str, mode, extra)

    cached = cache.get(key)
    if cached is not None:
        return cached

    # -------------------------------------------------------------
    # 2) IMPORTANT: resolve-aware execution
    #    (engine now owns DSL → DF alignment)
    # -------------------------------------------------------------
    result = engine._execute(df, spec_str)

    # -------------------------------------------------------------
    # 3) optional: enrich cache key with output shape stability
    #    (prevents silent mismatch in multi-TF + multi-output)
    # -------------------------------------------------------------
    if isinstance(result, pd.DataFrame):
        shape_sig = (result.shape, tuple(result.columns))
    else:
        shape_sig = (getattr(result, "shape", None),)

    if extra is None:
        extra = {}

    # extra["_out_sig"] = shape_sig                             # Deleted
    # # recompute final key (stable RL safety layer)            # Deleted
    # final_key = build_cache_key(df, spec_str, mode, extra)    # Deleted

    key = build_cache_key(df, spec_str, mode, extra)
    cached = cache.get(key)
    if cached is not None:
        return cached

    # cache.set(final_key, result)                              # Deleted
    cache.set(key, result)

    return result