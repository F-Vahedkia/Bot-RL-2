# f03_features/caching/feature_B_cache.py
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import hashlib
import pickle


# ============================================================
# Stable Hash (DSL-safe)
# ============================================================
def _hash(obj: Any) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


# ============================================================
# Feature Cache (Engine Shared State)
# ============================================================
class FeatureCache:
    """
    Shared cache used by FeatureEngine only.
    Must NOT be used by resolver or registry.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._store: Dict[str, Any] = {}
        self._order: list[str] = []

    def get(self, key: str) -> Optional[Any]:
        return self._store.get(key)

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            return

        if len(self._order) >= self.max_size:
            old = self._order.pop(0)
            self._store.pop(old, None)

        self._store[key] = value
        self._order.append(key)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


# ============================================================
# Global Cache Instance (Engine-only shared)
# ============================================================
GLOBAL_FEATURE_CACHE = FeatureCache()


# ============================================================
# Cache Key Builder (CRITICAL: engine + resolver + DSL aware)
# ============================================================
def build_cache_key(
    df: pd.DataFrame,
    spec: str,
    mode: str,
    *,
    resolved_spec: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> str:

    df_signature = (
        str(df.shape),
        tuple(df.columns),
        str(df.index[-1]) if len(df) else "empty"
    )

    payload = {
        # raw DSL (config input)
        "spec": spec,

        # resolver output (IMPORTANT for DSL normalization)
        "resolved_spec": resolved_spec or spec,

        # execution mode (train/live/backtest)
        "mode": mode,

        # dataframe identity
        "df": df_signature,

        # optional engine context (TF, symbol, session, etc.)
        "extra": extra or {}
    }

    return _hash(payload)


# ============================================================
# Cached Execution Wrapper (ENGINE ONLY ENTRY POINT)
# ============================================================
def cached_compute(
    engine,
    df: pd.DataFrame,
    spec: str,
    mode: str,
    *,
    resolved_spec: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    cache: FeatureCache = GLOBAL_FEATURE_CACHE
):

    # 1) key generation (includes resolver output)
    key = build_cache_key(df, spec, mode,
                          resolved_spec=resolved_spec,
                          extra=extra)

    cached = cache.get(key)
    if cached is not None:
        return cached

    # 2) engine execution (resolver-aware contract)
    if hasattr(engine, "_execute_resolved"):
        result = engine._execute_resolved(df, resolved_spec or spec)
    else:
        result = engine._execute(df, spec)

    # 3) cache store
    cache.set(key, result)

    return result