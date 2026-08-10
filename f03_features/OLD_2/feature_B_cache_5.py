# f03_features/feature_B_cache_5.py (feature_B_cache_v2)

from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
import hashlib
from collections import OrderedDict
import json

# =============================================================================
# HASH UTILITY (SAFE + DETERMINISTIC)
# =============================================================================
def _stable_hash(obj: Any) -> str:
    """
    Deterministic hash using JSON instead of pickle.
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()

# =============================================================================
# EXECUTION CONTRACT (EXTENDED)
# =============================================================================
class ExecutionContract:

    def __init__(
        self,
        engine_version: str,
        resolver_version: str,
        config_version: str,
        registry_version: str,
        feature_schema_version: str = "v2",
    ):
        self.engine_version = engine_version
        self.resolver_version = resolver_version
        self.config_version = config_version
        self.registry_version = registry_version
        self.feature_schema_version = feature_schema_version

    def signature(self) -> str:
        return _stable_hash({
            "engine": self.engine_version,
            "resolver": self.resolver_version,
            "config": self.config_version,
            "registry": self.registry_version,
            "schema": self.feature_schema_version,
        })

# =============================================================================
# CACHE CORE (LRU SAFE)
# =============================================================================
class FeatureCache:

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self._store: Dict[str, Any] = OrderedDict()

    def get(self, key: str):
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: str, value: Any):
        if key in self._store:
            self._store.move_to_end(key)
            return

        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)

        self._store[key] = value

    def clear(self):
        self._store.clear()


GLOBAL_FEATURE_CACHE = FeatureCache()

# =============================================================================
# SAFE DATA SIGNATURE (NO PICKLE)
# =============================================================================
def _df_signature(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Stable dataframe fingerprint.
    """
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "last_index": str(df.index[-1]) if len(df) else None,
        "dtypes": {c: str(df[c].dtype) for c in df.columns[:10]},  # lightweight sampling
    }

# =============================================================================
# CACHE KEY BUILDER
# =============================================================================
def build_cache_key(
    df: pd.DataFrame,
    spec: str,
    mode: str,
    contract: ExecutionContract,
    tf: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:

    payload = {
        "spec": spec,
        "mode": mode,
        "tf": tf,
        "df": _df_signature(df),
        "contract": contract.signature(),
        "extra": extra or {},
    }

    return _stable_hash(payload)

# =============================================================================
# CACHED EXECUTION WRAPPER
# =============================================================================
def cached_compute(
    engine,
    df: pd.DataFrame,
    spec: str,
    mode: str,
    contract: ExecutionContract,
    *,
    tf: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    cache: FeatureCache = GLOBAL_FEATURE_CACHE,
):

    key = build_cache_key(df, spec, mode, contract, tf=tf, extra=extra)

    cached = cache.get(key)
    if cached is not None:
        return cached

    # ENGINE ABSTRACTION (NO PRIVATE API DEPENDENCY)
    result = engine.execute(df, [spec], mode=mode)

    cache.set(key, result)
    return result
