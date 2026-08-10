# f03_features/caching/feature_B_cache.py
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import hashlib
import pickle


# ============================================================
# 🔒 INTERNAL HASH
# ============================================================
def _hash(obj: Any) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


# ============================================================
# 🔒 VERSIONED CONTRACT (CORE IDEA)
# ============================================================
class ExecutionContract:
    """
    Defines immutable interface binding between:
    - Engine
    - Resolver
    - Cache
    """

    def __init__(
        self,
        engine_version: str,
        resolver_version: str,
        config_version: str,
        registry_version: str,
    ):
        self.engine_version = engine_version
        self.resolver_version = resolver_version
        self.config_version = config_version
        self.registry_version = registry_version

    def signature(self) -> str:
        return _hash(
            (
                self.engine_version,
                self.resolver_version,
                self.config_version,
                self.registry_version,
            )
        )


# ============================================================
# 🔒 CACHE CORE
# ============================================================
class FeatureCache:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._store: Dict[str, Any] = {}
        self._order: list[str] = []

    def get(self, key: str):
        return self._store.get(key)

    def set(self, key: str, value: Any):
        if key in self._store:
            return

        if len(self._order) >= self.max_size:
            old = self._order.pop(0)
            self._store.pop(old, None)

        self._store[key] = value
        self._order.append(key)

    def clear(self):
        self._store.clear()
        self._order.clear()


GLOBAL_FEATURE_CACHE = FeatureCache()


# ============================================================
# 🔒 VERSIONED CACHE KEY (ENGINE+RESOLVER LOCK)
# ============================================================
def build_cache_key(
    df: pd.DataFrame,
    spec: str,
    mode: str,
    contract: ExecutionContract,
    tf: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:

    df_sig = (
        str(df.shape),
        tuple(df.columns),
        str(df.index[-1]) if len(df) else "empty",
    )

    payload = {
        "spec": spec,
        "mode": mode,
        "tf": tf,
        "df": df_sig,

        # 🔒 HARD CONTRACT LOCK
        "contract": contract.signature(),

        "extra": extra or {},
    }

    return _hash(payload)


# ============================================================
# 🔒 EXECUTION + CACHE BINDING (NO AMBIGUITY)
# ============================================================
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

    # 🔒 ENGINE MUST RESPECT RESOLVED DSL
    if hasattr(engine, "_execute_resolved"):
        result = engine._execute_resolved(df, spec, tf=tf)
    else:
        result = engine._execute(df, spec, tf=tf)

    cache.set(key, result)
    return result