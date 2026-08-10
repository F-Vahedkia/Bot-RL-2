# f03_features/caching/feature_B_cache.py
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import hashlib
import pickle


def _hash(obj: Any) -> str:
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


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
# 🔒 LOCKED CACHE KEY (VERSION-AWARE)
# ============================================================
def build_cache_key(
    df: pd.DataFrame,
    spec: str,
    mode: str,
    *,
    tf: Optional[str] = None,
    config_version: Optional[str] = None,
    registry_version: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:

    df_sig = (
        str(df.shape),
        tuple(df.columns),
        str(df.index[-1]) if len(df) else "empty",
    )

    payload = {
        "spec": spec,
        "tf": tf,  # 🔥 مهم برای M1/M5/H1
        "mode": mode,
        "df": df_sig,

        # 🔒 VERSION LOCK
        "config_v": config_version,
        "registry_v": registry_version,

        "extra": extra or {},
    }

    return _hash(payload)


# ============================================================
# 🔒 EXEC WRAPPER (LOCKED CONTRACT)
# ============================================================
def cached_compute(
    engine,
    df: pd.DataFrame,
    spec: str,
    mode: str,
    *,
    tf: Optional[str] = None,
    config_version: Optional[str] = None,
    registry_version: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    cache: FeatureCache = GLOBAL_FEATURE_CACHE,
):

    key = build_cache_key(
        df,
        spec,
        mode,
        tf=tf,
        config_version=config_version,
        registry_version=registry_version,
        extra=extra,
    )

    cached = cache.get(key)
    if cached is not None:
        return cached

    result = engine._execute(df, spec, tf=tf)

    cache.set(key, result)
    return result