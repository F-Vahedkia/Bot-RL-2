# VSCode run: python -m pytest -q f03_features/0tests/test_phase1_4_feature_B_cache_5.py

import pandas as pd
from f03_features.OLD_2.feature_B_cache_5 import (
    FeatureCache,
    ExecutionContract,
    build_cache_key,
    cached_compute
)


class DummyEngine:
    def __init__(self):
        self.calls = 0

    def execute(self, df, specs, mode="train"):
        self.calls += 1
        return df.copy()


def test_cache_hit_prevents_recompute():
    df = pd.DataFrame({"a": [1, 2, 3]})
    engine = DummyEngine()

    contract = ExecutionContract("1", "1", "1", "1")

    r1 = cached_compute(engine, df, "sma", "train", contract)
    r2 = cached_compute(engine, df, "sma", "train", contract)

    assert engine.calls == 1
    assert r1.equals(r2)


def test_cache_key_changes_with_data():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})

    contract = ExecutionContract("1", "1", "1", "1")

    k1 = build_cache_key(df1, "sma", "train", contract)
    k2 = build_cache_key(df2, "sma", "train", contract)

    assert k1 != k2


def test_cache_eviction_behavior():
    cache = FeatureCache(max_size=2)

    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert "a" not in cache._store  # evicted
    assert len(cache._store) == 2


def test_contract_signature_stability():
    c1 = ExecutionContract("1", "2", "3", "4")
    c2 = ExecutionContract("1", "2", "3", "4")

    assert c1.signature() == c2.signature()