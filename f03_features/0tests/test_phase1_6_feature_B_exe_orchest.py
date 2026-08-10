# VSCode run: python -m pytest -q f03_features/0tests/test_phase1_6_feature_B_exe_orchest.py

import pandas as pd
from types import SimpleNamespace

from f03_features.OLD_2.feature_B_exe_orchest import (
    ExecutionRequest,
    ExecutionOrchestrator,
)

# ---------------------------------------------------------
class DummyCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

# ---------------------------------------------------------
class DummyContractGate:
    def enter(self, contract):
        return True

# ---------------------------------------------------------
class DummyResolver:
    def resolve(self, spec, tf=None):
        return f"resolved::{spec}"

# ---------------------------------------------------------
class DummyEngine:
    def execute(self, df, spec):
        return df.copy()

    def _execute_resolved(self, df, resolved_spec, tf=None):
        df2 = df.copy()
        df2["ok"] = 1
        return df2

# ---------------------------------------------------------
def test_orchestrator_end_to_end_execution():
    engine = DummyEngine()
    cache = DummyCache()

    orch = ExecutionOrchestrator(
        contract_gate=DummyContractGate(),
        resolver=DummyResolver(),
        engine=engine,
        cache=cache,
    )

    df = pd.DataFrame({"close": [1, 2, 3]})

    req = ExecutionRequest(
        spec="sma(10)",
        mode="train",
        contract=SimpleNamespace(signature=lambda: "sig"),
        tf="M1",
    )

    result = orch.execute(req, df)

    assert "ok" in result.columns
    assert len(result) == 3

# ---------------------------------------------------------
def test_cache_hit():
    engine = DummyEngine()
    cache = DummyCache()

    orch = ExecutionOrchestrator(
        contract_gate=DummyContractGate(),
        resolver=DummyResolver(),
        engine=engine,
        cache=cache,
    )

    df = pd.DataFrame({"close": [1, 2, 3]})

    req = ExecutionRequest(
        spec="sma(10)",
        mode="train",
        contract=SimpleNamespace(signature=lambda: "sig"),
    )

    r1 = orch.execute(req, df)
    r2 = orch.execute(req, df)

    assert r1.equals(r2)
    assert len(cache.store) == 1
