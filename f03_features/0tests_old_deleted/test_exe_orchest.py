# Run in VSCode terminal:
# python -m f03_features.0tests.test_exe_orchest
"""
این تست تقریباً تمام رفتارهای مهم ExecutionOrchestrator را پوشش می‌دهد:
    - Contract enforcement
    - Resolver call
    - Engine call
    - Cache write
    - Cache hit
    - Cache key stability
    - Cache key invalidation
    - Empty DataFrame handling
"""
import pandas as pd

from f03_features.OLD_2.feature_B_exe_orchest import (
    ExecutionRequest,
    ExecutionOrchestrator,
)


# ============================================================
# Mocks
# ============================================================

class MockContract:
    def signature(self):
        return "contract_signature"


class MockContractGate:
    def __init__(self):
        self.called = False

    def enter(self, contract):
        self.called = True


class MockResolver:
    def __init__(self):
        self.called = False

    def resolve(self, spec, tf=None):
        self.called = True

        return {
            "resolved": True,
            "spec": spec,
            "tf": tf,
        }


class MockEngine:
    def __init__(self):
        self.called = False

    def _execute_resolved(self, df, resolved_spec, tf=None):
        self.called = True

        return {
            "rows": len(df),
            "resolved_spec": resolved_spec,
            "tf": tf,
        }


class MockCache:
    def __init__(self):
        self.storage = {}

    def get(self, key):
        return self.storage.get(key)

    def set(self, key, value):
        self.storage[key] = value


# ============================================================
# Tests
# ============================================================

def build_orchestrator():

    gate = MockContractGate()
    resolver = MockResolver()
    engine = MockEngine()
    cache = MockCache()

    orch = ExecutionOrchestrator(
        contract_gate=gate,
        resolver=resolver,
        engine=engine,
        cache=cache,
    )

    return orch, gate, resolver, engine, cache


# ------------------------------------------------------------
def test_normal_execution():

    orch, gate, resolver, engine, cache = build_orchestrator()

    df = pd.DataFrame({
        "close": [1, 2, 3, 4]
    })

    req = ExecutionRequest(
        spec="ema(close,20)@M5",
        mode="train",
        contract=MockContract(),
        tf="M5"
    )

    result = orch.execute(req, df)

    assert gate.called is True
    assert resolver.called is True
    assert engine.called is True

    assert result["rows"] == 4

    print("NORMAL EXECUTION TEST PASSED")


# ------------------------------------------------------------
def test_cache_hit():

    orch, gate, resolver, engine, cache = build_orchestrator()

    df = pd.DataFrame({
        "close": [1, 2, 3]
    })

    req = ExecutionRequest(
        spec="ema(close,20)@M5",
        mode="train",
        contract=MockContract(),
        tf="M5"
    )

    result1 = orch.execute(req, df)

    resolver.called = False
    engine.called = False

    result2 = orch.execute(req, df)

    assert result1 == result2

    assert resolver.called is False
    assert engine.called is False

    print("CACHE HIT TEST PASSED")


# ------------------------------------------------------------
def test_cache_key_stability():

    orch, _, _, _, _ = build_orchestrator()

    df = pd.DataFrame({
        "close": [1, 2, 3]
    })

    req = ExecutionRequest(
        spec="ema(close,20)@M5",
        mode="train",
        contract=MockContract(),
        tf="M5"
    )

    k1 = orch.cache_build_key(df, req)
    k2 = orch.cache_build_key(df, req)

    assert k1 == k2

    print("CACHE KEY STABILITY TEST PASSED")


# ------------------------------------------------------------
def test_cache_key_changes_if_spec_changes():

    orch, _, _, _, _ = build_orchestrator()

    df = pd.DataFrame({
        "close": [1, 2, 3]
    })

    req1 = ExecutionRequest(
        spec="ema(close,20)@M5",
        mode="train",
        contract=MockContract(),
        tf="M5"
    )

    req2 = ExecutionRequest(
        spec="ema(close,50)@M5",
        mode="train",
        contract=MockContract(),
        tf="M5"
    )

    k1 = orch.cache_build_key(df, req1)
    k2 = orch.cache_build_key(df, req2)

    assert k1 != k2

    print("CACHE KEY CHANGE TEST PASSED")


# ------------------------------------------------------------
def test_empty_dataframe_execution():

    orch, gate, resolver, engine, cache = build_orchestrator()

    df = pd.DataFrame()

    req = ExecutionRequest(
        spec="ema(close,20)@M5",
        mode="train",
        contract=MockContract(),
        tf="M5"
    )

    result = orch.execute(req, df)

    assert gate.called
    assert resolver.called
    assert engine.called

    print("EMPTY DATAFRAME TEST PASSED")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    test_normal_execution()

    test_cache_hit()

    test_cache_key_stability()

    test_cache_key_changes_if_spec_changes()

    test_empty_dataframe_execution()

    print("\nALL EXE_ORCHEST TESTS PASSED")