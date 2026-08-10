# f03_features/feature_B_exe_orchest.py (execution orchestrator)
# Status: Production-grade End-to-End Execution Orchestration Layer

#   
#
#
#
#
#                    DELETED
#
#
#









from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional

from f02_data.mtf_dataset import MTFDataset

# ============================================================
# 🔒 CONTRACT INPUT
# ============================================================
@dataclass(frozen=True)
class ExecutionRequest:
    spec: str
    mode: str
    contract: Any
    tf: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# ============================================================
# 🔒 ORCHESTRATOR
# ============================================================
class ExecutionOrchestrator_old1:

    def __init__(
        self,
        contract_gate,
        resolver,
        engine,
        cache,
    ):
        self.contract_gate = contract_gate
        self.resolver = resolver
        self.engine = engine
        self.cache = cache


    # --------------------------------------------------------
    def execute(self, request: ExecutionRequest, df: pd.DataFrame):

        # 1. CONTRACT ENFORCEMENT
        self.contract_gate.enter(request.contract)

        # 2. CACHE CHECK
        key = self.cache_build_key(df, request)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        # 3. RESOLVE DSL → EXEC SPEC
        resolved_spec = self.resolver.resolve(
            request.spec,
            tf=request.tf
        )

        # 4. ENGINE EXECUTION
        if hasattr(self.engine, "_execute_resolved"):
            result = self.engine._execute_resolved(df, resolved_spec, tf=request.tf)
        else:
            result = self.engine._execute(df, resolved_spec)

        # 5. CACHE STORE
        self.cache.set(key, result)

        return result

    # --------------------------------------------------------
    def cache_build_key(self, df: pd.DataFrame, request: ExecutionRequest):

        df_sig = (
            str(df.shape),
            tuple(df.columns),
            str(df.index[-1]) if len(df) else "empty",
        )

        contract_sig = request.contract.signature()

        return self._hash(
            (
                request.spec,
                request.mode,
                request.tf,
                df_sig,
                contract_sig,
                request.extra or {},
            )
        )

    # --------------------------------------------------------
    def _hash(self, obj: Any) -> str:
        import hashlib, pickle
        return hashlib.sha256(pickle.dumps(obj)).hexdigest()

class ExecutionOrchestrator:

    def __init__(self, engine):
        self.engine = engine

    def execute(
        self,
        dataset: MTFDataset,
        specs: list[str],
        mode: str = "train",
    ) -> MTFDataset:

        return self.engine.execute(
            dataset=dataset,
            specs=specs,
            mode=mode,
        )  