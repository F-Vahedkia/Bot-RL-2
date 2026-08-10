# f03_features/sys_contract.py (system contract)
# Status: Production-grade Contract Enforcement Layer (immutable gate)

from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Optional
import hashlib

# ============================================================
# 🔒 SYSTEM CONTRACT
# ============================================================
@dataclass(frozen=True)
class SystemContract:
    config_version: str
    engine_version: str
    resolver_version: str
    mode: str

    def signature(self) -> str:
        raw = f"{self.config_version}|{self.engine_version}|{self.resolver_version}|{self.mode}"
        return hashlib.sha256(raw.encode()).hexdigest()

# ============================================================
# 🔒 VALIDATOR
# ============================================================
class ContractValidator:
    """
    Hard gate between config → resolver → engine → cache
    """

    def __init__(
        self,
        engine_version: str,
        resolver_version: str,
        supported_config_versions: Set[str],
    ):
        self.engine_version = engine_version
        self.resolver_version = resolver_version
        self.supported_config_versions = supported_config_versions

    def validate(self, contract: SystemContract) -> None:

        if contract.engine_version != self.engine_version:
            raise RuntimeError("ENGINE_VERSION_MISMATCH")

        if contract.resolver_version != self.resolver_version:
            raise RuntimeError("RESOLVER_VERSION_MISMATCH")

        if contract.config_version not in self.supported_config_versions:
            raise RuntimeError("CONFIG_VERSION_NOT_SUPPORTED")

# ============================================================
# 🔒 GATE (single entry point)
# ============================================================
class ContractGate:
    def __init__(self, validator: ContractValidator):
        self.validator = validator

    def enter(self, contract: SystemContract) -> None:
        self.validator.validate(contract)

# ============================================================
# 🔒 HELPERS (engine binding)
# ============================================================
def enforce_contract(
    gate: ContractGate,
    contract: SystemContract,
) -> None:
    gate.enter(contract)



"""
این ترتیب طراحی Contract است، نه ترتیب اجرای برنامه.

مرحله	فایل	هدف
1	f02_data/mtf_dataset.py                 تعریف ظرف داده (Dataset Contract)
2	f02_data/data_handler_F_2_3_n.py        قرارداد خروجی لایه Data
3	f03_features/feature_sys_contract.py    قرارداد کل لایه Features
4	f03_features/feature_B_graph.py         قرارداد Feature Specification
5	f03_features/feature_B_registry_1.py    قرارداد ثبت Featureها
6	f03_features/feature_C_engine_4.py       قرارداد محاسبه Feature
7	f03_features/feature_B_store.py         قرارداد ذخیره و Merge Featureها
8	f03_features/observation_B_builder.py    قرارداد ساخت Observation
9	f03_features/feature_B_cache_6.py       قرارداد Cache
10	f03_features/feature_B_bootstrap.py     قرارداد مونتاژ کل سیستم
"""
"""
                            MTFDataset
                                │
                                ▼
                            DataHandler
                                │
                                ▼
                            Feature System Contract
                                │
                                ├──────────────┐
                                ▼              ▼
                            FeatureGraph    FeatureRegistry
                                │              │
                                └──────┬───────┘
                                        ▼
                                FeatureEngine
                                        │
                                        ▼
                                FeatureStore
                                        │
                                        ▼
                                ObservationBuilder
                                        │
                                        ▼
                                FeatureCache
                                        │
                                        ▼
                                Bootstrap
"""