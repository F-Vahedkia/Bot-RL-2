# f03_features/feature_B_bootstrap.py
# reviewed at 1405/04/28
"""
این فایل در حال حاضر موارد زیر را انجام میدهد:
    - Registry
    - Engine
    - Store
و خبری از موارد زیر در این فایل وجود ندارد:
    - Graph
    - ObservationBuilder
    - Alignment
"""
# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from typing import Optional, Dict, Any
import logging

from f03_features.feature_C_registry_1 import REGISTRY
from f03_features.feature_C_engine_6 import FeatureEngine
from f03_features.feature_B_cache_6 import GLOBAL_FEATURE_CACHE_MANAGER, ExecutionContract
from f03_features.feature_B_store import FeatureStoreV2
from f10_utils.config_loader import load_config

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN CLASS
# =============================================================================
class FeatureSystemBootstrap:
    """
    Bootstrap مسئول مقداردهی اولیه کل زیرسیستم Feature است.

        - Config -> Registry -> Engine -> Cache -> Store
    """

    # ----------------------------------------------------- 1
    def __init__(self, config_path: Optional[str] = None):

        self.cfg = load_config(config_path) if config_path else load_config()

        self.registry = REGISTRY
        # self.cache = GLOBAL_FEATURE_CACHE
        # self.cache = GLOBAL_FEATURE_CACHE_MANAGER.get(self.dataset.symbol)
        self.cache = None

        self.contract = ExecutionContract(
            engine_version="4",
            resolver_version="0",
            config_version=str(self.cfg.get("version", "1")),
            registry_version="1",
        )

        self.engine: Optional[FeatureEngine] = None
        self.store: Optional[FeatureStoreV2] = None

    # ----------------------------------------------------- 2
    def init_cache(self, symbol: str) -> None:
        """
        Initialize symbol-specific feature cache.
        """
        self.cache = GLOBAL_FEATURE_CACHE_MANAGER.get(symbol)

    # ----------------------------------------------------- 3
    def _init_registry(self) -> None:
        """
        Attach external feature families to global registry.
        """
        # register_price_action_to_indicators_registry(self.registry)
        pass

    # ----------------------------------------------------- 4
    def _init_engine(self) -> None:
        """
        Initialize FeatureEngine.
        """
        self.engine = FeatureEngine(config=self.cfg)

    # ----------------------------------------------------- 5
    def _init_store(self) -> None:
        """
        Persistent feature storage layer.
        """
        self.store = FeatureStoreV2(config=self.cfg)

    # ----------------------------------------------------- 6
    def build(self) -> "FeatureSystemBootstrap":     # Forward Reference
        """
        Build the complete feature system.

        Order:
            1. Registry
            2. Engine
            3. Store
        """
        self._init_registry()
        self._init_engine()
        self._init_store()

        logger.info("Feature system initialized successfully.")

        return self

    # ----------------------------------------------------- 7
    def get_registry(self):
        return self.registry

    # ----------------------------------------------------- 8
    def get_engine(self) -> FeatureEngine:
        return self.engine

    # ----------------------------------------------------- 9
    def get_store(self) -> FeatureStoreV2:
        return self.store

    # ----------------------------------------------------- 10
    def get_cache(self):
        return self.cache

    # ----------------------------------------------------- 11
    def get_config(self) -> Dict[str, Any]:
        return self.cfg

# =============================================================================
# Functional API (clean entrypoint)
# =============================================================================
def build_feature_system(config_path: Optional[str] = None) -> FeatureSystemBootstrap:
    """
    One-liner system initialization.
    """
    return FeatureSystemBootstrap(config_path).build()
# =============================================================================
