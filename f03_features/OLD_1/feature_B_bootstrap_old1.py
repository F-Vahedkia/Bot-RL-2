# f03_features/feature_B_bootstrap.py
# Status: Unified Feature System Bootstrap (v2 architecture)

from __future__ import annotations
from typing import Any, Dict, Optional

from f03_features.feature_B_registry_1 import REGISTRY
from f03_features.OLD.feature_B_engine_1 import FeatureEngine
from f03_features.OLD.feature_B_cache_1 import GLOBAL_FEATURE_CACHE
from f03_features.OLD.feature_store_old1 import FeatureStore
from f03_features.OLD.feature_B_resolver_1 import FeatureResolver

from f03_features.price_action.registry_adapter import (
    register_price_action_to_indicators_registry
)
from f10_utils.config_loader import load_config


# =============================================================================
# Bootstrap Core
# =============================================================================
class FeatureSystemBootstrap:
    """
    Initializes full feature pipeline:
        Config → Registry → Resolver → Engine → Cache → Store
    """
    # -------------------------------------------------------------------------
    def __init__(self, config_path: Optional[str] = None):
        self.cfg = load_config(config_path) if config_path else load_config()

        self.registry = REGISTRY
        self.cache = GLOBAL_FEATURE_CACHE

        self.resolver = None
        self.engine = None
        self.store = None

    # -------------------------------------------------------------------------
    def _init_registry(self):
        """
        Attach all external feature families to registry.
        """
        register_price_action_to_indicators_registry(self.registry)

    # -------------------------------------------------------------------------
    def _init_resolver(self):
        """
        Create resolver layer (DSL → DataHandler schema alignment).
        """
        self.resolver = FeatureResolver(
            config=self.cfg,
            registry=self.registry
        )

    # -------------------------------------------------------------------------
    def _init_engine(self):
        """
        Create execution engine (stateless runtime).
        """
        mode = (self.cfg.get("execution", {}) or {}).get("mode", "train")

        self.engine = FeatureEngine(mode=mode)

    # -------------------------------------------------------------------------
    def _init_store(self):
        """
        Persistence layer for RL datasets / caching / replay buffers.
        """
        self.store = FeatureStore(
            config=self.cfg
        )

    # -------------------------------------------------------------------------
    def build(self) -> "FeatureSystemBootstrap":
        """
        Full system initialization.
        """
        self._init_registry()
        self._init_resolver()
        self._init_engine()
        self._init_store()

        return self

    # -------------------------------------------------------------------------
    def get_engine(self) -> FeatureEngine:
        return self.engine

    def get_resolver(self) -> FeatureResolver:
        return self.resolver

    def get_store(self) -> FeatureStore:
        return self.store

    def get_registry(self):
        return self.registry

    def get_cache(self):
        return self.cache


# =============================================================================
# Functional API (clean entrypoint)
# =============================================================================
def build_feature_system(config_path: Optional[str] = None) -> FeatureSystemBootstrap:
    """
    One-liner system initialization.
    """
    return FeatureSystemBootstrap(config_path).build()

