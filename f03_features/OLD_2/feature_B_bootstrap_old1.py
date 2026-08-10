# f03_features/feature_B_bootstrap_old1.py

from __future__ import annotations

from typing import Optional, List, Dict, Any
import logging

from f03_features.feature_B_registry_1 import REGISTRY, get_indicator
from f03_features.feature_C_engine_4 import FeatureEngine
from f03_features.OLD_2.feature_B_cache_5 import GLOBAL_FEATURE_CACHE
from f03_features.feature_B_store import FeatureStoreV2
# from f03_features.price_action.registry_adapter import register_price_action_to_indicators_registry

from f10_utils.config_loader import load_config

# -------------------- Logger for this module -----------------------
logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

# ============================================================
# Bootstrap System
# ============================================================
"""
Full feature system initialization:

    Config
        ↓
    Registry (Batched + Live)
        ↓
    Engine (Execution Layer)
        ↓
    Cache + Store (Persistence Layer)
"""

class FeatureSystemBootstrap:
    # -------------------------------------------------------- 1
    def __init__(self, config_path: Optional[str] = None):

        self.cfg = load_config(config_path) if config_path else load_config()

        self.registry = REGISTRY
        self.cache = GLOBAL_FEATURE_CACHE

        self.engine: Optional[FeatureEngine] = None
        self.store: Optional[FeatureStoreV2] = None

    # -------------------------------------------------------- 2
    def _init_registry(self) -> None:
        """
        Attach external feature families to global registry.
        """
        # register_price_action_to_indicators_registry(self.registry)
        pass

    # -------------------------------------------------------- 3
    def _init_engine(self) -> None:
        """
        Initialize FeatureEngine with execution mode from config.
        Engine is fully registry-driven (no resolver layer).
        """
        execution_cfg = self.cfg.get("execution", {})
        mode = execution_cfg.get("mode", "train")

        self.engine = FeatureEngine(config=self.cfg)

        # attach runtime context (optional future extension)
        # self.engine.runtime_mode = mode
        self.engine.config = self.cfg

    # -------------------------------------------------------- 4
    def _init_store(self) -> None:
        """
        Persistent feature storage layer for RL datasets.
        """
        self.store = FeatureStoreV2(config=self.cfg)

    # -------------------------------------------------------- 5
    def build(self) -> "FeatureSystemBootstrap":
        """
        Full initialization pipeline.
        Order matters:
            1. Registry
            2. Engine
            3. Store
        """
        self._init_registry()
        self._init_engine()
        self._init_store()

        return self

    # -------------------------------------------------------- 6
    def get_engine(self) -> FeatureEngine:
        return self.engine

    # -------------------------------------------------------- 7
    def get_store(self) -> FeatureStoreV2:
        return self.store

    # -------------------------------------------------------- 8
    def get_registry(self):
        return self.registry

    # -------------------------------------------------------- 9
    def get_cache(self):
        return self.cache

    # -------------------------------------------------------- 10
    def get_config(self) -> Dict[str, Any]:
        return self.cfg

    # -------------------------------------------------------- 11 new
    def connect_to_data_handler(self, data_handler) -> "FeatureSystemBootstrap":
        """
        اتصال FeatureEngine به DataHandler از طریق ثبت کال‌بک.
        
        پارامترها:
            data_handler: نمونه‌ی کلاس DataHandler از لایه‌ی f02_data
            
        خروجی:
            خود شیء برای زنجیره‌ای کردن (Fluent API)
        """
        if self.engine is None:
            raise RuntimeError("FeatureEngine not initialized. Call build() first.")
        
        # بررسی وجود متد set_data_callback در data_handler
        if not hasattr(data_handler, "set_data_callback"):
            raise AttributeError(
                f"DataHandler object of type {type(data_handler).__name__} "
                "does not have 'set_data_callback' method. "
                "Make sure DataHandler from f02_data has this method implemented."
            )        
        # ثبت متد process_live_data به عنوان کال‌بک در DataHandler
        data_handler.set_data_callback(self.engine.process_live_data)
        logger.info("FeatureEngine successfully connected to DataHandler")
        return self


# ============================================================
# Functional API (clean entrypoint)
# ============================================================
def build_feature_system(config_path: Optional[str] = None) -> FeatureSystemBootstrap:
    """
    One-line initialization for full feature pipeline.
    """
    logger.info("function 'build_feature_system' is started")
    return FeatureSystemBootstrap(config_path).build()



############################################################## MY_NOTES
#### آموزشی ################################################# MY_NOTES
############################################################## MY_NOTES
def my_note():
    # نحوه‌ی استفاده نهایی (در فایل Main یا Orchestrator)
    # حالا در نقطه‌ی شروع برنامه (مثلاً main.py یا یک Orchestrator)، به این صورت همه چیز را به هم وصل می‌کنید:
    import threading
    from f02_data.market_data_engine.market_data_engine_2 import MarketDataEngine
    from f02_data.data_handler_F_2 import DataHandler
    from f03_features.feature_B_bootstrap import build_feature_system
    from datetime import time

    symbols = []
    timeframes = []

    # 1. ساخت لایه‌ی داده
    cfg = load_config()  # کانفیگ اصلی
    engine = MarketDataEngine(cfg)
    data_handler = DataHandler(cfg)

    # 2. ساخت لایه‌ی فیچرها (با پاس دادن کانفیگ)
    feature_system = build_feature_system()
    feature_system.build()  # اگر در build_feature_system خودکار نمی‌شود

    # 3. اتصال FeatureEngine به DataHandler
    feature_system.connect_to_data_handler(data_handler)

    # 4. اتصال DataHandler به EventBus (از طریق MarketDataEngine)
    engine.attach_data_handler(data_handler)

    # 5. اجرای Worker در یک Thread
    threading.Thread(
        target=engine.start,
        args=(symbols, timeframes, 2.0),
        daemon=True
    ).start()

    # 6. اجرای حلقه‌ی مصرف DataHandler در یک Thread جداگانه
    threading.Thread(
        target=data_handler.start_consuming2,
        daemon=True
    ).start()

    # 7. (اختیاری) اگر استراتژی دارید، آن را به خروجی FeatureEngine متصل کنید
    # feature_engine = feature_system.get_engine()
    # feature_engine.set_output_callback(my_strategy.on_features)  # اگر چنین متدی تعریف کنید

    logger.info("System fully initialized and running...")
    # برنامه را زنده نگه دارید (مثلاً با input() یا یک Event)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    
