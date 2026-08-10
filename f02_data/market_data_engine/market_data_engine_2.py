# 4
# f02_data/market_data_engine/market_data_engine.py

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
# ---------------------------
from f02_data.market_data_engine.event_bus_2 import EventBus
from f02_data.market_data_engine.mt5_stream_worker_2 import MT5StreamWorker
from f02_data.data_handler_F_2 import DataHandler  # ➕ Version-D

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# Market Data Engine (Orchestrator)
# ============================================================
class MarketDataEngine:
    """
    Central orchestrator of live market data pipeline.

    Responsibilities:
        - Initialize EventBus
        - Start MT5 stream worker
        - Manage lifecycle (start/stop)

    Some Data structures:
    warmups_dicts = {
        "XAUUSD" : {'M1': 14,  'M5':9 , 'H1':14, 'D1':12},
        "EURUSD" : {'M5': 12, 'M30':14, 'D1':5 },
        "BITCOIN": {'H2': 16,  'D1':9 },
    }
    symbols = ["XAUUSD", "EURUSD", "BITCOIN"]

    timeframes_dict = {
        "XAUUSD" : ['M1',  'M5', 'H1', 'D1],
        "EURUSD" : ['M5', 'M30', 'D1'],
        "BITCOIN": ['H2',  'D1'],
    }
    """
    # -------------------------------------------------------- OK
    def __init__(self, cfg: Dict[str, Any]) -> None:
        """
        یک شیئ self.event_bus = EventBus میسازد
        """
        self.cfg = cfg
        self.event_bus = EventBus(
            queue_size=(cfg.get("event_bus") or {}).get("queue_size", 1000) # config-readable
        )
        self.worker: Optional[MT5StreamWorker] = None
        self._running: bool = False
        self.data_handler: Optional[DataHandler] = None  # ➕ Version-D

    # -------------------------------------------------------- OK
    def start(
        self,
        warmups_dicts: Dict[str, Dict[str, int]],
        poll_interval_sec: float = 2.0,
    ) -> None:
        """
        این متد:
            - یک شیئ MT5StreamWorker میسازد و 
            - آنرا استارت میکند
            - وضعیت اجرای شیئ MarketDataEngine را در حالت True قرار میدهد
        """
        if self._running:
            logger.warning("MarketDataEngine already running")
            return

        self.worker = MT5StreamWorker(
            cfg=self.cfg,
            event_bus=self.event_bus,
            warmups_dicts=warmups_dicts,
            poll_interval_sec=poll_interval_sec,
        )
        self.worker.start()
        self._running = True
        
        logger.info("MarketDataEngine starting...")

    # -------------------------------------------------------- OK
    def stop(self) -> None:
        """
        - اجرای شیئ self.worker = MT5StreamWorker را متوقف میکند
        - وضعیت اجرای شیئ MarketDataEngine را در حالت False قرار میدهد
        """
        if not self._running:
            return
        logger.info("MarketDataEngine stopping...")
        if self.worker:
            self.worker.stop()
        self._running = False

    # -------------------------------------------------------- OK
    def get_event_bus(self) -> EventBus:
        """
        شیئ self.event_bus = EventBus را برمیگرداند
        """
        return self.event_bus
    
    # --------------------------------------------------------
    def attach_data_handler(self, data_handler: DataHandler) -> None:
        
        """اتصال DataHandler به EventBus"""
        self.data_handler = data_handler
        data_handler.subscribe_to_event_bus(self.event_bus)
        logger.info("DataHandler attached to MarketDataEngine")

    # --------------------------------------------------------


############################################################## MY_NOTES
#### آموزشی ################################################# MY_NOTES
############################################################## MY_NOTES
def my_notes():
    import threading

    cfg = {}
    symbols = []
    timeframes = []

    #✅ پیشنهاد نهایی برای استفاده:
    #✅ برای راه‌اندازی کل سیستم، کافی است این کار را انجام دهید:


    #✅  1. ساخت Engine و DataHandler
    engine = MarketDataEngine(cfg)
    data_handler = DataHandler(cfg)

    #✅  2. اتصال DataHandler به EventBus (از طریق Engine)
    engine.attach_data_handler(data_handler)

    #✅  3. اجرای Worker در یک Thread
    threading.Thread(target=engine.start, args=(symbols, timeframes, 1.0), daemon=True).start()

    #✅  4. اجرای حلقه‌ی مصرف DataHandler در یک Thread جداگانه
    threading.Thread(target=data_handler.start_consuming2, daemon=True).start()

    #✅  5. حالا استراتژی خود را به DataHandler متصل کنید (Callback)
    # my_strategy = MyTradingStrategy()
    # data_handler.set_data_callback(my_strategy.on_new_data)

