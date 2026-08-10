# 5
# f02_data/market_data_engine/data_handler_live_consumer_2.py
"""

با وجود متدهای موجود در کلاس DataHandler ، دیگر نیازی به این فایل نداریم

"""
from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import logging
# ---------------------------
from f02_data.market_data_engine.event_bus_2 import EventBus

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================
# Live DataHandler Consumer
# ============================================================
class DataHandlerLiveConsumer:
    """
    Consumes NEW_CANDLE events from EventBus
    and triggers DataHandler-style processing pipeline.
    """
    # --------------------------------------------------------
    def __init__(
        self,
        cfg: Dict[str, Any],
        event_bus: EventBus,
    ) -> None:

        self.cfg = cfg
        self.event_bus = event_bus
        self.subscriber_id = self.event_bus.subscribe()
        self.queue = self.event_bus.get_queue(self.subscriber_id)
        if self.queue is None:
            raise RuntimeError("Failed to create EventBus subscription")
        self._running = False

        # placeholder for future integration with real DataHandler
        self.data_handler = None

    # --------------------------------------------------------
    def attach_data_handler(self, data_handler: Any) -> None:
        """
        Inject real DataHandler instance (decoupled dependency).
        """
        self.data_handler = data_handler

    # --------------------------------------------------------
    def start(self) -> None:
        self._running = True
        logger.info("DataHandlerLiveConsumer started")
        self._loop()

    # --------------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            event = self.event_bus.get(
                self.subscriber_id,
                timeout=1.0,
            )
            if event is None:
                continue
            self._process_event(event)

    # --------------------------------------------------------
    def _process_event(self, event: Dict[str, Any]) -> None:
        if event.get("event_type") != "NEW_CANDLE":
            return
        # payload = event.get("payload") or {}
        symbol = event.get("symbol")
        timeframe = event.get("timeframe")
        all_dfs = event.get("all_dfs")    # Not Used

        if not symbol or not timeframe:
            return
        logger.info("Processing NEW_CANDLE %s/%s", symbol, timeframe)

        # ----------------------------------------------------
        # STEP 1: fetch latest raw snapshot (lightweight)
        # ----------------------------------------------------
        # (In production: could be cache or incremental update)

        # ----------------------------------------------------
        # STEP 2: trigger DataHandler if attached
        # ----------------------------------------------------
        if self.data_handler:
            try:
                self.data_handler.on_new_candle2(event)
            except Exception as ex:
                logger.exception("DataHandler processing failed: %s", ex)

    # --------------------------------------------------------
    def stop(self) -> None:
        self._running = False

    # --------------------------------------------------------
