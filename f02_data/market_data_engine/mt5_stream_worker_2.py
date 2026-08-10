# 3
# f02_data/market_data_engine/mt5_stream_worker_2.py

from __future__ import annotations
import time
import logging
import pandas as pd
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

# ---------------------------
from f02_data.mt5_connector import MT5Connector
from f02_data.market_data_engine.event_bus_2 import EventBus
from f02_data.market_data_engine.candle_detector_2 import CandleDetector
from f10_utils.parse_warmups import get_warmup_from_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# MT5 Live Stream Worker
# ============================================================
class MT5StreamWorker:
    """
    Live polling worker:  (polling: کشیدن  ,  pushing: فشار دادن)

    Responsibilities:
        - Connect to MT5
        - Poll latest candles
        - Detect new closed candles
        - Publish events via EventBus

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
    def __init__(
        self,
        cfg: Dict[str, Any],
        event_bus: EventBus,
        warmups_dicts: Dict[str, Dict[str, int]],
        poll_interval_sec: float = 2.0,
    ) -> None:
        """
        در ابتدا این اشیائ را میسازد: MT5Connector, CandleDetector
        """
        self.cfg = cfg
        self.event_bus = event_bus
        self.warmups_dicts = warmups_dicts
        self.symbols = list(warmups_dicts.keys())
        self.timeframes_dict = {sym: list(warmup.keys()) for sym, warmup in warmups_dicts.items()}
        self.poll_interval_sec = poll_interval_sec
        self.connector = MT5Connector(config=cfg)

        # --- broker_timezone ----------------------------- start
        project_cfg = cfg.get("project")
        if not project_cfg:
            raise ValueError("'project' key not found in config!")
            
        self.broker_timezone = project_cfg.get("broker_timezone")
        if not self.broker_timezone:
            raise ValueError("'broker_timezone' key not set in 'project' key!")

        tzinfo = ZoneInfo(self.broker_timezone)
        # ------------------------------------------------- end
        self.detector = CandleDetector(tzinfo)
        self._running = False
       
    # -------------------------------------------------------- OK
    def start(self) -> None:
        """
        این متد اتصال به متاتریدر را راه اندازی میکند و 
        حلقه اصلی را شروع میکند.
        در نهایت همیشه اتصال به متاتریدر را قطع میکند
        """
        if not self.connector.initialize():
            raise RuntimeError("MT5 connection failed")
        self._running = True # کنترل اجرای حقله- مادامیکه صحیح است، حلقه اجرا میشود
        logger.info("MT5StreamWorker started")
        try:
            self._loop() # حلقه پایش بازار
        finally:
            self.connector.shutdown()
            logger.info("MT5StreamWorker stopped")

    # -------------------------------------------------------- OK
    def _loop(self) -> None:
        """
        این متد این کارها را انجام میدهد:
            - پایش بازار هر چند ثانیه یکبار توسط دانلود کندل جدید
            - در صورت تشخیص کندل جدید، دانلود تمام تایم فرمهای مورد نیاز فیچرها در آن نماد
            -
        """
        logger.debug("===> start loop at streamworker")
        logger.debug(f"{self.symbols}")
        
        while self._running:
            # ---------------------------------/
            for symbol in self.symbols:
                for tf in self.timeframes_dict[symbol]:
                    # -------------------------//
                    try:
                        df = self._fetch_closed(symbol, tf, 3)
                        event = self.detector.detect(symbol, tf, df)  # will be "None" or "dict"
                        # event: is a dict contains: symbol, timeframe, candle_time, O,H,L,C,V,S
                        if event:
                            # سطر زیر:
                            # در تمام تایمفریهای وارم آپ، دیتاها را دانلود نموده
                            # و همگی را در قالب یک دیکشنری برمیگرداند
                            all_dfs = self._fetch_all_tfs(symbol)
                            self.event_bus.publish2(
                                event_type="NEW_CANDLE",
                                symbol=symbol,
                                timeframe=tf,
                                all_dfs=all_dfs,
                            )

                    except Exception as ex:
                        logger.exception(
                            "Stream error %s/%s: %s",
                            symbol,
                            tf,
                            ex,
                        )
                    # -------------------------//
            # ---------------------------------/
            time.sleep(self.poll_interval_sec)

    # -------------------------------------------------------- OK
    def _fetch_closed(self, symbol: str, timeframe: str, num_candles: Optional[int] = None) -> pd.DataFrame:
        """
        این متد فقط وظیفه دانلود داده های کندلی ((بسته شده)) را به عهده دارد.
        و فقط تعداد n کندل بسته شده نهایی را برمیگرداند.
        """
        n = num_candles if num_candles is not None else (self.cfg.get("executor") or {}).get("lookback_bars", 30)
        df = self.connector.get_candles_num(
            symbol=symbol,
            timeframe=timeframe,
            num_candles=n+1,
        )
        # سطر زیر کندل -1 را برنمی گرداند. چون هنوز بسته نشده است و کندل جاری است.
        return df.iloc[-n-1:-1] if df is not None else pd.DataFrame()
    
    # -------------------------------------------------------- OK
    def _fetch_all_tfs(self, symbol: str) -> Dict[str, pd.DataFrame]:
        all_dfs = {
            f"{symbol}:{tf.upper()}": self._fetch_closed(symbol, tf, self.warmups_dicts[symbol][tf]) for tf in self.warmups_dicts[symbol]
        }
        return all_dfs
    
    # -------------------------------------------------------- OK=
    def stop(self) -> None:
        """
        این متد، چوب لای چرخ حلقه اصلی میکند
        """
        self._running = False

    # -------------------------------------------------------- END
