# 2
# f02_data/market_data_engine/candle_detector_2.py

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================
# Candle State Tracker
# ============================================================
@dataclass
class CandleState:
    """
    نگهداری آخرین وضعیت کندل برای تشخیص بسته شدن کندل جدید
    """
    last_candle_time: Optional[datetime] = None

# ============================================================
# Candle Detector
# ============================================================
class CandleDetector:
    """
    Detects newly closed candles from streaming OHLCVS data.
    Logic:
        - receive latest dataframe snapshot
        - compare last timestamp
        - emit event only when a NEW candle is closed
        
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
    def __init__(self, tzinfo) -> None:
        """
        در ابتدای ساخت یک شیئ از این کلاس، این موارد معلوم یا ساخته میشود:
            - یک دیکشنری خالی به نام _state که :
                کلیدهای آن رشته های "نماد-تایمفریم" هستند و 
                مقادیر آن شیئی از کلاس CandleState است.
        """
        self._state: Dict[str, CandleState] = {}
        self.tzinfo = tzinfo

    # -------------------------------------------------------- OK
    def _get_state(self, symbol: str, timeframe: str) -> CandleState:
        """
        در ابتدا بررسی میکند که کلید "نماد-تایمفریم" در دیکشنری _state وجود دارد یا نه
        اگر وجود نداشته باشد، کلید مزبور را میسازد و همراه با مقدار متناظر با آن کلید در دیکشنری قرار میدهد
        اگر وجود داشته باشد، مقدار معادل با ان کلید را برمیگرداند.
        """
        key = f"{symbol}:{timeframe.upper()}"
        if key not in self._state:
            self._state[key] = CandleState()
        return self._state[key]

    # --------------------------------------------------------
    def detect(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        این متد، در صورت تشخیص کندل جدید، دیکشنری مربوطه را برمیگرداند
        در غیر اینصورت نَن را برمیگرداند
        Returns event "payload" if a new candle is detected.
        Otherwise returns None.
        """
        if df is None or df.empty:
            return None

        state = self._get_state(symbol, timeframe)

        # در تابع _fetch_closed این بازه خروجی داده شده است: df=[-n-1:-1]
        # یعنی در آنجا، اِن-تا کندل ((بسته شده،)) در قالب یک دیتافریم، برگردانده شده است
        if len(df) == 0:
            return None
    
        # --- گرفتن داده های آخرین کندل بسته شده
        last_time_raw = df.index[-1]

        # --- first run -------------------------
        if state.last_candle_time is None:
            state.last_candle_time = last_time_raw
            return None
        
        # --- no new candle ---------------------
        if last_time_raw <= state.last_candle_time:
            return None

        # --- کندل جدید تشخیص داده شد ---------
        # فقط در اینجا زمان را یک بار تبدیل کن
        try:
            # تبدیل به Timestamp اگر نبود
            if not isinstance(last_time_raw, pd.Timestamp):
                last_time = pd.to_datetime(last_time_raw)
            else:
                last_time = last_time_raw

            # اضافه کردن منطقه زمانی بروکر (اگر naive باشد)
            if last_time.tzinfo is None:
                last_time = last_time.tz_localize(self.tzinfo)

            # تبدیل به UTC
            last_time = last_time.tz_convert("UTC")

            # تبدیل به datetime پایتون برای خروجی
            if hasattr(last_time, 'to_pydatetime'):
                last_time = last_time.to_pydatetime()

        except Exception as e:
            logger.error(f"Time conversion failed for {symbol}/{timeframe}: {e}")
            return None

        # --- update state ----------------------
        state.last_candle_time = last_time_raw
        logger.info(f"New candle detected {symbol}/{timeframe} at {last_time} UTC / {last_time_raw} Broker")
    
        # --- استخراج داده های کندل ------------
        # داده‌های کندل بسته شده را از ردیف ماقبل آخر بگیر
        # در تابع _fetch_closed این بازه خروجی داده شده است: df=[-n-1:-1]
        # یعنی در آنجا، 3 تا کندل ((بسته شده،)) در قالب یک دیتافریم، برگردانده شده است
        row = df.iloc[-1]
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_time": last_time,
            "open"  : float(row["open"  ]) if "open"   in df.columns else None,
            "high"  : float(row["high"  ]) if "high"   in df.columns else None,
            "low"   : float(row["low"   ]) if "low"    in df.columns else None,
            "close" : float(row["close" ]) if "close"  in df.columns else None,
            "volume": float(row["volume"]) if "volume" in df.columns else None,
            "spread": float(row["spread"]) if "spread" in df.columns else None,
        }
    
    # -------------------------------------------------------- END
