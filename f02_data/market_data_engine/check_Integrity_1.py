# f02_data/market_data_engine/check_integrity_1.py
# Run: python -m f02_data.market_data_engine.check_Integrity_1
"""
    این ماژول، هماهنگی و اقدام مشترک چهار ماژول زیر را تست میکند
        - event_bus.py
        - candle_detector.py
        - mt5_stream_worker.py
        - market_data_engine.py
    و تا ابتدای اتصال به data_handler.py پیش میرود.
    یعنی تا آنجا که رویداد اتفاق افتاده را در سطر زیر
    event = event_bus.get_event(data_handlers[sym]._subscriber_id, timeout=0.1)
    از data_handler میگیرد.
"""
from typing import Dict
import threading
import logging
from zoneinfo import ZoneInfo
from f02_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f02_data.data_handler_F_2_2 import DataHandler
from f10_utils.config_completer import config_completer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-6s | %(message)s"
)
logger = logging.getLogger("tester")

def main():
    # 1) --- config
    cfg = config_completer()

    broker_timezone = cfg["project"]["broker_timezone"]
    print(broker_timezone)

    warmups_dicts: Dict[str, Dict[str, int]] = cfg["__warmups_dicts"]
    logger.info(warmups_dicts)

    # 2) --- symbols
    symbols = list(warmups_dicts.keys())
    logger.info(symbols)

    # 3) --- timeframes
    timeframes_dict = {sym: list(warmup.keys()) for sym, warmup in warmups_dicts.items()}
    logger.info(timeframes_dict)

    # 4) --- MarketDataEngine
    engine = MarketDataEngine(cfg)
    event_bus = engine.get_event_bus()


    logger.info("✅ Tester started. Press Ctrl+C to stop.")

    # 4) --- loop over all symbols to create DataHandlers
    data_handlers = {}
    for sym in symbols:
        data_handlers[sym] = DataHandler(cfg, symbol=sym, event_bus=None)
        engine.attach_data_handler(data_handlers[sym])  # این subscribe را انجام می‌دهد

    # 5) --- Start Engine (یک بار برای همه نمادها)
    threading.Thread(target=engine.start, args=(warmups_dicts, 2.0), daemon=True).start()
    logger.info("✅ Tester started. Press Ctrl+C to stop.")        


    # 6) --- Event loop for all symbols
    last_printed_time = {}

    while True:
        # برای هر نماد چک کن
        for sym in symbols:
            # event = event_bus.get_event(sym, timeout=0.1)
            event = event_bus.get_event(data_handlers[sym]._subscriber_id, timeout=0.1)
            if event and event.get("event_type") == "NEW_CANDLE":
                if sym == symbols[0]: print("="*60, "Symbols from first") # for debug

                tf = event.get("timeframe")
                all_dfs = event.get("all_dfs", {})
                key = f"{sym}:{tf}"
                df = all_dfs.get(key)

                if df is not None and not df.empty:
                    last_row = df.iloc[-1]      # Recent closed candle OHLCVS
                    candle_time = df.index[-1]  # Recent closed candle time

                    if last_printed_time.get(f"{sym}_{tf}") != candle_time:
                        last_printed_time[f"{sym}_{tf}"] = candle_time

                        # اصلاح زمان
                        candle_time_naive = candle_time.tz_localize(None)  # حذف منطقه زمانی
                        candle_time_broker = candle_time_naive.tz_localize(broker_timezone)  # تنظیم به بروکر
                        candle_time_utc = candle_time_broker.tz_convert("UTC")  # تبدیل به UTC

                        logger.info(f"\n📊 New Candle [{sym}/{tf}] at {candle_time_utc}")
                        print(f"   time : {last_row.get('candle_time', 'N/A')}") # در این سطر، اگر لازم است فرمت را اصلاح کن

                        print(f"   Open : {last_row.get('open', 'N/A'):.5f}")
                        print(f"   High : {last_row.get('high', 'N/A'):.5f}")
                        print(f"   Low  : {last_row.get('low', 'N/A'):.5f}")
                        print(f"   Close: {last_row.get('close', 'N/A'):.5f}")
                        print("-" * 40)

# ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Stopped by user.")

