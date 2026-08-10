# این برنامه نیمه کاره رها شد.

# Run: python -m f02_data.test_datahandler_marketdataengine

from __future__ import annotations
from f02_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f02_data.data_handler_F_3 import DataHandler
import threading
import logging

from f10_utils.parse_warmups import get_warmup_from_config
from f10_utils.config_loader import load_config


# ===================================================================
def my_test():

    # === initial checks ========================
    cfg = load_config(enable_env_override=True)
    if cfg is None:
        print(" ===> cfg is none")
        return
    else: print("cfg is OK")

    features = cfg.get("features") or {}
    if features is None:
        print(" ===> features is none")
        return
    else: print("features is OK")

    symbol = list(features.get("symbol" or {}))
    if symbol is None:
        print(" ===> symbol is none")
        return
    else: print("symbol is OK")

    warmup_dict = get_warmup_from_config(cfg)
    if warmup_dict is None:
        print(" ===> warmup_dict is none")
        return
    else: print("warmup_dict is OK")

    timeframes = list(warmup_dict.keys())
    if timeframes is None:
        print(" ===> timeframes is none")
        return
    else: print("timeframes is OK")

    print(f"   symbol: {symbol}")
    print(f"   warmup_dict: {warmup_dict}")
    print(f"   timeframes: {timeframes}")
    
    # === setting up logging ====================
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        format="%(asctime)s | %(levelname)-6s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )


    #✅  1. ساخت Engine و DataHandler
    engine = MarketDataEngine(cfg)
    data_handler = DataHandler(cfg)

    #✅  2. اتصال DataHandler به EventBus (از طریق Engine)
    engine.attach_data_handler(data_handler)

    #✅  3. اجرای Worker در یک Thread
    threading.Thread(target=engine.start, args=(symbol, timeframes, 2.0), daemon=True).start()

    #✅  4. اجرای حلقه‌ی مصرف DataHandler در یک Thread جداگانه
    threading.Thread(target=data_handler.start_consuming2, daemon=True).start()

    #✅  5. حالا استراتژی خود را به DataHandler متصل کنید (Callback)
    # my_strategy = MyTradingStrategy()
    # data_handler.set_data_callback(my_strategy.on_new_data)

my_test()