"""
تستر متد update_live از کلاس DataHandler - نسخه فشرده

Run: python -m f02_data.test_live_datahandler_B
"""
from __future__ import annotations
import sys
import os
import threading
import logging
from pathlib import Path
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f02_data.data_handler_F_3 import DataHandler, BuildParams
from f02_data.mt5_data_loader_E import MT5DataLoader_batch, DownloadPlan
from f02_data.market_data_engine.event_bus_2 import EventBus
from f02_data.market_data_engine.mt5_stream_worker_2 import MT5StreamWorker
from f10_utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-6s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s"
)
logger = logging.getLogger(__name__)


class LiveDataHandlerTester:
    # ------------------------------------------------------------------------- 1
    def __init__(self, cfg):
        self.cfg = cfg
        # dl = cfg.get("download_defaults", {})

        self.event_bus = EventBus()
        self.data_handler = DataHandler(cfg, event_bus=self.event_bus)

        self.symbol = cfg.get("features", {}).get("symbol", "XAUUSD_i")

        self._base_tf = self.data_handler._base_tf
        self._required_tfs = self.data_handler._required_timeframes

        # self.timeframes = [self._base_tf] + list(dl.get("timeframes", ["H1", "H4"]))
        self.num_live_candles = 3

        self.data_handler.set_data_callback(self._on_new_data)   # ← استفاده از callback

        self.live_rows = []
        self.collected = 0
        self.worker = None
        self._running = True

    # ------------------------------------------------------------------------- 2
    def _on_new_data(self, df: pd.DataFrame):
        """DataHandler هر بار دیتافریم جدید را از طریق این callback تحویل می‌دهد"""
        if not self._running:
            return
        self.collected += 1
        self.live_rows.append(df.iloc[-1:].copy())
        logger.info(f"Candle {self.collected}/{self.num_live_candles} collected")
        if self.collected >= self.num_live_candles:
            logger.info("Enough candles, stopping worker...")
            self._running = False
            if self.worker:
                self.worker.stop()

    # ------------------------------------------------------------------------- 3
    def _download_batch_data(self):
        """دانلود داده‌های batch برای همان بازه زمانی"""
        if not self.live_rows:
            return False
        start = self.live_rows[0].index[0] - timedelta(minutes=30)
        end = self.live_rows[-1].index[0] + timedelta(minutes=30)
        loader = MT5DataLoader_batch(cfg=self.cfg)
        plans = [DownloadPlan(symbol=self.symbol, timeframe=tf, date_from=start, date_to=end,
                              lookback_bars=None, range_policy="date") for tf in self.timeframes]
        results = loader.run(plans)
        return not any("error" in r for r in results)

    # ------------------------------------------------------------------------- 4
    def _compare_results(self) -> bool:
        """مقایسه خروجی update_live با build()"""
        # ساخت دیتافریم کامل از طریق build
        build_df = self.data_handler.build(BuildParams(
            symbol=self.symbol, base_tf=self._base_tf,
            timeframes=self.timeframes, format_="parquet"
        ))
        # فیلتر بر اساس بازه زمانی جمع‌آوری شده
        start, end = self.live_rows[0].index[0], self.live_rows[-1].index[0]
        build_df = build_df[(build_df.index >= start) & (build_df.index <= end)]

        if build_df.empty or not self.live_rows:
            logger.error("No data to compare")
            return False

        last_live = self.live_rows[-1].iloc[0]
        last_build = build_df.iloc[-1]

        # مقایسه ستون‌های عددی (به جز qc_ و session_)
        all_match = True
        for col in set(last_build.index) & set(last_live.index):
            if col.startswith(("qc_", "session_")):
                continue
            lv, bv = last_live[col], last_build[col]
            if pd.isna(lv) and pd.isna(bv):
                continue
            if pd.isna(lv) or pd.isna(bv) or (abs(lv - bv) > 1e-6 if isinstance(lv, float) else lv != bv):
                logger.warning(f"Mismatch in {col}: live={lv}, build={bv}")
                all_match = False

        logger.info("✓ All match!" if all_match else "✗ Mismatch found")
        # ذخیره نتایج برای دیباگ
        out_dir = Path(__file__).parent.parent / ""
        pd.concat(self.live_rows).to_csv(out_dir / "live_output.csv")
        build_df.to_csv(out_dir / "batch_output.csv")
        return all_match

    # ------------------------------------------------------------------------- 5
    def run(self) -> bool:
        logger.info(f"Testing {self.symbol} {self._base_tf} -> collecting {self.num_live_candles} candles")

        # 1) اتصال DataHandler به EventBus و شروع مصرف در ترد جداگانه
        self.data_handler.subscribe_to_event_bus(self.event_bus)
        threading.Thread(target=self.data_handler.start_consuming, daemon=True).start()

        # 2) راه‌اندازی worker (تنها base_tf را poll می‌کند)
        self.worker = MT5StreamWorker(
            cfg=self.cfg, event_bus=self.event_bus,
            symbols=[self.symbol], timeframes=[self._base_tf], poll_interval_sec=1.0
        )
        try:
            self.worker.start()   # مسدود می‌شود تا stop فراخوانی شود
        except Exception as e:
            logger.error(f"Worker failed: {e}")
            return False
        finally:
            self.worker.connector.shutdown()

        if self.collected < self.num_live_candles:
            logger.warning(f"Only {self.collected}/{self.num_live_candles} collected")
            return False

        # 3) دانلود داده‌های batch و مقایسه
        if not self._download_batch_data():
            logger.error("Batch download failed")
            return False
        return self._compare_results()

    # ------------------------------------------------------------------------- End of Methods

def main():
    tester = LiveDataHandlerTester(load_config())
    success = tester.run()
    print(f"\nTEST {'PASSED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()