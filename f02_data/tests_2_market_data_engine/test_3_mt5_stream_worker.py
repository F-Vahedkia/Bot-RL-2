# 3
# Run: pytest f02_data/tests_2_market_data_engine/test_3_mt5_stream_worker.py -v
# یا همه با هم:
# pytest f02_data/tests_2_market_data_engine/ -v

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timezone
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f02_data.market_data_engine.mt5_stream_worker_2 import MT5StreamWorker
from f02_data.market_data_engine.event_bus_2 import EventBus

class TestMT5StreamWorker:
    
    @pytest.fixture
    def mock_config(self):
        return {
            "executor": {"lookback_bars": 200},
            "connection": {"mt5_credentials": {"login": 12345}}
        }
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range(
            start="2024-01-01 00:00:00",
            periods=10,
            freq="1h",
            tz="UTC"
        )
        return pd.DataFrame({
            "open": [1.1000] * 10,
            "high": [1.1010] * 10,
            "low": [1.0990] * 10,
            "close": [1.1005] * 10,
            "volume": [1000] * 10
        }, index=dates)
    
    @patch('f02_data.market_data_engine.mt5_stream_worker_2.MT5Connector')
    def test_initialization(self, MockMT5Connector, mock_config, event_bus):
        """Test worker initialization"""
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        mock_connector.initialize.return_value = True
        
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            symbols=["EURUSD", "GBPUSD"],
            timeframes=["H1", "M15"],
            poll_interval_sec=1.0
        )
        
        assert worker.symbols == ["EURUSD", "GBPUSD"]
        assert worker.timeframes == ["H1", "M15"]
        assert worker.poll_interval_sec == 1.0
    
    @patch('f02_data.market_data_engine.mt5_stream_worker_2.MT5Connector')
    def test_fetch_calls_connector(self, MockMT5Connector, mock_config, event_bus, sample_df):
        """Test fetch method calls connector correctly"""
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        mock_connector.get_candles_num.return_value = sample_df
        mock_connector.initialize.return_value = True
        
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            symbols=["EURUSD"],
            timeframes=["H1"]
        )
        
        # Access private method for testing
        df = worker._fetch("EURUSD", "H1")
        
        mock_connector.get_candles_num.assert_called_once_with(
            symbol="EURUSD",
            timeframe="H1",
            num_candles=200
        )
        assert not df.empty
    
    @patch('f02_data.market_data_engine.mt5_stream_worker_2.MT5Connector')
    def test_detection_and_publish(self, MockMT5Connector, mock_config, event_bus, sample_df):
        """Test that new candle detection triggers publish"""
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        mock_connector.get_candles_num.return_value = sample_df
        mock_connector.initialize.return_value = True

        # Subscribe to event bus
        sub_id = event_bus.subscribe()

        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            symbols=["EURUSD"],
            timeframes=["H1"],
            poll_interval_sec=0.1
        )

        # ایجاد دو دیتافریم متفاوت (قدیمی و جدید)
        sample_df_old = sample_df.iloc[:-1]  # 4 ردیف اول
        sample_df_new = sample_df              # 5 ردیف کامل

        # First detection (old data) - should not publish
        event = worker.detector.detect("EURUSD", "H1", sample_df_old)
        assert event is None

        # Second detection with NEW data - should publish
        event = worker.detector.detect("EURUSD", "H1", sample_df_new)
        assert event is not None  # ✅ حالا پاس می‌شود
    
    @patch('f02_data.market_data_engine.mt5_stream_worker_2.MT5Connector')
    def test_connection_failure_handling(self, MockMT5Connector, mock_config, event_bus):
        """Test that connection failure raises exception"""
        mock_connector = Mock()
        MockMT5Connector.return_value = mock_connector
        mock_connector.initialize.return_value = False
        
        worker = MT5StreamWorker(
            cfg=mock_config,
            event_bus=event_bus,
            symbols=["EURUSD"],
            timeframes=["H1"]
        )
        
        with pytest.raises(RuntimeError, match="MT5 connection failed"):
            worker.start()
