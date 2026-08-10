# 4
# Run: pytest f02_data/tests_2_market_data_engine/test_4_market_data_engine.py -v
# یا همه با هم:
# pytest f02_data/tests_2_market_data_engine/ -v

import pytest
from unittest.mock import Mock, patch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f02_data.market_data_engine.market_data_engine_2 import MarketDataEngine
from f02_data.market_data_engine.event_bus_2 import EventBus

class TestMarketDataEngine:
    
    @pytest.fixture
    def mock_config(self):
        return {
            "event_bus": {"queue_size": 500},
            "executor": {"lookback_bars": 200}
        }
    
    def test_initialization(self, mock_config):
        """Test engine initialization"""
        engine = MarketDataEngine(cfg=mock_config)
        
        assert engine.event_bus is not None
        assert engine.worker is None
        assert engine._running is False
    
    def test_event_bus_queue_size_from_config(self, mock_config):
        """Test event bus uses queue_size from config"""
        engine = MarketDataEngine(cfg=mock_config)
        assert engine.event_bus._queue_size == 500
    
    def test_event_bus_default_queue_size(self):
        """Test event bus default queue size"""
        engine = MarketDataEngine(cfg={})
        assert engine.event_bus._queue_size == 1000
    
    @patch('f02_data.market_data_engine.market_data_engine_2.MT5StreamWorker')
    def test_start_creates_worker(self, MockWorker, mock_config):
        """Test start creates MT5StreamWorker"""
        mock_worker_instance = Mock()
        MockWorker.return_value = mock_worker_instance
        
        engine = MarketDataEngine(cfg=mock_config)
        
        engine.start(
            symbols=["EURUSD", "GBPUSD"],
            timeframes=["H1", "M15"],
            poll_interval_sec=2.0
        )
        
        MockWorker.assert_called_once_with(
            cfg=mock_config,
            event_bus=engine.event_bus,
            symbols=["EURUSD", "GBPUSD"],
            timeframes=["H1", "M15"],
            poll_interval_sec=2.0
        )
        
        mock_worker_instance.start.assert_called_once()
        assert engine._running is True
    
    def test_start_when_already_running(self, mock_config):
        """Test start does nothing if already running"""
        engine = MarketDataEngine(cfg=mock_config)
        engine._running = True
        
        engine.start(symbols=[], timeframes=[])
        # Should return without creating worker
        assert engine.worker is None
    
    @patch('f02_data.market_data_engine.market_data_engine_2.MT5StreamWorker')
    def test_stop(self, MockWorker, mock_config):
        """Test stop calls worker.stop"""
        mock_worker = Mock()
        MockWorker.return_value = mock_worker
        
        engine = MarketDataEngine(cfg=mock_config)
        engine.start(symbols=["EURUSD"], timeframes=["H1"])
        
        engine.stop()
        
        mock_worker.stop.assert_called_once()
        assert engine._running is False
    
    def test_get_event_bus(self, mock_config):
        """Test get_event_bus returns the event bus"""
        engine = MarketDataEngine(cfg=mock_config)
        bus = engine.get_event_bus()
        
        assert isinstance(bus, EventBus)
        assert bus is engine.event_bus
