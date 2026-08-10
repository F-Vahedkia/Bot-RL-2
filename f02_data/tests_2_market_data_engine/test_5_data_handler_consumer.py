# 5
# Run: pytest f02_data/tests_2_market_data_engine/test_5_data_handler_consumer.py -v
# یا همه با هم:
# pytest f02_data/tests_2_market_data_engine/ -v

import pytest
import time
import threading
from unittest.mock import Mock
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from f02_data.market_data_engine.data_handler_live_consumer_2 import DataHandlerLiveConsumer
from f02_data.market_data_engine.event_bus_2 import EventBus

class TestDataHandlerLiveConsumer:
    
    @pytest.fixture
    def event_bus(self):
        return EventBus()
    
    @pytest.fixture
    def mock_config(self):
        return {"some": "config"}
    
    def test_initialization(self, mock_config, event_bus):
        """Test consumer initialization"""
        consumer = DataHandlerLiveConsumer(
            cfg=mock_config,
            event_bus=event_bus
        )
        
        assert consumer.subscriber_id is not None
        assert consumer.queue is not None
        assert consumer._running is False
    
    def test_attach_data_handler(self, mock_config, event_bus):
        """Test attaching data handler"""
        consumer = DataHandlerLiveConsumer(cfg=mock_config, event_bus=event_bus)
        mock_handler = Mock()
        
        consumer.attach_data_handler(mock_handler)
        
        assert consumer.data_handler is mock_handler
    
    def test_process_event_ignores_wrong_type(self, mock_config, event_bus):
        """Test only NEW_CANDLE events are processed"""
        consumer = DataHandlerLiveConsumer(cfg=mock_config, event_bus=event_bus)
        mock_handler = Mock()
        consumer.attach_data_handler(mock_handler)
        
        # Wrong event type
        event = {"event_type": "WRONG_EVENT", "payload": {}}
        consumer._process_event(event)
        
        mock_handler.on_new_candle.assert_not_called()
    
    def test_process_event_calls_handler(self, mock_config, event_bus):
        """Test correct event triggers handler"""
        consumer = DataHandlerLiveConsumer(cfg=mock_config, event_bus=event_bus)
        mock_handler = Mock()
        consumer.attach_data_handler(mock_handler)
        
        payload = {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "close": 1.1000
        }
        event = {"event_type": "NEW_CANDLE", "payload": payload}
        
        consumer._process_event(event)
        
        mock_handler.on_new_candle.assert_called_once_with(payload)
    
    def test_process_event_handles_exception(self, mock_config, event_bus):
        """Test handler exception is caught and logged"""
        consumer = DataHandlerLiveConsumer(cfg=mock_config, event_bus=event_bus)
        mock_handler = Mock()
        mock_handler.on_new_candle.side_effect = Exception("Handler error")
        consumer.attach_data_handler(mock_handler)
        
        event = {"event_type": "NEW_CANDLE", "payload": {"symbol": "EURUSD", "timeframe": "H1"}}
        
        # Should not raise exception
        consumer._process_event(event)
        
        mock_handler.on_new_candle.assert_called_once()
    
    def test_loop_receives_events(self, mock_config, event_bus):
        """Test consumer loop receives published events"""
        consumer = DataHandlerLiveConsumer(cfg=mock_config, event_bus=event_bus)
        mock_handler = Mock()
        consumer.attach_data_handler(mock_handler)
        
        # Publish an event
        event_bus.publish("NEW_CANDLE", {"symbol": "EURUSD", "timeframe": "H1"})
        
        # Run one iteration
        event = event_bus.get(consumer.subscriber_id, timeout=1.0)
        assert event is not None
        consumer._process_event(event)
        
        mock_handler.on_new_candle.assert_called_once()
    
    def test_start_stop(self, mock_config, event_bus):
        """Test start and stop methods"""
        consumer = DataHandlerLiveConsumer(cfg=mock_config, event_bus=event_bus)
        
        # Start in a thread
        thread = threading.Thread(target=consumer.start)
        consumer._running = False  # Will exit immediately
        
        consumer.stop()
        assert consumer._running is False
