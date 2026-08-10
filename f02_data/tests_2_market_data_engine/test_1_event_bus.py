# 1
# Run: pytest f02_data/tests_2_market_data_engine/test_1_event_bus.py -v
# یا همه با هم:
# pytest f02_data/tests_2_market_data_engine/ -v

import pytest
import threading
import time
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")
from f02_data.market_data_engine.event_bus_2 import EventBus

class TestEventBus:
    
    def test_subscribe_and_get_queue(self):
        """Test subscribe creates queue"""
        bus = EventBus(queue_size=10)
        sub_id = bus.subscribe()
        
        queue = bus.get_queue(sub_id)
        assert queue is not None
        assert queue.maxsize == 10
    
    def test_publish_and_get(self):
        """Test publish event and consume it"""
        bus = EventBus()
        sub_id = bus.subscribe()
        
        bus.publish("TEST_EVENT", {"data": 123})
        
        event = bus.get(sub_id, timeout=1.0)
        assert event is not None
        assert event["event_type"] == "TEST_EVENT"
        assert event["payload"]["data"] == 123
    
    def test_unsubscribe(self):
        """Test unsubscribe removes queue"""
        bus = EventBus()
        sub_id = bus.subscribe()
        assert bus.subscriber_count() == 1
        
        bus.unsubscribe(sub_id)
        assert bus.subscriber_count() == 0
        assert bus.get_queue(sub_id) is None
    
    def test_queue_full_drops_event(self):
        """Test when queue is full, event is dropped"""
        bus = EventBus(queue_size=1)
        sub_id = bus.subscribe()
        queue = bus.get_queue(sub_id)
        
        # Fill the queue
        queue.put_nowait({"event_type": "first", "payload": {}})
        
        # This should be dropped (no exception)
        bus.publish("SECOND", {})
        
        assert queue.qsize() == 1  # Still only first event
    
    def test_concurrent_publish_and_subscribe(self):
        """Test thread safety"""
        bus = EventBus()
        events_received = []
        
        def consumer():
            sub_id = bus.subscribe()
            for _ in range(10):
                event = bus.get(sub_id, timeout=2.0)
                if event:
                    events_received.append(event)
        
        def producer():
            for i in range(10):
                bus.publish("EVENT", {"idx": i})
                time.sleep(0.01)
        
        consumer_thread = threading.Thread(target=consumer)
        producer_thread = threading.Thread(target=producer)
        
        consumer_thread.start()
        time.sleep(0.1)  # Ensure consumer subscribed
        producer_thread.start()
        
        producer_thread.join()
        time.sleep(0.5)
        consumer_thread.join(timeout=1)
        
        assert len(events_received) == 10
    
    def test_stats(self):
        """Test stats method"""
        bus = EventBus()
        sub1 = bus.subscribe()
        sub2 = bus.subscribe()
        
        bus.publish("TEST", {})
        bus.publish("TEST", {})
        
        stats = bus.stats()
        assert sub1 in stats
        assert sub2 in stats
        assert stats[sub1] == 2
        assert stats[sub2] == 2
