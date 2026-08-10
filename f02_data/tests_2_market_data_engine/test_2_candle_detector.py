# 2
# Run: pytest f02_data/tests_2_market_data_engine/test_2_candle_detector.py -v
# یا همه با هم:
# pytest f02_data/tests_2_market_data_engine/ -v

import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + "/../..")
from f02_data.market_data_engine.candle_detector_2 import CandleDetector

class TestCandleDetector:
    
    @pytest.fixture
    def sample_df(self):
        """Create sample OHLC dataframe"""
        dates = pd.date_range(
            start="2024-01-01 00:00:00", 
            periods=5, 
            freq="1h", 
            tz="UTC"
        )
        df = pd.DataFrame({
            "open": [1.1000, 1.1010, 1.1020, 1.1015, 1.1025],
            "high": [1.1010, 1.1020, 1.1030, 1.1025, 1.1035],
            "low": [1.0995, 1.1005, 1.1015, 1.1010, 1.1020],
            "close": [1.1005, 1.1015, 1.1025, 1.1020, 1.1030],
            "volume": [1000, 1200, 1100, 1300, 1250]
        }, index=dates)
        return df
    
    def test_first_call_returns_none(self, sample_df):
        """First call should return None (no previous candle)"""
        detector = CandleDetector()
        result = detector.detect("EURUSD", "H1", sample_df)
        assert result is None
    
    def test_new_candle_detection(self, sample_df):
        """Test detecting a new candle"""
        detector = CandleDetector()
        
        # First call - initialize state
        detector.detect("EURUSD", "H1", sample_df.iloc[:4])
        
        # Second call with new candle
        result = detector.detect("EURUSD", "H1", sample_df)
        
        assert result is not None
        assert result["symbol"] == "EURUSD"
        assert result["timeframe"] == "H1"
        assert result["close"] == 1.1030
    
    def test_no_new_candle(self, sample_df):
        """Test when no new candle, returns None"""
        detector = CandleDetector()
        
        # First call
        detector.detect("EURUSD", "H1", sample_df)
        
        # Second call with same data
        result = detector.detect("EURUSD", "H1", sample_df)
        
        assert result is None
    
    def test_different_symbols_independent(self, sample_df):
        """Test states for different symbols are independent"""
        detector = CandleDetector()
        
        # First symbol
        detector.detect("EURUSD", "H1", sample_df.iloc[:4])
        result1 = detector.detect("EURUSD", "H1", sample_df)
        
        # Second symbol (first call)
        result2 = detector.detect("GBPUSD", "H1", sample_df)
        
        assert result1 is not None
        assert result2 is None  # First call for GBPUSD
    
    def test_different_timeframes_independent(self, sample_df):
        """Test states for different timeframes are independent"""
        detector = CandleDetector()
        
        # H1 timeframe
        detector.detect("EURUSD", "H1", sample_df.iloc[:4])
        result_h1 = detector.detect("EURUSD", "H1", sample_df)
        
        # M5 timeframe (first call)
        result_m5 = detector.detect("EURUSD", "M5", sample_df)
        
        assert result_h1 is not None
        assert result_m5 is None  # First call for M5
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        detector = CandleDetector()
        result = detector.detect("EURUSD", "H1", pd.DataFrame())
        assert result is None
    
    def test_missing_columns(self):
        """Test dataframe missing some OHLC columns"""
        dates = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({
            "close": [1.1000, 1.1010],
            "volume": [100, 200]
        }, index=dates)
        
        detector = CandleDetector()
        detector.detect("EURUSD", "H1", df.iloc[:1])
        result = detector.detect("EURUSD", "H1", df)
        
        assert result is not None
        assert result["close"] == 1.1010
        assert result["open"] is None  # Missing column
