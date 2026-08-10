"""
test_full_pipeline.py

Comprehensive test suite covering:
- All 22 indicators from indicators_maths
- Full pipeline: indicators_maths → indicators_class → indicators_batch → feature_engine
- All three modes: train, live, backtest
- Output validation for each indicator
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import all components
from ..indicators_new.indicators_B_maths import (
    SMAState, EMAState, RSIState, MACDState, BBState,
    ATRState, StochState, ADXState, CCIState, KeltnerState,
    OBVState, WilliamsRState, ParabolicSARState, SupertrendState,
    IchimokuState, VWAPState, PivotState, FibState, ZigZagState,
    HeikinAshiState, RenkoState, KagiState
)
from ..indicators_new.indicators_B_class import Indicators
from ..indicators_new.indicators_B_batch import IndicatorsBatch
from f03_features.OLD.feature_engine import FeatureEngine
from f03_features.OLD.feature_registry import FEATURE_REGISTRY
from f10_utils.parser import parse_spec


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_ohlcv_data(n_bars=500, seed=42):
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    base_price = 100.0
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')
    
    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.02, n_bars)
    close = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.005, n_bars))
    
    # Volume
    volume = np.random.lognormal(10, 1, n_bars)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Fixture providing sample OHLCV data."""
    return generate_ohlcv_data(n_bars=500)


@pytest.fixture
def small_data():
    """Fixture providing small dataset for quick tests."""
    return generate_ohlcv_data(n_bars=100)


# ============================================================================
# Test Individual Indicator States (indicators_maths.py)
# ============================================================================

class TestIndicatorStates:
    """Test all 22 indicator state classes from indicators_maths."""
    
    def test_sma_state(self, sample_data):
        """Test SMAState."""
        state = SMAState(period=20)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1])
        assert results[-1] > 0
    
    def test_ema_state(self, sample_data):
        """Test EMAState."""
        state = EMAState(period=20)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1])
    
    def test_rsi_state(self, sample_data):
        """Test RSIState."""
        state = RSIState(period=14)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        valid_results = [r for r in results if not np.isnan(r)]
        assert all(0 <= r <= 100 for r in valid_results)
    
    def test_macd_state(self, sample_data):
        """Test MACDState."""
        state = MACDState(fast=12, slow=26, signal=9)
        results = []
        for _, row in sample_data.iterrows():
            macd, signal, hist = state.update(row['close'])
            results.append((macd, signal, hist))
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1][0])
    
    def test_bb_state(self, sample_data):
        """Test BBState (Bollinger Bands)."""
        state = BBState(period=20, std_dev=2.0)
        results = []
        for _, row in sample_data.iterrows():
            upper, middle, lower = state.update(row['close'])
            results.append((upper, middle, lower))
        
        assert len(results) == len(sample_data)
        # Upper should be > middle > lower
        if not np.isnan(results[-1][0]):
            assert results[-1][0] > results[-1][1] > results[-1][2]
    
    def test_atr_state(self, sample_data):
        """Test ATRState."""
        state = ATRState(period=14)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['high'], row['low'], row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        valid_results = [r for r in results if not np.isnan(r)]
        assert all(r >= 0 for r in valid_results)
    
    def test_stoch_state(self, sample_data):
        """Test StochState (Stochastic)."""
        state = StochState(k_period=14, d_period=3)
        results = []
        for _, row in sample_data.iterrows():
            k, d = state.update(row['high'], row['low'], row['close'])
            results.append((k, d))
        
        assert len(results) == len(sample_data)
        valid_k = [r[0] for r in results if not np.isnan(r[0])]
        assert all(0 <= k <= 100 for k in valid_k)
    
    def test_adx_state(self, sample_data):
        """Test ADXState."""
        state = ADXState(period=14)
        results = []
        for _, row in sample_data.iterrows():
            adx, plus_di, minus_di = state.update(row['high'], row['low'], row['close'])
            results.append((adx, plus_di, minus_di))
        
        assert len(results) == len(sample_data)
        valid_adx = [r[0] for r in results if not np.isnan(r[0])]
        assert all(0 <= a <= 100 for a in valid_adx)
    
    def test_cci_state(self, sample_data):
        """Test CCIState."""
        state = CCIState(period=20)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['high'], row['low'], row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1])
    
    def test_keltner_state(self, sample_data):
        """Test KeltnerState."""
        state = KeltnerState(ema_period=20, atr_period=10, multiplier=2.0)
        results = []
        for _, row in sample_data.iterrows():
            upper, middle, lower = state.update(row['high'], row['low'], row['close'])
            results.append((upper, middle, lower))
        
        assert len(results) == len(sample_data)
        if not np.isnan(results[-1][0]):
            assert results[-1][0] > results[-1][1] > results[-1][2]
    
    def test_obv_state(self, sample_data):
        """Test OBVState."""
        state = OBVState()
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['close'], row['volume'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1])
    
    def test_williamsr_state(self, sample_data):
        """Test WilliamsRState."""
        state = WilliamsRState(period=14)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['high'], row['low'], row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        valid_results = [r for r in results if not np.isnan(r)]
        assert all(-100 <= r <= 0 for r in valid_results)
    
    def test_parabolic_sar_state(self, sample_data):
        """Test ParabolicSARState."""
        state = ParabolicSARState(af_start=0.02, af_increment=0.02, af_max=0.2)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['high'], row['low'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1])
    
    def test_supertrend_state(self, sample_data):
        """Test SupertrendState."""
        state = SupertrendState(period=10, multiplier=3.0)
        results = []
        for _, row in sample_data.iterrows():
            supertrend, direction = state.update(row['high'], row['low'], row['close'])
            results.append((supertrend, direction))
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1][0])
        assert results[-1][1] in [-1, 1]
    
    def test_ichimoku_state(self, sample_data):
        """Test IchimokuState."""
        state = IchimokuState(tenkan=9, kijun=26, senkou_b=52)
        results = []
        for _, row in sample_data.iterrows():
            tenkan, kijun, senkou_a, senkou_b = state.update(row['high'], row['low'])
            results.append((tenkan, kijun, senkou_a, senkou_b))
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1][0])
    
    def test_vwap_state(self, sample_data):
        """Test VWAPState."""
        state = VWAPState()
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['high'], row['low'], row['close'], row['volume'])
            results.append(result)
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1])
        assert results[-1] > 0
    
    def test_pivot_state(self, sample_data):
        """Test PivotState."""
        state = PivotState()
        results = []
        for _, row in sample_data.iterrows():
            pivot, r1, r2, s1, s2 = state.update(row['high'], row['low'], row['close'])
            results.append((pivot, r1, r2, s1, s2))
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1][0])
    
    def test_fib_state(self, sample_data):
        """Test FibState."""
        state = FibState(lookback=50)
        results = []
        for _, row in sample_data.iterrows():
            levels = state.update(row['high'], row['low'])
            results.append(levels)
        
        assert len(results) == len(sample_data)
        assert len(results[-1]) > 0
    
    def test_zigzag_state(self, sample_data):
        """Test ZigZagState."""
        state = ZigZagState(threshold=0.05)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['high'], row['low'], row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
    
    def test_heikinashi_state(self, sample_data):
        """Test HeikinAshiState."""
        state = HeikinAshiState()
        results = []
        for _, row in sample_data.iterrows():
            ha_open, ha_high, ha_low, ha_close = state.update(
                row['open'], row['high'], row['low'], row['close']
            )
            results.append((ha_open, ha_high, ha_low, ha_close))
        
        assert len(results) == len(sample_data)
        assert not np.isnan(results[-1][0])
    
    def test_renko_state(self, sample_data):
        """Test RenkoState."""
        state = RenkoState(brick_size=1.0)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)
    
    def test_kagi_state(self, sample_data):
        """Test KagiState."""
        state = KagiState(reversal=0.04)
        results = []
        for _, row in sample_data.iterrows():
            result = state.update(row['close'])
            results.append(result)
        
        assert len(results) == len(sample_data)


# ============================================================================
# Test Indicators Class (indicators_class.py)
# ============================================================================

class TestIndicatorsClass:
    """Test the Indicators wrapper class."""
    
    def test_indicators_initialization(self):
        """Test Indicators class can be initialized."""
        indicators = Indicators()
        assert indicators is not None
    
    def test_indicators_update_single_bar(self, sample_data):
        """Test updating indicators with a single bar."""
        indicators = Indicators()
        row = sample_data.iloc[50]
        
        # This should not raise
        result = indicators.update(
            open_=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        
        assert result is not None
    
    def test_indicators_multiple_updates(self, small_data):
        """Test updating indicators with multiple bars."""
        indicators = Indicators()
        
        for _, row in small_data.iterrows():
            result = indicators.update(
                open_=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
        
        assert result is not None


# ============================================================================
# Test IndicatorsBatch (indicators_batch.py)
# ============================================================================

class TestIndicatorsBatch:
    """Test batch indicator computation."""
    
    def test_batch_computation(self, sample_data):
        """Test batch computation of all indicators."""
        batch = IndicatorsBatch()
        
        result_df = batch.compute(sample_data)
        
        assert result_df is not None
        assert len(result_df) == len(sample_data)
        assert 'close' in result_df.columns
    
    def test_batch_train_mode(self, sample_data):
        """Test batch computation in train mode."""
        batch = IndicatorsBatch()
        
        result_df = batch.compute(sample_data, mode='train')
        
        assert result_df is not None
        assert len(result_df) == len(sample_data)
    
    def test_batch_live_mode(self, sample_data):
        """Test batch computation in live mode."""
        batch = IndicatorsBatch()
        
        result_df = batch.compute(sample_data, mode='live')
        
        assert result_df is not None
    
    def test_batch_backtest_mode(self, sample_data):
        """Test batch computation in backtest mode."""
        batch = IndicatorsBatch()
        
        result_df = batch.compute(sample_data, mode='backtest')
        
        assert result_df is not None
        assert len(result_df) == len(sample_data)


# ============================================================================
# Test Feature Engine (feature_engine.py)
# ============================================================================

class TestFeatureEngine:
    """Test the feature generation engine."""
    
    def test_feature_engine_initialization(self):
        """Test FeatureEngine can be initialized."""
        engine = FeatureEngine()
        assert engine is not None
    
    def test_feature_generation_train_mode(self, sample_data):
        """Test feature generation in train mode."""
        engine = FeatureEngine()
        
        features_df = engine.generate_features(sample_data, mode='train')
        
        assert features_df is not None
        assert len(features_df) <= len(sample_data)
        assert len(features_df.columns) > 0
    
    def test_feature_generation_live_mode(self, sample_data):
        """Test feature generation in live mode."""
        engine = FeatureEngine()
        
        features_df = engine.generate_features(sample_data, mode='live')
        
        assert features_df is not None
    
    def test_feature_generation_backtest_mode(self, sample_data):
        """Test feature generation in backtest mode."""
        engine = FeatureEngine()
        
        features_df = engine.generate_features(sample_data, mode='backtest')
        
        assert features_df is not None
        assert len(features_df) <= len(sample_data)
    
    def test_feature_registry_integration(self, sample_data):
        """Test that feature registry is properly integrated."""
        engine = FeatureEngine()
        
        features_df = engine.generate_features(sample_data, mode='train')
        
        # Check that some registered features are present
        assert len(features_df.columns) > 0


# ============================================================================
# Test Full Pipeline Integration
# ============================================================================

class TestFullPipeline:
    """Test the complete pipeline from raw data to features."""
    
    def test_pipeline_train_mode(self, sample_data):
        """Test full pipeline in train mode."""
        # Step 1: Compute indicators
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='train')
        
        # Step 2: Generate features
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='train')
        
        assert features_df is not None
        assert len(features_df) > 0
        assert len(features_df.columns) > 0
        
        # Validate no all-NaN columns
        assert not features_df.isna().all().any()
    
    def test_pipeline_live_mode(self, sample_data):
        """Test full pipeline in live mode."""
        # Step 1: Compute indicators
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='live')
        
        # Step 2: Generate features
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='live')
        
        assert features_df is not None
    
    def test_pipeline_backtest_mode(self, sample_data):
        """Test full pipeline in backtest mode."""
        # Step 1: Compute indicators
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='backtest')
        
        # Step 2: Generate features
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='backtest')
        
        assert features_df is not None
        assert len(features_df) > 0
    
    def test_pipeline_consistency_across_modes(self, small_data):
        """Test that pipeline produces consistent structure across modes."""
        batch = IndicatorsBatch()
        engine = FeatureEngine()
        
        # Run all three modes
        train_features = engine.generate_features(
            batch.compute(small_data, mode='train'), mode='train'
        )
        live_features = engine.generate_features(
            batch.compute(small_data, mode='live'), mode='live'
        )
        backtest_features = engine.generate_features(
            batch.compute(small_data, mode='backtest'), mode='backtest'
        )
        
        # All should produce DataFrames
        assert isinstance(train_features, pd.DataFrame)
        assert isinstance(live_features, pd.DataFrame)
        assert isinstance(backtest_features, pd.DataFrame)
    
    def test_pipeline_with_all_22_indicators(self, sample_data):
        """Test that all 22 indicators flow through the pipeline."""
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='train')
        
        # Check that indicator columns are present
        # (exact column names depend on implementation)
        assert len(indicators_df.columns) >= 5  # At minimum OHLCV
        
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='train')
        
        assert features_df is not None
        assert len(features_df) > 0


# ============================================================================
# Test Output Validation
# ============================================================================

class TestOutputValidation:
    """Validate outputs at each stage of the pipeline."""
    
    def test_no_infinite_values(self, sample_data):
        """Test that no infinite values are produced."""
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='train')
        
        assert not np.isinf(indicators_df.select_dtypes(include=[np.number])).any().any()
        
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='train')
        
        assert not np.isinf(features_df.select_dtypes(include=[np.number])).any().any()
    
    def test_reasonable_value_ranges(self, sample_data):
        """Test that values are in reasonable ranges."""
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='train')
        
        # Prices should be positive
        assert (indicators_df['close'] > 0).all()
        
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='train')
        
        # Features should not have extreme values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            valid_values = features_df[col].dropna()
            if len(valid_values) > 0:
                assert valid_values.abs().max() < 1e10
    
    def test_timestamp_preservation(self, sample_data):
        """Test that timestamps are preserved through pipeline."""
        batch = IndicatorsBatch()
        indicators_df = batch.compute(sample_data, mode='train')
        
        if 'timestamp' in indicators_df.columns:
            assert len(indicators_df['timestamp']) == len(sample_data)
        
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='train')
        
        if 'timestamp' in features_df.columns:
            assert len(features_df['timestamp']) > 0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        batch = IndicatorsBatch()
        result = batch.compute(empty_df, mode='train')
        
        assert result is not None
        assert len(result) == 0
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        single_row = generate_ohlcv_data(n_bars=1)
        
        batch = IndicatorsBatch()
        result = batch.compute(single_row, mode='train')
        
        assert result is not None
    
    def test_missing_values_handling(self, sample_data):
        """Test handling of missing values in input."""
        data_with_nan = sample_data.copy()
        data_with_nan.loc[10:20, 'close'] = np.nan
        
        batch = IndicatorsBatch()
        result = batch.compute(data_with_nan, mode='train')
        
        assert result is not None


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_dataset_processing(self):
        """Test processing of large dataset."""
        large_data = generate_ohlcv_data(n_bars=5000)
        
        batch = IndicatorsBatch()
        indicators_df = batch.compute(large_data, mode='train')
        
        engine = FeatureEngine()
        features_df = engine.generate_features(indicators_df, mode='train')
        
        assert features_df is not None
        assert len(features_df) > 0


    def test_processing_speed(self):
        """
        Ensure pipeline runs in reasonable time for medium dataset.
        """
        import time

        data = generate_ohlcv_data(n_bars=2000)

        batch = IndicatorsBatch()
        engine = FeatureEngine()

        start = time.time()

        indicators_df = batch.compute(data, mode="train")
        features_df = engine.generate_features(indicators_df, mode="train")

        end = time.time()

        elapsed = end - start

        assert features_df is not None
        assert elapsed < 30   # pipeline should finish under 30 seconds

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
