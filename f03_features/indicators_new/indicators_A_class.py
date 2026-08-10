"""
Classic Technical Indicators - Wrapper Classes
Stateful wrappers around indicators_maths state classes.
Each wrapper receives candle data and maintains internal state.
"""

from typing import Literal, Optional, Tuple
# import numpy as np
# import pandas as pd
from .indicators_A_maths import (
    SMAState, WMAState, EMAState, ROCState, RSIState,
    TrueRangeState, ATRState, MACDState, BollingerState,
    KeltnerState, StochasticState, CCIState, MFIState,
    OBVState, WilliamsRState, ParabolicSARState, HeikinAshiState,

    SupertrendState, AroonState, DEMAState, TEMAState,
    HMAState, ZigZagState
)

# ----------------- 1-new
class SMA:
    """Simple Moving Average wrapper"""
    def __init__(self, period: int = 20, min_periods=None):
        self.period = period
        self.min_periods = min_periods
        self.state = SMAState(n=period, min_periods=min_periods)
    
    def update(self, close: float) -> float:
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 2-new
class WMA:
    """Weighted Moving Average wrapper"""
    def __init__(self, period: int = 20, min_periods=None):
        self.period = period
        self.min_periods = min_periods
        self.state = WMAState(n=period, min_periods=min_periods)
    
    def update(self, close: float) -> float:
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 3-new
class EMA:
    """Exponential Moving Average wrapper"""
    def __init__(self, period: int = 20, min_periods=None):
        self.period = period
        self.min_periods = min_periods
        self.state = EMAState(n=period, min_periods=min_periods)
    
    def update(self, close: float) -> float:
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 4
class ROC:
    """Rate of Change wrapper"""
    def __init__(self, period: int = 12):
        self.period = period
        self.state = ROCState(n=period)
    
    def update(self, close: float) -> float:
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 5-new
class RSI:
    """Relative Strength Index wrapper"""
    def __init__(self, period: int = 14, method: Literal["ema", "wilders"] = "ema"):
        self.period = period
        self.method = method
        self.state = RSIState(n=period, method=method)
    
    def update(self, close: float) -> float:
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()
# ----------------- 6
class TrueRange:
    """True Range wrapper"""
    def __init__(self):
        self.state = TrueRangeState()
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 7-new
class ATR:
    """Average True Range wrapper"""
    def __init__(self, period: int = 14, method: Literal["classic", "wilder", "ema"] = "wilder"):
        self.period = period
        self.method = method
        self.state = ATRState(n=period, method=method)
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 8
class MACD:
    """Moving Average Convergence Divergence wrapper"""
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.state = MACDState(fast=fast, slow=slow, signal=signal)
    
    def update(self, close: float) -> Tuple[float, float, float]:
        """Returns (macd, signal, histogram)"""
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 9
class BollingerBands:
    """Bollinger Bands wrapper"""
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.state = BollingerState(n=period, k=std_dev)
    
    def update(self, close: float) -> Tuple[float, float, float, float, float]:
        """Returns (upper, middle, lower)"""
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 10
class KeltnerChannel:
    """Keltner Channel wrapper"""
    def __init__(self, period: int = 20, multiplier: float = 2.0):
        self.period = period
        self.multiplier = multiplier
        self.state = KeltnerState(n=period, m=multiplier)
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, float, float, float, float]:
        """Returns (upper, middle, lower)"""
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 11-avdanced
class Stochastic:
    """Stochastic Oscillator wrapper"""
    def __init__(
        self, 
        k_period: int = 14, 
        d_period: int = 3,
        smooth_k: int = 3,
        method: Literal["sma", "ema"] = "sma",
        min_periods: Optional[int] = None
    ):
        """
        Stochastic Oscillator
        
        Parameters:
        -----------
        k_period : int
            دوره محاسبه %K (پیش‌فرض: 14)
        d_period : int
            دوره هموارسازی %D (پیش‌فرض: 3)
        smooth_k : int
            دوره هموارسازی %K (پیش‌فرض: 3)
            - smooth_k=1 → Fast Stochastic
            - smooth_k=3 → Slow Stochastic
        method : {"sma", "ema"}
            روش هموارسازی (پیش‌فرض: "sma")
        min_periods : int, optional
            حداقل داده برای شروع محاسبه
        """
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.method = method
        self.min_periods = min_periods
        
        self.state = StochasticState(
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
            method=method,
            min_periods=min_periods
        )
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, float]:
        """
        Returns:
        --------
        k : float
            مقدار %K (هموارشده)
        d : float
            مقدار %D (سیگنال)
        """
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 12-avdanced
class CCI:
    """Commodity Channel Index wrapper"""
    def __init__(self, period: int = 20, min_periods: int = None):
        self.period = period
        self.min_periods = min_periods
        self.state = CCIState(n=period, min_periods=min_periods)
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)
    
    def reset(self):
        """Reset indicator state"""
        self.state.reset()
#
#  ----------------- 13-avdanced
class MFI:
    """Money Flow Index wrapper"""
    def __init__(self, period: int = 14, min_periods: int = None):
        self.period = period
        self.min_periods = min_periods
        self.state = MFIState(n=period, min_periods=min_periods)
    
    def update(self, high: float, low: float, close: float, volume: float) -> float:
        return self.state.update(high, low, close, volume)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 14
class OBV:
    """On-Balance Volume wrapper"""
    def __init__(self):
        self.state = OBVState()
    
    def update(self, close: float, volume: float) -> float:
        return self.state.update(close, volume)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()
# ----------------- 15
class WilliamsR:
    """Williams %R wrapper"""
    def __init__(self, period: int = 14):
        self.period = period
        self.state = WilliamsRState(n=period)
    
    def update(self, high: float, low: float, close: float) -> float:
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()

# ----------------- 16
class ParabolicSAR:
    """Parabolic SAR wrapper"""
    def __init__(self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.state = ParabolicSARState(af_start=af_start, af_step=af_step, af_max=af_max)
    
    def update(self, high: float, low: float) -> float:
        return self.state.update(high, low)

    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()

# ----------------- 17
class HeikinAshi:
    """Heikin-Ashi wrapper"""
    def __init__(self):
        self.state = HeikinAshiState()
    
    def update(self, open_: float, high: float, low: float, close: float) -> Tuple[float, float, float, float]:
        """Returns (ha_open, ha_high, ha_low, ha_close)"""
        return self.state.update(open_, high, low, close)

    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()

# ----------------- 18
class Supertrend:
    """Supertrend wrapper with full state capabilities"""
    
    def __init__(self, 
                 period: int = 10, 
                 multiplier: float = 3.0,
                 atr_method: Literal["classic", "wilder", "ema"] = "wilder",
                 min_periods: Optional[int] = None):
        """
        Args:
            period: ATR period
            multiplier: Band multiplier
            atr_method: ATR calculation method ("classic", "wilder", "ema")
            min_periods: Minimum periods before valid output (default: period)
        """
        self.period = period
        self.multiplier = multiplier
        self.atr_method = atr_method
        self.min_periods = min_periods
        
        self.state = SupertrendState(
            period=period,
            multiplier=multiplier,
            atr_method=atr_method,
            min_periods=min_periods
        )
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, int]:
        """
        Update with new candle.
        
        Returns:
            (supertrend, direction): Supertrend value and direction (1=up, -1=down)
        """
        return self.state.update(high, low, close)
    
    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()

# ----------------- 19
class Aroon:
    """Aroon Indicator wrapper with full state capabilities"""
    
    def __init__(self, period: int = 25, min_periods: Optional[int] = None):
        """
        Args:
            period: Lookback period
            min_periods: Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods
        
        self.state = AroonState(period=period, min_periods=min_periods)
    
    def update(self, high: float, low: float) -> Tuple[float, float, float]:
        """
        Update with new candle.
        
        Returns:
            (aroon_up, aroon_down, oscillator)
        """
        return self.state.update(high, low)
    
    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()

# ----------------- 20
class DEMA:
    """Double Exponential Moving Average wrapper with full state capabilities"""
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None):
        """
        Args:
            period: EMA period
            min_periods: Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods
        
        self.state = DEMAState(n=period, min_periods=min_periods)
    
    def update(self, value: float) -> float:
        """
        Update with new value.
        
        Returns:
            DEMA value (NaN until min_periods reached)
        """
        return self.state.update(value)
    
    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()


# ----------------- 21
class TEMA:
    """Triple Exponential Moving Average wrapper with full state capabilities"""
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None):
        """
        Args:
            period: EMA period
            min_periods: Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods
        
        self.state = TEMAState(n=period, min_periods=min_periods)
    
    def update(self, value: float) -> float:
        """
        Update with new value.
        
        Returns:
            TEMA value (NaN until min_periods reached)
        """
        return self.state.update(value)
    
    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()


# ----------------- 22
class HMA:
    """Hull Moving Average wrapper with full state capabilities"""
    
    def __init__(self, period: int = 20, min_periods: Optional[int] = None):
        """
        Args:
            period: WMA period
            min_periods: Minimum periods before valid output (default: period)
        """
        self.period = period
        self.min_periods = min_periods
        
        self.state = HMAState(n=period, min_periods=min_periods)
    
    def update(self, value: float) -> float:
        """
        Update with new value.
        
        Returns:
            HMA value (NaN until min_periods reached)
        """
        return self.state.update(value)
    
    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()

# ----------------- 
