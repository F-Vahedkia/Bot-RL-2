
# این فایل همان فایل core.py است. منتها برای حالت live

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

EPS = 1e-12

# ------------------------------------- 0
def _safe_div(a, b):
    return 0.0 if abs(b) < EPS else a / b

"""
class:
SMA, WMA, EMA, ROC, RSI, TR, ATR, MACD, Bollinger, Keltner, Stochastic,
CCI, MFI, OBV, WilliamsR, ParabolicSARS, HeikinAshi

dataclass:
Supertrend, Aroon, DEMA, TEMA, HMA, ZigZag
"""
# ============================================================================
# Indicator States (Classic Technical Indicators) PART-1
# ============================================================================
# ------------------------------------- 1
class SMAState:
    def __init__(self, n):
        self.n = n
        self.buf = deque(maxlen=n)
        self.sum = 0.0

    def update(self, x):
        x = float(x)
        if len(self.buf) == self.n:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        return np.nan if len(self.buf) < self.n else self.sum / self.n


# ------------------------------------- 2
class WMAState:
    def __init__(self, n):
        self.n = n
        self.buf = deque(maxlen=n)
        self.weights = np.arange(1, n + 1, dtype=np.float64)
        self.weights /= self.weights.sum()

    def update(self, x):
        self.buf.append(float(x))
        if len(self.buf) < self.n:
            return np.nan
        return float(np.dot(np.asarray(self.buf), self.weights))


# ------------------------------------- 3
class EMAState:
    def __init__(self, n):
        self.alpha = 2.0 / (n + 1.0)
        self.value = None

    def update(self, x):
        x = float(x)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


# ------------------------------------- 4
class ROCState:
    def __init__(self, n):
        self.n = n
        self.buf = deque(maxlen=n + 1)

    def update(self, x):
        x = float(x)
        self.buf.append(x)
        if len(self.buf) <= self.n:
            return np.nan
        return (x - self.buf[0]) / self.buf[0] * 100.0


# ------------------------------------- 5
class RSIState:
    def __init__(self, n):
        self.n = n
        self.prev = None
        self.gains = deque(maxlen=n)
        self.losses = deque(maxlen=n)
        self.avg_gain = None
        self.avg_loss = None

    def update(self, close):
        close = float(close)
        if self.prev is None:
            self.prev = close
            return np.nan

        d = close - self.prev
        gain = max(d, 0.0)
        loss = max(-d, 0.0)

        self.gains.append(gain)
        self.losses.append(loss)

        if len(self.gains) < self.n:
            self.prev = close
            return np.nan

        if self.avg_gain is None:
            self.avg_gain = np.mean(self.gains)
            self.avg_loss = np.mean(self.losses)
        else:
            a = 1.0 / self.n
            self.avg_gain = (1 - a) * self.avg_gain + a * gain
            self.avg_loss = (1 - a) * self.avg_loss + a * loss

        self.prev = close
        rs = _safe_div(self.avg_gain, self.avg_loss)
        return 100.0 - (100.0 / (1.0 + rs))


# ------------------------------------- 6
class TrueRangeState:
    def __init__(self):
        self.prev_close = None

    def update(self, h, l, c):
        h, l, c = float(h), float(l), float(c)
        if self.prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.prev_close), abs(l - self.prev_close))
        self.prev_close = c
        return tr


# ------------------------------------- 7
class ATRState:
    def __init__(self, n):
        self.n = n
        self.tr = TrueRangeState()
        self.values = deque(maxlen=n)
        self.atr = None

    def update(self, h, l, c):
        tr = self.tr.update(h, l, c)
        self.values.append(tr)

        if len(self.values) < self.n:
            return np.nan

        if self.atr is None:
            self.atr = np.mean(self.values)
        else:
            a = 1.0 / self.n
            self.atr = (1 - a) * self.atr + a * tr

        return self.atr


# ------------------------------------- 8
class MACDState:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = EMAState(fast)
        self.slow = EMAState(slow)
        self.sig = EMAState(signal)

    def update(self, c):
        f = self.fast.update(c)
        s = self.slow.update(c)
        macd = f - s
        sig = self.sig.update(macd)
        hist = macd - sig
        return macd, sig, hist


# ------------------------------------- 9
class BollingerState:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.buf = deque(maxlen=n)

    def update(self, c):
        self.buf.append(float(c))
        if len(self.buf) < self.n:
            return np.nan, np.nan, np.nan
        arr = np.asarray(self.buf)
        mid = arr.mean()
        std = arr.std()
        return mid + self.k * std, mid, mid - self.k * std


# ------------------------------------- 10
class KeltnerState:
    def __init__(self, n, m):
        self.ema = EMAState(n)
        self.atr = ATRState(n)
        self.m = m

    def update(self, h, l, c):
        mid = self.ema.update(c)
        atr = self.atr.update(h, l, c)
        if np.isnan(atr):
            return np.nan, np.nan, np.nan
        return mid + self.m * atr, mid, mid - self.m * atr


# ------------------------------------- 11
class StochasticState:
    def __init__(self, n, d):
        self.n = n
        self.highs = deque(maxlen=n)
        self.lows = deque(maxlen=n)
        self.d_sma = SMAState(d)

    def update(self, h, l, c):
        self.highs.append(float(h))
        self.lows.append(float(l))
        c = float(c)

        if len(self.highs) < self.n:
            return np.nan, np.nan

        hh = max(self.highs)
        ll = min(self.lows)
        k = 100.0 * _safe_div(c - ll, hh - ll)
        dline = self.d_sma.update(k)
        return k, dline


# ------------------------------------- 12
class CCIState:
    def __init__(self, n):
        self.n = n
        self.tp = deque(maxlen=n)

    def update(self, h, l, c):
        tp = (float(h) + float(l) + float(c)) / 3.0
        self.tp.append(tp)

        if len(self.tp) < self.n:
            return np.nan

        arr = np.asarray(self.tp)
        ma = arr.mean()
        md = np.mean(np.abs(arr - ma))
        return (tp - ma) / (0.015 * md + EPS)


# ------------------------------------- 13
class MFIState:
    def __init__(self, n):
        self.n = n
        self.prev_tp = None
        self.pos = deque(maxlen=n)
        self.neg = deque(maxlen=n)

    def update(self, h, l, c, v):
        tp = (float(h) + float(l) + float(c)) / 3.0
        mf = tp * float(v)

        if self.prev_tp is None:
            self.prev_tp = tp
            return np.nan

        self.pos.append(mf if tp > self.prev_tp else 0.0)
        self.neg.append(mf if tp < self.prev_tp else 0.0)

        self.prev_tp = tp

        if len(self.pos) < self.n:
            return np.nan

        ratio = _safe_div(sum(self.pos), sum(self.neg))
        return 100.0 - (100.0 / (1.0 + ratio))


# ------------------------------------- 14
class OBVState:
    def __init__(self):
        self.prev = None
        self.value = 0.0

    def update(self, c, v):
        c = float(c)
        v = float(v)

        if self.prev is None:
            self.prev = c
            return 0.0

        if c > self.prev:
            self.value += v
        elif c < self.prev:
            self.value -= v

        self.prev = c
        return self.value


# ------------------------------------- 15
class WilliamsRState:
    def __init__(self, n):
        self.n = n
        self.highs = deque(maxlen=n)
        self.lows = deque(maxlen=n)

    def update(self, h, l, c):
        self.highs.append(float(h))
        self.lows.append(float(l))
        c = float(c)

        if len(self.highs) < self.n:
            return np.nan

        hh = max(self.highs)
        ll = min(self.lows)
        return -100.0 * _safe_div(hh - c, hh - ll)


# ------------------------------------- 16
class ParabolicSARState:
    def __init__(self, af_start=0.02, af_step=0.02, af_max=0.2):
        self.af_start = af_start
        self.af_step = af_step
        self.af_max = af_max
        self.initialized = False

    def update(self, h, l):
        h, l = float(h), float(l)

        if not self.initialized:
            self.sar = l
            self.ep = h
            self.af = self.af_start
            self.uptrend = True
            self.initialized = True
            return np.nan

        self.sar = self.sar + self.af * (self.ep - self.sar)

        if self.uptrend:
            if h > self.ep:
                self.ep = h
                self.af = min(self.af + self.af_step, self.af_max)
            if l < self.sar:
                self.uptrend = False
                self.sar = self.ep
                self.ep = l
                self.af = self.af_start
        else:
            if l < self.ep:
                self.ep = l
                self.af = min(self.af + self.af_step, self.af_max)
            if h > self.sar:
                self.uptrend = True
                self.sar = self.ep
                self.ep = h
                self.af = self.af_start

        return self.sar


# ------------------------------------- 17
class HeikinAshiState:
    def __init__(self):
        self.prev_open = None
        self.prev_close = None

    def update(self, o, h, l, c):
        o, h, l, c = map(float, (o, h, l, c))
        ha_close = (o + h + l + c) / 4.0

        if self.prev_open is None:
            ha_open = (o + c) / 2.0
        else:
            ha_open = (self.prev_open + self.prev_close) / 2.0

        ha_high = max(h, ha_open, ha_close)
        ha_low = min(l, ha_open, ha_close)

        self.prev_open = ha_open
        self.prev_close = ha_close

        return ha_open, ha_high, ha_low, ha_close
    

# ============================================================================
# Additional Indicator States (Classic Technical Indicators) PART-2
# ============================================================================
# ------------------------------------- 18
@dataclass
class SupertrendState:
    """State for Supertrend indicator (ATR-based trend following)."""
    atr: float = 0.0
    basic_upper: float = 0.0
    basic_lower: float = 0.0
    final_upper: float = 0.0
    final_lower: float = 0.0
    direction: int = 1  # 1 = uptrend, -1 = downtrend
    supertrend: float = 0.0
    atr_state: Optional['ATRState'] = None

    def update(self, high: float, low: float, close: float, period: int = 10, multiplier: float = 3.0) -> tuple:
        """Update Supertrend state incrementally."""
        if self.atr_state is None:
            self.atr_state = ATRState()
        
        # Update ATR
        tr, atr = self.atr_state.update(high, low, close, period)
        self.atr = atr
        
        # Calculate basic bands
        hl_avg = (high + low) / 2.0
        self.basic_upper = hl_avg + multiplier * atr
        self.basic_lower = hl_avg - multiplier * atr
        
        # Calculate final bands
        if self.final_upper == 0.0:
            self.final_upper = self.basic_upper
        else:
            self.final_upper = self.basic_upper if self.basic_upper < self.final_upper or close > self.final_upper else self.final_upper
        
        if self.final_lower == 0.0:
            self.final_lower = self.basic_lower
        else:
            self.final_lower = self.basic_lower if self.basic_lower > self.final_lower or close < self.final_lower else self.final_lower
        
        # Determine direction
        if self.supertrend == 0.0:
            self.direction = 1
            self.supertrend = self.final_lower
        else:
            if self.direction == 1:
                if close <= self.final_lower:
                    self.direction = -1
                    self.supertrend = self.final_upper
                else:
                    self.supertrend = self.final_lower
            else:
                if close >= self.final_upper:
                    self.direction = 1
                    self.supertrend = self.final_lower
                else:
                    self.supertrend = self.final_upper
        
        return self.supertrend, self.direction

# ------------------------------------- 19
@dataclass
class AroonState:
    """State for Aroon indicator (periods since high/low)."""
    periods_since_high: int = 0
    periods_since_low: int = 0
    aroon_up: float = 0.0
    aroon_down: float = 0.0
    aroon_oscillator: float = 0.0
    high_buffer: list = field(default_factory=list)
    low_buffer: list = field(default_factory=list)

    def update(self, high: float, low: float, period: int = 25) -> tuple:
        """Update Aroon state incrementally."""
        self.high_buffer.append(high)
        self.low_buffer.append(low)
        
        if len(self.high_buffer) > period:
            self.high_buffer.pop(0)
            self.low_buffer.pop(0)
        
        if len(self.high_buffer) < period:
            return 0.0, 0.0, 0.0
        
        # Find periods since highest high and lowest low
        max_high = max(self.high_buffer)
        min_low = min(self.low_buffer)
        
        self.periods_since_high = period - 1 - self.high_buffer[::-1].index(max_high)
        self.periods_since_low = period - 1 - self.low_buffer[::-1].index(min_low)
        
        # Calculate Aroon values
        self.aroon_up = ((period - self.periods_since_high) / period) * 100.0
        self.aroon_down = ((period - self.periods_since_low) / period) * 100.0
        self.aroon_oscillator = self.aroon_up - self.aroon_down
        
        return self.aroon_up, self.aroon_down, self.aroon_oscillator

# ------------------------------------- 20
@dataclass
class DEMAState:
    """State for Double Exponential Moving Average."""
    ema1: float = 0.0
    ema2: float = 0.0
    dema: float = 0.0
    ema1_state: Optional['EMAState'] = None
    ema2_state: Optional['EMAState'] = None

    def update(self, value: float, period: int = 20) -> float:
        """Update DEMA state incrementally."""
        if self.ema1_state is None:
            self.ema1_state = EMAState()
            self.ema2_state = EMAState()
        
        # First EMA
        self.ema1 = self.ema1_state.update(value, period)
        
        # Second EMA (EMA of EMA)
        self.ema2 = self.ema2_state.update(self.ema1, period)
        
        # DEMA = 2*EMA1 - EMA2
        self.dema = 2.0 * self.ema1 - self.ema2
        
        return self.dema

# ------------------------------------- 21
@dataclass
class TEMAState:
    """State for Triple Exponential Moving Average."""
    ema1: float = 0.0
    ema2: float = 0.0
    ema3: float = 0.0
    tema: float = 0.0
    ema1_state: Optional['EMAState'] = None
    ema2_state: Optional['EMAState'] = None
    ema3_state: Optional['EMAState'] = None

    def update(self, value: float, period: int = 20) -> float:
        """Update TEMA state incrementally."""
        if self.ema1_state is None:
            self.ema1_state = EMAState()
            self.ema2_state = EMAState()
            self.ema3_state = EMAState()
        
        # First EMA
        self.ema1 = self.ema1_state.update(value, period)
        
        # Second EMA
        self.ema2 = self.ema2_state.update(self.ema1, period)
        
        # Third EMA
        self.ema3 = self.ema3_state.update(self.ema2, period)
        
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        self.tema = 3.0 * self.ema1 - 3.0 * self.ema2 + self.ema3
        
        return self.tema

# ------------------------------------- 22
@dataclass
class HMAState:
    """State for Hull Moving Average."""
    wma_half: float = 0.0
    wma_full: float = 0.0
    raw_hma: float = 0.0
    hma: float = 0.0
    wma_half_state: Optional['WMAState'] = None
    wma_full_state: Optional['WMAState'] = None
    wma_sqrt_state: Optional['WMAState'] = None

    def update(self, value: float, period: int = 20) -> float:
        """Update HMA state incrementally."""
        import math
        
        if self.wma_half_state is None:
            half_period = period // 2
            sqrt_period = int(math.sqrt(period))
            self.wma_half_state = WMAState()
            self.wma_full_state = WMAState()
            self.wma_sqrt_state = WMAState()
        
        half_period = period // 2
        sqrt_period = int(math.sqrt(period))
        
        # WMA with half period
        self.wma_half = self.wma_half_state.update(value, half_period)
        
        # WMA with full period
        self.wma_full = self.wma_full_state.update(value, period)
        
        # Raw HMA = 2*WMA(n/2) - WMA(n)
        self.raw_hma = 2.0 * self.wma_half - self.wma_full
        
        # Final HMA = WMA(sqrt(n)) of raw_hma
        self.hma = self.wma_sqrt_state.update(self.raw_hma, sqrt_period)
        
        return self.hma

# ------------------------------------- 23
@dataclass
class ZigZagState:
    """State for ZigZag indicator (swing detection)."""
    last_pivot_high: float = 0.0
    last_pivot_low: float = 0.0
    last_pivot_high_idx: int = -1
    last_pivot_low_idx: int = -1
    current_trend: int = 0  # 1 = up, -1 = down, 0 = undefined
    current_extreme: float = 0.0
    current_extreme_idx: int = -1
    zigzag_value: Optional[float] = None
    current_idx: int = 0

    def update(self, high: float, low: float, close: float, deviation_pct: float = 5.0) -> Optional[float]:
        """Update ZigZag state incrementally."""
        self.zigzag_value = None
        
        if self.current_trend == 0:
            # Initialize
            self.current_extreme = high
            self.current_extreme_idx = self.current_idx
            self.current_trend = 1
        
        elif self.current_trend == 1:
            # Uptrend
            if high > self.current_extreme:
                self.current_extreme = high
                self.current_extreme_idx = self.current_idx
            
            # Check for reversal
            deviation = ((self.current_extreme - low) / self.current_extreme) * 100.0
            if deviation >= deviation_pct:
                # Confirm pivot high
                self.last_pivot_high = self.current_extreme
                self.last_pivot_high_idx = self.current_extreme_idx
                self.zigzag_value = self.last_pivot_high
                
                # Switch to downtrend
                self.current_trend = -1
                self.current_extreme = low
                self.current_extreme_idx = self.current_idx
        
        else:
            # Downtrend
            if low < self.current_extreme:
                self.current_extreme = low
                self.current_extreme_idx = self.current_idx
            
            # Check for reversal
            deviation = ((high - self.current_extreme) / self.current_extreme) * 100.0
            if deviation >= deviation_pct:
                # Confirm pivot low
                self.last_pivot_low = self.current_extreme
                self.last_pivot_low_idx = self.current_extreme_idx
                self.zigzag_value = self.last_pivot_low
                
                # Switch to uptrend
                self.current_trend = 1
                self.current_extreme = high
                self.current_extreme_idx = self.current_idx
        
        self.current_idx += 1
        return self.zigzag_value

# ============================================================================
