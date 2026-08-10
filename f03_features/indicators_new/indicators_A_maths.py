
# این فایل همان فایل core.py است. منتها برای حالت live

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple

EPS = 1e-12

# ------------------------------------- 0
def _safe_div(a, b):
    return 0.0 if abs(b) < EPS else a / b

"""
class:
SMA, WMA, EMA, ROC, RSI, TR, ATR, MACD, Bollinger, Keltner, Stochastic,
CCI, MFI, OBV, WilliamsR, ParabolicSARS, HeikinAshi

dataclass:
Supertrend, Aroon, DEMA, TEMA, HMA,
ZigZag
"""
# ============================================================================
# Indicator States (Classic Technical Indicators) PART-1
# ============================================================================
# ------------------------------------- 1-new
class SMAState:
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = n if min_periods is None else max(1, min(min_periods, n))
        self.buf = deque(maxlen=n)
        self.sum = 0.0

    def update(self, x):
        x = float(x)
        if len(self.buf) == self.n:
            self.sum -= self.buf[0]
        self.buf.append(x)
        self.sum += x
        
        current_len = len(self.buf)
        if current_len < self.min_periods:
            return np.nan
        return self.sum / current_len

    def reset(self):
        self.buf.clear()
        self.sum = 0.0

# ------------------------------------- 2-new
class WMAState:
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = n if min_periods is None else max(1, min(min_periods, n))
        self.buf = deque(maxlen=n)
        self.weights_full = np.arange(1, n + 1, dtype=np.float64)
        self.weights_full /= self.weights_full.sum()

    def update(self, x):
        self.buf.append(float(x))
        current_len = len(self.buf)
        
        if current_len < self.min_periods:
            return np.nan
        
        # وزن‌های متناسب با طول فعلی buffer
        weights = self.weights_full[-current_len:]
        weights = weights / weights.sum()
        
        return float(np.dot(np.asarray(self.buf), weights))

    def reset(self):
        self.buf.clear()

# ------------------------------------- 3-new
class EMAState:
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = n if min_periods is None else max(1, min(min_periods, n))
        self.alpha = 2.0 / (n + 1.0)
        self.value = None
        self.count = 0

    def update(self, x):
        x = float(x)
        self.count += 1
        
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        
        if self.count < self.min_periods:
            return np.nan
        return self.value

    def reset(self):
        self.value = None
        self.count = 0

# ------------------------------------- 4-ok
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

    def reset(self):
        self.buf.clear()

# ------------------------------------- 5-new
class RSIState:
    def __init__(self, n, method: Literal["ema", "wilders"] = "ema"):
        self.n = n
        self.method = method
        self.prev = None
        self.gains = deque(maxlen=n)
        self.losses = deque(maxlen=n)
        self.avg_gain = None
        self.avg_loss = None

        # محاسبه alpha بر اساس method
        if method == "wilders":
            self.alpha = 1.0 / n
        else:  # ema
            self.alpha = 2.0 / (n + 1)

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
            self.avg_gain = (1 - self.alpha) * self.avg_gain + self.alpha * gain
            self.avg_loss = (1 - self.alpha) * self.avg_loss + self.alpha * loss

        self.prev = close
        rs = _safe_div(self.avg_gain, self.avg_loss)
        return 100.0 - (100.0 / (1.0 + rs))

    def reset(self):
        self.prev = None
        self.gains.clear()
        self.losses.clear()
        self.avg_gain = None
        self.avg_loss = None

# ------------------------------------- 6-ok
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

    def reset(self):
        self.prev_close = None

# ------------------------------------- 7-new
class ATRState:
    def __init__(self, n, method: Literal["classic", "wilder", "ema"] = "wilder", min_periods: Optional[int] = None):
        self.n = n
        self.method = method
        self.min_periods = min_periods if min_periods is not None else n
        self.tr = TrueRangeState()
        self.values = deque(maxlen=n)
        self.atr = None

        # محاسبه alpha بر اساس method
        if method == "wilder":
            self.alpha = 1.0 / n
        elif method == "ema":
            self.alpha = 2.0 / (n + 1)
        else:  # classic
            self.alpha = None  # برای classic از میانگین ساده استفاده می‌شود

    def update(self, h, l, c):
        tr = self.tr.update(h, l, c)
        self.values.append(tr)

        if len(self.values) < self.min_periods:
            return np.nan

        if self.method == "classic":
            # میانگین ساده همیشه
            return np.mean(self.values)
        else:
            # wilder یا ema
            if self.atr is None:
                self.atr = np.mean(self.values)
            else:
                self.atr = (1 - self.alpha) * self.atr + self.alpha * tr
            return self.atr

    def reset(self):
        self.tr.reset()
        self.values.clear()
        self.atr = None

# ------------------------------------- 8-ok
class MACDState:
    def __init__(self, fast=12, slow=26, signal=9):
        self.fast = EMAState(fast)
        self.slow = EMAState(slow)
        self.sig = EMAState(signal)

    def update(self, close):
        f = self.fast.update(close)
        s = self.slow.update(close)
        macd = f - s
        sig = self.sig.update(macd)
        hist = macd - sig
        return macd, sig, hist

    def reset(self):
        self.fast.reset()
        self.slow.reset()
        self.sig.reset()

# ------------------------------------- 9-ok
class BollingerState:
    def __init__(self, n, k, min_periods: Optional[int] = None):
        self.n = n
        self.k = k
        self.min_periods = min_periods if min_periods is not None else n
        self.buf = deque(maxlen=n)

    def update(self, close):
        self.buf.append(float(close))
        if len(self.buf) < self.min_periods:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        arr = np.asarray(self.buf)
        mid = arr.mean()
        std = arr.std()

        upper = mid + self.k * std
        lower = mid - self.k * std
                
        w = upper - lower
        p = (close - lower) / w if w != 0 else np.nan

        return upper, mid, lower, w, p

    def reset(self):
        self.buf.clear()

# ------------------------------------- 10-ok
class KeltnerState:
    def __init__(self, n, m, min_periods: Optional[int] = None):
        self.ema = EMAState(n, min_periods=min_periods)
        self.atr = ATRState(n, min_periods=min_periods)
        self.m = m

    def update(self, h, l, c):
        mid = self.ema.update(c)
        atr = self.atr.update(h, l, c)
        if np.isnan(atr):
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        upper = mid + self.m * atr
        lower = mid - self.m * atr

        w = upper - lower
        p = (c - lower) / w if w != 0 else np.nan

        return upper, mid, lower, w, p

    def reset(self):
        self.ema.reset()
        self.atr.reset()

# ------------------------------------- 11-ok-avdanced
class StochasticState:
    def __init__(
        self, 
        k_period: int = 14, 
        d_period: int = 3, 
        smooth_k: int = 3,
        method: Literal["sma", "ema"] = "sma",
        min_periods: Optional[int] = None
    ):
        """
        Stochastic Oscillator State
        
        Parameters:
        -----------
        k_period : int
            دوره محاسبه %K (معمولاً 14)
        d_period : int
            دوره هموارسازی برای %D (معمولاً 3)
        smooth_k : int
            دوره هموارسازی %K قبل از محاسبه %D (معمولاً 3)
            اگر 1 باشد، Fast Stochastic تولید می‌شود
        method : {"sma", "ema"}
            روش هموارسازی برای %K و %D
        min_periods : int, optional
            حداقل داده برای شروع محاسبه (پیش‌فرض: k_period)
        """
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.method = method
        self.min_periods = min_periods if min_periods is not None else k_period
        
        # بافرهای high/low/close
        self.highs = deque(maxlen=k_period)
        self.lows = deque(maxlen=k_period)
        
        # هموارساز %K (Fast K -> Slow K)
        if smooth_k > 1:
            if method == "sma":
                self.k_smoother = SMAState(period=smooth_k, min_periods=1)
            else:
                self.k_smoother = EMAState(period=smooth_k, min_periods=1)
        else:
            self.k_smoother = None
        
        # هموارساز %D
        if method == "sma":
            self.d_smoother = SMAState(period=d_period, min_periods=1)
        else:
            self.d_smoother = EMAState(period=d_period, min_periods=1)

    def update(self, h: float, l: float, c: float) -> Tuple[float, float]:
        """
        به‌روزرسانی با داده جدید
        
        Returns:
        --------
        k : float
            مقدار %K (هموارشده اگر smooth_k > 1)
        d : float
            مقدار %D (هموارسازی %K)
        """
        self.highs.append(float(h))
        self.lows.append(float(l))
        c = float(c)

        # چک min_periods
        if len(self.highs) < self.min_periods:
            return np.nan, np.nan

        # محاسبه %K خام
        hh = max(self.highs)
        ll = min(self.lows)
        raw_k = 100.0 * _safe_div(c - ll, hh - ll)
        
        # هموارسازی %K (اگر smooth_k > 1)
        if self.k_smoother is not None:
            k = self.k_smoother.update(raw_k)
        else:
            k = raw_k
        
        # محاسبه %D (هموارسازی %K)
        d = self.d_smoother.update(k)
        
        return k, d

    def reset(self):
        self.highs.clear()
        self.lows.clear()
        if self.k_smoother is not None:
            self.k_smoother.reset()
        self.d_smoother.reset()

# ------------------------------------- 12-ok-avdanced
class CCIState:
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = min_periods if min_periods is not None else n
        self.tp = deque(maxlen=n)

    def update(self, h, l, c):
        tp = (float(h) + float(l) + float(c)) / 3.0
        self.tp.append(tp)

        if len(self.tp) < self.min_periods:
            return np.nan

        arr = np.asarray(self.tp)
        ma = arr.mean()
        md = np.mean(np.abs(arr - ma))
        return (tp - ma) / (0.015 * md + EPS)
    
    def reset(self):
        self.tp.clear()

# ------------------------------------- 13-ok-avdanced
class MFIState:
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = min_periods if min_periods is not None else n
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

        if len(self.pos) < self.min_periods:
            return np.nan

        ratio = _safe_div(sum(self.pos), sum(self.neg))
        return 100.0 - (100.0 / (1.0 + ratio))

    def reset(self):
        self.prev_tp = None
        self.pos.clear()
        self.neg.clear()

# ------------------------------------- 14-ok
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

    def reset(self):
        self.prev = None
        self.value = 0.0

# ------------------------------------- 15-ok
class WilliamsRState:
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = min_periods if min_periods is not None else n
        self.highs = deque(maxlen=n)
        self.lows = deque(maxlen=n)

    def update(self, h, l, c):
        self.highs.append(float(h))
        self.lows.append(float(l))
        c = float(c)

        if len(self.highs) < self.min_periods:
            return np.nan

        hh = max(self.highs)
        ll = min(self.lows)
        return -100.0 * _safe_div(hh - c, hh - ll)

    def reset(self):
        self.highs.clear()
        self.lows.clear()

# ------------------------------------- 16-ok
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

    def reset(self):
        self.initialized = False

# ------------------------------------- 17-ok
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

    def reset(self):
        self.prev_open = None
        self.prev_close = None

# ============================================================================
# Additional Indicator States (Classic Technical Indicators) PART-2
# ============================================================================
# ------------------------------------- 18-recoded
class SupertrendState:
    """State for Supertrend indicator (ATR-based trend following)."""
    
    def __init__(self,
                 period: int = 10,
                 multiplier: float = 3.0, 
                 atr_method: Literal["classic", "wilder", "ema"] = "wilder",
                 min_periods: Optional[int] = None):
        """
        Args:
            period: ATR period
            multiplier: Band multiplier
            atr_method: ATR calculation method
            min_periods: Minimum periods before returning valid values (default: period)
        """
        self.period = period
        self.multiplier = multiplier
        self.min_periods = min_periods if min_periods is not None else period
        
        # ATR state
        self.atr_state = ATRState(n=period, method=atr_method)
        
        # Band state
        self.basic_upper = 0.0
        self.basic_lower = 0.0
        self.final_upper = 0.0
        self.final_lower = 0.0
        
        # Trend state
        self.direction = 1  # 1 = uptrend, -1 = downtrend
        self.supertrend = 0.0
        
        # Warm-up counter
        self.count = 0
    
    def update(self, high: float, low: float, close: float) -> tuple[float, int]:
        """
        Update Supertrend state incrementally.
        
        Returns:
            (supertrend, direction): Supertrend value and direction (1 or -1)
        """
        high, low, close = float(high), float(low), float(close)
        self.count += 1
        
        # Update ATR
        tr, atr = self.atr_state.update(high, low, close)
        
        # Warm-up period
        if self.count < self.min_periods or np.isnan(atr):
            return np.nan, self.direction
        
        # Calculate basic bands
        hl_avg = (high + low) / 2.0
        self.basic_upper = hl_avg + self.multiplier * atr
        self.basic_lower = hl_avg - self.multiplier * atr
        
        # Calculate final bands
        if self.final_upper == 0.0:
            self.final_upper = self.basic_upper
        else:
            self.final_upper = (self.basic_upper 
                               if self.basic_upper < self.final_upper or close > self.final_upper 
                               else self.final_upper)
        
        if self.final_lower == 0.0:
            self.final_lower = self.basic_lower
        else:
            self.final_lower = (self.basic_lower 
                               if self.basic_lower > self.final_lower or close < self.final_lower 
                               else self.final_lower)
        
        # Determine direction and supertrend
        if self.supertrend == 0.0:
            # First valid value
            self.direction = 1
            self.supertrend = self.final_lower
        else:
            if self.direction == 1:
                if close <= self.final_lower:
                    self.direction = -1
                    self.supertrend = self.final_upper
                else:
                    self.supertrend = self.final_lower
            else:  # direction == -1
                if close >= self.final_upper:
                    self.direction = 1
                    self.supertrend = self.final_lower
                else:
                    self.supertrend = self.final_upper
        
        return self.supertrend, self.direction
    
    def reset(self):
        """Reset state for new sequence."""
        self.atr_state.reset()
        self.basic_upper = 0.0
        self.basic_lower = 0.0
        self.final_upper = 0.0
        self.final_lower = 0.0
        self.direction = 1
        self.supertrend = 0.0
        self.count = 0

# ------------------------------------- 19-recoded
class AroonState:
    """
    Aroon Indicator State - Production-grade stateful implementation
    
    Aroon Up: ((period - bars_since_high) / period) * 100
    Aroon Down: ((period - bars_since_low) / period) * 100
    Aroon Oscillator: Aroon Up - Aroon Down
    
    Parameters:
    -----------
    period : int
        Lookback period (معمولاً 25)
    min_periods : int, optional
        حداقل تعداد داده برای شروع محاسبه (پیش‌فرض: period)
    
    Returns:
    --------
    tuple: (aroon_up, aroon_down, oscillator) یا (np.nan, np.nan, np.nan)
    """
    
    def __init__(self, period: int = 25, min_periods: int = None):
        if period < 1:
            raise ValueError("period must be >= 1")
        
        self.period = period
        self.min_periods = period if min_periods is None else max(1, min(min_periods, period))
        
        # Rolling buffers برای high و low
        self.highs = deque(maxlen=period)
        self.lows = deque(maxlen=period)
        
        # Track آخرین موقعیت high/low برای بهینه‌سازی
        self.bars_since_high = 0
        self.bars_since_low = 0
        self.current_high = None
        self.current_low = None
    
    def update(self, high: float, low: float) -> tuple[float, float, float]:
        """
        Update state با یک کندل جدید
        
        Parameters:
        -----------
        high : float
            قیمت بالای کندل
        low : float
            قیمت پایین کندل
        
        Returns:
        --------
        tuple: (aroon_up, aroon_down, oscillator)
        """
        high = float(high)
        low = float(low)
        
        # اضافه کردن به buffers
        self.highs.append(high)
        self.lows.append(low)
        
        current_len = len(self.highs)
        
        # بررسی min_periods
        if current_len < self.min_periods:
            return (np.nan, np.nan, np.nan)
        
        # پیدا کردن آخرین موقعیت highest high
        max_high = self.highs[0]
        bars_since_high = current_len - 1
        for i in range(current_len):
            if self.highs[i] >= max_high:
                max_high = self.highs[i]
                bars_since_high = current_len - 1 - i
        
        # پیدا کردن آخرین موقعیت lowest low
        min_low = self.lows[0]
        bars_since_low = current_len - 1
        for i in range(current_len):
            if self.lows[i] <= min_low:
                min_low = self.lows[i]
                bars_since_low = current_len - 1 - i
        
        # محاسبه Aroon Up و Aroon Down
        aroon_up = ((self.period - bars_since_high) / self.period) * 100.0
        aroon_down = ((self.period - bars_since_low) / self.period) * 100.0
        
        # محاسبه Oscillator
        oscillator = aroon_up - aroon_down
        
        return (aroon_up, aroon_down, oscillator)
    
    def reset(self):
        """Reset state به حالت اولیه"""
        self.highs.clear()
        self.lows.clear()
        self.bars_since_high = 0
        self.bars_since_low = 0
        self.current_high = None
        self.current_low = None

# ------------------------------------- 20-recoded
class DEMAState:
    """Double Exponential Moving Average - stateful incremental calculation."""
    
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = n if min_periods is None else max(1, min(min_periods, n))
        
        # Two nested EMAs
        self.ema1_state = EMAState(n=n, min_periods=1)  # First EMA
        self.ema2_state = EMAState(n=n, min_periods=1)  # EMA of EMA
        
        self.count = 0
    
    def update(self, x):
        """Update with new value and return DEMA."""
        x = float(x)
        self.count += 1
        
        # First EMA
        ema1 = self.ema1_state.update(x)
        
        # Second EMA (EMA of first EMA)
        ema2 = self.ema2_state.update(ema1)
        
        # DEMA = 2*EMA1 - EMA2
        dema = 2.0 * ema1 - ema2
        
        # Return NaN until min_periods reached
        if self.count < self.min_periods:
            return np.nan
        
        return dema
    
    def reset(self):
        """Reset state."""
        self.ema1_state = EMAState(n=self.n, min_periods=1)
        self.ema2_state = EMAState(n=self.n, min_periods=1)
        self.count = 0

# ------------------------------------- 21-recoded
class TEMAState:
    """Triple Exponential Moving Average - stateful incremental calculation."""
    
    def __init__(self, n, min_periods=None):
        self.n = n
        self.min_periods = n if min_periods is None else max(1, min(min_periods, n))
        
        # Three nested EMAs
        self.ema1_state = EMAState(n=n, min_periods=1)
        self.ema2_state = EMAState(n=n, min_periods=1)
        self.ema3_state = EMAState(n=n, min_periods=1)
        
        self.count = 0
    
    def update(self, x):
        """Update with new value and return TEMA."""
        x = float(x)
        self.count += 1
        
        # First EMA
        ema1 = self.ema1_state.update(x)
        
        # Second EMA (EMA of first EMA)
        ema2 = self.ema2_state.update(ema1)
        
        # Third EMA (EMA of second EMA)
        ema3 = self.ema3_state.update(ema2)
        
        # TEMA = 3*EMA1 - 3*EMA2 + EMA3
        tema = 3.0 * ema1 - 3.0 * ema2 + ema3
        
        if self.count < self.min_periods:
            return np.nan
        
        return tema
    
    def reset(self):
        """Reset state."""
        self.ema1_state = EMAState(n=self.n, min_periods=1)
        self.ema2_state = EMAState(n=self.n, min_periods=1)
        self.ema3_state = EMAState(n=self.n, min_periods=1)
        self.count = 0

# ------------------------------------- 22-recoded
class HMAState:
    """Hull Moving Average - stateful incremental calculation."""
    
    def __init__(self, n, min_periods=None):
        import math
        
        self.n = n
        self.min_periods = n if min_periods is None else max(1, min(min_periods, n))
        
        self.half_period = n // 2
        self.sqrt_period = int(math.sqrt(n))
        
        # Three WMA states
        self.wma_half_state = WMAState(n=self.half_period, min_periods=1)
        self.wma_full_state = WMAState(n=n, min_periods=1)
        self.wma_sqrt_state = WMAState(n=self.sqrt_period, min_periods=1)
        
        self.count = 0
    
    def update(self, x):
        """Update with new value and return HMA."""
        x = float(x)
        self.count += 1
        
        # WMA with half period
        wma_half = self.wma_half_state.update(x)
        
        # WMA with full period
        wma_full = self.wma_full_state.update(x)
        
        # Raw HMA = 2*WMA(n/2) - WMA(n)
        raw_hma = 2.0 * wma_half - wma_full
        
        # Final HMA = WMA(sqrt(n)) of raw_hma
        hma = self.wma_sqrt_state.update(raw_hma)
        
        if self.count < self.min_periods:
            return np.nan
        
        return hma
    
    def reset(self):
        """Reset state."""
        import math
        
        self.wma_half_state = WMAState(n=self.half_period, min_periods=1)
        self.wma_full_state = WMAState(n=self.n, min_periods=1)
        self.wma_sqrt_state = WMAState(n=self.sqrt_period, min_periods=1)
        self.count = 0

# ============================================================================
