""" indicators_B_batch_NoComment.py """
import numpy as np
import pandas as pd
from numba import njit
from .indicators_B_maths import (
    SMAState       , WMAState       , EMAState       , ROCState,
    RSIState       , TrueRangeState , ATRState       , MACDState,
    BollingerState , KeltnerState   , StochasticState, CCIState,
    MFIState       , OBVState       , WilliamsRState , ParabolicSARState,
    HeikinAshiState, SupertrendState, AroonState     , DEMAState,
    TEMAState      , KAMAState      , HMAState)
# ============================================================================
# 1. SMA - Simple Moving Average
@njit
def sma_batch(data, period, min_periods=-1):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    state = SMAState(period, min_periods)
    for i in range(n):
        result[i] = state.update(data[i])
    return result

def sma_batch_df(df, column,
                 period, min_periods=-1,
                 result_col='sma', add_para_to_names = False):
    data = df[column].values
    result = sma_batch(data, period, min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
         return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)       
# ============================================================================
# 2. WMA - Weighted Moving Average
@njit
def wma_batch(data, period, min_periods=-1):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    state = WMAState(period, min_periods)
    for i in range(n):
        result[i] = state.update(data[i])
    return result

def wma_batch_df(df, column,
                 period, min_periods=-1,
                 result_col='wma', add_para_to_names = False):
    data = df[column].values
    result = wma_batch(data, period, min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 3. EMA - Exponential Moving Average
@njit
def ema_batch(data, period, min_periods=-1):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    state = EMAState(period, min_periods)
    for i in range(n):
        result[i] = state.update(data[i])
    return result

def ema_batch_df(df, column,
                 period, min_periods=-1,
                 result_col='ema', add_para_to_names = False):
    data = df[column].values
    result = ema_batch(data, period, min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 4. ROC - Rate of Change
@njit
def roc_batch(data, period):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    state = ROCState(period)
    for i in range(n):
        result[i] = state.update(data[i])
    return result

def roc_batch_df(df, column,
                 period,
                 result_col='roc', add_para_to_names = False):
    data = df[column].values
    result = roc_batch(data, period)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 5. RSI - Relative Strength Index ("ema=0", wilder=1)
@njit
def rsi_batch(data, period, method=0):
    n = len(data)
    result = np.empty(n, dtype=np.float64)
    state = RSIState(period, method)
    for i in range(n):
        result[i] = state.update(data[i])
    return result

def rsi_batch_df(df, column='close',
                 period=14, method='ema',
                 result_col='rsi', add_para_to_names = False):
    # تبدیل method از string به int
    method_int = 1 if method.lower() == 'wilders' else 0
    data = df[column].values
    result = rsi_batch(data, period, method_int)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{period}_{method}': result
        }, index=df.index)
    else:
         return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 6. True Range
@njit
def truerange_batch(high, low, close):
    n = len(high)
    result = np.empty(n, dtype=np.float64)
    state = TrueRangeState()
    for i in range(n):
        result[i] = state.update(high[i], low[i], close[i])
    return result

def truerange_batch_df(df, high_col='high', low_col='low', close_col='close',
                       result_col='true_range', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    result = truerange_batch(high, low, close)
    return pd.DataFrame({
        f'{result_col}': result
    }, index=df.index)
# ============================================================================
# 7. ATR - Average True Range (classic=0, "wilder=1", ema=2)
@njit
def atr_batch(high, low, close, n=14, method=1, min_periods=-1):
    length = len(high)
    result = np.empty(length, dtype=np.float64)
    state = ATRState(n, method, min_periods)
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i])
    return result

def atr_batch_df(df, high_col='high', low_col='low', close_col='close', 
                 n=14, method='wilder', min_periods=-1, 
                 result_col='atr', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    # تبدیل method از string به int
    method_map = {'classic': 0, 'wilder': 1, 'ema': 2}
    method_int = method_map.get(str(method).lower(), 1)  # default: 'wilder'
    if method_int is None:
        raise ValueError(f"Invalid method: {method}. Choose from {list(method_map.keys())}")
    result = atr_batch(high, low, close, n=n, method=method_int, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}_{method}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 8. MACD - Moving Average Convergence Divergence
@njit
def macd_batch(close, fast=12, slow=26, signal=9):
    length = len(close)
    macd_line = np.empty(length, dtype=np.float64)
    signal_line = np.empty(length, dtype=np.float64)
    histogram = np.empty(length, dtype=np.float64)
    state = MACDState(fast, slow, signal)
    for i in range(length):
        macd, sig, hist = state.update(close[i])
        macd_line[i] = macd
        signal_line[i] = sig
        histogram[i] = hist
    return macd_line, signal_line, histogram

def macd_batch_df(df, close_col='close',
                  fast=12, slow=26, signal=9, 
                  macd_col='macd', signal_col='macd_signal',
                  hist_col='macd_hist', add_para_to_names = False):
    close = df[close_col].values
    macd_line, signal_line, histogram = macd_batch(close, fast=fast, slow=slow, signal=signal)
    if add_para_to_names:
        return pd.DataFrame({
            f'{macd_col}_{fast}_{slow}_{signal}': macd_line,
            f'{signal_col}_{fast}_{slow}_{signal}': signal_line,
            f'{hist_col}_{fast}_{slow}_{signal}': histogram
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{macd_col}': macd_line,
            f'{signal_col}': signal_line,
            f'{hist_col}': histogram
        }, index=df.index)
# ============================================================================
# 9. Bollinger Bands
@njit
def bollinger_batch(close, n=20, k=2.0, min_periods=-1):
    length = len(close)
    upper = np.empty(length, dtype=np.float64)
    middle = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    width = np.empty(length, dtype=np.float64)
    percent = np.empty(length, dtype=np.float64)
    state = BollingerState(n, k, min_periods)
    for i in range(length):
        u, m, l, w, p = state.update(close[i])
        upper[i] = u
        middle[i] = m
        lower[i] = l
        width[i] = w
        percent[i] = p
    return upper, middle, lower, width, percent

def bollinger_batch_df(df, close_col='close',
                       n=20, k=2.0, min_periods=-1,
                       upper_col='bb_upper', middle_col='bb_middle', 
                       lower_col='bb_lower', width_col='bb_width', 
                       percent_col='bb_percent', add_para_to_names = False):
    close = df[close_col].values
    upper, middle, lower, width, percent = bollinger_batch(
        close, n=n, k=k, min_periods=min_periods
    )
    if add_para_to_names:
        return pd.DataFrame({
            f'{upper_col}_{n}_{k}': upper,
            f'{middle_col}_{n}_{k}': middle,
            f'{lower_col}_{n}_{k}': lower,
            f'{width_col}_{n}_{k}': width,
            f'{percent_col}_{n}_{k}': percent
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{upper_col}': upper,
            f'{middle_col}': middle,
            f'{lower_col}': lower,
            f'{width_col}': width,
            f'{percent_col}': percent
        }, index=df.index)
# ============================================================================
# 10. Keltner Channel
@njit
def keltner_batch(high, low, close, n=20, m=2.0, min_periods=-1):
    length = len(close)
    upper = np.empty(length, dtype=np.float64)
    middle = np.empty(length, dtype=np.float64)
    lower = np.empty(length, dtype=np.float64)
    width = np.empty(length, dtype=np.float64)
    percent = np.empty(length, dtype=np.float64)
    state = KeltnerState(n, m, min_periods)
    for i in range(length):
        u, mid, l, w, p = state.update(high[i], low[i], close[i])
        upper[i] = u
        middle[i] = mid
        lower[i] = l
        width[i] = w
        percent[i] = p
    return upper, middle, lower, width, percent

def keltner_batch_df(df, high_col='high', low_col='low', close_col='close',
                     n=20, m=2.0, min_periods=-1,
                     upper_col='kc_upper', middle_col='kc_middle',
                     lower_col='kc_lower', width_col='kc_width',
                     percent_col='kc_percent', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    upper, middle, lower, width, percent = keltner_batch(
        high, low, close, n=n, m=m, min_periods=min_periods
    )
    if add_para_to_names:
        return pd.DataFrame({
            f'{upper_col}_{n}_{m}': upper,
            f'{middle_col}_{n}_{m}': middle,
            f'{lower_col}_{n}_{m}': lower,
            f'{width_col}_{n}_{m}': width,
            f'{percent_col}_{n}_{m}': percent
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{upper_col}': upper,
            f'{middle_col}': middle,
            f'{lower_col}': lower,
            f'{width_col}': width,
            f'{percent_col}': percent
        }, index=df.index)
# ============================================================================
# 11. Stochastic Oscillator ("sma=0", ema=1)
@njit
def stochastic_batch(high, low, close, k_period=14, d_period=3, smooth_k=3, method=0, min_periods=-1):
    length = len(close)
    k = np.empty(length, dtype=np.float64)
    d = np.empty(length, dtype=np.float64)
    state = StochasticState(k_period, d_period, smooth_k, method, min_periods)
    for i in range(length):
        k_val, d_val = state.update(high[i], low[i], close[i])
        k[i] = k_val
        d[i] = d_val
    return k, d

def stochastic_batch_df(df, high_col='high', low_col='low', close_col='close',
                       k_period=14, d_period=3, smooth_k=3, method='sma', min_periods=-1,
                       k_col='stoch_k', d_col='stoch_d', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    method_map = {'sma': 0, 'ema': 1}
    method_int = method_map.get(str(method).lower(), 0)  # default: 'sma'
    if method_int is None:
        raise ValueError(f"Invalid method: {method}. Choose from {list(method_map.keys())}")
    k, d = stochastic_batch(high, low, close, k_period=k_period, 
                           d_period=d_period, smooth_k=smooth_k, 
                           method=method_int, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{k_col}_{k_period}_{d_period}_{smooth_k}_{method}': k,
            f'{d_col}_{k_period}_{d_period}_{smooth_k}_{method}': d
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{k_col}': k,
            f'{d_col}': d
        }, index=df.index)
# ============================================================================
# 12. CCI (Commodity Channel Index)
@njit
def cci_batch(high, low, close, n=20, min_periods=-1):
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    state = CCIState(n, min_periods)
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i])
    return result

def cci_batch_df(df, high_col='high', low_col='low', close_col='close',
                 n=20, min_periods=-1,
                 result_col='cci', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    result = cci_batch(high, low, close, n=n, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 13. MFI (Money Flow Index)
@njit
def mfi_batch(high, low, close, volume, n=14, min_periods=-1):
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    state = MFIState(n, min_periods)
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i], volume[i])
    return result

def mfi_batch_df(df, high_col='high', low_col='low', close_col='close', volume_col='volume',
                 n=14, min_periods=-1,
                 result_col='mfi', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    volume = df[volume_col].values
    result = mfi_batch(high, low, close, volume, n=n, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}': result
        }, index=df.index)
    else:
         return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 14. OBV (On-Balance Volume)
@njit
def obv_batch(close, volume):
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    state = OBVState()
    for i in range(length):
        result[i] = state.update(close[i], volume[i])
    return result

def obv_batch_df(df, close_col='close', volume_col='volume',
                 result_col='obv', add_para_to_names = False):
    close = df[close_col].values
    volume = df[volume_col].values
    result = obv_batch(close, volume)
    return pd.DataFrame({
        f'{result_col}': result
    }, index=df.index)
# ============================================================================
# 15. Williams %R
@njit
def williamsr_batch(high, low, close, n=14, min_periods=-1):
    length = len(close)
    result = np.empty(length, dtype=np.float64)
    state = WilliamsRState(n, min_periods)
    for i in range(length):
        result[i] = state.update(high[i], low[i], close[i])
    return result

def williamsr_batch_df(df, high_col='high', low_col='low', close_col='close',
                       n=14, min_periods=-1,
                       result_col='williamsr', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    result = williamsr_batch(high, low, close, n=n, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result,
        }, index=df.index)
# ============================================================================
# 16. Parabolic SAR
@njit
def psar_batch(high, low, af_start=0.02, af_step=0.02, af_max=0.2):
    length = len(high)
    result = np.empty(length, dtype=np.float64)
    state = ParabolicSARState(af_start, af_step, af_max)
    for i in range(length):
        result[i] = state.update(high[i], low[i])
    return result

def psar_batch_df(df, high_col='high', low_col='low',
                  af_start=0.02, af_step=0.02, af_max=0.2,
                  result_col='psar', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    result = psar_batch(high, low, af_start=af_start, af_step=af_step, af_max=af_max)
    if add_para_to_names:    
        return pd.DataFrame({
            f'{result_col}_{af_start}_{af_step}_{af_max}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 17. Heikin-Ashi
@njit
def heikinashi_batch(open_, high, low, close):
    length = len(open_)
    ha_open = np.empty(length, dtype=np.float64)
    ha_high = np.empty(length, dtype=np.float64)
    ha_low = np.empty(length, dtype=np.float64)
    ha_close = np.empty(length, dtype=np.float64)
    state = HeikinAshiState()
    for i in range(length):
        o, h, l, c = state.update(open_[i], high[i], low[i], close[i])
        ha_open[i] = o
        ha_high[i] = h
        ha_low[i] = l
        ha_close[i] = c
    return ha_open, ha_high, ha_low, ha_close

def heikinashi_batch_df(df, open_col='open', high_col='high', low_col='low', close_col='close',
                        result_prefix='ha', add_para_to_names = False):
    open_arr = df[open_col].values
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    ha_o, ha_h, ha_l, ha_c = heikinashi_batch(open_arr, high, low, close)
    return pd.DataFrame({
        f'{result_prefix}_open': ha_o,
        f'{result_prefix}_high': ha_h,
        f'{result_prefix}_low': ha_l,
        f'{result_prefix}_close': ha_c
    }, index=df.index)
# ============================================================================
# 18. Supertrend (classic=0, "wilder=1", ema=2)
@njit
def supertrend_batch(high, low, close, period=10, multiplier=3.0, 
                     atr_method=1, min_periods=-1):
    length = len(high)
    supertrend = np.empty(length, dtype=np.float64)
    direction = np.empty(length, dtype=np.int32)
    state = SupertrendState(period, multiplier, atr_method, min_periods)
    for i in range(length):
        st_val, dir_val = state.update(high[i], low[i], close[i])
        supertrend[i] = st_val
        direction[i] = dir_val
    return supertrend, direction

def supertrend_batch_df(df, high_col='high', low_col='low', close_col='close',
                        period=10, multiplier=3.0, atr_method='wilder', min_periods=-1,
                        super_col='supertrend', direction_col='st_direction', add_para_to_names = False):
    # تبدیل method از string به int
    if isinstance(atr_method, str):
        method_map = {'classic': 0, 'wilder': 1, 'ema': 2}
        atr_method_int = method_map.get(str(atr_method).lower(), 1)  # default: 'wilder'
    high = df[high_col].values
    low = df[low_col].values
    close = df[close_col].values
    st, direction = supertrend_batch(high, low, close, period=period, 
                                     multiplier=multiplier, atr_method=atr_method_int,
                                     min_periods=min_periods) 
    if add_para_to_names:
        return pd.DataFrame({
            f'{super_col}_{period}_{multiplier}_{atr_method}': st,
            f'{direction_col}_{period}_{multiplier}_{atr_method}': direction
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{super_col}': st,
            f'{direction_col}': direction
        }, index=df.index)
# ============================================================================
# 19. Aroon
@njit
def aroon_batch(high, low, period=25, min_periods=-1):
    length = len(high)
    aroon_up = np.empty(length, dtype=np.float64)
    aroon_down = np.empty(length, dtype=np.float64)
    oscillator = np.empty(length, dtype=np.float64)
    state = AroonState(period, min_periods)
    for i in range(length):
        up, down, osc = state.update(high[i], low[i])
        aroon_up[i] = up
        aroon_down[i] = down
        oscillator[i] = osc
    return aroon_up, aroon_down, oscillator

def aroon_batch_df(df, high_col='high', low_col='low',
                   period=25, min_periods=-1,
                   aroon_up_col='aroon_up', aroon_down_col='aroon_down',
                   aroon_osc_col='aroon_oscillator', add_para_to_names = False):
    high = df[high_col].values
    low = df[low_col].values
    aroon_up, aroon_down, oscillator = aroon_batch(high, low, 
                                                    period=period, 
                                                    min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{aroon_up_col}_{period}': aroon_up,
            f'{aroon_down_col}_{period}': aroon_down,
            f'{aroon_osc_col}_{period}': oscillator
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{aroon_up_col}': aroon_up,
            f'{aroon_down_col}': aroon_down,
            f'{aroon_osc_col}': oscillator
        }, index=df.index)
# ============================================================================
# 20. DEMA (Double Exponential Moving Average)
@njit
def dema_batch(data, n=20, min_periods=-1):
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    state = DEMAState(n, min_periods)
    for i in range(length):
        result[i] = state.update(data[i])
    return result

def dema_batch_df(df, price_col='close',
                  n=20, min_periods=-1,
                  result_col='dema', add_para_to_names = False):
    data = df[price_col].values
    result = dema_batch(data, n=n, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 21. TEMA (Triple Exponential Moving Average)
@njit
def tema_batch(data, n=20, min_periods=-1):
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    state = TEMAState(n, min_periods)
    for i in range(length):
        result[i] = state.update(data[i])
    return result

def tema_batch_df(df, price_col='close',
                  n=20, min_periods=-1,
                  result_col='tema', add_para_to_names = False):
    data = df[price_col].values
    result = tema_batch(data, n=n, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 22. KAMA (Kaufman’s Adaptive Moving Average)
@njit
def kama_batch(data, n=10, fast_span=2, slow_span=30, min_periods=-1):
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    state = KAMAState(n, fast_span, slow_span, min_periods)
    for i in range(length):
        result[i] = state.update(data[i])
    return result

def kama_batch_df(df, price_col='close',
                  n=10, fast_span=2, slow_span=30, min_periods=-1,
                  result_col='kama', add_para_to_names = False):
    data = df[price_col].values
    result = kama_batch(data, n=n, fast_span=fast_span, slow_span=slow_span, 
                       min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}_{fast_span}_{slow_span}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# 23. HMA (Hull Moving Average)
@njit
def hma_batch(data, n=20, min_periods=-1):
    length = len(data)
    result = np.empty(length, dtype=np.float64)
    state = HMAState(n, min_periods)
    for i in range(length):
        result[i] = state.update(data[i])
    return result

def hma_batch_df(df, price_col='close',
                 n=20, min_periods=-1,
                 result_col='hma', add_para_to_names = False):
    result = hma_batch(df[price_col].values, n=n, min_periods=min_periods)
    if add_para_to_names:
        return pd.DataFrame({
            f'{result_col}_{n}': result
        }, index=df.index)
    else:
        return pd.DataFrame({
            f'{result_col}': result
        }, index=df.index)
# ============================================================================
# Registry-compatible exports
# ============================================================================
__all__ = [
    "sma_batch", "wma_batch", "ema_batch", "roc_batch", "rsi_batch", "true_range_batch",
    "atr_batch", "macd_batch", "bollinger_batch", "keltner_batch", "stochastic_batch",
    "cci_batch", "mfi_batch", "obv_batch", "williamsr_batch", "parabolic_sar_batch",
    "heikin_ashi_batch", "supertrend_batch", "aroon_batch", "dema_batch", "tema_batch",
    "kama_batch", "hma_batch"]
# ============================================================================
# Registry mapping for easy access
# ============================================================================
INDICATOR_REGISTRY = {
    "sma": sma_batch,
    "wma": wma_batch,
    "ema": ema_batch,
    "roc": roc_batch,
    "rsi": rsi_batch,
    "true_range": truerange_batch,
    "atr": atr_batch,
    "macd": macd_batch,
    "bollinger": bollinger_batch,
    "keltner": keltner_batch,
    "stochastic": stochastic_batch,
    "cci": cci_batch,
    "mfi": mfi_batch,
    "obv": obv_batch,
    "williamsr": williamsr_batch,
    "parabolic_sar": psar_batch,
    "heikin_ashi": heikinashi_batch,
    "supertrend": supertrend_batch,
    "aroon": aroon_batch,
    "dema": dema_batch,
    "tema": tema_batch,
    "hma": hma_batch,
}