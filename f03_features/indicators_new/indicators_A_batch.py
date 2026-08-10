"""
indicators_batch.py
===================
Adapter layer: wraps scalar-based indicator classes from indicators_class.py
into DataFrame-oriented functions compatible with feature_registry.py

Each function signature: fn(df: pd.DataFrame, **kwargs) -> pd.DataFrame | pd.Series
"""
import numpy as np
import pandas as pd
from typing import Literal, Optional
from indicators_A_class import (
    SMA, WMA, EMA, ROC, RSI, TrueRange, ATR, MACD, BollingerBands, KeltnerChannel,
    
    Stochastic, CCI, MFI, OBV, WilliamsR, ParabolicSAR,
    HeikinAshi, Supertrend, Aroon, DEMA, TEMA, HMA
)

# ============================================================================
# 1. SMA - Simple Moving Average
# ============================================================================

def sma_batch_py(
    df: pd.DataFrame,
    n: int = 20,
    source: str = "close",
    min_periods=None,
    **kwargs
) -> pd.Series:
    """
    Compute SMA using scalar-based SMA class.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    source : str
        Column name (default: 'close')
    min_periods : int, optional
        Minimum number of observations required
        
    Returns
    -------
    pd.Series
        SMA values
    """
    sma_obj = SMA(period=n, min_periods=min_periods)
    result = []
    
    for val in df[source]:
        result.append(sma_obj.update(val))
    
    return pd.Series(result, index=df.index, name=f"sma_{n}")

# -----
def sma_batch_pd(
    df: pd.DataFrame,
    n: int = 20,
    source: str = "close",
    min_periods=None,
    **kwargs
) -> pd.Series:
    """
    Compute SMA using vectorized pandas rolling.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    source : str
        Column name (default: 'close')
    min_periods : int, optional
        Minimum number of observations required
        
    Returns
    -------
    pd.Series
        SMA values
    """
    if min_periods is None:
        min_periods = n
    
    return df[source].rolling(window=n, min_periods=min_periods).mean()

# -----
def sma_batch(
    df: pd.DataFrame,
    n: int = 20,
    source: str = "close",
    min_periods=None,
    **kwargs
) -> pd.Series:
    """
    Compute SMA using NumPy for better performance.
    """
    values = df[source].values
    result = np.empty(len(values))
    result[:] = np.nan
    
    if min_periods is None:
        min_periods = n
    
    # محاسبه cumsum برای سرعت بالا
    cumsum = np.nancumsum(values)
    
    for i in range(len(values)):
        if i < min_periods - 1:
            continue
        elif i < n:
            result[i] = cumsum[i] / (i + 1)
        else:
            result[i] = (cumsum[i] - cumsum[i - n]) / n
    
    return pd.Series(result, index=df.index, name=f"sma_{n}")



# ============================================================================
# 2. WMA - Weighted Moving Average
# ============================================================================
def wma_batch(
    df: pd.DataFrame,
    n: int = 20,
    source: str = "close",
    min_periods=None,
    **kwargs
) -> pd.Series:
    """
    Compute WMA using scalar-based WMA class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    source : str
        Column name (default: 'close')
    min_periods : int, optional
        Minimum number of observations required
        
    Returns
    -------
    pd.Series
        WMA values
    """
    wma_obj = WMA(period=n, min_periods=min_periods)
    result = []
    
    for val in df[source]:
        result.append(wma_obj.update(val))
    
    return pd.Series(result, index=df.index, name=f"wma_{n}")


# ============================================================================
# 3. EMA - Exponential Moving Average
# ============================================================================
def ema_batch(
    df: pd.DataFrame,
    n: int = 20,
    source: str = "close",
    min_periods=None,
    **kwargs
) -> pd.Series:
    """
    Compute EMA using scalar-based EMA class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    source : str
        Column name (default: 'close')
    min_periods : int, optional
        Minimum number of observations required

    Returns
    -------
    pd.Series
        EMA values
    """
    ema_obj = EMA(period=n, min_periods=min_periods)
    result = []
    
    for val in df[source]:
        result.append(ema_obj.update(val))
    
    return pd.Series(result, index=df.index, name=f"ema_{n}")


# ============================================================================
# 4. ROC - Rate of Change
# ============================================================================
def roc_batch(
    df: pd.DataFrame,
    n: int = 10,
    source: str = "close",
    **kwargs
) -> pd.Series:
    """
    Compute ROC using scalar-based ROC class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    source : str
        Column name (default: 'close')
    
    Returns
    -------
    pd.Series
        ROC values (percentage)
    """
    roc_obj = ROC(period=n)
    result = []
    
    for val in df[source]:
        result.append(roc_obj.update(val))
    
    return pd.Series(result, index=df.index, name=f"roc_{n}")


# ============================================================================
# 5. RSI - Relative Strength Index
# ============================================================================
def rsi_batch(
    df: pd.DataFrame,
    n: int = 14,
    source: str = "close",
    method: Literal["ema", "wilders"] = "ema",
    **kwargs
) -> pd.Series:
    """
    Compute RSI using scalar-based RSI class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    source : str
        Column name (default: 'close')
    method : {'ema', 'wilders'}
        Smoothing method

    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    rsi_obj = RSI(period=n, method=method)
    result = []
    
    for val in df[source]:
        result.append(rsi_obj.update(val))
    
    return pd.Series(result, index=df.index, name=f"rsi_{n}")


# ============================================================================
# 6. True Range
# ============================================================================
def true_range_batch(
    df: pd.DataFrame,
    **kwargs
) -> pd.Series:
    """
    Compute True Range using scalar-based TrueRange class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (must have 'high', 'low', 'close')
    
    Returns
    -------
    pd.Series
        True Range values
    """
    tr_obj = TrueRange()
    result = []
    
    for _, row in df.iterrows():
        result.append(tr_obj.update(row['high'], row['low'], row['close']))
    
    return pd.Series(result, index=df.index, name="true_range")


# ============================================================================
# 7. ATR - Average True Range
# ============================================================================
def atr_batch(
    df: pd.DataFrame,
    n: int = 14,
    method: Literal["classic", "wilder", "ema"] = "wilder",
    **kwargs
) -> pd.Series:
    """
    Compute ATR using scalar-based ATR class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (must have 'high', 'low', 'close')
    n : int
        Period
    method : {'classic', 'wilder', 'ema'}
        Smoothing method

    Returns
    -------
    pd.Series
        ATR values
    """
    atr_obj = ATR(period=n, method=method)
    result = []
    
    for _, row in df.iterrows():
        result.append(atr_obj.update(row['high'], row['low'], row['close']))
    
    return pd.Series(result, index=df.index, name=f"atr_{n}")


# ============================================================================
# 8. MACD - Moving Average Convergence Divergence
# ============================================================================
def macd_batch(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    source: str = "close",
    **kwargs
) -> pd.DataFrame:
    """
    Compute MACD using scalar-based MACD class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    signal : int
        Signal line period
    source : str
        Column name (default: 'close')
    
    Returns
    -------
    pd.DataFrame
        Columns: macd, signal, histogram
    """
    macd_obj = MACD(fast=fast, slow=slow, signal=signal)
    macd_vals = []
    signal_vals = []
    hist_vals = []
    
    for val in df[source]:
        m, s, h = macd_obj.update(val)
        macd_vals.append(m)
        signal_vals.append(s)
        hist_vals.append(h)
    
    return pd.DataFrame({
        f"macd_{fast}_{slow}_{signal}": macd_vals,
        f"macd_signal_{fast}_{slow}_{signal}": signal_vals,
        f"macd_hist_{fast}_{slow}_{signal}": hist_vals
    }, index=df.index)


# ============================================================================
# 9. Bollinger Bands
# ============================================================================
def bollinger_batch(
    df: pd.DataFrame,
    n: int = 20,
    k: float = 2.0,
    source: str = "close",
    **kwargs
) -> pd.DataFrame:
    """
    Compute Bollinger Bands using scalar-based BollingerBands class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    n : int
        Period
    k : float
        Standard deviation multiplier
    source : str
        Column name (default: 'close')
    
    Returns
    -------
    pd.DataFrame
        Columns: bb_upper, bb_middle, bb_lower, bb_width, bb_pct
    """
    bb_obj = BollingerBands(period=n, std_dev=k)
    upper_vals = []
    middle_vals = []
    lower_vals = []
    width_vals = []
    pct_vals = []
    
    for val in df[source]:
        u, m, l, w, p = bb_obj.update(val)
        upper_vals.append(u)
        middle_vals.append(m)
        lower_vals.append(l)
        width_vals.append(w)
        pct_vals.append(p)
    
    return pd.DataFrame({
        f"bb_upper_{n}_{k}": upper_vals,
        f"bb_middle_{n}_{k}": middle_vals,
        f"bb_lower_{n}_{k}": lower_vals,
        f"bb_width_{n}_{k}": width_vals,
        f"bb_pct_{n}_{k}": pct_vals
    }, index=df.index)


# ============================================================================
# 10. Keltner Channel
# ============================================================================
def keltner_batch(
    df: pd.DataFrame,
    n: int = 20,
    m: float = 2.0,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Keltner Channel using scalar-based KeltnerChannel class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data (must have 'high', 'low', 'close')
    n : int
        EMA period
    m : float
        ATR multiplier
    
    Returns
    -------
    pd.DataFrame
        Columns: kc_upper, kc_middle, kc_lower, kc_width, kc_pct
    """
    kc_obj = KeltnerChannel(period=n, multiplier=m)
    upper_vals = []
    middle_vals = []
    lower_vals = []
    width_vals = []
    pct_vals = []
    
    for _, row in df.iterrows():
        u, mid, l, w, p = kc_obj.update(row['high'], row['low'], row['close'])
        upper_vals.append(u)
        middle_vals.append(mid)
        lower_vals.append(l)
        width_vals.append(w)
        pct_vals.append(p)
    
    return pd.DataFrame({
        f"kc_upper_{n}_{m}": upper_vals,
        f"kc_middle_{n}_{m}": middle_vals,
        f"kc_lower_{n}_{m}": lower_vals,
        f"kc_width_{n}_{m}": width_vals,
        f"kc_pct_{n}_{m}": pct_vals
    }, index=df.index)


# ============================================================================
# 11. Stochastic Oscillator
# ============================================================================ avdanced
def stochastic_batch(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
    method: Literal["sma", "ema"] = "sma",
    min_periods: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator using scalar-based Stochastic class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low', 'close'
    k_period : int
        %K period (default: 14)
    d_period : int
        %D smoothing period (default: 3)
    smooth_k : int
        %K smoothing period (default: 3)
        - smooth_k=1 → Fast Stochastic
        - smooth_k=3 → Slow Stochastic
    method : {"sma", "ema"}
        Smoothing method (default: "sma")
    min_periods : int, optional
        Minimum periods before returning valid values
    
    Returns
    -------
    pd.DataFrame
        Columns: stoch_k_{k_period}_{d_period}, stoch_d_{k_period}_{d_period}
    """
    stoch_obj = Stochastic(
        k_period=k_period,
        d_period=d_period,
        smooth_k=smooth_k,
        method=method,
        min_periods=min_periods
    )
    
    k_vals = []
    d_vals = []
    
    for _, row in df.iterrows():
        k, d = stoch_obj.update(row['high'], row['low'], row['close'])
        k_vals.append(k)
        d_vals.append(d)
    
    return pd.DataFrame({
        f"stoch_k_{k_period}_{d_period}": k_vals,
        f"stoch_d_{k_period}_{d_period}": d_vals
    }, index=df.index)


# ============================================================================
# 12. CCI (Commodity Channel Index)
# ============================================================================ avdanced
def cci_batch(
    df: pd.DataFrame,
    period: int = 20,
    min_periods: int = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compute CCI (Commodity Channel Index) using scalar-based CCI class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low', 'close'
    period : int
        CCI period
    min_periods : int, optional
        Minimum number of observations required for valid output
    
    Returns
    -------
    pd.DataFrame
        Column: cci_{period}
    """
    cci_obj = CCI(period=period, min_periods=min_periods)
    cci_vals = []
    
    for _, row in df.iterrows():
        val = cci_obj.update(row['high'], row['low'], row['close'])
        cci_vals.append(val)
    
    return pd.DataFrame({
        f"cci_{period}": cci_vals
    }, index=df.index)



# ============================================================================
# 13. MFI (Money Flow Index)
# ============================================================================ avdanced
def mfi_batch(
    df: pd.DataFrame,
    period: int = 14,
    min_periods: int = None,
    **kwargs
) -> pd.DataFrame:
    """
    Compute MFI using scalar-based MFI class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low', 'close', 'volume'
    period : int
        MFI period
    min_periods : int, optional
        Minimum periods required for valid output
    
    Returns
    -------
    pd.DataFrame
        Column: mfi
    """
    mfi_obj = MFI(period=period, min_periods=min_periods)
    mfi_vals = []
    
    for _, row in df.iterrows():
        val = mfi_obj.update(row['high'], row['low'], row['close'], row['volume'])
        mfi_vals.append(val)
    
    return pd.DataFrame({
        f"mfi_{period}": mfi_vals
    }, index=df.index)


# ============================================================================
# 14. OBV (On-Balance Volume)
# ============================================================================
def obv_batch(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Compute OBV using scalar-based OBV class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'close', 'volume'
    
    Returns
    -------
    pd.DataFrame
        Column: obv
    """
    obv_obj = OBV()
    obv_vals = []
    
    for _, row in df.iterrows():
        val = obv_obj.update(row['close'], row['volume'])
        obv_vals.append(val)
    
    return pd.DataFrame({
        "obv": obv_vals
    }, index=df.index)


# ============================================================================
# 15. Williams %R
# ============================================================================
def williamsr_batch(
    df: pd.DataFrame,
    period: int = 14,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Williams %R using scalar-based WilliamsR class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low', 'close'
    period : int
        Lookback period
    
    Returns
    -------
    pd.DataFrame
        Column: williams_r
    """
    wr_obj = WilliamsR(period=period)
    wr_vals = []
    
    for _, row in df.iterrows():
        val = wr_obj.update(row['high'], row['low'], row['close'])
        wr_vals.append(val)
    
    return pd.DataFrame({
        f"williams_r_{period}": wr_vals
    }, index=df.index)


# ============================================================================
# 16. Parabolic SAR
# ============================================================================
def parabolic_sar_batch(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Parabolic SAR using scalar-based ParabolicSAR class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low'
    af_start : float
        Initial acceleration factor
    af_increment : float
        AF increment per step
    af_max : float
        Maximum AF
    
    Returns
    -------
    pd.DataFrame
        Column: psar
    """
    psar_obj = ParabolicSAR(af_start=af_start, af_increment=af_increment, af_max=af_max)
    psar_vals = []
    
    for _, row in df.iterrows():
        val = psar_obj.update(row['high'], row['low'])
        psar_vals.append(val)
    
    return pd.DataFrame({
        f"psar_{af_start}_{af_max}": psar_vals
    }, index=df.index)


# ============================================================================
# 17. Heikin-Ashi
# ============================================================================
def heikin_ashi_batch(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Heikin-Ashi candles using scalar-based HeikinAshi class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'open', 'high', 'low', 'close'
    
    Returns
    -------
    pd.DataFrame
        Columns: ha_open, ha_high, ha_low, ha_close
    """
    ha_obj = HeikinAshi()
    ha_open_vals = []
    ha_high_vals = []
    ha_low_vals = []
    ha_close_vals = []
    
    for _, row in df.iterrows():
        ha_o, ha_h, ha_l, ha_c = ha_obj.update(row['open'], row['high'], row['low'], row['close'])
        ha_open_vals.append(ha_o)
        ha_high_vals.append(ha_h)
        ha_low_vals.append(ha_l)
        ha_close_vals.append(ha_c)
    
    return pd.DataFrame({
        "ha_open": ha_open_vals,
        "ha_high": ha_high_vals,
        "ha_low": ha_low_vals,
        "ha_close": ha_close_vals
    }, index=df.index)


# ============================================================================
# 18. Supertrend
# ============================================================================
def supertrend_batch(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Supertrend using scalar-based Supertrend class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low', 'close'
    period : int
        ATR period
    multiplier : float
        ATR multiplier
    
    Returns
    -------
    pd.DataFrame
        Columns: supertrend, supertrend_direction
    """
    st_obj = Supertrend(period=period, multiplier=multiplier)
    st_vals = []
    dir_vals = []
    
    for _, row in df.iterrows():
        st, direction = st_obj.update(row['high'], row['low'], row['close'])
        st_vals.append(st)
        dir_vals.append(direction)
    
    return pd.DataFrame({
        f"supertrend_{period}_{multiplier}": st_vals,
        f"supertrend_dir_{period}_{multiplier}": dir_vals
    }, index=df.index)


# ============================================================================
# 19. Aroon
# ============================================================================
def aroon_batch(
    df: pd.DataFrame,
    period: int = 25,
    **kwargs
) -> pd.DataFrame:
    """
    Compute Aroon indicator using scalar-based Aroon class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with 'high', 'low'
    period : int
        Aroon period
    
    Returns
    -------
    pd.DataFrame
        Columns: aroon_up, aroon_down, aroon_osc
    """
    aroon_obj = Aroon(period=period)
    up_vals = []
    down_vals = []
    osc_vals = []
    
    for _, row in df.iterrows():
        up, down, osc = aroon_obj.update(row['high'], row['low'])
        up_vals.append(up)
        down_vals.append(down)
        osc_vals.append(osc)
    
    return pd.DataFrame({
        f"aroon_up_{period}": up_vals,
        f"aroon_down_{period}": down_vals,
        f"aroon_osc_{period}": osc_vals
    }, index=df.index)


# ============================================================================
# 20. DEMA (Double Exponential Moving Average)
# ============================================================================
def dema_batch(
    df: pd.DataFrame,
    period: int = 20,
    source: str = "close",
    **kwargs
) -> pd.DataFrame:
    """
    Compute DEMA using scalar-based DEMA class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    period : int
        DEMA period
    source : str
        Column name (default: 'close')
    
    Returns
    -------
    pd.DataFrame
        Column: dema
    """
    dema_obj = DEMA(period=period)
    dema_vals = []
    
    for val in df[source]:
        result = dema_obj.update(val)
        dema_vals.append(result)
    
    return pd.DataFrame({
        f"dema_{period}": dema_vals
    }, index=df.index)


# ============================================================================
# 21. TEMA (Triple Exponential Moving Average)
# ============================================================================
def tema_batch(
    df: pd.DataFrame,
    period: int = 20,
    source: str = "close",
    **kwargs
) -> pd.DataFrame:
    """
    Compute TEMA using scalar-based TEMA class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    period : int
        TEMA period
    source : str
        Column name (default: 'close')
    
    Returns
    -------
    pd.DataFrame
        Column: tema
    """
    tema_obj = TEMA(period=period)
    tema_vals = []
    
    for val in df[source]:
        result = tema_obj.update(val)
        tema_vals.append(result)
    
    return pd.DataFrame({
        f"tema_{period}": tema_vals
    }, index=df.index)


# ============================================================================
# 22. HMA (Hull Moving Average)
# ============================================================================
def hma_batch(
    df: pd.DataFrame,
    period: int = 20,
    source: str = "close",
    **kwargs
) -> pd.DataFrame:
    """
    Compute HMA using scalar-based HMA class.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
    period : int
        HMA period
    source : str
        Column name (default: 'close')
    
    Returns
    -------
    pd.DataFrame
        Column: hma
    """
    hma_obj = HMA(period=period)
    hma_vals = []
    
    for val in df[source]:
        result = hma_obj.update(val)
        hma_vals.append(result)
    
    return pd.DataFrame({
        f"hma_{period}": hma_vals
    }, index=df.index)


# ============================================================================
# Registry-compatible exports
# ============================================================================
__all__ = [
    "sma_batch",
    "wma_batch",
    "ema_batch",
    "roc_batch",
    "rsi_batch",
    "true_range_batch",
    "atr_batch",
    "macd_batch",
    "bollinger_batch",
    "keltner_batch",

    "stochastic_batch",
    "cci_batch",
    "mfi_batch",
    "obv_batch",
    "williamsr_batch",
    "parabolic_sar_batch",
    "heikin_ashi_batch",
    "supertrend_batch",
    "aroon_batch",
    "dema_batch",
    "tema_batch",
    "hma_batch",
]

# ============================================================================
# Registry mapping for easy access
# ============================================================================
INDICATOR_REGISTRY = {
    "sma": sma_batch,
    "wma": wma_batch,
    "ema": ema_batch,
    "roc": roc_batch,
    "rsi": rsi_batch,
    "true_range": true_range_batch,
    "atr": atr_batch,
    "macd": macd_batch,
    "bollinger": bollinger_batch,
    "keltner": keltner_batch,

    "stochastic": stochastic_batch,
    "cci": cci_batch,
    "mfi": mfi_batch,
    "obv": obv_batch,
    "williamsr": williamsr_batch,
    "parabolic_sar": parabolic_sar_batch,
    "heikin_ashi": heikin_ashi_batch,
    "supertrend": supertrend_batch,
    "aroon": aroon_batch,
    "dema": dema_batch,
    "tema": tema_batch,
    "hma": hma_batch,
}
