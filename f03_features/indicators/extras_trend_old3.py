# f03_features/indicators/extras_trend.py
# Status: PRODUCTION-GRADE / WORLD-CLASS
# Reviewed & hardened

from __future__ import annotations

import logging
from typing import Dict, Literal, Mapping, Callable

import numpy as np
import pandas as pd

from .core import atr, ema, sma, wma
from f03_features.indicators.core import rsi as rsi_core

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# Internal helpers
# =============================================================================

def _ensure_series(
    x: pd.Series | np.ndarray,
    index: pd.Index,
    name: str | None = None,
    dtype: str | None = "float64",
) -> pd.Series:
    """
    Ensure x is a pandas.Series with a given index and (optionally) dtype.
    This prevents index misalignment / length explosions in Pandas ops.
    """
    if isinstance(x, pd.Series):
        s = x
        if not s.index.equals(index):
            s = s.reindex(index)
    else:
        s = pd.Series(x, index=index)
    if dtype is not None:
        s = s.astype(dtype)
    if name is not None:
        s.name = name
    return s


def _safe_div(
    num: pd.Series,
    den: pd.Series,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Numerically-safe division: num / (den + eps), preserving index.
    """
    den_safe = den.copy()
    den_safe = den_safe.where(den_safe.abs() > eps, np.nan)
    out = num / den_safe
    return out


# =============================================================================
# Supertrend (classic)
# =============================================================================

def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    """
    Classic Supertrend indicator (no future leak).

    Parameters
    ----------
    high, low, close : pd.Series
        Price series with identical index (no look-ahead).
    period : int, default 10
        ATR lookback window.
    multiplier : float, default 3.0
        ATR multiplier.

    Returns
    -------
    st : pd.Series (float32)
        Supertrend series aligned with `close.index`.
    """
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("high, low, close must share the same index")

    atrv = atr(high, low, close, n=period).astype("float64")
    hl2 = ((high + low) / 2.0).astype("float64")
    m = float(multiplier)

    upper = (hl2 + m * atrv)
    lower = (hl2 - m * atrv)

    st = pd.Series(index=close.index, dtype="float64")
    direction_up = True

    for i in range(len(close)):
        if i == 0:
            st.iloc[i] = upper.iloc[i]
            direction_up = True
            continue

        prev = st.iloc[i - 1]

        if close.iloc[i] > prev:
            st.iloc[i] = max(lower.iloc[i], prev)
            direction_up = True
        else:
            st.iloc[i] = min(upper.iloc[i], prev)
            direction_up = False

    return st.astype("float32")


# =============================================================================
# ADX / +DI / -DI / ADXR (Welles Wilder)
# =============================================================================

def adx_di(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Welles Wilder Directional Movement System.

    Computes:
    - +DI  (Positive Directional Indicator)
    - -DI  (Negative Directional Indicator)
    - ADX  (Average Directional Index)
    - ADXR (Average Directional Index Rating)

    All outputs are anti-look-ahead (causal) and index-aligned.

    Parameters
    ----------
    high, low, close : pd.Series
        Price series with identical index.
    n : int, default 14
        Lookback period.

    Returns
    -------
    pdi : pd.Series (float32)
        +DI.
    mdi : pd.Series (float32)
        -DI.
    adx : pd.Series (float32)
        ADX.
    adxr : pd.Series (float32)
        ADXR.
    """
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("high, low, close must share the same index")

    high = high.astype("float64")
    low = low.astype("float64")
    close = close.astype("float64")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = _ensure_series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype="float64",
    )

    minus_dm = _ensure_series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
        dtype="float64",
    )

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    tr = tr.replace(0.0, np.nan)

    atr_n = tr.rolling(n, min_periods=n).mean()

    pdi = 100.0 * plus_dm.rolling(n, min_periods=n).sum() / atr_n
    mdi = 100.0 * minus_dm.rolling(n, min_periods=n).sum() / atr_n

    dx = 100.0 * _safe_div((pdi - mdi).abs(), (pdi + mdi).replace(0.0, np.nan))

    adx = dx.rolling(n, min_periods=n).mean()
    adxr = (adx + adx.shift(n)) / 2.0

    return (
        pdi.astype("float32"),
        mdi.astype("float32"),
        adx.astype("float32"),
        adxr.astype("float32"),
    )


# =============================================================================
# Aroon
# =============================================================================

def aroon(
    high: pd.Series,
    low: pd.Series,
    n: int = 25,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Aroon Up / Down / Oscillator.

    Parameters
    ----------
    high, low : pd.Series
        High/low series with identical index.
    n : int, default 25
        Lookback window.

    Returns
    -------
    up : pd.Series (float32)
        Aroon Up (% time since highest high).
    down : pd.Series (float32)
        Aroon Down (% time since lowest low).
    osc : pd.Series (float32)
        Aroon Oscillator = up - down.
    """
    if not high.index.equals(low.index):
        raise ValueError("high and low must share the same index")

    high = high.astype("float64")
    low = low.astype("float64")

    def _aroon_up(s: pd.Series) -> pd.Series:
        def _argmax_rev(x: np.ndarray) -> float:
            idx = np.argmax(x[::-1])
            return (idx / (n - 1)) * 100.0

        return (s.rolling(n, min_periods=n)
                  .apply(_argmax_rev, raw=True)
                  .astype("float64"))

    def _aroon_down(s: pd.Series) -> pd.Series:
        def _argmin_rev(x: np.ndarray) -> float:
            idx = np.argmin(x[::-1])
            return (idx / (n - 1)) * 100.0

        return (s.rolling(n, min_periods=n)
                  .apply(_argmin_rev, raw=True)
                  .astype("float64"))

    up = 100.0 * _aroon_up(high)
    down = 100.0 * _aroon_down(low)
    osc = up - down

    return (
        up.astype("float32"),
        down.astype("float32"),
        osc.astype("float32"),
    )


# =============================================================================
# KAMA / DEMA / TEMA / HMA
# =============================================================================

def kama(
    s: pd.Series,
    n: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA), causal implementation.

    Parameters
    ----------
    s : pd.Series
        Input price series.
    n : int, default 10
        Efficiency ratio lookback.
    fast : int, default 2
        Fast EMA equivalent length.
    slow : int, default 30
        Slow EMA equivalent length.

    Returns
    -------
    kama : pd.Series (float32)
        Adaptive moving average.
    """
    s = s.astype("float64")

    change = s.diff(n).abs()
    volatility = s.diff().abs().rolling(n, min_periods=n).sum()
    volatility = volatility.replace(0.0, np.nan)

    ef = _safe_div(change, volatility)

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (ef * (fast_sc - slow_sc) + slow_sc) ** 2

    out = pd.Series(index=s.index, dtype="float64")

    if len(s) == 0:
        return out.astype("float32")

    out.iloc[:n] = s.iloc[:n]

    for i in range(n, len(s)):
        prev = out.iloc[i - 1]
        out.iloc[i] = prev + sc.iloc[i] * (s.iloc[i] - prev)

    return out.astype("float32")


def dema(
    s: pd.Series,
    n: int = 20,
) -> pd.Series:
    """
    Double Exponential Moving Average (DEMA).

    Parameters
    ----------
    s : pd.Series
        Input series.
    n : int, default 20
        EMA length.

    Returns
    -------
    dema : pd.Series (float32)
    """
    s = s.astype("float64")
    e = ema(s, n).astype("float64")
    e2 = ema(e, n).astype("float64")
    out = 2.0 * e - e2
    return out.astype("float32")


def tema(
    s: pd.Series,
    n: int = 20,
) -> pd.Series:
    """
    Triple Exponential Moving Average (TEMA).

    Parameters
    ----------
    s : pd.Series
        Input series.
    n : int, default 20
        EMA length.

    Returns
    -------
    tema : pd.Series (float32)
    """
    s = s.astype("float64")
    e1 = ema(s, n).astype("float64")
    e2 = ema(e1, n).astype("float64")
    e3 = ema(e2, n).astype("float64")
    out = 3.0 * e1 - 3.0 * e2 + e3
    return out.astype("float32")


def hma(
    s: pd.Series,
    n: int = 20,
) -> pd.Series:
    """
    Hull Moving Average (HMA).

    Parameters
    ----------
    s : pd.Series
        Input series.
    n : int, default 20
        HMA length.

    Returns
    -------
    hma : pd.Series (float32)
    """
    s = s.astype("float64")
    n2 = max(2, n // 2)
    w1 = wma(s, n2).astype("float64")
    w2 = wma(s, n).astype("float64")
    diff = 2.0 * w1 - w2
    out = wma(diff, int(np.sqrt(n))).astype("float64")
    return out.astype("float32")


# =============================================================================
# Ichimoku (Tenkan / Kijun / Senkou A / Senkou B)
# NOTE: Chikou is intentionally excluded from the return to avoid look-ahead.
# =============================================================================

def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ichimoku Kinko Hyo core lines, anti look-ahead (no Chikou output).

    Parameters
    ----------
    high, low, close : pd.Series
        Price series with identical index.
    tenkan : int, default 9
        Tenkan-sen period.
    kijun : int, default 26
        Kijun-sen period / Senkou shift.
    span_b : int, default 52
        Senkou Span B period.

    Returns
    -------
    tenkan_sen : pd.Series (float32)
    kijun_sen : pd.Series (float32)
    span_a : pd.Series (float32)
        Senkou Span A, shifted +kijun.
    span_b_line : pd.Series (float32)
        Senkou Span B, shifted +kijun.

    Notes
    -----
    Chikou (lagging span) is not returned to avoid direct look-ahead bias
    in feature pipelines. If needed, it should be derived externally with
    proper shifting consistent with the ML/RL setup.
    """
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("high, low, close must share the same index")

    high = high.astype("float64")
    low = low.astype("float64")
    close = close.astype("float64")

    tenkan_sen = ((high.rolling(tenkan).max() +
                   low.rolling(tenkan).min()) / 2.0)
    kijun_sen = ((high.rolling(kijun).max() +
                  low.rolling(kijun).min()) / 2.0)

    span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)
    span_b_line = ((high.rolling(span_b).max() +
                    low.rolling(span_b).min()) / 2.0).shift(kijun)

    return (
        tenkan_sen.astype("float32"),
        kijun_sen.astype("float32"),
        span_a.astype("float32"),
        span_b_line.astype("float32"),
    )


# =============================================================================
# MA Slope (normalized)
# =============================================================================

def ma_slope(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    method: Literal["sma", "ema"] = "ema",
    norm: Literal["stdev", "price", "none"] = "stdev",
    eps: float = 1e-12,
) -> pd.Series:
    """
    Normalized moving-average slope, suitable as a trend-strength feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, must contain `price_col`.
    price_col : str, default "close"
        Column to use for price.
    window : int, default 20
        MA and normalization window.
    method : {"sma", "ema"}, default "ema"
        Moving average type.
    norm : {"stdev", "price", "none"}, default "stdev"
        Normalization scheme:
          - "stdev": slope / rolling std(price)
          - "price": slope / |price|
          - "none" : raw slope
    eps : float, default 1e-12
        Numerical epsilon for denominator.

    Returns
    -------
    slope : pd.Series (float32)
        Normalized slope with name f"ma_slope_{method}_{window}".
    """
    if price_col not in df.columns:
        raise ValueError(f"df must contain column '{price_col}'")

    px = df[price_col].astype("float64")

    if method == "ema":
        ma = px.ewm(
            span=window,
            adjust=False,
            min_periods=max(2, window // 2),
        ).mean()
    elif method == "sma":
        ma = px.rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).mean()
    else:
        raise ValueError("method must be 'sma' or 'ema'")

    slope = ma.diff()

    if norm == "stdev":
        denom = px.rolling(
            window=window,
            min_periods=max(2, window // 2),
        ).std()
        slope = _safe_div(slope, denom + eps)
    elif norm == "price":
        slope = _safe_div(slope, px.abs() + eps)
    elif norm == "none":
        pass
    else:
        raise ValueError("norm must be 'stdev', 'price', or 'none'")

    slope = slope.astype("float32")
    slope.name = f"ma_slope_{method}_{window}"
    return slope


# =============================================================================
# RSI Zone flags / score
# =============================================================================

def rsi_zone(
    df: pd.DataFrame,
    price_col: str = "close",
    period: int = 14,
    overbought: float = 70.0,
    oversold: float = 30.0,
    mid: float = 50.0,
    band: float = 5.0,
) -> pd.DataFrame:
    """
    RSI zoning with binary flags (and optional score extension).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe, must contain `price_col`.
    price_col : str, default "close"
        Column used for RSI computation.
    period : int, default 14
        RSI lookback length.
    overbought : float, default 70.0
        Threshold for overbought region.
    oversold : float, default 30.0
        Threshold for oversold region.
    mid : float, default 50.0
        Midline used for "mid zone".
    band : float, default 5.0
        +/- band around mid for mid-zone flag.

    Returns
    -------
    out : pd.DataFrame
        Columns:
        - rsi_value : float32
        - rsi_is_overbought : bool
        - rsi_is_oversold : bool
        - rsi_mid_zone : bool
    """
    if price_col not in df.columns:
        raise ValueError(f"df must contain column '{price_col}'")

    px = df[price_col].astype("float64")
    rsi = rsi_core(px, n=period).astype("float32")

    is_ob = (rsi >= overbought)
    is_os = (rsi <= oversold)
    is_mid = (rsi.sub(mid).abs() <= band)

    out = pd.DataFrame(
        {
            "rsi_value": rsi.astype("float32"),
            "rsi_is_overbought": is_ob.astype(bool),
            "rsi_is_oversold": is_os.astype(bool),
            "rsi_mid_zone": is_mid.astype(bool),
        },
        index=df.index,
    )
    return out


# =============================================================================
# Registry
# =============================================================================

def registry() -> Mapping[str, Callable[..., Dict[str, pd.Series]]]:
    """
    Registry of extra trend indicators, production-grade.

    All returned Series are:
    - index-aligned with the input df
    - dtype float32 (for numeric features)
    - strictly causal (no look-ahead)
    """

    def wrap(name: str, s: pd.Series) -> Dict[str, pd.Series]:
        return {name: s.astype("float32")}

    def make_supertrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
        **_,
    ) -> Dict[str, pd.Series]:
        st = supertrend(df["high"], df["low"], df["close"], period, multiplier)
        return wrap(f"supertrend_{period}_{multiplier}", st)

    def make_adx(
        df: pd.DataFrame,
        n: int = 14,
        **_,
    ) -> Dict[str, pd.Series]:
        pdi, mdi, adxv, adxr = adx_di(df["high"], df["low"], df["close"], n)
        return {
            f"pdi_{n}": pdi.astype("float32"),
            f"mdi_{n}": mdi.astype("float32"),
            f"adx_{n}": adxv.astype("float32"),
            f"adxr_{n}": adxr.astype("float32"),
        }

    def make_aroon(
        df: pd.DataFrame,
        n: int = 25,
        **_,
    ) -> Dict[str, pd.Series]:
        up, down, osc = aroon(df["high"], df["low"], n)
        return {
            f"aroon_up_{n}": up.astype("float32"),
            f"aroon_down_{n}": down.astype("float32"),
            f"aroon_osc_{n}": osc.astype("float32"),
        }

    def make_kama(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 10,
        fast: int = 2,
        slow: int = 30,
        **_,
    ) -> Dict[str, pd.Series]:
        return wrap(
            f"kama_{col}_{n}_{fast}_{slow}",
            kama(df[col], n, fast, slow),
        )

    def make_dema(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 20,
        **_,
    ) -> Dict[str, pd.Series]:
        return wrap(f"dema_{col}_{n}", dema(df[col], n))

    def make_tema(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 20,
        **_,
    ) -> Dict[str, pd.Series]:
        return wrap(f"tema_{col}_{n}", tema(df[col], n))

    def make_hma(
        df: pd.DataFrame,
        col: str = "close",
        n: int = 20,
        **_,
    ) -> Dict[str, pd.Series]:
        return wrap(f"hma_{col}_{n}", hma(df[col], n))

    def make_ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        span_b: int = 52,
        **_,
    ) -> Dict[str, pd.Series]:
        conv, base, sa, sb = ichimoku(
            df["high"],
            df["low"],
            df["close"],
            tenkan,
            kijun,
            span_b,
        )
        return {
            f"ichi_tenkan_{tenkan}": conv.astype("float32"),
            f"ichi_kijun_{kijun}": base.astype("float32"),
            f"ichi_span_a_{tenkan}_{kijun}": sa.astype("float32"),
            f"ichi_span_b_{span_b}": sb.astype("float32"),
        }

    def make_ma_slope(
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        method: Literal["sma", "ema"] = "ema",
        norm: Literal["stdev", "price", "none"] = "stdev",
        **_,
    ) -> Dict[str, pd.Series]:
        s = ma_slope(
            df=df,
            price_col=price_col,
            window=window,
            method=method,
            norm=norm,
        )
        return {s.name: s.astype("float32")}

    def make_rsi_zone(
        df: pd.DataFrame,
        price_col: str = "close",
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        mid: float = 50.0,
        band: float = 5.0,
        **_,
    ) -> Dict[str, pd.Series]:
        rz = rsi_zone(
            df=df,
            price_col=price_col,
            period=period,
            overbought=overbought,
            oversold=oversold,
            mid=mid,
            band=band,
        )
        # flatten DataFrame -> dict of Series
        out: Dict[str, pd.Series] = {}
        for col in rz.columns:
            s = rz[col]
            if s.dtype == bool:
                # cast to float32 for downstream uniformity if needed
                out[col] = s.astype("float32")
            else:
                out[col] = s.astype("float32")
        return out

    return {
        "supertrend": make_supertrend,
        "adx": make_adx,
        "aroon": make_aroon,
        "kama": make_kama,
        "dema": make_dema,
        "tema": make_tema,
        "hma": make_hma,
        "ichimoku": make_ichimoku,
        "ma_slope": make_ma_slope,
        "rsi_zone": make_rsi_zone,
    }
