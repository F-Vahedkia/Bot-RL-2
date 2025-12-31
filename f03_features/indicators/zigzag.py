# f03_features/indicators/zigzag.py

import numpy as np
import pandas as pd
from numba import njit, types
from numba.typed import List as TypedList
from typing import Literal


#==============================================================================
# MQL ZIGZAG
#==============================================================================
def _zigzag_mql_njit(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
) -> pd.Series:

    high_arr = high.values
    low_arr = low.values
    n = len(high_arr)

    @njit
    def _zigzag(high_arr, low_arr, depth, deviation, backstep, point):
        zz = np.zeros(n)
        high_map = np.zeros(n)
        low_map = np.zeros(n)
        state_series = np.zeros(n, dtype=np.int8)
        
        last_high = last_low = 0.0
        last_high_pos = last_low_pos = -1
        last_swing_type = 0

        leg_count = 0
        legs_ts_start = np.zeros(n, dtype=np.int32)
        legs_ts_end = np.zeros(n, dtype=np.int32)
        legs_start_point = np.zeros(n)
        legs_end_point = np.zeros(n)
        legs_direction = np.zeros(n, dtype=np.int8)

        def highest(arr, start, depth):
            i0 = max(0, start - depth + 1)
            max_idx = i0
            max_val = arr[i0]
            for i in range(i0 + 1, start + 1):
                if arr[i] > max_val:
                    max_val = arr[i]
                    max_idx = i
            return max_idx

        def lowest(arr, start, depth):
            i0 = max(0, start - depth + 1)
            min_idx = i0
            min_val = arr[i0]
            for i in range(i0 + 1, start + 1):
                if arr[i] < min_val:
                    min_val = arr[i]
                    min_idx = i
            return min_idx

        # populate high_map and low_map
        for i in range(depth, n):
            idx = lowest(low_arr, i, depth)
            val = low_arr[idx]
            if val == last_low or (low_arr[i] - val) > deviation * point:
                val = 0.0
            else:
                last_low = val
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and low_map[j] != 0 and low_map[j] > val:
                        low_map[j] = 0.0
            low_map[i] = val if low_arr[i] == val else 0.0

            idx = highest(high_arr, i, depth)
            val = high_arr[idx]
            if val == last_high or (val - high_arr[i]) > deviation * point:
                val = 0.0
            else:
                last_high = val
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and high_map[j] != 0 and high_map[j] < val:
                        high_map[j] = 0.0
            high_map[i] = val if high_arr[i] == val else 0.0

        EXTREMUM, PEAK, BOTTOM = 0, 1, -1
        state = EXTREMUM
        last_high = last_low = 0.0
        last_high_pos = last_low_pos = -1
        last_swing_type = 0

        for i in range(depth, n):
            if state == EXTREMUM:
                if last_low == 0 and last_high == 0:
                    if high_map[i] != 0 and last_swing_type != 1:
                        last_high = high_map[i]
                        last_high_pos = i
                        zz[i] = 1
                        state_series[i] = BOTTOM
                        state = BOTTOM
                        last_swing_type = 1
                    elif low_map[i] != 0 and last_swing_type != -1:
                        last_low = low_map[i]
                        last_low_pos = i
                        zz[i] = -1
                        state_series[i] = PEAK
                        state = PEAK
                        last_swing_type = -1

            elif state == PEAK:
                if low_map[i] != 0 and (last_swing_type != -1 or low_map[i] < last_low) and high_map[i] == 0:
                    if last_low_pos >= 0:
                        zz[last_low_pos] = 0.0
                    zz[i] = -1
                    state_series[i] = PEAK
                    last_low = low_map[i]
                    last_low_pos = i
                    last_swing_type = -1
                elif high_map[i] != 0 and low_map[i] == 0 and last_swing_type != 1:
                    last_high = high_map[i]
                    last_high_pos = i
                    zz[i] = 1
                    state_series[i] = BOTTOM
                    state = BOTTOM
                    last_swing_type = 1
                    if last_low_pos >= 0:
                        legs_ts_start[leg_count] = last_low_pos
                        legs_ts_end[leg_count] = i
                        legs_start_point[leg_count] = last_low
                        legs_end_point[leg_count] = last_high
                        legs_direction[leg_count] = 1
                        leg_count += 1

            elif state == BOTTOM:
                if high_map[i] != 0 and (last_swing_type != 1 or high_map[i] > last_high) and low_map[i] == 0:
                    if last_high_pos >= 0:
                        zz[last_high_pos] = 0.0
                    zz[i] = 1
                    state_series[i] = BOTTOM
                    last_high = high_map[i]
                    last_high_pos = i
                    last_swing_type = 1
                elif low_map[i] != 0 and high_map[i] == 0 and last_swing_type != -1:
                    last_low = low_map[i]
                    last_low_pos = i
                    zz[i] = -1
                    state_series[i] = PEAK
                    state = PEAK
                    last_swing_type = -1
                    if last_high_pos >= 0:
                        legs_ts_start[leg_count] = last_high_pos
                        legs_ts_end[leg_count] = i
                        legs_start_point[leg_count] = last_high
                        legs_end_point[leg_count] = last_low
                        legs_direction[leg_count] = -1
                        leg_count += 1

        return zz, state_series, high_map, low_map, \
               legs_ts_start[:leg_count], legs_ts_end[:leg_count], \
               legs_start_point[:leg_count], legs_end_point[:leg_count], legs_direction[:leg_count]

    zz, state_series, high_map, low_map, legs_start_idx, legs_end_idx, legs_start_pt, legs_end_pt, legs_dir = _zigzag(
        high_arr, low_arr, depth, deviation, backstep, point
    )

    zz_series = pd.Series(zz, index=high.index)
    legs = []
    for i in range(len(legs_start_idx)):
        legs.append({
            "ts_start": legs_start_idx[i],
            "ts_end": legs_end_idx[i],
            "start_point": legs_start_pt[i],
            "end_point": legs_end_pt[i],
            "direction": legs_dir[i]
        })

    zz_series.attrs["legs"] = legs
    zz_series.attrs["state_series"] = state_series.tolist()
    zz_series.attrs["high_map"] = high_map.tolist()
    zz_series.attrs["low_map"] = low_map.tolist()

    return zz_series

def _zigzag_mql(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
) -> pd.Series:
    
    n = len(high)
    high_arr = high.values
    low_arr = low.values

    zz = np.zeros(n)
    high_map = np.zeros(n)
    low_map = np.zeros(n)
    state_series = np.zeros(n, dtype=np.int8)

    # --- initial swing tracking ---
    last_high = last_low = 0.0
    last_high_pos = last_low_pos = -1
    last_swing_type = 0  # 1=High, -1=Low, 0=None

    # --- populate high_map and low_map ---
    def highest(arr, start, depth):
        i0 = max(0, start - depth + 1) # سمت چپ بازه
        max_idx = i0
        max_val = arr[i0]
        for i in range(i0 + 1, start + 1):
            if arr[i] > max_val:
                max_val = arr[i]
                max_idx = i
        return max_idx

    def lowest(arr, start, depth):
        i0 = max(0, start - depth + 1)
        min_idx = i0
        min_val = arr[i0]
        for i in range(i0 + 1, start + 1):
            if arr[i] < min_val:
                min_val = arr[i]
                min_idx = i
        return min_idx

    for i in range(depth, n):
        # low extremum
        idx = lowest(low_arr, i, depth)
        val = low_arr[idx]
        if val == last_low or (val - low_arr[i]) > deviation * point:
            val = 0.0
        else:
            last_low = val
            for b in range(1, backstep + 1):
                j = i - b
                if j >= 0 and low_map[j] != 0 and low_map[j] > val:
                    low_map[j] = 0.0
        low_map[i] = val if low_arr[i] == val else 0.0

        # high extremum
        idx = highest(high_arr, i, depth)
        val = high_arr[idx]
        if val == last_high or (high_arr[i] - val) > deviation * point:
            val = 0.0
        else:
            last_high = val
            for b in range(1, backstep + 1):
                j = i - b
                if j >= 0 and high_map[j] != 0 and high_map[j] < val:
                    high_map[j] = 0.0
        high_map[i] = val if high_arr[i] == val else 0.0

    # --- final zigzag selection and leg calculation ---
    EXTREMUM, PEAK, BOTTOM = 0, 1, -1
    state = EXTREMUM
    last_high = last_low = 0.0
    last_high_pos = last_low_pos = -1
    last_swing_type = 0

    legs = []

    for i in range(depth, n):
        if state == EXTREMUM:
            if last_low == 0 and last_high == 0:
                if high_map[i] != 0 and last_swing_type != 1:
                    last_high = high_map[i]
                    last_high_pos = i
                    zz[i] = 1
                    state_series[i] = BOTTOM
                    state = BOTTOM
                    last_swing_type = 1
                elif low_map[i] != 0 and last_swing_type != -1:
                    last_low = low_map[i]
                    last_low_pos = i
                    zz[i] = -1
                    state_series[i] = PEAK
                    state = PEAK
                    last_swing_type = -1

        elif state == PEAK:
            if low_map[i] != 0 and (last_swing_type != -1 or low_map[i] < last_low) and high_map[i] == 0:
                if last_low_pos >= 0:
                    zz[last_low_pos] = 0.0
                zz[i] = -1
                state_series[i] = PEAK
                last_low = low_map[i]
                last_low_pos = i
                last_swing_type = -1
            elif high_map[i] != 0 and low_map[i] == 0 and last_swing_type != 1:
                last_high = high_map[i]
                last_high_pos = i
                zz[i] = 1
                state_series[i] = BOTTOM
                state = BOTTOM
                last_swing_type = 1
                if last_low_pos >= 0:
                    legs.append({
                        "ts_start": last_low_pos,
                        "ts_end": i,
                        "start_point": last_low,
                        "end_point": last_high,
                        "direction": 1
                    })

        elif state == BOTTOM:
            if high_map[i] != 0 and (last_swing_type != 1 or high_map[i] > last_high) and low_map[i] == 0:
                if last_high_pos >= 0:
                    zz[last_high_pos] = 0.0
                zz[i] = 1
                state_series[i] = BOTTOM
                last_high = high_map[i]
                last_high_pos = i
                last_swing_type = 1
            elif low_map[i] != 0 and high_map[i] == 0 and last_swing_type != -1:
                last_low = low_map[i]
                last_low_pos = i
                zz[i] = -1
                state_series[i] = PEAK
                state = PEAK
                last_swing_type = -1
                if last_high_pos >= 0:
                    legs.append({
                        "ts_start": last_high_pos,
                        "ts_end": i,
                        "start_point": last_high,
                        "end_point": last_low,
                        "direction": -1
                    })

    zz_series = pd.Series(zz, index=high.index)
    zz_series.attrs["legs"] = legs
    zz_series.attrs["state_series"] = state_series.tolist()
    zz_series.attrs["high_map"] = high_map.tolist()
    zz_series.attrs["low_map"] = low_map.tolist()

    return zz_series

#==============================================================================
# Wrapper function to choose between njit and non-njit based on data size
#==============================================================================
def zigzag(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
) -> pd.Series:
    
    if(len(high) < 150_000):
        return _zigzag_mql     (high, low, depth, deviation, backstep, point)
    else:
        return _zigzag_mql_njit(high, low, depth, deviation, backstep, point)

#==============================================================================
# MULTI-TIMEFRAME ADAPTER
#==============================================================================
def zigzag_mtf_adapter(
    high: pd.Series,
    low: pd.Series,
    tf_higher: str,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
    mode: Literal["last", "forward_fill"] = "forward_fill",
) -> pd.Series:
    """
    Multi-Timeframe ZigZag Adapter
    --------------------------------
    - Computes ZigZag on higher timeframe (HTF)
    - Projects HTF swing points onto lower timeframe (LTF)

    Parameters
    ----------
    high, low : pd.Series
        LTF price series (DatetimeIndex required)
    tf_higher : str
        Pandas resample rule (e.g. '5T', '15T', '1H', '4H', '1D')
    mode : 'last' | 'forward_fill'
        Projection mode to LTF
        'last':         → Marks the swing only at the first LTF candle where the HTF swing is confirmed.
                    (فقط نقطه شروع swing در LTF علامت می‌خورد)
        'forward_fill': → Propagates the HTF swing signal forward on all LTF candles until the next HTF swing appears.
                    (سیگنال swing در تمام کندل‌های LTF تا ظهور swing بعدی ادامه می‌یابد)
        
    Returns
    -------
    pd.Series
        MTF-aware ZigZag aligned to LTF index
    """
    
    # --- validation ---
    if not isinstance(high.index, pd.DatetimeIndex):
        raise ValueError("high/low must have DatetimeIndex")
    
    # --- build HTF OHLC ---
    htf = pd.DataFrame({"high": high, "low": low}).resample(tf_higher) \
                  .agg({"high": "max", "low": "min"}).dropna()
    
    # --- run original ZigZag on HTF ---
    zz_htf = zigzag(
        high=htf["high"],
        low=htf["low"],
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point
    )
    
    # --- prepare LTF container ---
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}"
    )
    
    # --- project HTF swings onto LTF ---
    for ts, signal in zz_htf.items():
        if signal == 0:
            continue

        if ts not in zz_ltf.index:
            ts = zz_ltf.index[zz_ltf.index.get_indexer([ts], method="ffill")[0]]

        if mode == "last":
            zz_ltf.loc[ts] = signal

        elif mode == "forward_fill":
            zz_ltf.loc[ts:] = signal
    
    # --- attach metadata ---
    zz_ltf.attrs["source_tf"] = tf_higher
    zz_ltf.attrs["mode"] = mode
    zz_ltf.attrs["htf_legs"] = zz_htf.attrs.get("legs", [])

    return zz_ltf

#==============================================================================






# OK ++++++
def _zigzag_mql_2(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    point: float = 0.01,
) -> pd.Series:
    
    n = len(high)
    high_arr = high.values
    low_arr = low.values

    # بافرهای اصلی
    zz_buffer = np.zeros(n)        # نقاط ZigZag
    high_map = np.zeros(n)         # نقاط سقف
    low_map = np.zeros(n)          # نقاط کف
    state_series = np.zeros(n, dtype=np.int8)  # state: -1=high, 1=low, 0=none
    
    # توابع کمکی برای یافتن highest/lowest
    def highest(arr, start, depth):
        end = max(0, start - depth + 1)
        max_idx = end
        max_val = arr[end]
        for i in range(end + 1, start + 1):
            if arr[i] > max_val:
                max_val = arr[i]
                max_idx = i
        return max_idx
    
    def lowest(arr, start, depth):
        end = max(0, start - depth + 1)
        min_idx = end
        min_val = arr[end]
        for i in range(end + 1, start + 1):
            if arr[i] < min_val:
                min_val = arr[i]
                min_idx = i
        return min_idx
    
    # متغیرهای ردیابی
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0  # 0=Extremum, 1=Peak, -1=Bottom
    
    # --- مرحله ۱: شناسایی نقاط high/low محلی ---
    for i in range(depth - 1, n):
        # --- LOW بررسی ---
        idx = lowest(low_arr, i, depth)
        val = low_arr[idx]
        
        if val == last_low:
            val = 0.0
        else:
            last_low = val
            if (low_arr[i] - val) > deviation * point:
                val = 0.0
            else:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and low_map[j] != 0 and low_map[j] > val:
                        low_map[j] = 0.0
        
        if low_arr[i] == val:
            low_map[i] = val
        else:
            low_map[i] = 0.0
        
        # --- HIGH بررسی ---
        idx = highest(high_arr, i, depth)
        val = high_arr[idx]
        
        if val == last_high:
            val = 0.0
        else:
            last_high = val
            if (val - high_arr[i]) > deviation * point:
                val = 0.0
            else:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and high_map[j] != 0 and high_map[j] < val:
                        high_map[j] = 0.0
        
        if high_arr[i] == val:
            high_map[i] = val
        else:
            high_map[i] = 0.0
    
    # --- مرحله ۲: انتخاب نهایی نقاط ZigZag ---
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0  # Extremum
    
    legs = []
    
    for i in range(depth - 1, n):
        if search_mode == 0:  # Extremum
            if last_low == 0.0 and last_high == 0.0:
                if high_map[i] != 0.0:
                    last_high = high_map[i]
                    last_high_pos = i
                    zz_buffer[i] = last_high
                    state_series[i] = -1  # high point
                    search_mode = -1  # Bottom
                
                if low_map[i] != 0.0:
                    last_low = low_map[i]
                    last_low_pos = i
                    zz_buffer[i] = last_low
                    state_series[i] = 1   # low point
                    search_mode = 1  # Peak
        
        elif search_mode == 1:  # Peak - به دنبال Low
            if low_map[i] != 0.0 and low_map[i] < last_low and high_map[i] == 0.0:
                if last_low_pos >= 0:
                    zz_buffer[last_low_pos] = 0.0
                    state_series[last_low_pos] = 0
                
                last_low = low_map[i]
                last_low_pos = i
                zz_buffer[i] = last_low
                state_series[i] = 1
            
            elif high_map[i] != 0.0 and low_map[i] == 0.0:
                last_high = high_map[i]
                last_high_pos = i
                zz_buffer[i] = last_high
                state_series[i] = -1
                search_mode = -1
                
                if last_low_pos >= 0 and last_high_pos >= 0:
                    legs.append({
                        "start_idx": last_low_pos,
                        "end_idx": last_high_pos,
                        "start_price": last_low,
                        "end_price": last_high,
                        "direction": 1
                    })
        
        elif search_mode == -1:  # Bottom - به دنبال High
            if high_map[i] != 0.0 and high_map[i] > last_high and low_map[i] == 0.0:
                if last_high_pos >= 0:
                    zz_buffer[last_high_pos] = 0.0
                    state_series[last_high_pos] = 0
                
                last_high = high_map[i]
                last_high_pos = i
                zz_buffer[i] = last_high
                state_series[i] = -1
            
            elif low_map[i] != 0.0 and high_map[i] == 0.0:
                last_low = low_map[i]
                last_low_pos = i
                zz_buffer[i] = last_low
                state_series[i] = 1
                search_mode = 1
                
                if last_high_pos >= 0 and last_low_pos >= 0:
                    legs.append({
                        "start_idx": last_high_pos,
                        "end_idx": last_low_pos,
                        "start_price": last_high,
                        "end_price": last_low,
                        "direction": -1
                    })
    
    # ایجاد Series خروجی
    zz_series = pd.Series(zz_buffer, index=high.index)
    
    # اضافه کردن تمام metadataهای مورد نیاز
    zz_series.attrs.update({
        "legs": legs,
        "state_series": state_series.tolist(),
        "high_map": high_map.tolist(),
        "low_map": low_map.tolist(),
        "params": {
            "depth": depth,
            "deviation": deviation,
            "backstep": backstep,
            "point": point
        }
    })
    
    return zz_series








def _zigzag_mql_njit_2(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    point: float = 0.00001,
) -> pd.Series:
    #==========================================================================
    """
    نسخه njit کامل ZigZag با API یکسان با نسخه اصلی
    
    Parameters:
    -----------
    high : pd.Series
        قیمت‌های high
    low : pd.Series
        قیمت‌های low
    depth : int
        عمق جستجو (default: 12)
    deviation : float
        حداقل انحراف (default: 5.0)
    backstep : int
        قدم بازگشت (default: 3)
    point : float
        ارزش هر pip (default: 0.00001 برای 5-digit)
    
    Returns:
    --------
    pd.Series
        سری ZigZag با metadata کامل در attrs:
        - state_series: آرایه stateها (-1, 0, 1)
        - high_map: نقاط high
        - low_map: نقاط low
        - legs: لیست legها با فرمت اصلی
        - params: پارامترهای ورودی
    """
    #==========================================================================
    @njit
    def _highest_njit(arr: np.ndarray, start: int, depth: int) -> int:
        """پیدا کردن index بیشترین مقدار در بازه depth"""
        end = max(0, start - depth + 1)
        max_idx = end
        max_val = arr[end]
        for i in range(end + 1, start + 1):
            if arr[i] > max_val:
                max_val = arr[i]
                max_idx = i
        return max_idx

    @njit
    def _lowest_njit(arr: np.ndarray, start: int, depth: int) -> int:
        """پیدا کردن index کمترین مقدار در بازه depth"""
        end = max(0, start - depth + 1)
        min_idx = end
        min_val = arr[end]
        for i in range(end + 1, start + 1):
            if arr[i] < min_val:
                min_val = arr[i]
                min_idx = i
        return min_idx

    @njit
    def _zigzag_njit_core(
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        depth: int,
        deviation: float,
        backstep: int,
        point: float
    ) -> tuple:
        """هسته اصلی ZigZag با njit - کاملاً مشابه منطق اصلی"""
        n = len(high_arr)
        
        # بافرهای اصلی
        zz_buffer = np.zeros(n)
        high_map = np.zeros(n)
        low_map = np.zeros(n)
        state_series = np.zeros(n)  # -1: high, 1: low, 0: none
        
        # متغیرهای موقت برای مرحله اول
        last_high_temp = 0.0
        last_low_temp = 0.0
        
        # --- مرحله 1: شناسایی نقاط high/low محلی ---
        for i in range(depth - 1, n):
            # LOW بررسی
            idx = _lowest_njit(low_arr, i, depth)
            val = low_arr[idx]
            
            if val == last_low_temp:
                val = 0.0
            else:
                last_low_temp = val
                if (low_arr[i] - val) > deviation * point:
                    val = 0.0
                else:
                    # backstep برای low
                    for b in range(1, backstep + 1):
                        j = i - b
                        if j >= 0 and low_map[j] != 0 and low_map[j] > val:
                            low_map[j] = 0.0
            
            if low_arr[i] == val:
                low_map[i] = val
            
            # HIGH بررسی
            idx = _highest_njit(high_arr, i, depth)
            val = high_arr[idx]
            
            if val == last_high_temp:
                val = 0.0
            else:
                last_high_temp = val
                if (val - high_arr[i]) > deviation * point:
                    val = 0.0
                else:
                    # backstep برای high
                    for b in range(1, backstep + 1):
                        j = i - b
                        if j >= 0 and high_map[j] != 0 and high_map[j] < val:
                            high_map[j] = 0.0
            
            if high_arr[i] == val:
                high_map[i] = val
        
        # --- مرحله 2: انتخاب نهایی نقاط ZigZag و ساخت legs ---
        last_high = 0.0
        last_low = 0.0
        last_high_pos = -1
        last_low_pos = -1
        search_mode = 0  # 0=Extremum, 1=Peak, -1=Bottom
        
        # ایجاد لیست‌های typed برای legs (محدودیت njit)
        leg_starts = TypedList.empty_list(types.int64)
        leg_ends = TypedList.empty_list(types.int64)
        leg_start_prices = TypedList.empty_list(types.float64)
        leg_end_prices = TypedList.empty_list(types.float64)
        leg_directions = TypedList.empty_list(types.int64)
        
        for i in range(depth - 1, n):
            if search_mode == 0:  # Extremum
                if last_low == 0.0 and last_high == 0.0:
                    if high_map[i] != 0.0:
                        last_high = high_map[i]
                        last_high_pos = i
                        zz_buffer[i] = last_high
                        state_series[i] = -1  # high point
                        search_mode = -1  # Bottom
                    
                    if low_map[i] != 0.0:
                        last_low = low_map[i]
                        last_low_pos = i
                        zz_buffer[i] = last_low
                        state_series[i] = 1   # low point
                        search_mode = 1  # Peak
            
            elif search_mode == 1:  # Peak - به دنبال Low
                if low_map[i] != 0.0 and low_map[i] < last_low and high_map[i] == 0.0:
                    if last_low_pos >= 0:
                        zz_buffer[last_low_pos] = 0.0
                        state_series[last_low_pos] = 0
                    
                    last_low = low_map[i]
                    last_low_pos = i
                    zz_buffer[i] = last_low
                    state_series[i] = 1  # low point
                
                elif high_map[i] != 0.0 and low_map[i] == 0.0:
                    last_high = high_map[i]
                    last_high_pos = i
                    zz_buffer[i] = last_high
                    state_series[i] = -1  # high point
                    search_mode = -1  # Bottom
                    
                    # ثبت leg (از low به high - روند صعودی)
                    if last_low_pos >= 0:
                        leg_starts.append(last_low_pos)
                        leg_ends.append(last_high_pos)
                        leg_start_prices.append(last_low)
                        leg_end_prices.append(last_high)
                        leg_directions.append(1)  # صعودی
            
            elif search_mode == -1:  # Bottom - به دنبال High
                if high_map[i] != 0.0 and high_map[i] > last_high and low_map[i] == 0.0:
                    if last_high_pos >= 0:
                        zz_buffer[last_high_pos] = 0.0
                        state_series[last_high_pos] = 0
                    
                    last_high = high_map[i]
                    last_high_pos = i
                    zz_buffer[i] = last_high
                    state_series[i] = -1  # high point
                
                elif low_map[i] != 0.0 and high_map[i] == 0.0:
                    last_low = low_map[i]
                    last_low_pos = i
                    zz_buffer[i] = last_low
                    state_series[i] = 1  # low point
                    search_mode = 1  # Peak
                    
                    # ثبت leg (از high به low - روند نزولی)
                    if last_high_pos >= 0:
                        leg_starts.append(last_high_pos)
                        leg_ends.append(last_low_pos)
                        leg_start_prices.append(last_high)
                        leg_end_prices.append(last_low)
                        leg_directions.append(-1)  # نزولی
        
        # تبدیل لیست‌های typed به آرایه‌های numpy برای خروجی
        legs_count = len(leg_starts)
        legs_array = np.zeros((legs_count, 5))
        for idx in range(legs_count):
            legs_array[idx, 0] = leg_starts[idx]
            legs_array[idx, 1] = leg_ends[idx]
            legs_array[idx, 2] = leg_start_prices[idx]
            legs_array[idx, 3] = leg_end_prices[idx]
            legs_array[idx, 4] = leg_directions[idx]
        
        return zz_buffer, high_map, low_map, state_series, legs_array
    
    #==========================================================================

    # تبدیل به آرایه‌های numpy با نوع مناسب برای njit
    high_arr = high.values.astype(np.float64)
    low_arr = low.values.astype(np.float64)
    
    # اجرای هسته njit
    zz_buffer, high_map, low_map, state_series, legs_array = _zigzag_njit_core(
        high_arr, low_arr, depth, deviation, backstep, point
    )
    
    # ایجاد Series خروجی
    zz_series = pd.Series(zz_buffer, index=high.index)
    
    # تبدیل legs_array به لیست دیکت‌شوند (فرمت اصلی)
    legs = []
    for i in range(len(legs_array)):
        legs.append({
            "start_idx": int(legs_array[i, 0]),
            "end_idx": int(legs_array[i, 1]),
            "start_price": float(legs_array[i, 2]),
            "end_price": float(legs_array[i, 3]),
            "direction": int(legs_array[i, 4])
        })
    
    # اضافه کردن تمام metadataهای مورد نیاز (فرمت دقیق نسخه اصلی)
    zz_series.attrs.update({
        "legs": legs,  # لیست legها
        "state_series": state_series.tolist(),  # آرایه stateها
        "high_map": high_map.tolist(),  # نقاط high
        "low_map": low_map.tolist(),    # نقاط low
        "params": {  # پارامترهای ورودی
            "depth": depth,
            "deviation": deviation,
            "backstep": backstep,
            "point": point
        }
    })
    
    return zz_series

