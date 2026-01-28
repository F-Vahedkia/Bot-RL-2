# f03_features/indicators/zigzag.py
# This file is checked and OK (1404/11/08)
# Check: By use of f15_testcheck/unit/test_zigzag.py

import numpy as np
import pandas as pd
from numba import njit
from typing import Literal

#==================================================================== 2 (==> OK & Final)
# Vectorized by Numpy
#====================================================================
def _zigzag_mql_numpy(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    point: float = 0.01,
) -> np.ndarray:
    
    n = len(high)

    # بافرهای اصلی
    zz_buffer    = np.zeros(n)                 # نقاط ZigZag
    high_map     = np.zeros(n)                 # نقاط کاندید برای سقف
    low_map      = np.zeros(n)                 # نقاط کاندید برای کف
    state_series = np.zeros(n, dtype=np.int8)  # state: -1=high, 1=low, 0=none
        
    # متغیرهای ردیابی
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0  # 0=Extremum, 1=Search next peak, -1=Search next bottom

    # --------------------------------------------------------------------
    # --- مرحله ۱: شناسایی نقاط high/low محلی کاملاً NumPy-vectorized ---
    # ایجاد ماتریس rolling window برای low و high
    # هر سطر = window از طول depth
    low_windows  = np.lib.stride_tricks.sliding_window_view(low, depth)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    # مقادیر min/max هر پنجره
    low_idx_in_window  = np.argmin(low_windows, axis=1)  # index در window
    high_idx_in_window = np.argmax(high_windows, axis=1)

    # اندیس واقعی در کل آرایه
    low_idx  = np.arange(depth - 1, n) - (depth - 1) + low_idx_in_window
    high_idx = np.arange(depth - 1, n) - (depth - 1) + high_idx_in_window

    # مقادیر local min/max
    low_vals  = low[low_idx]
    high_vals = high[high_idx]

    # اعمال deviation و حذف duplicate با backstep
    low_map  = np.zeros(n)
    high_map = np.zeros(n)

    for i in range(depth - 1, n):
        val = low_vals[i - (depth - 1)]
        # if val != last_low and (low[i] - val) <= deviation * point: #==========
        if (low[i] - val) <= deviation * point:
            # حذف backstep قبلی
            back_range = slice(max(0, i - backstep), i)
            mask = (low_map[back_range] != 0) & (low_map[back_range] > val)
            idx = np.where(mask)[0] + back_range.start
            low_map[idx] = 0.0
            # last_low = val #==========

        if low[i] == val:
            low_map[i] = val

        val = high_vals[i - (depth - 1)]
        # if val != last_high and (val - high[i]) <= deviation * point: #==========
        if (val - high[i]) <= deviation * point:
            back_range = slice(max(0, i - backstep), i)
            mask = (high_map[back_range] != 0) & (high_map[back_range] < val)
            idx = np.where(mask)[0] + back_range.start
            high_map[idx] = 0.0
            # last_high = val #==========

        if high[i] == val:
            high_map[i] = val
    
    # --------------------------------------------------------------------
    # --- مرحله ۲: انتخاب نهایی نقاط ZigZag با سرعت بالاتر -------------
    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0  # Extremum

    # پیدا کردن اندیس‌های candidate highs/lows
    high_idx = np.flatnonzero(high_map)
    low_idx  = np.flatnonzero(low_map)

    for i in range(depth - 1, n):
        h_val = high_map[i]
        l_val = low_map[i]

        if search_mode == 0:
            if last_high == 0.0 and h_val != 0.0:
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
                search_mode = -1

            elif last_low == 0.0 and l_val != 0.0:
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
                search_mode = 1

        elif search_mode == 1:  # Peak -> دنبال Low
            if l_val != 0.0 and l_val < last_low and h_val == 0.0:
                if last_low_pos >= 0:
                    # حذف مقدار قبلی به صورت masked
                    zz_buffer[last_low_pos] = 0.0
                    state_series[last_low_pos] = 0
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
            elif h_val != 0.0:
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
                search_mode = -1

        elif search_mode == -1:  # Bottom -> دنبال High
            if h_val != 0.0 and h_val > last_high and l_val == 0.0:
                if last_high_pos >= 0:
                    zz_buffer[last_high_pos] = 0.0
                    state_series[last_high_pos] = 0
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
            elif l_val != 0.0:
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
                search_mode = 1

    # ایجاد Series خروجی
    # zz_series = pd.Series(zz_buffer, index=high.index)
    
    # state_series شامل کد +1 یا -1 است که بیانگر جستجو برای کف یا سقف بعدی است
    # اگر در کندل جاری دارای مقدار -1 باشد، یعنی این کندل سقف است و اکسترمم بعدی باید کف باشد
    # و بالعکس
    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == +1, low , 0)
    
    return state_series, high_actual, low_actual

#==================================================================== 5 (==> OK & Final)
# Loop-wise and njit
#====================================================================
def _zigzag_mql_njit_loopwise(
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    point: float = 0.00001,
) -> np.ndarray:

    high_arr = high.astype(np.float64)
    low_arr  = low.astype(np.float64)

    @njit(cache=True)
    def _zigzag_core(high_arr, low_arr, depth, deviation, backstep, point) -> np.ndarray:
        n = high_arr.shape[0]

        zz_buffer   = np.zeros(n, dtype=np.float64)
        high_map    = np.zeros(n, dtype=np.float64)
        low_map     = np.zeros(n, dtype=np.float64)
        state       = np.zeros(n, dtype=np.int8)

        # =========================
        # Stage 1: local extrema
        # =========================
        for i in range(depth - 1, n):

            # ---- LOW -----------
            lo_idx = i
            lo_val = low_arr[i]
            for j in range(i - 1, i - depth, -1):
                if j < 0:
                    break
                if low_arr[j] < lo_val:
                    lo_val = low_arr[j]
                    lo_idx = j

            val = lo_val
            if (low_arr[i] - val) <= deviation * point:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and low_map[j] != 0.0 and low_map[j] > val:
                        low_map[j] = 0.0
            else:
                val = 0.0

            if low_arr[i] == val:
                low_map[i] = val

            # ---- HIGH ----------
            hi_idx = i
            hi_val = high_arr[i]
            for j in range(i - 1, i - depth, -1):
                if j < 0:
                    break
                if high_arr[j] > hi_val:
                    hi_val = high_arr[j]
                    hi_idx = j

            val = hi_val
            if (val - high_arr[i]) <= deviation * point:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and high_map[j] != 0.0 and high_map[j] < val:
                        high_map[j] = 0.0
            else:
                val = 0.0

            if high_arr[i] == val:
                high_map[i] = val

        # =========================
        # Stage 2: zigzag resolve
        # =========================
        last_high = 0.0
        last_low  = 0.0
        last_hp   = -1
        last_lp   = -1
        mode      = 0

        for i in range(depth - 1, n):

            if mode == 0:
                if last_high == 0.0 and high_map[i] != 0.0:
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                    mode = -1

                elif last_low == 0.0 and low_map[i] != 0.0:
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                    mode = 1

            elif mode == 1:
                if low_map[i] != 0.0 and low_map[i] < last_low and high_map[i] == 0.0:
                    if last_lp >= 0:
                        zz_buffer[last_lp] = 0.0
                        state[last_lp] = 0
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1

                elif high_map[i] != 0.0:
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                    mode = -1

            else:
                if high_map[i] != 0.0 and high_map[i] > last_high and low_map[i] == 0.0:
                    if last_hp >= 0:
                        zz_buffer[last_hp] = 0.0
                        state[last_hp] = 0
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1

                elif low_map[i] != 0.0:
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                    mode = 1

        return state, zz_buffer, high_map, low_map

    
    state, zz, hmap, lmap = _zigzag_core(
        high_arr, low_arr, depth, deviation, backstep, point
    )

    high_actual = np.zeros_like(high_arr)
    low_actual  = np.zeros_like(low_arr)

    for i in range(state.shape[0]):
        if state[i] == -1:
            high_actual[i] = high_arr[i]
        elif state[i] == 1:
            low_actual[i] = low_arr[i]

    return state, high_actual, low_actual

#====================================================================
# Wrapper function to choose between njit and non-njit based on data size
#==================================================================== 6
def zigzag(
    high: pd.Series,
    low: pd.Series,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
) -> pd.DataFrame:
    
    idx = high.index
    high_np = np.ascontiguousarray(high.values, dtype=np.float64)
    low_np  = np.ascontiguousarray(low.values,  dtype=np.float64)

    if len(high_np) <= 1_000_000:
        state, high_actual, low_actual = \
            _zigzag_mql_numpy(
                # dt_np,
                high_np, low_np,
                depth, deviation, backstep, point
            )
    else:
        state, high_actual, low_actual = \
            _zigzag_mql_njit_loopwise(
                # dt_np,
                high_np, low_np,
                depth, deviation, backstep, point
            )

    return pd.DataFrame(
        index=idx,
        data={
            "state": -state,
            # در مورد state:
            # از اینجا به بعد وضعیت سوئینگ را از (جستجوی آینده) به (وضعیت فعلی) اصلاح میکنیم
            "high" : high_actual,
            "low"  : low_actual,
            })

#====================================================================
# MULTI-TIMEFRAME ADAPTER
#====================================================================
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

    # --- get metadata from HTF zigzag ---
    state_series = zz_htf.attrs.get("state_series", [])  # -1: high, 1: low, 0: none
    htf_timestamps = list(zz_htf.index)
            
    # --- prepare LTF container ---
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}",
        dtype=np.float32
    )

    # --- project HTF swings onto LTF ----------- start of old part
    # for ts, signal in zz_htf.items():
    #     if signal == 0:
    #         continue
    #     if ts not in zz_ltf.index:
    #         ts = zz_ltf.index[zz_ltf.index.get_indexer([ts], method="ffill")[0]]
    #     if mode == "last":
    #         zz_ltf.loc[ts] = signal
    #     elif mode == "forward_fill":
    #         zz_ltf.loc[ts:] = signal
    #
    # --- attach metadata ---
    # zz_ltf.attrs["source_tf"] = tf_higher
    # zz_ltf.attrs["mode"] = mode
    # zz_ltf.attrs["htf_legs"] = zz_htf.attrs.get("legs", [])

    # return zz_ltf
    # ------------------------------------------- end of old part

    
    # --- project HTF swings onto LTF ---
    if mode == "last":
        # فقط کندل LTF متناظر با کندل HTF علامت می‌خورد
        for ts_htf, signal in zip(htf_timestamps, state_series):
            if signal == 0:
                continue
            
            # پیدا کردن کندل LTF مربوطه (آخرین کندل LTF قبل از یا در زمان ts_htf)
            ltf_idx = zz_ltf.index[zz_ltf.index <= ts_htf]
            if len(ltf_idx) > 0:
                last_ltf_idx = ltf_idx[-1]
                zz_ltf.loc[last_ltf_idx] = signal
    
    elif mode == "forward_fill":
        # Forward fill تا ظهور swing بعدی
        prev_signal = 0
        prev_ts = None
        
        # ایجاد لیست زمانی از swingها
        swing_points = []
        for ts_htf, signal in zip(htf_timestamps, state_series):
            if signal != 0:
                swing_points.append((ts_htf, signal))
        
        # پروجکشن forward fill
        for i in range(len(swing_points)):
            current_ts, current_signal = swing_points[i]
            
            # پیدا کردن محدوده زمانی برای این swing
            start_idx = None
            if i > 0:
                # شروع از کندل بعد از swing قبلی
                prev_ts, _ = swing_points[i-1]
                ltf_after_prev = zz_ltf.index[zz_ltf.index > prev_ts]
                if len(ltf_after_prev) > 0:
                    start_idx = ltf_after_prev[0]
            else:
                # اولین swing - از ابتدای داده شروع کن
                start_idx = zz_ltf.index[0]
            
            # پایان در کندل مربوط به swing جاری
            ltf_up_to_current = zz_ltf.index[zz_ltf.index <= current_ts]
            if len(ltf_up_to_current) > 0:
                end_idx = ltf_up_to_current[-1]
                
                if start_idx is not None:
                    # اعمال forward fill در محدوده
                    mask = (zz_ltf.index >= start_idx) & (zz_ltf.index <= end_idx)
                    zz_ltf.loc[mask] = current_signal
    
    # --- attach metadata ---
    zz_ltf.attrs["source_tf"] = tf_higher
    zz_ltf.attrs["mode"] = mode
    zz_ltf.attrs["htf_legs"] = zz_htf.attrs.get("legs", [])
    zz_ltf.attrs["htf_state_series"] = state_series
    zz_ltf.attrs["htf_timestamps"] = htf_timestamps
    zz_ltf.attrs["params"] = {
        "depth": depth,
        "deviation": deviation,
        "backstep": backstep,
        "point": point
    }

    return zz_ltf

#====================================================================
