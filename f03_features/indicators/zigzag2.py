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
def _zigzag_mql_numpy(       # Depricated
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

    # ✨ جدید: confirmed و developing
    # confirmed_at    = np.full(n, -1, dtype=np.int32)
    # developing_leg  = np.zeros(n, dtype=np.int8)  # +1/-1

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
    high_idx = np.flatnonzero(high_map)   # Deleted
    low_idx  = np.flatnonzero(low_map)    # Deleted

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

   
    # state_series شامل کد +1 یا -1 است که بیانگر جستجو برای کف یا سقف بعدی است
    # اگر در کندل جاری دارای مقدار -1 باشد، یعنی این کندل سقف است و اکسترمم بعدی باید کف باشد
    # و بالعکس
    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == +1, low , 0)
    
    return state_series, high_actual, low_actual, #confirmed_at, developing_leg


def _zigzag_mql_numpy_complete(       # New
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    point: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numpy-based ZigZag with confirmed and developing leg tracking (retrospective confirmed_at).

    Returns:
        state_series: +1/-1 for lows/highs
        high_actual: values at highs
        low_actual: values at lows
        confirmed_at: index where pivot is confirmed (retrospective)
        developing_leg: +1/-1 provisional leg before confirmation
    """
    
    n = len(high)

    # بافرهای اصلی
    zz_buffer    = np.zeros(n)                 # نقاط ZigZag
    high_map     = np.zeros(n)                 # candidate highs
    low_map      = np.zeros(n)                 # candidate lows
    state_series = np.zeros(n, dtype=np.int8)  # -1=high, 1=low, 0=none

    # confirmed و developing
    confirmed_at   = np.full(n, -1, dtype=np.int32)
    developing_leg = np.zeros(n, dtype=np.int8)

    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1
    search_mode = 0  # 0=Extremum, 1=Search next peak, -1=Search next bottom

    # ------------------------------
    # مرحله ۱: محاسبه local highs/lows
    low_windows  = np.lib.stride_tricks.sliding_window_view(low, depth)
    high_windows = np.lib.stride_tricks.sliding_window_view(high, depth)

    low_idx_in_window  = np.argmin(low_windows, axis=1)
    high_idx_in_window = np.argmax(high_windows, axis=1)

    low_idx  = np.arange(depth - 1, n) - (depth - 1) + low_idx_in_window
    high_idx = np.arange(depth - 1, n) - (depth - 1) + high_idx_in_window

    low_vals  = low[low_idx]
    high_vals = high[high_idx]

    low_map  = np.zeros(n)
    high_map = np.zeros(n)

    for i in range(depth - 1, n):
        # LOW
        val = low_vals[i - (depth - 1)]
        if (low[i] - val) <= deviation * point:
            back_range = slice(max(0, i - backstep), i)
            mask = (low_map[back_range] != 0) & (low_map[back_range] > val)
            idx = np.where(mask)[0] + back_range.start
            low_map[idx] = 0.0
        if low[i] == val:
            low_map[i] = val

        # HIGH
        val = high_vals[i - (depth - 1)]
        if (val - high[i]) <= deviation * point:
            back_range = slice(max(0, i - backstep), i)
            mask = (high_map[back_range] != 0) & (high_map[back_range] < val)
            idx = np.where(mask)[0] + back_range.start
            high_map[idx] = 0.0
        if high[i] == val:
            high_map[i] = val

    # ------------------------------
    # مرحله ۲: resolve zigzag + developing (retrospective confirmed_at)
    for i in range(depth - 1, n):
        h_val = high_map[i]
        l_val = low_map[i]

        # developing_leg
        if search_mode == 1:
            developing_leg[i] = 1
        elif search_mode == -1:
            developing_leg[i] = -1

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
                    confirmed_at[last_low_pos] = i
                    zz_buffer[last_low_pos] = 0.0
                    state_series[last_low_pos] = 0
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
            elif h_val != 0.0:
                if last_low_pos >= 0:
                    confirmed_at[last_low_pos] = i
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
                search_mode = -1

        elif search_mode == -1:  # Bottom -> دنبال High
            if h_val != 0.0 and h_val > last_high and l_val == 0.0:
                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i
                    zz_buffer[last_high_pos] = 0.0
                    state_series[last_high_pos] = 0
                last_high = h_val
                last_high_pos = i
                zz_buffer[i] = h_val
                state_series[i] = -1
            elif l_val != 0.0:
                if last_high_pos >= 0:
                    confirmed_at[last_high_pos] = i
                last_low = l_val
                last_low_pos = i
                zz_buffer[i] = l_val
                state_series[i] = 1
                search_mode = 1

    high_actual = np.where(state_series == -1, high, 0)
    low_actual  = np.where(state_series == +1, low , 0)

    return state_series, high_actual, low_actual, confirmed_at, developing_leg


#==================================================================== 5 (==> OK & Final)
# Loop-wise and njit
#====================================================================
def _zigzag_mql_njit_loopwise(       # Depricated
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

    # state شامل کد +1 یا -1 است که بیانگر جستجو برای کف یا سقف بعدی است
    # اگر در کندل جاری دارای مقدار -1 باشد، یعنی این کندل سقف است و اکسترمم بعدی باید کف باشد
    # و بالعکس
    return state, high_actual, low_actual


def _zigzag_mql_njit_loopwise_complete(       # New
    high: np.ndarray,
    low: np.ndarray,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 3,
    point: float = 0.00001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    high_arr = high.astype(np.float64)
    low_arr  = low.astype(np.float64)

    @njit(cache=True)
    def _zigzag_core(high_arr, low_arr, depth, deviation, backstep, point):
        n = high_arr.shape[0]

        zz_buffer   = np.zeros(n, dtype=np.float64)
        high_map    = np.zeros(n, dtype=np.float64)
        low_map     = np.zeros(n, dtype=np.float64)
        state       = np.zeros(n, dtype=np.int8)
        confirmed_at = np.full(n, -1, dtype=np.int32)
        developing_leg = np.zeros(n, dtype=np.int8)

        # ------------------
        # Stage 1: local extrema
        # ------------------
        for i in range(depth - 1, n):
            # LOW
            lo_val = low_arr[i]
            lo_idx = i
            for j in range(i - 1, i - depth, -1):
                if j < 0: break
                if low_arr[j] < lo_val:
                    lo_val = low_arr[j]
                    lo_idx = j

            if (low_arr[i] - lo_val) <= deviation * point:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and low_map[j] != 0.0 and low_map[j] > lo_val:
                        low_map[j] = 0.0
            else:
                lo_val = 0.0

            if low_arr[i] == lo_val:
                low_map[i] = lo_val

            # HIGH
            hi_val = high_arr[i]
            hi_idx = i
            for j in range(i - 1, i - depth, -1):
                if j < 0: break
                if high_arr[j] > hi_val:
                    hi_val = high_arr[j]
                    hi_idx = j

            if (hi_val - high_arr[i]) <= deviation * point:
                for b in range(1, backstep + 1):
                    j = i - b
                    if j >= 0 and high_map[j] != 0.0 and high_map[j] < hi_val:
                        high_map[j] = 0.0
            else:
                hi_val = 0.0

            if high_arr[i] == hi_val:
                high_map[i] = hi_val

        # ------------------
        # Stage 2: zigzag resolve + developing + retrospective confirmed_at
        # ------------------
        last_high = 0.0
        last_low = 0.0
        last_hp = -1
        last_lp = -1
        mode = 0  # 0=Extremum, 1=Search Low, -1=Search High

        for i in range(depth - 1, n):
            # developing_leg
            if mode == 1:
                developing_leg[i] = 1
            elif mode == -1:
                developing_leg[i] = -1

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
                        confirmed_at[last_lp] = i
                        zz_buffer[last_lp] = 0.0
                        state[last_lp] = 0
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                elif high_map[i] != 0.0:
                    if last_lp >= 0:
                        confirmed_at[last_lp] = i
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                    mode = -1

            elif mode == -1:
                if high_map[i] != 0.0 and high_map[i] > last_high and low_map[i] == 0.0:
                    if last_hp >= 0:
                        confirmed_at[last_hp] = i
                        zz_buffer[last_hp] = 0.0
                        state[last_hp] = 0
                    last_high = high_map[i]
                    last_hp = i
                    zz_buffer[i] = last_high
                    state[i] = -1
                elif low_map[i] != 0.0:
                    if last_hp >= 0:
                        confirmed_at[last_hp] = i
                    last_low = low_map[i]
                    last_lp = i
                    zz_buffer[i] = last_low
                    state[i] = 1
                    mode = 1

        return state, zz_buffer, high_map, low_map, confirmed_at, developing_leg

    state, zz, hmap, lmap, confirmed_at, developing_leg = _zigzag_core(
        high_arr, low_arr, depth, deviation, backstep, point
    )

    high_actual = np.zeros_like(high_arr)
    low_actual  = np.zeros_like(low_arr)

    for i in range(state.shape[0]):
        if state[i] == -1:
            high_actual[i] = high_arr[i]
        elif state[i] == 1:
            low_actual[i] = low_arr[i]

    return state, high_actual, low_actual, confirmed_at, developing_leg


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
    addmeta: bool = True,
) -> pd.DataFrame:
    
    idx = high.index
    high_np = np.ascontiguousarray(high.values, dtype=np.float64)
    low_np  = np.ascontiguousarray(low.values,  dtype=np.float64)
    bytes_used = high_np.nbytes + low_np.nbytes

    # Using "bytes_used <= 1_280_000"  isntead of "len(high_np) <= 80_000"
    if bytes_used <= 1_280_000:
        state, high_actual, low_actual, confirmed_at, developing_leg= \
            _zigzag_mql_numpy_complete(
                high_np, low_np, depth, deviation, backstep, point
            )
    else:
        state, high_actual, low_actual, confirmed_at, developing_leg = \
            _zigzag_mql_njit_loopwise_complete(
                high_np, low_np, depth, deviation, backstep, point
            )

    # --- state correction ----------------------------------------------------
    # از اینجا به بعد وضعیت سوئینگ را از (جستجوی آینده) به (وضعیت فعلی) اصلاح میکنیم
    # یعنی در سقفها برابر با +1 است و در کفها برابر با -1 است    
    state = -1 * state
    
    # --- main result of this function ----------------------------------------
    zz_df = pd.DataFrame(
        index=idx,
        data={
            "state": state,        # at HIGHs: state = +1, at LOWs: state = -1
            "high" : high_actual,
            "low"  : low_actual,
            "confirmed_at": confirmed_at,
            "developing_leg": developing_leg,
        }
    )

    if not addmeta:
        return zz_df
    
    # --- build zigzag legs metadata ------------------------------------------
    # leg = فاصله بین دو اکسترمم متوالی با جهت مشخص
    legs = []
    # precompute positional map (O(1) lookup)
    pos_map = {ts: i for i, ts in enumerate(idx)}

    last_idx = None
    last_state = 0
    for idx_val, state_val in zip(idx, state):
        if state_val == 0:
            continue

        if last_idx is not None:
            start_pos = pos_map[last_idx]
            end_pos   = pos_map[idx_val]

            if end_pos > start_pos:
                legs.append({
                    "start_ts": last_idx,               # timestamp
                    "end_ts": idx_val,                  # timestamp
                    "direction": int(state_val),        # -1/+1
                    "start_pos": int(start_pos),                  # pos
                    "end_pos": int(end_pos),                      # pos
                    "start_confirmed_at": int(confirmed_at[start_pos]),   # pos
                    "end_confirmed_at": int(confirmed_at[end_pos]),       # pos
                    "start_developing_leg": int(developing_leg[start_pos]),      # pos
                    "end_developing_leg": int(developing_leg[end_pos]),          # pos
                })

        last_idx = idx_val
        last_state = state_val

    # --- attach metadata -----------------------------------------------------
    zz_df.attrs["legs"] = legs
    return zz_df

#====================================================================
# MULTI-TIMEFRAME ZIGZAG ADAPTER
"""
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
#====================================================================
# برای ساخت لگهای ساختار بازار باید از متادیتای ضمیمه شده به خروجی این تابع استفاده کنیم
# چرا که اگر از خروجی اصلی تابع استفاده کنیم، نمیتوانیم لگهای نامعتبر را بحساب نیاوریم
# و در نتیجه لگهای نامعتبر هم به اجبار به عنوان ساختار بازار در نظر گرفته میشوند

def zigzag_mtf_adapter_old(
    high: pd.Series,
    low: pd.Series,
    tf_higher: str,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
    mode: Literal["last", "forward_fill"] = "forward_fill",
    extend_last_leg: bool = False,
) -> pd.Series:
    
    # --- validation ----------------------------------------------------------
    if not isinstance(high.index, pd.DatetimeIndex) or not isinstance(low.index, pd.DatetimeIndex):
        raise ValueError("high/low must have DatetimeIndex")
    
    # --- build HTF OHLC ------------------------------------------------------
    htf = (
        pd.DataFrame({"high": high, "low": low})
            .resample(tf_higher)
            .agg({"high": "max", "low": "min"})
            .dropna()
    )
    # --- run original ZigZag on HTF ------------------------------------------
    zz_htf = zigzag(
        high=htf["high"],
        low=htf["low"],
        depth=depth, deviation=deviation, backstep=backstep, point=point,
        addmeta=True,
    )

    # --- get metadata from HTF zigzag ----------------------------------------
    htf_state_series = zz_htf["state"]
    # htf_state_series.to_csv("1__htf_state_series.csv")

    # --- prepare LTF container -----------------------------------------------
    # ساختن یک دیتافریم خالی، البته با اندکسهای زمانی معلوم
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}",
        dtype=np.float32
    )

    # --- project HTF swings onto LTF -----------------------------------------
    # if mode == "last":
    #     for ts_htf, signal in htf_state_series.items():
    #         # if signal != 0 and ts_htf in zz_ltf.index:
    #         if signal != 0 and ts_htf <= zz_ltf.index[-1]:
    #             zz_ltf.at[ts_htf] = signal

    if mode == "last":
        for ts_htf, signal in htf_state_series.items():
            if signal != 0:
                # فقط چک کن که تایم‌استمپ در ایندکس LTF وجود داره یا نه
                if ts_htf in zz_ltf.index:
                    zz_ltf.at[ts_htf] = signal
                # یا اگر می‌خواهی نزدیکترین تایم LTF رو پیدا کنی:
                # else:
                #     nearest_idx = zz_ltf.index.get_indexer([ts_htf], method='pad')[0]
                #     zz_ltf.iloc[nearest_idx] = signal


    elif mode == "forward_fill":
        # --- collect valid HTF swings (exact timestamp only)
        swings = [(ts, sig) for ts, sig in htf_state_series.items()
                if sig != 0 and ts in zz_ltf.index]

        if swings:
            # --- forward fill between consecutive swings
            for (ts0, sig0), (ts1, _) in zip(swings[:-1], swings[1:]):
                zz_ltf.loc[ts0:ts1] = sig0

            # --- fill after end of last leg until end of series with same signal ---
            # جهت و سوی آخرین لگ، تا پایان سری داده ها ادامه می یابد
            if extend_last_leg and swings:
                # last_ts, last_sig = swings[-1]
                last_ts, _        = swings[-1]
                _      , last_sig = swings[-2]    

                # _ts_end = zz_ltf.index[-1]                     # part-1
                # zz_ltf.loc[last_ts:_ts_end] = last_sig         # part-1

                # last_loc = zz_ltf.index.get_indexer([last_ts])[0]    # part-2
                # if last_loc != -1:                                   # part-2
                #     zz_ltf.iloc[last_loc:] = last_sig                # part-2

                if last_ts in zz_ltf.index:                       # part-3
                    last_loc = zz_ltf.index.get_loc(last_ts)      # part-3
                    zz_ltf.iloc[last_loc:] = last_sig             # part-3




    # zz_ltf.to_csv("2__after_mod.csv")
    
    # --- project HTF legs onto LTF (metadata only, no signal logic) ----------
    htf_legs = zz_htf.attrs.get("legs", [])
    ltf_index = zz_ltf.index

    ltf_legs = []

    for leg in htf_legs:
        start_ts = leg["start_ts"]
        end_ts   = leg["end_ts"]

        if start_ts not in ltf_index or end_ts not in ltf_index:
            continue

        start_pos = ltf_index.get_loc(start_ts)
        end_pos   = ltf_index.get_loc(end_ts)

        if end_pos <= start_pos:
            continue

        ltf_legs.append({
            "start_ts": start_ts,
            "end_ts": end_ts,
            "direction": leg["direction"],
            "start_pos": int(start_pos),
            "end_pos": int(end_pos),
        })
    # --- FIX: Add the last leg as a metadata leg ---------------------------
    # This ensures the final leg is also included in the metadata
    if mode == "forward_fill" and len(swings) >= 2:
        # Get the last two swings
        (ts0, sig0), (ts1, sig1) = swings[-2], swings[-1]
        
        if ts0 in ltf_index and ts1 in ltf_index:
            start_pos = ltf_index.get_loc(ts0)
            end_pos = ltf_index.get_loc(ts1)
            
            if end_pos > start_pos:
                ltf_legs.append({
                    "start_ts": ts0,
                    "end_ts": ts1,
                    "direction": int(sig1),  # direction of the end point
                    "start_pos": int(start_pos),
                    "end_pos": int(end_pos),
                })

    # --- attach metadata to zz_ltf -------------------------------------------
    zz_ltf.attrs["legs"] = ltf_legs

    return zz_ltf

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
    extend_last_leg: bool = False,
) -> pd.Series:
    
    # --- Validation ---------------------------------------------------------- ok1
    if not isinstance(high.index, pd.DatetimeIndex) or not isinstance(low.index, pd.DatetimeIndex):
        raise ValueError("high/low must have DatetimeIndex")
    
    # --- Build HTF OHLC ------------------------------------------------------ ok2
    htf = (
        pd.DataFrame({"high": high, "low": low})
            .resample(tf_higher)
            .agg({"high": "max", "low": "min"})
            .dropna()
    )
    # --- Run original ZigZag on HTF ------------------------------------------ ok3
    zz_htf = zigzag(
        high=htf["high"],
        low=htf["low"],
        depth=depth, deviation=deviation, backstep=backstep, point=point,
        addmeta=True,
    )
    # --- for debuging- part-1 ----------------------------
    # print("===== Debuging Part 1 =====")
    # print(f" ---> zz_htf shape = {zz_htf.shape}")               # 2001*5
    # print(f" ---> zz_htf columns = {zz_htf.columns.tolist()}")  # 'state', 'high', 'low', 'confirmed_at', 'developing_leg'

    # htf_legs = zz_htf.attrs.get("legs", [])
    # df_legs = pd.DataFrame(htf_legs)

    # print(f" ---> HTF legs metadata shape = {df_legs.shape}")    # 91*9 : 92=extremums , 91=legs
    # print(f" ---> df_legs columns = {df_legs.columns.tolist()}") # 'start_ts', 'end_ts', 'direction', 'start_pos',
    #                                                              # 'end_pos', 'start_confirmed_at', 'end_confirmed_at',
    #                                                              # 'start_developing_leg', 'end_developing_leg'

    # zz_htf.to_csv("1__zz_htf.csv")
    # df_legs.to_csv("2__df_legs.csv")
    
    # --- prepare LTF container ----------------------------------------------- ok4
    # build an empty series with known time index, for "final main result" (container_1)
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}",
        dtype=np.float32
    )

    # build an empty series with known time index, for "developing_leg" (container_2)
    zz_ltf_developing = pd.Series(
        data=0,
        index=high.index,
        name="zigzag_mtf_developing_leg",
        dtype=np.int8,
    )

    # --- get metadata from HTF zigzag ---------------------------------------- ok
    ltf_index = zz_ltf.index                               # (10001,) pandas...DatatimeIndex
    htf_index = zz_htf.index                               # (2001,)  pandas...DatatimeIndex
    htf_state_series = zz_htf["state"         ].values     # (2001,) numpy.ndarray   contain: -1/0/+1
    htf_confirmed_at = zz_htf["confirmed_at"  ].values     # (2001,) numpy.ndarray   contain: pos
    htf_developing_leg = zz_htf["developing_leg"].values   # (2001,) numpy.ndarray   contain: pos
    
    # --- for debuging- part-2 ----------------------------
    # print("===== Debuging Part 2 =====")    
    # print(f" ---> htf_index shape = {htf_index.shape} and it's type is {type(htf_index)}")
    # print(f" ---> htf_state_series shape = {htf_state_series.shape} and it's type is {type(htf_state_series)}")
    # print(f" ---> htf_confirmed_at shape = {htf_confirmed_at.shape} and it's type is {type(htf_confirmed_at)}")
    # print(f" ---> htf_developing_leg shape = {htf_developing_leg.shape} and it's type is {type(htf_developing_leg)}")
    
    # =========================================================================
    # Project HTF swings onto LTF
    # ========================================================================= ok
    if mode == "last":
        # pos_cnt=0            # debug
        # signal_cnt=0         # debug
        # my_list_pos=[]       # debug
        # my_list_confirmed=[] # debug

        for pos, signal in enumerate(htf_state_series):    # pos=0 to pos=2000  (loop iteration = 2001)
            # pos_cnt +=1 # debug
            if signal == 0:
                continue
            # signal_cnt+=1 # debug
            # my_list_pos.append({"pos":pos, "signal":signal}) # debug , is OK

            confirmed_pos = htf_confirmed_at[pos]
            if confirmed_pos < 0:
                continue
            confirmed_ts = htf_index[confirmed_pos]

            # --- استفاده از searchsorted برای نزدیکترین LTF ---
            loc = ltf_index.searchsorted(confirmed_ts, side="right") - 1     # loc=0 to loc=9999
            if 0 <= loc < len(zz_ltf):
                zz_ltf.iloc[loc] = signal
            
            # my_list_confirmed.append({"confirmed_pos":confirmed_pos,     # debug , is OK
            #                           "confirmed_ts":confirmed_ts,       # debug , is OK
            #                           "ltf_loc":loc,                     # debug , is OK
            #                           "ltf_ts":ltf_index[loc],           # debug , is OK
            #                           "signal":signal,                   # debug , is OK
            #                           })                                 # debug , is OK

        # --- for debuging- part-3 ----------------------------
        # print("===== Debuging Part 3 =====")
        # print(f"at mode=last, pos_cnt: {pos_cnt}")                    # debug  Len=2001
        # print(f"at mode=last, signal_cnt: {signal_cnt}")              # debug  Len=92
        # pd.DataFrame(my_list_pos)      .to_csv("3__my_list_pos.csv")  # debug , is OK
        # pd.DataFrame(my_list_confirmed).to_csv("4__my_list_conf.csv") # debug

    elif mode == "forward_fill":
        swings = []

        # pos_cnt=0            # debug
        # signal_cnt=0         # debug
        # my_list_pos=[]       # debug
        # my_list_confirmed=[] # debug
        # swings_cnt =0        # debug

        for pos, signal in enumerate(htf_state_series):    # pos=0 to pos=2000  (loop iteration = 2001)
            # pos_cnt +=1 # debug
            if signal == 0:
                continue
            # signal_cnt+=1 # debug
            # my_list_pos.append({"pos":pos, "signal":signal}) #debug

            confirmed_pos = htf_confirmed_at[pos]
            if confirmed_pos < 0:
                continue
            confirmed_ts = htf_index[confirmed_pos]

            # --- استفاده از searchsorted برای نزدیکترین LTF ---
            loc = ltf_index.searchsorted(confirmed_ts, side="right") - 1     # loc=0 to loc=9999
            # swings_cnt +=1                                         # debug
            # print(f" {swings_cnt}---> loc={loc}, signal={signal}") # debug
            if 0 <= loc < len(zz_ltf):
                swings.append((ltf_index[loc], signal))
            
            # my_list_confirmed.append({"confirmed_pos":confirmed_pos,     # debug , is OK
            #                           "confirmed_ts":confirmed_ts,       # debug , is OK
            #                           "ltf_loc":loc,                     # debug , is OK
            #                           "ltf_ts":ltf_index[loc],           # debug , is OK
            #                           "signal":signal,                   # debug , is OK
            #                           })                                 # debug , is OK

        # --- for debuging- part-3 ----------------------------
        # print("===== Debuging Part 3 =====")
        # print(f"at mode=forward_fill, pos_cnt: {pos_cnt}")            # debug  Len=2001
        # print(f"at mode=forward_fill, signal_cnt: {signal_cnt}")      # debug  Len=92
        # pd.DataFrame(my_list_pos)      .to_csv("3__my_list_pos.csv")  # debug
        # pd.DataFrame(my_list_confirmed).to_csv("4__my_list_conf.csv") # debug
        
        # pd.DataFrame(swings).to_csv("5__swings.csv")

        if swings:       # "swings": it contains ("confirmed_ts","signal")
            if len(swings) == 1:
                ts0, sig0 = swings[0]
                start_idx = ltf_index.get_loc(ts0)
                zz_ltf.iloc[start_idx:] = sig0
            else:
                for (ts0, sig0), (ts1, _) in zip(swings[:-1], swings[1:]):
                    start_idx = ltf_index.get_loc(ts0)
                    end_idx   = ltf_index.get_loc(ts1)
                    zz_ltf.iloc[start_idx:end_idx] = sig0  # end exclusive

                # fill after last swing if extend_last_leg
                if extend_last_leg:
                    last_ts, last_sig = swings[-1]
                    last_loc = ltf_index.get_loc(last_ts)
                    zz_ltf.iloc[last_loc:] = last_sig
    
    # print(" finished modes") # debug
    # zz_ltf.to_csv("__after_mod.csv")

    # =========================================================================
    # causal forward propagation of developing_leg 
    # ========================================================================= ok
    for pos, leg_dir in enumerate(htf_developing_leg):   # htf_developing_leg:  (2001,) numpy.ndarray   contain: pos
        if leg_dir == 0:
            continue

        confirmed_pos = htf_confirmed_at[pos]
        if confirmed_pos < 0:
            continue

        confirmed_ts = htf_index[confirmed_pos] + pd.Timedelta(tf_higher)
        if confirmed_ts not in ltf_index:
            continue
        # --- getting "start_loc" ---
        start_loc = ltf_index.get_loc(confirmed_ts)

        # --- finding "end_loc" ---
        end_loc = len(zz_ltf_developing)  # مقدار پیش‌فرض
        next_conf_pos = pos + 1
        while next_conf_pos < len(htf_confirmed_at) and htf_confirmed_at[next_conf_pos] < 0:
            next_conf_pos += 1
        if next_conf_pos < len(htf_confirmed_at):
            next_ts = htf_index[htf_confirmed_at[next_conf_pos]]
            if next_ts in ltf_index:
                end_loc = ltf_index.get_loc(next_ts)
        
        end_loc = min(end_loc, len(zz_ltf_developing))

        if start_loc < end_loc:
            zz_ltf_developing.iloc[start_loc:end_loc] = leg_dir

    # =========================================================================
    # project HTF legs onto LTF (metadata only, no signal logic)
    # =========================================================================
    htf_legs = zz_htf.attrs.get("legs", [])
    ltf_legs = []
    """ htf_legs: 
              'start_ts',             'end_ts',             # timestamp
              'direction',                                  # -1/+1
              'start_pos',            'end_pos',            # pos higher timeframe
              'start_confirmed_at',   'end_confirmed_at',   # pos htf
              'start_developing_leg', 'end_developing_leg'  # pos htf
    """
    
    for leg in htf_legs:
        start_ts = leg["start_ts"]
        end_ts   = leg["end_ts"]

        # --- map start/end to LTF --------------------------------------------
        start_loc = ltf_index.searchsorted(start_ts, side="right") - 1
        end_loc   = ltf_index.searchsorted(end_ts,   side="right") - 1

        if not (0 <= start_loc < len(ltf_index)):
            continue
        if not (0 <= end_loc < len(ltf_index)):
            continue
        if end_loc <= start_loc:
            continue

        # --- map confirmed_at HTF -> LTF -------------------------------------
        start_conf_ltf = -1
        end_conf_ltf   = -1

        start_conf_htf = leg.get("start_confirmed_at", -1)
        end_conf_htf   = leg.get("end_confirmed_at", -1)
        
        if start_conf_htf >= 0:
            start_conf_ts = htf_index[start_conf_htf] + pd.Timedelta(tf_higher)
            loc = ltf_index.searchsorted(start_conf_ts, side="right") - 1
            if 0 <= loc < len(ltf_index):
                start_conf_ltf = int(loc)

        if end_conf_htf >= 0:
            end_conf_ts = htf_index[end_conf_htf]
            loc = ltf_index.searchsorted(end_conf_ts, side="right") - 1
            if 0 <= loc < len(ltf_index):
                end_conf_ltf = int(loc)

        # --- map developing_leg HTF -> LTF --------------------------------------------- my version
        start_dev_ltf = 0
        end_dev_ltf   = 0

        loc = ltf_index.searchsorted(start_ts + pd.Timedelta(tf_higher), side="right") - 1
        if 0 <= loc < len(ltf_index):
            start_dev_ltf = int(loc)
            
        loc = ltf_index.searchsorted(end_ts - pd.Timedelta(tf_higher), side="right") - 1
        if 0 <= loc < len(ltf_index):
            end_dev_ltf = int(loc)

        # --- store in metadata LTF -------------------------------------------
        ltf_legs.append({
            "start_ltf_pos": int(start_loc),
            "end_ltf_pos": int(end_loc),
            "direction": int(leg["direction"]),
            "start_confirmed_ltf_pos": int(start_conf_ltf),
            "end_confirmed_ltf_pos": int(end_conf_ltf),
            "start_developing_ltf_pos": int(start_dev_ltf),
            "end_developing_ltf_pos": int(end_dev_ltf),
        })

    base_series = zz_ltf
    zz_ltf = pd.DataFrame({
        f"zigzag_mtf_{tf_higher}": base_series,
        "zigzag_mtf_developing_leg": zz_ltf_developing
    }, index=high.index)

    # --- attach metadata to zz_ltf -------------------------------------------
    zz_ltf.attrs["legs"] = ltf_legs

    return zz_ltf




def zigzag_mtf_adapter(
    high: pd.Series,
    low: pd.Series,
    tf_higher: str,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01,
    mode: Literal["last", "forward_fill"] = "forward_fill",
    extend_last_leg: bool = False,
) -> pd.Series:
    
    # --- Validation ---------------------------------------------------------- ok1
    if not isinstance(high.index, pd.DatetimeIndex) or not isinstance(low.index, pd.DatetimeIndex):
        raise ValueError("high/low must have DatetimeIndex")
    
    # --- Build HTF OHLC ------------------------------------------------------ ok2
    htf = (
        pd.DataFrame({"high": high, "low": low})
            .resample(tf_higher)
            .agg({"high": "max", "low": "min"})
            .dropna()
    )
    # --- Run original ZigZag on HTF ------------------------------------------ ok3
    zz_htf = zigzag(
        high=htf["high"],
        low=htf["low"],
        depth=depth, deviation=deviation, backstep=backstep, point=point,
        addmeta=True,
    )
    
    # --- prepare LTF container ----------------------------------------------- ok4
    # build an empty series with known time index, for "final main result" (container_1)
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}",
        dtype=np.float32
    )

    # build an empty series with known time index, for "developing_leg" (container_2)
    zz_ltf_developing = pd.Series(
        data=0,
        index=high.index,
        name="zigzag_mtf_developing_leg",
        dtype=np.int8,
    )

    # --- get metadata from HTF zigzag ---------------------------------------- ok
    ltf_index = zz_ltf.index                               # pandas...DatatimeIndex
    htf_index = zz_htf.index                               # pandas...DatatimeIndex
    htf_state_series = zz_htf["state"         ].values     # numpy.ndarray   contain: -1/0/+1
    htf_confirmed_at = zz_htf["confirmed_at"  ].values     # numpy.ndarray   contain: pos
    htf_developing_leg = zz_htf["developing_leg"].values   # numpy.ndarray   contain: pos
    
    # =========================================================================
    # Project HTF swings onto LTF
    # ========================================================================= ok
    if mode == "last":

        for pos, signal in enumerate(htf_state_series):
            # pos_cnt +=1 # debug
            if signal == 0:
                continue

            confirmed_pos = htf_confirmed_at[pos]
            if confirmed_pos < 0:
                continue
            confirmed_ts = htf_index[confirmed_pos]

            # --- استفاده از searchsorted برای نزدیکترین LTF ---
            loc = ltf_index.searchsorted(confirmed_ts, side="right") - 1
            if 0 <= loc < len(zz_ltf):
                zz_ltf.iloc[loc] = signal
            
    elif mode == "forward_fill":
        swings = []

        for pos, signal in enumerate(htf_state_series):
            if signal == 0:
                continue

            confirmed_pos = htf_confirmed_at[pos]
            if confirmed_pos < 0:
                continue
            confirmed_ts = htf_index[confirmed_pos]

            loc = ltf_index.searchsorted(confirmed_ts, side="right") - 1
            if 0 <= loc < len(zz_ltf):
                swings.append((ltf_index[loc], signal))

        if swings:       # "swings": it contains ("confirmed_ts","signal")
            if len(swings) == 1:
                ts0, sig0 = swings[0]
                start_idx = ltf_index.get_loc(ts0)
                zz_ltf.iloc[start_idx:] = sig0
            else:
                for (ts0, sig0), (ts1, _) in zip(swings[:-1], swings[1:]):
                    start_idx = ltf_index.get_loc(ts0)
                    end_idx   = ltf_index.get_loc(ts1)
                    zz_ltf.iloc[start_idx:end_idx] = sig0  # end exclusive

                # fill after last swing if extend_last_leg
                if extend_last_leg:
                    last_ts, last_sig = swings[-1]
                    last_loc = ltf_index.get_loc(last_ts)
                    zz_ltf.iloc[last_loc:] = last_sig
    
    # =========================================================================
    # causal forward propagation of developing_leg 
    # ========================================================================= ok
    for pos, leg_dir in enumerate(htf_developing_leg):   # htf_developing_leg:  (2001,) numpy.ndarray   contain: pos
        if leg_dir == 0:
            continue

        confirmed_pos = htf_confirmed_at[pos]
        if confirmed_pos < 0:
            continue

        confirmed_ts = htf_index[confirmed_pos] + pd.Timedelta(tf_higher)
        if confirmed_ts not in ltf_index:
            continue
        # --- getting "start_loc" ---
        start_loc = ltf_index.get_loc(confirmed_ts)

        # --- finding "end_loc" ---
        end_loc = len(zz_ltf_developing)  # مقدار پیش‌فرض
        next_conf_pos = pos + 1
        while next_conf_pos < len(htf_confirmed_at) and htf_confirmed_at[next_conf_pos] < 0:
            next_conf_pos += 1
        if next_conf_pos < len(htf_confirmed_at):
            next_ts = htf_index[htf_confirmed_at[next_conf_pos]]
            if next_ts in ltf_index:
                end_loc = ltf_index.get_loc(next_ts)
        
        end_loc = min(end_loc, len(zz_ltf_developing))

        if start_loc < end_loc:
            zz_ltf_developing.iloc[start_loc:end_loc] = leg_dir

    # =========================================================================
    # project HTF legs onto LTF (metadata only, no signal logic)
    # =========================================================================
    htf_legs = zz_htf.attrs.get("legs", [])
    ltf_legs = []

    for leg in htf_legs:
        start_ts = leg["start_ts"]
        end_ts   = leg["end_ts"]

        # --- map start/end to LTF --------------------------------------------
        start_loc = ltf_index.searchsorted(start_ts, side="right") - 1
        end_loc   = ltf_index.searchsorted(end_ts,   side="right") - 1

        if not (0 <= start_loc < len(ltf_index)):
            continue
        if not (0 <= end_loc < len(ltf_index)):
            continue
        if end_loc <= start_loc:
            continue

        # --- map confirmed_at HTF -> LTF -------------------------------------
        start_conf_ltf = -1
        end_conf_ltf   = -1

        start_conf_htf = leg.get("start_confirmed_at", -1)
        end_conf_htf   = leg.get("end_confirmed_at", -1)
        
        if start_conf_htf >= 0:
            start_conf_ts = htf_index[start_conf_htf] + pd.Timedelta(tf_higher)
            loc = ltf_index.searchsorted(start_conf_ts, side="right") - 1
            if 0 <= loc < len(ltf_index):
                start_conf_ltf = int(loc)

        if end_conf_htf >= 0:
            end_conf_ts = htf_index[end_conf_htf]
            loc = ltf_index.searchsorted(end_conf_ts, side="right") - 1
            if 0 <= loc < len(ltf_index):
                end_conf_ltf = int(loc)

        # --- map developing_leg HTF -> LTF --------------------------------------------- my version
        start_dev_ltf = 0
        end_dev_ltf   = 0

        loc = ltf_index.searchsorted(start_ts + pd.Timedelta(tf_higher), side="right") - 1
        if 0 <= loc < len(ltf_index):
            start_dev_ltf = int(loc)

        loc = ltf_index.searchsorted(end_ts, side="right") - 1
        if 0 <= loc < len(ltf_index):
            end_dev_ltf = int(loc)

        # --- store in metadata LTF -------------------------------------------
        ltf_legs.append({
            "start_ltf_pos": int(start_loc),
            "end_ltf_pos": int(end_loc),
            "direction": int(leg["direction"]),
            "start_confirmed_ltf_pos": int(start_conf_ltf),
            "end_confirmed_ltf_pos": int(end_conf_ltf),
            "start_developing_ltf_pos": int(start_dev_ltf),
            "end_developing_ltf_pos": int(end_dev_ltf),
        })

    base_series = zz_ltf
    zz_ltf = pd.DataFrame({
        f"zigzag_mtf_{tf_higher}": base_series,
        "zigzag_mtf_developing_leg": zz_ltf_developing
    }, index=high.index)

    # --- attach metadata to zz_ltf -------------------------------------------
    zz_ltf.attrs["legs"] = ltf_legs

    return zz_ltf

