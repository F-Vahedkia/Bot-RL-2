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

    # state شامل کد +1 یا -1 است که بیانگر جستجو برای کف یا سقف بعدی است
    # اگر در کندل جاری دارای مقدار -1 باشد، یعنی این کندل سقف است و اکسترمم بعدی باید کف باشد
    # و بالعکس
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
    addmeta: bool = True,
) -> pd.DataFrame:
    
    idx = high.index
    high_np = np.ascontiguousarray(high.values, dtype=np.float64)
    low_np  = np.ascontiguousarray(low.values,  dtype=np.float64)
    bytes_used = high_np.nbytes + low_np.nbytes

    # Using "bytes_used <= 1_280_000"  isntead of "len(high_np) <= 80_000"
    if bytes_used <= 1_280_000:
        state, high_actual, low_actual = _zigzag_mql_numpy(
            high_np, low_np, depth, deviation, backstep, point
        )
    else:
        state, high_actual, low_actual = _zigzag_mql_njit_loopwise(
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
        }
    )

    if not addmeta:
        return zz_df
    
    # --- build zigzag legs metadata ------------------------------------------
    # leg = فاصله بین دو اکسترمم متوالی با جهت مشخص
    legs = []
    last_idx = None
    last_state = 0
    for idx_val, state_val in zip(idx, state):
        if state_val == 0:
            continue

        if last_idx is not None:
            legs.append({
                "start_idx": last_idx,
                "end_idx": idx_val,
                "direction": int(state_val),
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
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
        addmeta=True,
    )

    # --- get metadata from HTF zigzag ----------------------------------------
    htf_state_series = zz_htf["state"]
            
    # --- prepare LTF container -----------------------------------------------
    # ساختن یک دیتافریم خالی، البته با اندکسهای زمانی معلوم
    zz_ltf = pd.Series(
        data=0.0,
        index=high.index,
        name=f"zigzag_mtf_{tf_higher}",
        dtype=np.float32
    )

    # --- project HTF swings onto LTF -----------------------------------------
    if mode == "last":
        for ts_htf, signal in htf_state_series.items():
            if signal != 0 and ts_htf in zz_ltf.index:
                zz_ltf.at[ts_htf] = signal


    elif mode == "forward_fill":
        # --- collect valid HTF swings (exact timestamp only)
        swings = [(ts, sig) for ts, sig in htf_state_series.items()
                if sig != 0 and ts in zz_ltf.index]

        # --- forward fill between consecutive swings
        for (ts0, sig0), (ts1, _) in zip(swings[:-1], swings[1:]):
            zz_ltf.loc[ts0:ts1] = sig0


    # --- project HTF legs onto LTF (metadata only, no signal logic) ----------
    htf_legs = zz_htf.attrs.get("legs", [])
    ltf_index = zz_ltf.index

    ltf_legs = []

    for leg in htf_legs:
        start_ts = leg["start_idx"]
        end_ts   = leg["end_idx"]

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
            "start_ltf_pos": int(start_pos),
            "end_ltf_pos": int(end_pos),
        })

    # --- attach metadata to zz_ltf -------------------------------------------
    zz_ltf.attrs["legs"] = ltf_legs

    return zz_ltf

#====================================================================
