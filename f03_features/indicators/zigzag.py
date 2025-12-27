import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from datetime import datetime

# --- MAIN FUNCTION ---
def zigzag_mql(high: pd.Series,
                low: pd.Series,
                depth: int = 12,
                deviation: float = 5.0,
                backstep: int = 10,
                point: float = 0.01,
                ):
    n = len(high)
    zz = np.zeros(n)
    high_map = np.zeros(n)
    low_map = np.zeros(n)
    state_series = np.zeros(n, dtype=np.int8)  # 0=EXTREMUM, 1=PEAK, -1=BOTTOM

    last_high = 0.0
    last_low = 0.0
    last_high_pos = -1
    last_low_pos = -1

    legs = []

    # --- helper functions ---
    @njit
    def highest(arr, start, depth):
        i0 = max(0, start - depth + 1)
        return i0 + np.argmax(arr[i0:start + 1])

    @njit
    def lowest(arr, start, depth):
        i0 = max(0, start - depth + 1)
        return i0 + np.argmin(arr[i0:start + 1])

    # --- populate high_map and low_map ---
    for i in range(depth, n):
        # --- low extremum ----------------------
        idx = lowest(low.values, i, depth)
        val = low.values[idx]
        if val == last_low or (low.values[i] - val) > deviation * point:
            val = 0.0
        else:
            last_low = val
            for b in range(1, backstep + 1):
                j = i - b
                if j >= 0 and low_map[j] != 0 and low_map[j] > val:
                    low_map[j] = 0.0
        low_map[i] = val if low.values[i] == val else 0.0

        # --- high extremum ---------------------
        idx = highest(high.values, i, depth)
        val = high.values[idx]
        if val == last_high or (val - high.values[i]) > deviation * point:
            val = 0.0
        else:
            last_high = val
            for b in range(1, backstep + 1):
                j = i - b
                if j >= 0 and high_map[j] != 0 and high_map[j] < val:
                    high_map[j] = 0.0
        high_map[i] = val if high.values[i] == val else 0.0

    # --- final zigzag selection ---
    EXTREMUM, PEAK, BOTTOM = 0, 1, -1
    state = EXTREMUM
    last_high = last_low = 0.0
    last_high_pos = last_low_pos = -1

    for i in range(depth, n):
        if state == EXTREMUM:
            if last_low == 0 and last_high == 0:
                if high_map[i] != 0:
                    last_high = high_map[i]
                    last_high_pos = i
                    zz[i] = 1  # mark as high
                    state_series[i] = BOTTOM
                    state = BOTTOM
                elif low_map[i] != 0:
                    last_low = low_map[i]
                    last_low_pos = i
                    zz[i] = -1  # mark as low
                    state_series[i] = PEAK
                    state = PEAK

        elif state == PEAK:
            if low_map[i] != 0 and low_map[i] < last_low and high_map[i] == 0:
                zz[last_low_pos] = 0.0
                zz[i] = -1
                state_series[i] = PEAK
                last_low = low_map[i]
                last_low_pos = i
            elif high_map[i] != 0 and low_map[i] == 0:
                last_high = high_map[i]
                last_high_pos = i
                zz[i] = 1
                state_series[i] = BOTTOM
                state = BOTTOM
                # record previous leg
                legs.append({
                    "ts_start": last_low_pos,
                    "ts_end": i,
                    "start_point": last_low,
                    "end_point": last_high,
                    "direction": 1
                })

        elif state == BOTTOM:
            if high_map[i] != 0 and high_map[i] > last_high and low_map[i] == 0:
                zz[last_high_pos] = 0.0
                zz[i] = 1
                state_series[i] = BOTTOM
                last_high = high_map[i]
                last_high_pos = i
            elif low_map[i] != 0 and high_map[i] == 0:
                last_low = low_map[i]
                last_low_pos = i
                zz[i] = -1
                state_series[i] = PEAK
                state = PEAK
                # record previous leg
                legs.append({
                    "ts_start": last_high_pos,
                    "ts_end": i,
                    "start_point": last_high,
                    "end_point": last_low,
                    "direction": -1
                })

    # --- convert to Pandas Series ---
    zz_series = pd.Series(zz, index=high.index)
    zz_series.attrs["legs"] = legs
    zz_series.attrs["state_series"] = state_series.tolist()
    zz_series.attrs["high_map"]     = high_map.tolist()
    zz_series.attrs["low_map"]      = low_map.tolist()

    return zz_series


def zigzag_mql_njit(high: pd.Series,
                    low: pd.Series,
                    depth: int = 12,
                    deviation: float = 5.0,
                    backstep: int = 10,
                    point: float = 0.01,
                    ):
    
    # --- convert series to numpy arrays for njit speed ---
    high_arr = high.values
    low_arr = low.values
    n = len(high_arr)
    
    # --- njit core function ---
    @njit
    def _zigzag(high_arr, low_arr, depth, deviation, backstep, point):
        zz = np.zeros(n)
        high_map = np.zeros(n)
        low_map = np.zeros(n)
        state_series = np.zeros(n, dtype=np.int8)  # 0=EXTREMUM, 1=PEAK, -1=BOTTOM

        last_high = 0.0
        last_low = 0.0
        last_high_pos = -1
        last_low_pos = -1

        legs_ts_start = np.zeros(n, dtype=np.int32)
        legs_ts_end = np.zeros(n, dtype=np.int32)
        legs_start_point = np.zeros(n)
        legs_end_point = np.zeros(n)
        legs_direction = np.zeros(n, dtype=np.int8)
        leg_count = 0

        # --- helper functions ---
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

        # --- populate high_map and low_map ---
        for i in range(depth, n):
            # low extremum
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

            # high extremum
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

        # --- final zigzag selection ---
        EXTREMUM, PEAK, BOTTOM = 0, 1, -1
        state = EXTREMUM
        last_high = last_low = 0.0
        last_high_pos = last_low_pos = -1

        for i in range(depth, n):
            if state == EXTREMUM:
                if last_low == 0 and last_high == 0:
                    if high_map[i] != 0:
                        last_high = high_map[i]
                        last_high_pos = i
                        zz[i] = 1
                        state_series[i] = BOTTOM
                        state = BOTTOM
                    elif low_map[i] != 0:
                        last_low = low_map[i]
                        last_low_pos = i
                        zz[i] = -1
                        state_series[i] = PEAK
                        state = PEAK

            elif state == PEAK:
                if low_map[i] != 0 and low_map[i] < last_low and high_map[i] == 0:
                    zz[last_low_pos] = 0.0
                    zz[i] = -1
                    state_series[i] = PEAK
                    last_low = low_map[i]
                    last_low_pos = i
                elif high_map[i] != 0 and low_map[i] == 0:
                    last_high = high_map[i]
                    last_high_pos = i
                    zz[i] = 1
                    state_series[i] = BOTTOM
                    state = BOTTOM
                    # record previous leg
                    legs_ts_start[leg_count] = last_low_pos
                    legs_ts_end[leg_count] = i
                    legs_start_point[leg_count] = last_low
                    legs_end_point[leg_count] = last_high
                    legs_direction[leg_count] = 1
                    leg_count += 1

            elif state == BOTTOM:
                if high_map[i] != 0 and high_map[i] > last_high and low_map[i] == 0:
                    zz[last_high_pos] = 0.0
                    zz[i] = 1
                    state_series[i] = BOTTOM
                    last_high = high_map[i]
                    last_high_pos = i
                elif low_map[i] != 0 and high_map[i] == 0:
                    last_low = low_map[i]
                    last_low_pos = i
                    zz[i] = -1
                    state_series[i] = PEAK
                    state = PEAK
                    # record previous leg
                    legs_ts_start[leg_count] = last_high_pos
                    legs_ts_end[leg_count] = i
                    legs_start_point[leg_count] = last_high
                    legs_end_point[leg_count] = last_low
                    legs_direction[leg_count] = -1
                    leg_count += 1

        return zz, state_series, high_map, low_map, legs_ts_start[:leg_count], legs_ts_end[:leg_count], legs_start_point[:leg_count], legs_end_point[:leg_count], legs_direction[:leg_count]

    zz, state_series, high_map, low_map, legs_start_idx, legs_end_idx, legs_start_pt, legs_end_pt, legs_dir = _zigzag(high_arr, low_arr, depth, deviation, backstep, point)

    # --- convert to Pandas Series and attach legs ---
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
    zz_series.attrs["high_map"]     = high_map.tolist()
    zz_series.attrs["low_map"]      = low_map.tolist()

    return zz_series

# --- Wrapper function to choose between njit and non-njit based on data size ---
def zigzag(high: pd.Series,
            low: pd.Series,
            depth: int = 12,
            deviation: float = 5.0,
            backstep: int = 10,
            point: float = 0.01,):
    if(len(high) < 150_000):
        return zigzag_mql(high, low, depth, deviation, backstep, point)
    else:
        return zigzag_mql_njit(high, low, depth, deviation, backstep, point)


# ============================================================
# Load data
# ============================================================
t1 = datetime.now()
data3 = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()
df3 = data3[-2_000:].copy()
t3 = datetime.now()
df3["time"] = pd.to_datetime(df3["time"], utc=True)
df3.set_index("time", inplace=True)
t4 = datetime.now()
print(f"Time taken to read CSV with {len(data3)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
print(f"Time taken to slice data with {len(df3)} rows: {round((t3 - t2).total_seconds(), 1)} seconds")
print(f"Time taken to process datetime/index: {round((t4 - t3).total_seconds(), 1)} seconds")
print("\n")

# ============================================================
# Call ZigZag (MQL-compatible)
# ============================================================
t1 = datetime.now()
zzg1 = zigzag_mql(df3["high"], df3["low"], depth=12, deviation=5.0, backstep=3, point=0.01)
t2 = datetime.now()
print(f"Time taken to run zigzag whitout njit: {round((t2 - t1).total_seconds(), 1)} seconds")

t1 = datetime.now()
zzg = zigzag_mql_njit(df3["high"], df3["low"], depth=12, deviation=5.0, backstep=3, point=0.01)
t2 = datetime.now()
print(f"Time taken to run zigzag with njit: {round((t2 - t1).total_seconds(), 1)} seconds")
print("\n")

# ============================================================
# Extract swing prices correctly
# ============================================================/////
high_idx = zzg[zzg > 0].index
low_idx  = zzg[zzg < 0].index

high_prices = df3.loc[high_idx, "high"]
low_prices  = df3.loc[low_idx,  "low"]

# ============================================================
# Build debug DataFrame
# ============================================================
debug_df = pd.DataFrame(
    index=df3.index,
    data={
        "open": df3["open"],
        "high": df3["high"],
        "low": df3["low"],
        "close": df3["close"],
        "zigzag_signal": zzg.values,
        "state": np.asarray(zzg.attrs["state_series"]),
        "high_map": np.asarray(zzg.attrs["high_map"]),
        "low_map": np.asarray(zzg.attrs["low_map"]),
    }
)
# ============================================================
# Compute real swing price
# ============================================================
debug_df["swing_price"] = 0.0

temp_filter = debug_df["zigzag_signal"] > 0
debug_df.loc[temp_filter, "swing_price"] = debug_df.loc[temp_filter, "high"]

temp_filter = debug_df["zigzag_signal"] < 0
debug_df.loc[temp_filter, "swing_price"] = debug_df.loc[temp_filter, "low"]

# --- Save CSV ---
out_path = "zigzag_debug_output_2.csv"
t1 = datetime.now()
debug_df.to_csv(out_path)
t2 = datetime.now()
print(f"Time taken to write debug CSV with {len(debug_df)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
print(f"[OK] ZigZag debug CSV written to: {out_path}")

# ============================================================
# Final styling
# ============================================================
zigzag_points = debug_df[debug_df['swing_price'] != 0]   # انتخاب فقط نقاط زیگزاگ
zigzag_points = zigzag_points.sort_index()               # مرتب‌سازی بر اساس زمان
plt.figure(figsize=(12,6))                               # فقط نقاط زیگزاگ به صورت خط ممتد با marker

plt.plot(df3.index, df3["close"], color="lightgray", label="Close")  # نمودار قیمت کل
plt.plot(zigzag_points.index, zigzag_points["swing_price"],color="orange", label="ZigZag", linewidth=2)  # خط زیگزاگ ممتد
plt.scatter(high_idx, high_prices, marker="^", s=120, color="green", label="HIGH Swing")
plt.scatter(low_idx, low_prices, marker="v", s=120, color="red", label="LOW Swing")

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("ZigZag Continuous Plot")
plt.legend()
plt.grid(True)

plt.show()
# ============================================================
