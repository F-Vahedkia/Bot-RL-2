# f15_testcheck/unit/test_zigzag.py
# Test script for ZigZag indicator with debug output and visualization
# Author: Farhad Vahedkia
# Date: 1404/10/06- 23:09

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from datetime import datetime
from f03_features.indicators.zigzag import zigzag


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
zzg = zigzag(df3["high"], df3["low"], depth=12, deviation=5.0, backstep=3, point=0.01)
t2 = datetime.now()
print(f"Time taken to run zigzag whitout njit: {round((t2 - t1).total_seconds(), 1)} seconds")

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
out_path = "test_zigzag_output.csv"
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
