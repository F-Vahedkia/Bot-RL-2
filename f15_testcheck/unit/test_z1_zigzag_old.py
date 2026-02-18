# f15_testcheck/unit/test_z1_zigzag.py
# Test script for ZigZag indicator with debug output and visualization
# Run: python -m f15_testcheck.unit.test_z1_zigzag_old
# Author: Farhad Vahedkia
# Date Start: 1404/10/09- 17:40
# Date End  : 1404/11/08- 07:49

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from f03_features.indicators.zigzag2 import (
    # _zigzag_mql,
    _zigzag_mql_numpy,
    # _zigzag_mql_njit,
    # _zigzag_mql_njit_vectorized,
    _zigzag_mql_njit_loopwise,
    zigzag,
)

# ============================================================
# Load data
# ============================================================
t1 = datetime.now()
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()
df = data[-80_000:].copy()    # عدد مرزی برابر است با 80_000
t3 = datetime.now()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
t4 = datetime.now()
print(f"Time taken to read CSV with {len(data)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
print(f"Time taken to slice data with {len(df)} rows: {round((t3 - t2).total_seconds(), 1)} seconds")
print(f"Time taken to process datetime/index: {round((t4 - t3).total_seconds(), 1)} seconds")
print("============================================================================")

# ============================================================
# Call ZigZag (MQL-compatible)
# ============================================================
'''
from f03_features.indicators.zigzag import (
    _zigzag_mql,
    _zigzag_mql_numpy,
    _zigzag_mql_njit,
    _zigzag_mql_njit_vectorized,
    _zigzag_mql_njit_loopwise,
    zigzag,
)

zigzag_funcs = {"no_njit": _zigzag_mql,       # 1
                "by_njit": _zigzag_mql_njit,  # 3
                }
for key in zigzag_funcs.keys():
    continue
    func = zigzag_funcs[key]
    t1 = datetime.now()
    zzg, h_act, l_act = func(df["high"], df["low"], depth=12, deviation=5.0, backstep=10, point=0.01)
    t2 = datetime.now()
    print(f"Time taken to run zigzag ({key}): {round((t2 - t1).total_seconds(), 3)} seconds")

    # ایجاد DataFrame - اصلاح نام ستون‌ها
    zzg_df = pd.DataFrame(
        index=df.index,
        data={
            "zzg Result": zzg,
            "high_actual": h_act,
            "low_actual": l_act,
            "high": df["high"],
            "low": df["low"],
            "state": np.asarray(zzg.attrs["state_series"]),
            "high_map": np.asarray(zzg.attrs["high_map"]),
            "low_map": np.asarray(zzg.attrs["low_map"]),
        }
    )

    zzg_legs = pd.DataFrame(zzg.attrs["legs"])
    zzg_legs["check"] = False
    zzg_legs["leg_peak"] = 0.0
    zzg_legs["leg_bttm"] = 0.0
    zzg_legs["interval_peak"] = 0.0
    zzg_legs["interval_bttm"] = 0.0
    
    for i in range(len(zzg_legs)):
        leg_srt_pos = zzg_legs["start_ts"][i]
        leg_end_pos   = zzg_legs["end_ts"][i]
        leg_peak = max(df["high"].iloc[leg_srt_pos], df["high"].iloc[leg_end_pos])
        leg_bttm = min(df["low" ].iloc[leg_srt_pos], df["low" ].iloc[leg_end_pos])
        interval_peak = df["high"].iloc[leg_srt_pos : leg_end_pos + 1].max()
        interval_bttm = df["low" ].iloc[leg_srt_pos : leg_end_pos + 1].min()
        
        zzg_legs.loc[i, "leg_peak"] = leg_peak
        zzg_legs.loc[i, "leg_bttm"] = leg_bttm
        zzg_legs.loc[i, "interval_peak"] = interval_peak
        zzg_legs.loc[i, "interval_bttm"] = interval_bttm

        zzg_legs.loc[i, "check"] = (leg_peak >= interval_peak and leg_bttm <= interval_bttm)

    # counting True, False legs
    valid = zzg_legs["check"].sum()
    invalid = len(zzg_legs) - valid
    print(f"Valid: {valid}, Invalid: {invalid}, total:{len(zzg_legs)}")

    # فیلتر
    valid_legs = zzg_legs[zzg_legs["check"]]
    invalid_legs = zzg_legs[~zzg_legs["check"]]


    name_swing = f"zigzag_swings_{key}.csv"
    name_leg = f"zigzag_legs_{key}.csv"
    zzg_df.to_csv(name_swing)
    zzg_legs.to_csv(name_leg)
    print(f"Output swings CSV written to: {name_swing} with {len(zzg)} rows")
    print(f"Output legs CSV written to: {name_leg} with {len(zzg_legs)} rows \n")
'''
# ============================================================
# Call _zigzag_mql_numpy()
# ============================================================
zigzag_funcs = {
                "no_njit": _zigzag_mql_numpy,                # 2
                # "New_by_njit_vec": _zigzag_mql_njit_vectorized,  # 4
                "by_njit_loop": _zigzag_mql_njit_loopwise,   # 5
                }

for key in zigzag_funcs.keys():
    # continue
    df_index = df.index
    func = zigzag_funcs[key]
    t1 = datetime.now()
    zzg, h_act, l_act = func(
                            df["high"].values,
                            df["low"].values,
                            depth=12, deviation=5.0, backstep=10, point=0.01)
    t2 = datetime.now()
    print(f"Time taken to run zigzag ({key}): {round((t2 - t1).total_seconds(), 3)} seconds")
    df_new = pd.DataFrame(
        index = df_index,
        data = {
            "high": df["high"],
            "low": df["low"],
            "state": zzg,
            "high_actual": h_act,
            "low_actual": l_act
        }
    )
    df_new.to_csv(f"z1_zigzag_{key}.csv")

# ============================================================
# Call _zigzag_mql_numpy()
# ============================================================
t1 = datetime.now()
zzg_df = zigzag(df["high"], df["low"], depth=12, deviation=5.0, backstep=10, point=0.01)
t2 = datetime.now()
print(f"Time taken to run zigzag for {len(df)} candles: {round((t2 - t1).total_seconds(), 3)} seconds")
zzg_df.to_csv("z1_zigzag_Main.csv")
pd.DataFrame(zzg_df.attrs["legs"]).to_csv("z1_zigzag_legs.csv")
# ============================================================
# Extract swing prices correctly
# ============================================================/////
# high_idx = zzg[zzg > 0].index
# low_idx  = zzg[zzg < 0].index

# high_prices = df3.loc[high_idx, "high"]
# low_prices  = df3.loc[low_idx,  "low"]

# ============================================================
# Build debug DataFrame
# ============================================================
# debug_df = pd.DataFrame(
#     index=df3.index,
#     data={
#         # "open": df3["open"],
#         "high": df3["high"],
#         "low": df3["low"],
#         # "close": df3["close"],
#         "zigzag_signal": zzg.values,
#         "state": np.asarray(zzg.attrs["state_series"]),
#         "high_map": np.asarray(zzg.attrs["high_map"]),
#         "low_map": np.asarray(zzg.attrs["low_map"]),

#         "high_map": np.asarray(high_prices),
#         "low_map": np.asarray(low_prices),
#     }
# )
# ============================================================
# Compute real swing price
# ============================================================
# debug_df["swing_price"] = 0.0

# temp_filter = debug_df["zigzag_signal"] > 0
# debug_df.loc[temp_filter, "swing_price"] = debug_df.loc[temp_filter, "high"]

# temp_filter = debug_df["zigzag_signal"] < 0
# debug_df.loc[temp_filter, "swing_price"] = debug_df.loc[temp_filter, "low"]

# # --- Save CSV ---
# out_path = "test_zigzag_output.csv"
# t1 = datetime.now()
# debug_df.to_csv(out_path)
# t2 = datetime.now()
# print(f"Time taken to write debug CSV with {len(debug_df)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
# print(f"[OK] ZigZag debug CSV written to: {out_path}")

# ============================================================
# Final styling
# ============================================================
# zigzag_points = debug_df[debug_df['swing_price'] != 0]   # انتخاب فقط نقاط زیگزاگ
# zigzag_points = zigzag_points.sort_index()               # مرتب‌سازی بر اساس زمان
# plt.figure(figsize=(12,6))                               # فقط نقاط زیگزاگ به صورت خط ممتد با marker

# plt.plot(df3.index, df3["close"], color="lightgray", label="Close")  # نمودار قیمت کل
# plt.plot(zigzag_points.index, zigzag_points["swing_price"],color="orange", label="ZigZag", linewidth=2)  # خط زیگزاگ ممتد
# plt.scatter(high_idx, high_prices, marker="^", s=120, color="green", label="HIGH Swing")
# plt.scatter(low_idx, low_prices, marker="v", s=120, color="red", label="LOW Swing")

# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.title("ZigZag Continuous Plot")
# plt.legend()
# plt.grid(True)

# plt.show()
# ============================================================
