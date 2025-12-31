# Written by: DeepSeek at 1404/10/10- 17:40

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from datetime import datetime
from f03_features.indicators.zigzag import _zigzag_mql, _zigzag_mql_njit

# ============================================================
# Load data
# ============================================================
t1 = datetime.now()
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()
df = data[-400_000:].copy()
t3 = datetime.now()
df["time"] = pd.to_datetime(data["time"], utc=True)
df.set_index("time", inplace=True)
t4 = datetime.now()
print(f"Time taken to read CSV with {len(data)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
print(f"Time taken to slice data with {len(df)} rows: {round((t3 - t2).total_seconds(), 1)} seconds")
print(f"Time taken to process datetime/index: {round((t4 - t3).total_seconds(), 1)} seconds")
print("============================================================================")

# ============================================================
# Call ZigZag (MQL-compatible)
# ============================================================
zigzag = {"no_njit": _zigzag_mql,
          "by_njit": _zigzag_mql_njit,
          }

for key in zigzag.keys():  # اصلاح: .keys() به جای .items
    func = zigzag[key]
    t1 = datetime.now()
    zzg = func(df["high"], df["low"], depth=20, deviation=5.0, backstep=4, point=0.01)
    t2 = datetime.now()
    print(f"Time taken to run zigzag ({key}): {round((t2 - t1).total_seconds(), 3)} seconds")

    # ایجاد DataFrame - اصلاح نام ستون‌ها
    zzg_df = pd.DataFrame(
        index=df.index,
        data={
            "high": df["high"],
            "low": df["low"],
            "state": np.asarray(zzg.attrs["state_series"]),
            "high_map": np.asarray(zzg.attrs["high_map"]),
            "low_map": np.asarray(zzg.attrs["low_map"]),
        }
    )

    # ایجاد DataFrame از legs - با مدیریت خطا
    if zzg.attrs.get("legs"):
        zzg_legs = pd.DataFrame(zzg.attrs["legs"])
        
        # بررسی نام ستون‌ها (ممکن است ts_start/ts_end یا start_idx/end_idx باشد)
        if "ts_start" in zzg_legs.columns:
            start_col = "ts_start"
            end_col = "ts_end"
        elif "start_idx" in zzg_legs.columns:
            start_col = "start_idx"
            end_col = "end_idx"
        else:
            print(f"Warning: Unknown column names in legs for {key}")
            start_col = zzg_legs.columns[0] if len(zzg_legs.columns) > 0 else None
            end_col = zzg_legs.columns[1] if len(zzg_legs.columns) > 1 else None
        
        # اضافه کردن ستون‌های جدید
        zzg_legs["check"] = False
        zzg_legs["leg_peak"] = 0.0
        zzg_legs["leg_bttm"] = 0.0
        zzg_legs["interval_peak"] = 0.0
        zzg_legs["interval_bttm"] = 0.0

        if start_col and end_col:
            for i in range(len(zzg_legs)):
                leg_srt_pos = int(zzg_legs[start_col].iloc[i])
                leg_end_pos = int(zzg_legs[end_col].iloc[i])
                
                # اطمینان از محدوده معتبر
                if 0 <= leg_srt_pos < len(df) and 0 <= leg_end_pos < len(df):
                    leg_peak = max(df["high"].iloc[leg_srt_pos], df["high"].iloc[leg_end_pos])
                    leg_bttm = min(df["low"].iloc[leg_srt_pos], df["low"].iloc[leg_end_pos])
                    
                    # محاسبه peak/bottom در بازه
                    start_idx = min(leg_srt_pos, leg_end_pos)
                    end_idx = max(leg_srt_pos, leg_end_pos)
                    interval_peak = df["high"].iloc[start_idx:end_idx + 1].max()
                    interval_bttm = df["low"].iloc[start_idx:end_idx + 1].min()
                    
                    zzg_legs.loc[i, "leg_peak"] = leg_peak
                    zzg_legs.loc[i, "leg_bttm"] = leg_bttm
                    zzg_legs.loc[i, "interval_peak"] = interval_peak
                    zzg_legs.loc[i, "interval_bttm"] = interval_bttm
                    zzg_legs.loc[i, "check"] = (leg_peak >= interval_peak and leg_bttm <= interval_bttm)
                else:
                    print(f"Warning: Invalid indices in leg {i}: {leg_srt_pos}, {leg_end_pos}")
        
        name_swing = f"zigzag_swings_{key}.csv"
        name_leg = f"zigzag_legs_{key}.csv"
        
        zzg_df.to_csv(name_swing)
        zzg_legs.to_csv(name_leg)
        
        print(f"Output swings CSV written to: {name_swing} with {len(zzg_df)} rows")
        print(f"Output legs CSV written to: {name_leg} with {len(zzg_legs)} rows")
        print(f"Valid legs: {zzg_legs['check'].sum()} out of {len(zzg_legs)}")
    else:
        print(f"No legs found for {key}")
    
    print("-" * 60)

print("\n✓ All tests completed successfully!")