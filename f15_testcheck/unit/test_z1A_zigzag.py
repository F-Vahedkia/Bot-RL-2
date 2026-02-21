# f15_testcheck/unit/test_z1A_zigzag_completed.py
# Run: python -m f15_testcheck.unit.test_z1A_zigzag

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from f03_features.indicators.zigzag import (
    # _zigzag_mql_numpy,
    _zigzag_mql_numpy_complete,
    _zigzag_mql_njit_loopwise_complete,
    zigzag,
)

# ============================================================
# Load data
# ============================================================
t1 = datetime.now()
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
t2 = datetime.now()
df = data[-10_000:].copy()
t3 = datetime.now()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
t4 = datetime.now()
# print(f"Time taken to read CSV with {len(data)} rows: {round((t2 - t1).total_seconds(), 1)} seconds")
# print(f"Time taken to slice data with {len(df)} rows: {round((t3 - t2).total_seconds(), 1)} seconds")
# print(f"Time taken to process datetime/index: {round((t4 - t3).total_seconds(), 1)} seconds")
# print("============================================================================")

# ============================================================
# Call _zigzag_mql_numpy()
# ============================================================
zigzag_funcs = {
                "no_njit": _zigzag_mql_numpy_complete,
                "by_njit":  _zigzag_mql_njit_loopwise_complete,
                }

bytes_used = df["high"].nbytes + df["low"].nbytes

for key in zigzag_funcs.keys():
    # continue
    df_index = df.index
    func = zigzag_funcs[key]
    t1 = datetime.now()
    zzg, h_act, l_act = \
        func(
            df["high"].values,
            df["low"].values,
            depth=12, deviation=5.0, backstep=10, point=0.01
        )
    t2 = datetime.now()
    print(f"Time taken to run zigzag ({key}): {round((t2 - t1).total_seconds(), 3)} seconds, \
          bytes_used:{bytes_used}, \
          length_df:{len(df)}")
    df_new = pd.DataFrame(
        index = df_index,
        data = {
            "high": df["high"],
            "low": df["low"],
            "state": zzg,
            "high_actual": h_act,
            "low_actual": l_act,
        }
    )
    df_new.reset_index().to_csv(f"z1_zigzag_{key}.csv", index_label="no.")

# ============================================================
# Call _zigzag_mql_numpy()
# ============================================================
t1 = datetime.now()
zzg_df = zigzag(df["high"], df["low"], depth=12, deviation=5.0, backstep=10, point=0.01)
t2 = datetime.now()
print(f"Time taken to run zigzag for {len(df)} candles: {round((t2 - t1).total_seconds(), 3)} seconds")
zzg_df.reset_index().to_csv("z1_zigzag_Main.csv", index_label="no.")
pd.DataFrame(zzg_df.attrs["legs"]).to_csv("z1_zigzag_legs.csv")


