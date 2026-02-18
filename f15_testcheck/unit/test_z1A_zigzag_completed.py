# f15_testcheck/unit/test_z1A_zigzag_completed.py
# Run: python -m f15_testcheck.unit.test_z1A_zigzag_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from f03_features.indicators.zigzag2 import (
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
                # "no_njit_old":       _zigzag_mql_numpy,
                "no_njit_completed": _zigzag_mql_numpy_complete,
                "by_njit_complete":  _zigzag_mql_njit_loopwise_complete,
                }

bytes_used = df["high"].nbytes + df["low"].nbytes

for key in zigzag_funcs.keys():
    # continue
    df_index = df.index
    func = zigzag_funcs[key]
    t1 = datetime.now()
    zzg, h_act, l_act, conf_at, dev_leg = \
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
            "confirmed_at": conf_at,
            "developing_leg": dev_leg,
        }
    )
    df_new.to_csv(f"z1_zigzag_{key}.csv")
    # ============================================================
    # ============================================================
    # فرض: df_new همان دیتافریم خروجی تابع zigzag جدید است
    # ستون‌ها: confirmed_at و developing_leg

    # بررسی اینکه همه developing_leg=1 قبل از confirmed_at مرتبط هستند

    # اضافه کردن ستون اندیس عددی
    df_new['idx_numeric'] = np.arange(len(df_new))

    errors = []

    for i, row in df_new.iterrows():
        if row['developing_leg'] == 1:
            if row['confirmed_at'] == -1:
                continue  # هنوز تأیید نشده، صحیح
            if row['idx_numeric'] > row['confirmed_at']:
                errors.append((i, row['developing_leg'], row['confirmed_at']))


    confirmed_without_dev = []

    for i, row in df_new.iterrows():
        if row['confirmed_at'] != -1:
            pivot_idx = df_new.index.get_loc(i)
            confirm_idx = int(row['confirmed_at'])

            # باید در فاصله pivot تا confirmation developing_leg فعال باشد
            segment = df_new.iloc[pivot_idx+1:confirm_idx+1]

            if not (segment['developing_leg'] != 0).any():
                confirmed_without_dev.append(i)

    print("تعداد confirmed بدون developing معتبر:", len(confirmed_without_dev))

    print("تعداد مشکلات developing_leg قبل از confirmed:", len(errors))
    print("تعداد confirmed_at بدون developing_leg مربوطه:", len(confirmed_without_dev))
    print("========")


# ============================================================
# Call _zigzag_mql_numpy()
# ============================================================
t1 = datetime.now()
zzg_df = zigzag(df["high"], df["low"], depth=12, deviation=5.0, backstep=10, point=0.01)
t2 = datetime.now()
print(f"Time taken to run zigzag for {len(df)} candles: {round((t2 - t1).total_seconds(), 3)} seconds")
zzg_df.to_csv("z1_zigzag_Main.csv")
pd.DataFrame(zzg_df.attrs["legs"]).to_csv("z1_zigzag_legs.csv")


