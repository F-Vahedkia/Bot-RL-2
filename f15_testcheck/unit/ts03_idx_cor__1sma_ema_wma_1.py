# f03_features/indicators/ts03_idx_cor__1sma_ema_wma_1.py
# Run: python -m f15_testcheck.unit.ts03_idx_cor__1sma_ema_wma_1

#####################################################################
# فایل تستر براساس داده های واقعی
#####################################################################

import pandas as pd
from datetime import datetime

from f03_features.indicators.core import (
    sma_orig_slow, sma,
    ema, ema_numpy_slow,
    wma_slow, wma
)

# --- Load data -----------------------------------------------------
t1 = datetime.now()
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-1_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"open", "high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain open, high, low, close")
t2 = datetime.now()
print(f"Time taken to load data: {round((t2 - t1).total_seconds(), 5)} seconds, \
          length_df:{len(df)}")


# --- Calling functions ---------------------------------------------
funcs = {
    "sma_orig_slow": sma_orig_slow,
    "sma": sma,
    "ema": ema,
    "eema_numpy_slow": ema_numpy_slow,
    "wwma_slowa": wma_slow,
    "wma": wma,
}
# دیتافریم برای ذخیره همه نتایج
results_df = pd.DataFrame(index=df.index)

for name, func in funcs.items():
    t1 = datetime.now()
    result = func(df["open"], n=20)
    t2 = datetime.now()

    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to run {name}: {elapsed} seconds, length_df:{len(df)}")
    # ذخیره خروجی در دیتافریم مشترک
    results_df[name] = result


# --- Save results --------------------------------------------------
results_df.to_csv("ts03_idx_cor__1sma_ema_wma_1.csv")
