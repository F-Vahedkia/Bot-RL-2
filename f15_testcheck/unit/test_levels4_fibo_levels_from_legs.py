# f15_testcheck/unit/test_levels4_fibo_levels_from_legs.py
# Run: python -m f15_testcheck.unit.test_levels4_fibo_levels_from_legs

import pandas as pd
import numpy as np
from f03_features.indicators.zigzag import zigzag
from f03_features.indicators.levels import fibo_levels_from_legs_orig, fibo_levels_from_legs_njit

# --- بارگذاری داده واقعی -------------------------------------------
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-40_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# --- ساخت ZigZag ----------------------------------------------------
zz_df = zigzag(
    high=df["high"],
    low=df["low"],
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
    addmeta=True
)

# --- محاسبه فیبو با هر دو تابع --------------------------------------
fibo_df_orig = fibo_levels_from_legs_orig(df, zz_df)
fibo_df_njit = fibo_levels_from_legs_njit(df, zz_df)

# --- بررسی و مقایسه نتایج ------------------------------------------
# اختلاف بین دو روش (nan-safe)
diff = (fibo_df_orig.fillna(0) - fibo_df_njit.fillna(0)).abs()

# ترکیب نتایج در یک DataFrame
compare_df = pd.concat([
    fibo_df_orig.add_suffix("_orig"),
    fibo_df_njit.add_suffix("_njit"),
    diff.add_suffix("_diff")
], axis=1)

# --- ذخیره در CSV در ریشه پروژه ------------------------------------
compare_df.to_csv("fibo_levels_comparison.csv")
print("✅ فایل fibo_levels_comparison.csv با موفقیت ساخته شد.")
