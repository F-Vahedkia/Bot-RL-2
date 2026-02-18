# test_levels1_sr_from_zigzag_legs.py
# Run: python -m f15_testcheck.unit.test_levels1_sr_from_zigzag_legs
# Date modified: 1404/11/19

import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime

from f03_features.indicators.levels import (
    sr_from_zigzag_legs_orig,
    sr_from_zigzag_legs_njit,
    sr_from_zigzag_legs,
)
from f03_features.indicators.zigzag2 import zigzag_mtf_adapter

#--- ساختن داده های واقعی --------------------------------------------------- Start
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# --- ZigZag parameters -------------------------------------------------------
tf_higher = "15min"      # example higher timeframe
depth = 12
deviation = 5.0
backstep = 10

# --- Run original function ---------------------------------------------------
t1 = datetime.now()
sr_original = sr_from_zigzag_legs_orig(
    df,
    tf=tf_higher,
    depth=depth,
    deviation=deviation,
    backstep=backstep,
    extend_last_leg=True,
)
t2 = datetime.now()
temp = round((t2 - t1).total_seconds(), 1)
print(f"Time taken to run 'sr_from_zigzag_legs_orig' with {len(df)} rows: {temp} seconds")

# --- Run Numba-optimized function --------------------------------------------
t3 = datetime.now()
sr_njit = sr_from_zigzag_legs_njit(
    df,
    tf=tf_higher,
    depth=depth,
    deviation=deviation,
    backstep=backstep,
    extend_last_leg=True,
)
t4 = datetime.now()
temp = round((t4 - t3).total_seconds(), 1)
print(f"Time taken to run 'sr_from_zigzag_legs_njit' with {len(df)} rows: {temp} seconds")

# --- Test 1: Shape equality --------------------------------------------------
assert sr_original.shape == sr_njit.shape, "Shape mismatch between original and njit outputs"
print("Shape check passed.")

# --- Test 2: Value equality --------------------------------------------------
def allclose_nan(a, b, rtol=1e-5, atol=1e-8):
    mask = np.isnan(a) & np.isnan(b)
    if np.any(mask):
        return np.allclose(a[~mask], b[~mask], rtol=rtol, atol=atol) and np.all(mask == mask)
    return np.allclose(a, b, rtol=rtol, atol=atol)

sup_equal = allclose_nan(sr_original["sr_support"].values, sr_njit["sr_support"].values)
res_equal = allclose_nan(sr_original["sr_resistance"].values, sr_njit["sr_resistance"].values)

if sup_equal and res_equal:
    print("All values match between original and Numba-optimized sr_from_zigzag_legs.")
else:
    print("Mismatch detected in support or resistance arrays!")

# --- Write results to CSV ----------------------------------------------------
output_dir = Path(".")  # root of project

sr_original.to_csv(output_dir / "sr_original.csv")
sr_njit.to_csv(output_dir / "sr_njit.csv")

print(f"Original SR saved to: {output_dir / 'sr_original.csv'}")
print(f"NJIT SR saved to: {output_dir / 'sr_njit.csv'}")

# --- Optional: quick summary -------------------------------------------------
print("Original SR head:\n", sr_original.head())
print("NJIT SR head:\n", sr_njit.head())








# --- Run Wrapper function ----------------------------------------------------
t3 = datetime.now()
sr_njit = sr_from_zigzag_legs(
    df,
    tf=tf_higher,
    depth=depth,
    deviation=deviation,
    backstep=backstep
)
t4 = datetime.now()
temp = round((t4 - t3).total_seconds(), 1)
print(f"Time taken to run 'sr_from_zigzag_legs' with {len(df)} rows: {temp} seconds")