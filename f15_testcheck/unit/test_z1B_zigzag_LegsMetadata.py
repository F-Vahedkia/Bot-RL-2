# f15_testcheck/unit/test_z1B_zigzag_LegsMetadata.py
# Run: python -m f15_testcheck.unit.test_z1B_zigzag_LegsMetadata

import pandas as pd
from pathlib import Path
from f03_features.indicators.zigzag2 import zigzag

# ------------------------------------------------------------
# Loading Data
# ------------------------------------------------------------
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

# ------------------------------------------------------------
# Run ZigZag
# ------------------------------------------------------------
zzg = zigzag(
    high=df["high"], low=df["low"],
    depth=12, deviation=5.0, backstep=10, point=0.01, addmeta=True,
)
zzg.to_csv(f"z1_zigzag_Result.csv")

# ------------------------------------------------------------
# Extract legs metadata
# ------------------------------------------------------------
legs = zzg.attrs.get("legs", [])
legs_df = pd.DataFrame(legs)

# ------------------------------------------------------------
# Save to CSV
# ------------------------------------------------------------
legs_df.to_csv("z1_zigzag_legs_metadata.csv")
print(f"ZigZag legs metadata exported successfully to: zigzag_legs_metadata.csv")

