# f15_testcheck/unit/ts03_PA_MkSt__3detect_bos_choch_1.py
# Run: python -m f15_testcheck.unit.ts03_PA_MkSt__3detect_bos_choch_1

import pandas as pd
# import numpy as np

from f03_features.price_action.market_structure import (
    build_market_structure,
    detect_bos_choch,
)

_PATH = "f16_test_results/"
# ---------------------------------------------------
# Load data
# ---------------------------------------------------
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain high, low, close")

# ---------------------------------------------------
# Build structure first (required input)
# ---------------------------------------------------
structure = build_market_structure(
    df,
    depth=12,
    deviation=5.0,
    backstep=10,
    point=0.01,
)

bos = detect_bos_choch(structure)

# ---------------------------------------------------
# Validation
# ---------------------------------------------------
fault_counter = 0

required_cols = ["bos_up", "bos_down", "choch_up", "choch_down"]
for col in required_cols:
    if col not in bos.columns:
        raise ValueError(f"{col} column missing")

# 1) BOS/CHOCH must only appear on structure rows
structure_mask = bos[["HH", "HL", "LH", "LL"]].any(axis=1)
for col in required_cols:
    invalid_flag = (~structure_mask) & (bos[col] == 1)
    fault_counter += invalid_flag.sum()

# 2) No simultaneous opposite signals
opposite_conflict = (
    (bos["bos_up"] == 1) & (bos["bos_down"] == 1)
) | (
    (bos["choch_up"] == 1) & (bos["choch_down"] == 1)
)
fault_counter += opposite_conflict.sum()

# 3) Logical consistency replay check
last_structure = None

for idx, row in bos.iterrows():

    current_structure = None

    if row.get("HH", False):
        current_structure = "HH"
    elif row.get("HL", False):
        current_structure = "HL"
    elif row.get("LH", False):
        current_structure = "LH"
    elif row.get("LL", False):
        current_structure = "LL"

    if current_structure is None:
        continue

    if last_structure is None:
        last_structure = current_structure
        continue

    if current_structure == "HH" and last_structure in ("HL", "HH"):
        if row["bos_up"] != 1:
            fault_counter += 1

    elif current_structure == "LL" and last_structure in ("LH", "LL"):
        if row["bos_down"] != 1:
            fault_counter += 1

    elif current_structure == "HH" and last_structure in ("LH", "LL"):
        if row["choch_up"] != 1:
            fault_counter += 1

    elif current_structure == "LL" and last_structure in ("HL", "HH"):
        if row["choch_down"] != 1:
            fault_counter += 1

    last_structure = current_structure

print(f"fault_counter = {fault_counter}")

bos.to_csv(f"{_PATH}ts03_PA_MkSt__3detect_bos_choch_1.csv")
print("--------------------------------------------------")
print("Added 1 test result files to main project root:")
print("✅   ts03_PA_MkSt__3detect_bos_choch_1.csv")
print("--------------------------------------------------")
