# f15_testcheck/unit/test_levels2_sr_distance_from_levels.py
# Run: python -m f15_testcheck.unit.test_levels2_sr_distance_from_levels
# Date modified: 1404/11/20

import pandas as pd
import numpy as np

from datetime import datetime
from f03_features.indicators.levels import (
    sr_from_zigzag_legs,
    sr_distance_from_levels,
)

def main():

    #--- ساختن داده های واقعی ---
    data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
    df = data[-10_000:].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)

    # --- build SR levels ---
    sr = sr_from_zigzag_legs(
        df,
        tf="15min",
        depth=12,
        deviation=5.0,
        backstep=10,
        extend_last_leg=True,
    )

    # --- compute distances ---
    t1 = datetime.now()
    dist = sr_distance_from_levels(
        df,
        sr=sr,
    )
    t2 = datetime.now()
    temp = round((t2 - t1).total_seconds(), 1)
    print(f"Time taken to run 'sr_distance_from_levels' with {len(df)} rows: {temp} seconds")

    # --- save result ---
    out = pd.concat([df["close"], sr, dist], axis=1)
    out.to_csv("test_levels2_sr_distance_from_levels.csv")

    print("test_levels2_sr_distance_from_levels.csv generated successfully")


if __name__ == "__main__":
    main()
