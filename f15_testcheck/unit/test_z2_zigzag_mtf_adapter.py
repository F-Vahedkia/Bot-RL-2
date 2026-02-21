# f15_testcheck/unit/test_z2A_zigzag_mtf_adapter.py
# Run: python -m f15_testcheck.unit.test_z2_zigzag_mtf_adapter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f03_features.indicators.zigzag import zigzag_mtf_adapter


#--- ساختن داده های واقعی ----------------------------- Start
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-10_000:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)
#-------------------------------------------------------

high = df["high"]
low  = df["low"]
tf_higher = "5min"

zz_last = zigzag_mtf_adapter(
    high=high, low=low, tf_higher=tf_higher,
    depth=12, deviation=5.0, backstep=10, point=0.01,
    mode="last",
    extend_last_leg=False,
)
zz_last.reset_index().to_csv("9__zz_last.csv", index_label="no.")
legs = pd.DataFrame(zz_last.attrs["legs"])
legs.to_csv("9__legs_last.csv", index_label="no.")


zz_ffill = zigzag_mtf_adapter(
    high=high, low=low, tf_higher=tf_higher,
    depth=12, deviation=5.0, backstep=10, point=0.01,
    mode="forward_fill",
    extend_last_leg=True,
)
zz_ffill.reset_index().to_csv("9__zz_ffill.csv", index_label="no.")
legs = pd.DataFrame(zz_ffill.attrs["legs"])
legs.to_csv("9__legs_ffill.csv", index_label="no.")
