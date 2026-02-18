# f15_testcheck/unit/test_z2_zigzag_mtf_adapter.py
# Run: python -m f15_testcheck.unit.test_z2_zigzag_mtf_adapter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f03_features.indicators.zigzag2 import zigzag_mtf_adapter as zigzag_mtf_adapter


# -------------------------------
# Run test
# -------------------------------
def test_zigzag_mtf_adapter():
    #--- ساختن داده های واقعی ----------------------------- Start
    data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
    df = data[-10_000:].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    #-------------------------------------------------------

    high = df["high"]
    low  = df["low"]
    tf_higher = "15min"

    zz_last = zigzag_mtf_adapter(
        high=high,
        low=low,
        tf_higher=tf_higher,
        depth=12,
        deviation=5.0,
        backstep=10,
        point=0.01,
        mode="last",
        extend_last_leg=False,
    )
    df = pd.DataFrame({"high": high, "low": low, "zz_last": zz_last})
    df.to_csv("z2_zz_last.csv")
    meta = pd.DataFrame(zz_last.attrs["legs"])
    meta.to_csv("z2_meta_last.csv")

    zz_ffill = zigzag_mtf_adapter(
        high=high,
        low=low,
        tf_higher=tf_higher,
        depth=12,
        deviation=5.0,
        backstep=10,
        point=0.01,
        mode="forward_fill",
        extend_last_leg=False,
    )
    df = pd.DataFrame({"high": high, "low": low, "zz_ffill": zz_ffill})
    df.to_csv("z2_zz_ffill.csv")
    meta = pd.DataFrame(zz_ffill.attrs["legs"])
    meta.to_csv("z2_meta_ffill.csv")

    # -------------------------------
    # Assertions (Logical correctness)
    # -------------------------------

    # 1) Index alignment
    assert zz_last.index.equals(high.index)
    assert zz_ffill.index.equals(high.index)

    # 2) No NaN allowed
    assert not zz_last.isna().any()
    assert not zz_ffill.isna().any()

    # 3) forward_fill must have >= number of non-zero than last
    assert (zz_ffill != 0).sum() >= (zz_last != 0).sum()

    # 4) last-mode must be sparse (only single candle per HTF swing)
    assert (zz_last != 0).sum() < (len(high) * 0.05)

    # 5) forward_fill must be piecewise constant
    diffs = zz_ffill.diff().fillna(0)
    assert (diffs != 0).sum() <= (zz_last != 0).sum()

    print("✅ All logical tests passed")

    # -------------------------------
    # Visualization
    # -------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax[0].plot(high.index, high.values, label="High", alpha=0.6)
    ax[0].plot(low.index, low.values, label="Low", alpha=0.6)
    ax[0].set_title("Synthetic Price")
    ax[0].legend()

    ax[1].plot(zz_last.index, zz_last.values, label="ZZ MTF (last)")
    ax[1].set_title("ZigZag MTF - last mode")

    ax[2].plot(zz_ffill.index, zz_ffill.values, label="ZZ MTF (forward_fill)")
    ax[2].set_title("ZigZag MTF - forward_fill")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Execute
# -------------------------------
test_zigzag_mtf_adapter()
