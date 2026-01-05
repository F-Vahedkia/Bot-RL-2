# f15_testcheck/unit/test_zigzag_mtf_adapter.py
# Run: python -m f15_testcheck.unit.test_zigzag_mtf_adapter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f03_features.indicators.zigzag import zigzag_mtf_adapter

# -------------------------------
# Synthetic OHLC generator
# -------------------------------
def generate_price_series(
    start="2024-01-01",
    periods=2000,
    freq="1min",
    trend_slope=0.0003,
    noise_scale=0.02,
    seed=42,
):
    np.random.seed(seed)
    idx = pd.date_range(start, periods=periods, freq=freq)

    trend = np.linspace(0, trend_slope * periods, periods)
    noise = np.random.normal(0, noise_scale, periods)
    price = 100 + trend + np.cumsum(noise)

    high = pd.Series(price + np.random.uniform(0, 0.05, periods), index=idx)
    low = pd.Series(price - np.random.uniform(0, 0.05, periods), index=idx)

    return high, low


# -------------------------------
# Run test
# -------------------------------
def test_zigzag_mtf_adapter():
    high, low = generate_price_series()

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
    )

    zz_ffill = zigzag_mtf_adapter(
        high=high,
        low=low,
        tf_higher=tf_higher,
        depth=12,
        deviation=5.0,
        backstep=10,
        point=0.01,
        mode="forward_fill",
    )

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
    assert (diffs != 0).sum() == (zz_last != 0).sum()

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
    ax[1].set_title("ZigZag MTF – last mode")

    ax[2].plot(zz_ffill.index, zz_ffill.values, label="ZZ MTF (forward_fill)")
    ax[2].set_title("ZigZag MTF – forward_fill")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Execute
# -------------------------------
test_zigzag_mtf_adapter()
