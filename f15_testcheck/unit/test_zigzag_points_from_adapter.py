# f15_testcheck/unit/test_zigzag_points_from_adapter.py
# Run: python -m f15_testcheck.unit.test_zigzag_points_from_adapter
# Purpose: Unit test for zigzag_points_from_adapter function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from f03_features.indicators.zigzag import zigzag_mtf_adapter
from f03_features.indicators.levels import zigzag_points_from_adapter

# --------------------------------------------------
# Synthetic data generator (trend + regime change)
# --------------------------------------------------
def generate_price_series(
    start="2024-01-01",
    periods=3000,
    freq="1min",
    seed=123,
):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=freq)

    # piecewise trend (up → range → down)
    trend = np.concatenate([
        np.linspace(0, 5, periods // 3),
        np.zeros(periods // 3),
        np.linspace(0, -5, periods - 2 * (periods // 3)),
    ])

    noise = rng.normal(0, 0.15, periods)
    price = 100 + trend + np.cumsum(noise)

    high = pd.Series(price + rng.uniform(0.01, 0.08, periods), index=idx)
    low  = pd.Series(price - rng.uniform(0.01, 0.08, periods), index=idx)

    return high, low


# --------------------------------------------------
# Main test
# --------------------------------------------------
def test_zigzag_points_from_adapter():
    high, low = generate_price_series()

    tf = "15min"

    hh, ll = zigzag_points_from_adapter(
        high=high,
        low=low,
        tf=tf,
        depth=12,
        deviation=5.0,
        backstep=10,
    )

    # --------------------------------------------------
    # 1️⃣ Structural tests
    # --------------------------------------------------
    assert isinstance(hh, pd.Series)
    assert isinstance(ll, pd.Series)

    assert hh.index.equals(high.index)
    assert ll.index.equals(high.index)

    assert hh.dtype == np.float32
    assert ll.dtype == np.float32

    assert not hh.isna().any()
    assert not ll.isna().any()

    # --------------------------------------------------
    # 2️⃣ Value-domain tests
    # --------------------------------------------------
    assert set(hh.unique()).issubset({0.0, 1.0})
    assert set(ll.unique()).issubset({0.0, 1.0})

    # --------------------------------------------------
    # 3️⃣ Logical consistency
    # --------------------------------------------------

    # HH and LL must never be 1 at the same time
    assert not ((hh == 1.0) & (ll == 1.0)).any()

    # At least some swings must exist
    assert hh.sum() > 0
    assert ll.sum() > 0

    # HH + LL should be sparse
    assert ((hh + ll) <= 1.0).all()

    zz = zigzag_mtf_adapter(
        high=high,
        low=low,
        tf_higher=tf,
        depth=12,
        deviation=5.0,
        backstep=10,
    )

    assert hh.equals((zz > 0).astype(np.float32))
    assert ll.equals((zz < 0).astype(np.float32))

    transitions = ((zz != zz.shift(1)) & (zz != 0)).sum()
    assert transitions > 0

    # --------------------------------------------------
    # 4️⃣ Edge sanity checks
    # --------------------------------------------------
    # If zz is zero → hh & ll must be zero
    zero_mask = zz == 0
    assert (hh[zero_mask] == 0).all()
    assert (ll[zero_mask] == 0).all()

    print("✅ All zigzag_points_from_adapter tests passed")

    # --------------------------------------------------
    # 5️⃣ Visual inspection (optional but powerful)
    # --------------------------------------------------
    fig, ax = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    ax[0].plot(high.index, high.values, alpha=0.6, label="High")
    ax[0].plot(low.index, low.values, alpha=0.6, label="Low")
    ax[0].set_title("Price")
    ax[0].legend()

    ax[1].plot(zz.index, zz.values)
    ax[1].set_title("ZigZag MTF")

    ax[2].stem(hh.index, hh.values, linefmt="g-", markerfmt="go", basefmt=" ")
    ax[2].set_title("HH (swing highs)")

    ax[3].stem(ll.index, ll.values, linefmt="r-", markerfmt="ro", basefmt=" ")
    ax[3].set_title("LL (swing lows)")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Execute
# --------------------------------------------------
test_zigzag_points_from_adapter()
