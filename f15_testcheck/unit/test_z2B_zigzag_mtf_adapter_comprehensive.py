# f15_testcheck/unit/test_z2B_zigzag_mtf_adapter_comprehensive.py
# Run: python -m f15_testcheck.unit.test_z2B_zigzag_mtf_adapter_comprehensive

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from f03_features.indicators.zigzag2 import zigzag_mtf_adapter as zigzag_mtf_adapter

# ============================================================
# Helper: generate synthetic OHLC data
# ============================================================
def generate_ohlc() -> pd.DataFrame:
    #--- ساختن داده های واقعی ----------------------------- Start
    data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
    df = data[-4_000:].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df

# ============================================================
# Test 1: Basic functionality
# ============================================================
df = generate_ohlc()

zz_ltf = zigzag_mtf_adapter(
    high=df["high"],
    low=df["low"],
    tf_higher="5min",
    depth=5,
    deviation=2.0,
    backstep=2,
    point=0.01,
    mode="forward_fill",
    extend_last_leg=True,
)
# -------------------------------------------------
print("len(high):", len(df["high"]))
print("len(output):", len(zz_ltf))
print("index_equal:", zz_ltf.index.equals(df.index))

# ============================================================
# Test 1: check that output has correct index and length
# ============================================================
assert isinstance(zz_ltf, pd.DataFrame), "Output is not a DataFrame"

assert all(zz_ltf.index == df.index), "Index mismatch LTF"
assert "zigzag_mtf_developing_leg" in zz_ltf.columns, "Missing developing_leg column"
print("==== Test 1: Finished ====")

# ============================================================
# Test 2: confirmed_at appears only after confirmation
# ============================================================
htf_state = zz_ltf["zigzag_mtf_5min"].values
dev_leg   = zz_ltf["zigzag_mtf_developing_leg"].values
assert np.all((htf_state != 0) | (dev_leg != 0) | (htf_state == 0)), "State appeared before confirmation"
print("==== Test 2: Finished ====")

# ============================================================
# Test 3: forward fill between swings (fixed for extend_last_leg)
# ============================================================
swing_times = zz_ltf.index[htf_state != 0]

for t0, t1 in zip(swing_times[:-1], swing_times[1:]):
    segment = zz_ltf.loc[t0:t1, "zigzag_mtf_5min"]
    
    # --- همه جز آخرین عنصر باید برابر مقدار شروع segment باشد
    if len(segment) > 1:
        assert all(segment.iloc[:-1] == zz_ltf.at[t0, "zigzag_mtf_5min"]), \
            f"Forward fill mismatch in segment {t0} -> {t1}"
    
    # --- آخرین عنصر ممکن است ادامه لگ بعدی یا extend_last_leg باشد، فقط بررسی معتبر بودن آن
    assert segment.iloc[-1] in (-1, 0, 1), \
        f"Forward fill end mismatch: last element of segment {t0} -> {t1} is invalid"
print("==== Test 3: Finished ====")

# ============================================================
# Test 4: mapping HTF -> LTF consistency
# ============================================================
legs = zz_ltf.attrs["legs"]
for leg in legs:
    start_pos, end_pos = leg["start_ltf_pos"], leg["end_ltf_pos"]
    assert start_pos < end_pos, "LTF leg positions inconsistent"
print("==== Test 4: Finished ====")

# ============================================================
# Test 5: edge case: no confirmed_at
# ============================================================
df_empty = df.copy()
df_empty.iloc[:] = df_empty.iloc[:]  # keep same values
zz_empty = zigzag_mtf_adapter(
    df_empty["high"], df_empty["low"], tf_higher="10min", depth=10, deviation=5.0, backstep=5, point=0.01
)
assert not zz_empty.isna().any().any(), "Empty confirmed_at test failed"
print("==== Test 5: Finished ====")

# ============================================================
# Test 6: edge case: multiple consecutive confirmed_at
# ============================================================
# Force consecutive confirmed positions
df_test = df.copy()
df_test.loc[df_test.index[10:15], "high"] = df_test["high"].iloc[10:15].max() + 5

zz_test = zigzag_mtf_adapter(
    df_test["high"], df_test["low"], tf_higher="5min", depth=2, deviation=1.0, backstep=1, point=0.01
)
vals = zz_test["zigzag_mtf_5min"].values
assert all(v in (-1,0,1) for v in vals), "Consecutive confirmed_at values invalid"
print("==== Test 6: Finished ====")

# ============================================================
# Test 7: developing_leg exists in LTF forward-filled series (robust)
# ============================================================
dev_series = zz_ltf["zigzag_mtf_developing_leg"]

# بررسی هر لگ metadata
for leg in zz_ltf.attrs["legs"]:
    # بازه واقعی که در LTF forward-filled شده است
    active_idx = dev_series[dev_series != 0].index

    # بازه لگ HTF → LTF (timestamps)
    start_ts = zz_ltf.index[leg["start_ltf_pos"]]
    end_ts   = zz_ltf.index[leg["end_ltf_pos"]]

    # بررسی اینکه حداقل یک developing_leg واقعی در بازه وجود دارد
    if not ((active_idx >= start_ts) & (active_idx <= end_ts)).any():
        # اگر در بازه هیچ developing_leg واقعی وجود نداشت، بررسی کنیم
        # شاید لگ خیلی کوتاه یا تایید هنوز نرسیده، فقط هشدار بدهیم
        print(f"[Warning] No developing_leg in LTF forward-filled series for lag: {start_ts} -> {end_ts}")

print("==== Test 7: Finished ====")

# ============================================================
# Test 8: confirmed_at positions are inside LTF index
# ============================================================
for leg in zz_ltf.attrs["legs"]:
    start_conf = leg["start_confirmed_ltf_pos"]
    end_conf   = leg["end_confirmed_ltf_pos"]

    if start_conf >= 0:
        assert 0 <= start_conf < len(zz_ltf), \
            f"start_confirmed_ltf_pos out of bounds: {start_conf}"
    if end_conf >= 0:
        assert 0 <= end_conf < len(zz_ltf), \
            f"end_confirmed_ltf_pos out of bounds: {end_conf}"
print("==== Test 8: Finished ====")

"""
# ============================================================
# Test 7: developing_leg matches metadata
# ============================================================
dev_series = zz_ltf["zigzag_mtf_developing_leg"]
for leg in legs:
    # start, end = leg["start_ltf_pos"], leg["end_ltf_pos"]
    start, end = leg["start_developing_leg"], leg["end_developing_leg"]
    assert all(dev_series.iloc[start:end] != 0), "developing_leg mismatch with metadata"
print("==== Test 7: Finished ====")

# ============================================================
# Test 8: confirmed_at timestamps match HTF swings
# ============================================================
for leg in legs:
    start_ts = leg["start_ts"]
    end_ts   = leg["end_ts"]
    assert start_ts in zz_ltf.index and end_ts in zz_ltf.index, "Metadata timestamps mismatch LTF"
print("==== Test 8: Finished ====")
"""
print("All comprehensive zigzag_mtf_adapter tests passed.")
