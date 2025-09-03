# check_wiring_fib_cluster.py

# Wiring fib_cluster with MA/RSI/SR weighting (Bot-RL-2)
# Persian comments inside; prints are English.
# ------ تغییر ریشه برای رسیدن به مسیر ایمپورتها
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ------
import pandas as pd
import numpy as np

from f04_features.indicators.fibonacci import fib_cluster, DEFAULT_RETR_RATIOS
from f04_features.indicators.levels import round_levels
from f04_features.indicators.utils import detect_swings, compute_atr

from f04_features.indicators.utils import levels_from_recent_legs
#from f04_features.indicators.levels import round_levels
#from f04_features.indicators.fibonacci import fib_cluster

# ---------- helpers ----------
def ohlc_view(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """استخراج نمای استاندارد OHLC از ستون‌های prefix‌دارِ همان TF."""
    cols = {}
    for k in ["open","high","low","close","tick_volume","spread"]:
        c = f"{tf}_{k}"
        if c in df.columns: cols[k] = df[c]
    out = pd.DataFrame(cols).dropna(how="all")
    if out.empty:
        raise ValueError(f"OHLC for TF={tf} not found")
    return out

def last_leg_levels(ohlc_df: pd.DataFrame,
                    prominence: float | None = None,
                    min_distance: int = 5,
                    atr_mult: float | None = 1.0,
                    ratios=DEFAULT_RETR_RATIOS) -> pd.DataFrame:
    """ساخت سطوح رتریسمنت برای «آخرین لگ معتبر» از روی سوئینگ‌های بسته."""
    #atr = compute_atr(ohlc_df, window=14)
    #swings = detect_swings(ohlc_df["close"], prominence=prominence, min_distance=min_distance,
    #                       atr=atr, atr_mult=atr_mult, tf=None)
    
    atr = compute_atr(ohlc_df, window=14) if 'compute_atr' in globals() else None
    swings = detect_swings(
        close, prominence=prominence, min_distance=min_distance,
        atr=atr, atr_mult=atr_mult, tf=None
    )
    
    
    if swings.empty or len(swings) < 2:
        return pd.DataFrame(columns=["ratio","price","leg_up"])
    s = swings.sort_index()
    p1, p2 = float(s["price"].iloc[-2]), float(s["price"].iloc[-1])
    leg_up = p2 > p1
    low, high = (p1, p2) if leg_up else (p2, p1)
    rows = []
    for r in ratios:
        price = (high - r*(high-low)) if leg_up else (low + r*(high-low))
        rows.append({"ratio": float(r), "price": float(price), "leg_up": bool(leg_up)})
    return pd.DataFrame(rows)

# ---------- load processed dataset ----------
df = pd.read_parquet(r"f02_data/processed/XAUUSD/M1.parquet")

# OHLC views for H1 and H4 (adjust TFs as needed)
h1 = ohlc_view(df, "H1")
h4 = ohlc_view(df, "H4")

h1 = h1.tail(4000)  # seems to bo temporary
h4 = h4.tail(3200)   # seems to bo temporary
last_close = float(h1["close"].iloc[-1])                            # seems to be temporary
sr_levels = round_levels(last_close, step=10.0, n=25)  # XAUUSD≈10  # seems to be temporary

# Build TF levels from last valid leg on each TF
tf_levels = {
    "H1": last_leg_levels(h1, prominence=None, min_distance=5, atr_mult=1.0),
    "H4": last_leg_levels(h4, prominence=None, min_distance=5, atr_mult=1.0),
}
tf_levels = {                                                                    # seems to bo temporary
    "H1": levels_from_recent_legs(h1, n_legs=10, min_distance=5, atr_mult=1.0),  # seems to bo temporary
    "H4": levels_from_recent_legs(h4, n_legs=10, min_distance=5, atr_mult=1.0),  # seems to bo temporary
}


# Weighting series (already created by your previous CLI run)
ma_slope_series = df.get("__ma_slope@M5")
rsi_score_series = df.get("__rsi_zone@H1__rsi_zone_score")

# SR levels around last close (tune step per symbol; e.g., 10.0 for XAUUSD)
last_close = float(h1["close"].dropna().iloc[-1])
sr_levels = round_levels(last_close, step=10.0, n=25)

# Reference time = last index (use any timestamp you want)
ref_ts = pd.to_datetime(df.index[-1], utc=True)

# ---------- run clustering ----------
clusters = fib_cluster(
    tf_levels=tf_levels,
    tol_pct=0.20,                  # 0.10% price tolerance for clustering window
    prefer_ratio=0.618,
    ma_slope=ma_slope_series,      # optional weighting
    rsi_zone_score=rsi_score_series,
    sr_levels=sr_levels,
    #ref_time=ref_ts,
    ref_time=pd.to_datetime(df.index[-1], utc=True),
    w_trend=10.0, w_rsi=10.0, w_sr=10.0,
    sr_tol_pct=0.05,
)

print("Top clusters (by score):")
print(clusters.head(12).to_string(index=False))
