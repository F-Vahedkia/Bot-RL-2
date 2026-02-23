# f03_features/indicators/extras_channel.py
# Status in (Bot-RL-2): Reviewed at 1404/09/28

"""کانال‌ها: Donchian, Chaikin Volatility"""

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
import pandas as pd
from typing import Dict

#------------------------------------------------------------------------------
def donchian(high: pd.Series, low: pd.Series, n: int = 20):
    upper = high.rolling(n, min_periods=n).max().astype("float32")
    lower = low.rolling(n, min_periods=n).min().astype("float32")
    mid = ((upper + lower)/2.0).astype("float32")
    return upper, mid, lower

#------------------------------------------------------------------------------
def chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10, roc: int = 10):
    ema_range = (high - low).ewm(span=n, adjust=False, min_periods=n).mean()
    return (100.0 * (ema_range - ema_range.shift(roc)) / ema_range.shift(roc)).astype("float32")

#------------------------------------------------------------------------------
def registry() -> Dict[str, callable]:
    
    def make_donchian(df, n: int = 20, **_):
        up, mid, lo = donchian(df["high"], df["low"], n)
        return {f"donch_up_{n}": up, f"donch_mid_{n}": mid, f"donch_lo_{n}": lo}
    
    def make_ch_vol(df, n: int = 10, roc: int = 10, **_):
        return {f"chaikin_vol_{n}_{roc}": chaikin_volatility(df["high"], df["low"], n, roc)}
        
    return {
            "chaikin_vol": make_ch_vol,
            "donchian": make_donchian,
            }

#------------------------------------------------------------------------------

# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                           Used in Functions: ...
                       1   2   3   4   5   6   7   8   9  10  11  11  12  13  14
1  donchian           --  --  ok
2  chaikin_volatility --  --  ok
3  registry           --  --  --


"""