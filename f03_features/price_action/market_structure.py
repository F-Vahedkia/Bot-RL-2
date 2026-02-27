# f03_features/price_action/market_structure.py
# Modified at 1404/12/07

"""
Market Structure Features (Price Action)
========================================
- تشخیص سوئینگ‌های HH, HL, LH, LL
- شناسایی BOS و CHoCH
"""

import pandas as pd
import numpy as np
from f03_features.indicators.zigzag import zigzag

# ============================================================================= Test at 04/12/08
# Swing Detection
# ============================================================================= Func1
def detect_swings(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:
    """
    Detect swing highs and lows using zigzag indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLC dataframe.
    **zigzag_kwargs :
        Parameters forwarded directly to zigzag().

    Returns
    -------
    pd.DataFrame
        Columns:
            - swing_high (bool)
            - swing_low  (bool)
            - swing_price (float)
    """

    # --- Run zigzag ---
    zz = zigzag(
        df["high"],
        df["low"],
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
    )

    # --- Extract swing states ---
    swing_high = zz["state"] == 1
    swing_low  = zz["state"] == -1

    # --- Build result ---
    result = pd.DataFrame(index=df.index)

    result["swing_high"] = swing_high
    result["swing_low"]  = swing_low

    result["swing_price"] = np.where(
        swing_high,
        zz["high"],
        np.where(swing_low, zz["low"], np.nan)
    )

    return result


# =============================================================================
# Market Structure (HH / HL / LH / LL)
# ============================================================================= Func2
def build_market_structure_old1(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:
    """
    Build market structure (HH, HL, LH, LL) using zigzag-based swings.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLC dataframe.
    **zigzag_kwargs :
        Parameters forwarded directly to zigzag().

    Returns
    -------
    pd.DataFrame
        Columns:
            - swing_high (bool)
            - swing_low  (bool)
            - swing_price (float)
            - HH (bool)
            - HL (bool)
            - LH (bool)
            - LL (bool)
    """

    swings = detect_swings(
        df,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,        
    )

    result = swings.copy()

    # Initialize structure columns
    result["HH"] = False
    result["HL"] = False
    result["LH"] = False
    result["LL"] = False

    # Extract only pivot rows
    pivots = result[
        result["swing_high"] | result["swing_low"]
    ].copy()

    if len(pivots) < 2:
        return result

    prev_price = None
    prev_type = None  # "high" or "low"

    for idx, row in pivots.iterrows():

        current_price = row["swing_price"]
        current_type = "high" if row["swing_high"] else "low"

        if prev_price is not None:

            # High after High → HH / LH
            if current_type == "high" and prev_type == "high":
                if current_price > prev_price:
                    result.at[idx, "HH"] = True
                elif current_price < prev_price:
                    result.at[idx, "LH"] = True

            # Low after Low → HL / LL
            elif current_type == "low" and prev_type == "low":
                if current_price > prev_price:
                    result.at[idx, "HL"] = True
                elif current_price < prev_price:
                    result.at[idx, "LL"] = True

        prev_price = current_price
        prev_type = current_type

    return result

def build_market_structure(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:

    swings = detect_swings(
        df,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
    )

    result = swings.copy()

    result["HH"] = False
    result["HL"] = False
    result["LH"] = False
    result["LL"] = False

    pivots = result[result["swing_high"] | result["swing_low"]]

    if len(pivots) < 2:
        return result

    last_high_price = None
    last_low_price = None

    for idx, row in pivots.iterrows():

        price = row["swing_price"]

        # ---------------------------
        # HIGH pivot (state == +1)
        # ---------------------------
        if row["swing_high"]:

            if last_high_price is not None:
                if price > last_high_price:
                    result.at[idx, "HH"] = True
                elif price < last_high_price:
                    result.at[idx, "LH"] = True

            last_high_price = price

        # ---------------------------
        # LOW pivot (state == -1)
        # ---------------------------
        elif row["swing_low"]:

            if last_low_price is not None:
                if price > last_low_price:
                    result.at[idx, "HL"] = True
                elif price < last_low_price:
                    result.at[idx, "LL"] = True

            last_low_price = price

    return result

# =============================================================================
# BOS / CHOCH Detection
# ============================================================================= Func3
def detect_bos_choch(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Detect:
        - Break of Structure (BOS)
        - Change of Character (CHOCH)

    Requires:
        HH, HL, LH, LL columns
    """

    out = df.copy()

    out["bos_up"] = 0
    out["bos_down"] = 0
    out["choch_up"] = 0
    out["choch_down"] = 0

    last_structure = None

    for idx, row in out.iterrows():

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

        # Bullish continuation
        if current_structure == "HH" and last_structure in ("HL", "HH"):
            out.at[idx, "bos_up"] = 1

        # Bearish continuation
        elif current_structure == "LL" and last_structure in ("LH", "LL"):
            out.at[idx, "bos_down"] = 1

        # Bullish reversal
        elif current_structure == "HH" and last_structure in ("LH", "LL"):
            out.at[idx, "choch_up"] = 1

        # Bearish reversal
        elif current_structure == "LL" and last_structure in ("HL", "HH"):
            out.at[idx, "choch_down"] = 1

        last_structure = current_structure

    return out


# =============================================================================
# Unified Pipeline (Optional Convenience)
# ============================================================================= Func4
def market_structure_pipeline(
    df: pd.DataFrame,
    depth: int = 12,
    deviation: float = 5.0,
    backstep: int = 10,
    point: float = 0.01
) -> pd.DataFrame:
    """
    Full pipeline:
        zigzag → swings → structure → bos/choch
    """

    structure = build_market_structure(
        df,
        depth=depth,
        deviation=deviation,
        backstep=backstep,
        point=point,
    )
    final = detect_bos_choch(structure)

    return final


# =============================================================================
# Regime State Machine (Bull / Bear)
# ============================================================================= Func5
def build_regime_state(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build persistent market regime state machine.

    Requires:
        bos_up, bos_down, choch_up, choch_down

    Returns
    -------
    pd.DataFrame
        Adds:
            - regime (int)
                1  = bullish
               -1  = bearish
                0  = neutral (before first structure event)
    """

    out = df.copy()

    regime = 0
    regimes = []

    for _, row in out.iterrows():

        if row.get("bos_up", 0) == 1 or row.get("choch_up", 0) == 1:
            regime = 1

        elif row.get("bos_down", 0) == 1 or row.get("choch_down", 0) == 1:
            regime = -1

        regimes.append(regime)

    out["regime"] = regimes

    return out

