# f03_features/resolver/feature_B_resolver.py
# DSL-aware Column Resolver Layer (MTF + Prefixed DataHandler compatible)

"""
Engine → مسئول execution logic
Cache → مسئول semantic identity
Resolve layer → مسئول schema mapping
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ...f10_utils.parser import parse_spec

# =============================================================================
# Resolver Core
# =============================================================================
class FeatureResolver:
    """
    Converts DSL spec → actual dataframe column mapping.

    Works with:
    - DataHandler prefixed columns (M5_close, H1_high, ...)
    - parser.py ParsedSpec
    - FeatureEngine execution layer
    """
    # ---------------------------------------------------------------------
    def __init__(self, available_columns: List[str]):
        self.columns = set(available_columns)

    # ---------------------------------------------------------------------
    def resolve(self, spec_str: str) -> Dict[str, List[str]]:
        """
        Returns required dataframe columns for a DSL spec.

        Output:
        {
            "inputs": ["M5_close", "M5_open"],
            "missing": [...]
        }
        """

        parsed = parse_spec(spec_str)

        tf = parsed.timeframe
        if tf is None:
            raise ValueError(f"Timeframe missing in DSL: {spec_str}")

        prefix = tf.upper()

        inputs: List[str] = []
        missing: List[str] = []

        # --- mapping required_cols from registry ---
        # example: ["close", "high"] → ["M5_close", "M5_high"]
        for col in self._expand_required_cols(parsed.name, parsed, prefix):
            if col in self.columns:
                inputs.append(col)
            else:
                missing.append(col)

        return {
            "inputs": inputs,
            "missing": missing
        }

    # ---------------------------------------------------------------------
    def _expand_required_cols(self, name: str, parsed, prefix: str) -> List[str]:
        """
        Maps logical required_cols → prefixed DataHandler columns.
        """

        # base mapping rules
        result: List[str] = []

        for col in self._get_required_cols_stub(name):
            # already prefixed?
            if "_" in col:
                result.append(col)
                continue

            # map logical → prefixed
            result.append(f"{prefix}_{col}")

        return result

    # ---------------------------------------------------------------------
    def _get_required_cols_stub(self, name: str) -> List[str]:
        """
        Stub layer:
        In production engine we will inject registry spec.

        This avoids circular dependency with feature_registry.
        """

        # minimal safe fallback (engine will override later)
        default_map = {
            "sma": ["close"],
            "wma": ["close"],
            "ema": ["close"],
            "roc": ["close"],
            "rsi": ["close"],
            "true_range": ["high", "low", "close"],
            "atr": ["high", "low", "close"],
            "macd": ["close"],
            "bollinger_bands": ["close"],
            "keltner_channel": ["high", "low", "close"],
            "stochastic": ["high", "low", "close"],
            "cci": ["high", "low", "close"],
            "mfi": ["high", "low", "close", "volume"],
            "obv": ["close", "volume"],
            "williams_r": ["high", "low", "close"],
            "parabolic_sar": ["high", "low", "close"],
            "heikin_ashi": ["open", "high", "low", "close"],
            "supertrend": ["high", "low", "close"],
            "aroon": ["high", "low"],
            "dema": ["close"],
            "tema": ["close"],
            "kama": ["close"],
            "hma": ["close"],
        }

        return default_map.get(name, ["close"])


# =============================================================================
# Utility function (stateless API)
# =============================================================================
def resolve_feature_columns(
    df: pd.DataFrame,
    spec_str: str
) -> Dict[str, List[str]]:
    """
    Stateless helper for quick integration in engine.
    """
    resolver = FeatureResolver(list(df.columns))
    return resolver.resolve(spec_str)

#


