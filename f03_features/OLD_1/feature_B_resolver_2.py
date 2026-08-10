# f03_features/feature_B_resolver.py

from __future__ import annotations
from typing import Dict, List
import pandas as pd
from ...f10_utils.parser import parse_spec
from ..feature_B_registry_1 import get_indicator


class FeatureResolver:
    """
    DSL → required columns ONLY (no dataframe logic)
    """

    def __init__(self, available_columns: List[str]):
        self.columns = set(available_columns)

    def resolve(self, spec_str: str) -> Dict[str, List[str]]:
        parsed = parse_spec(spec_str)

        mode = getattr(parsed, "mode", None)
        if mode is None:
            mode = "train"
        if isinstance(spec, dict):
            spec = spec.get("batch") or spec.get("live")
        required = spec.required_cols if spec else []

        inputs = []
        missing = []

        for col in self._expand(required, parsed):
            if col in self.columns:
                inputs.append(col)
            else:
                missing.append(col)

        return {
            "inputs": inputs,
            "missing": missing
        }

    def _expand(self, cols: List[str], parsed) -> List[str]:
        """
        ONLY prefix expansion (no logic duplication with engine)
        """
        tf = (parsed.timeframe or "").upper()

        out = []
        for c in cols:
            if "_" in c:
                out.append(c)
            else:
                out.append(f"{tf}_{c}")
        return out


def resolve_feature_columns(df: pd.DataFrame, spec_str: str):
    return FeatureResolver(list(df.columns)).resolve(spec_str)

