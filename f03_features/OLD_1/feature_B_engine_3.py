# f03_features/engine/feature_B_engine.py

from __future__ import annotations

from typing import Dict, Any
import pandas as pd

from f10_utils.parser import parse_spec
from f03_features.feature_B_registry_1 import get_indicator
from f03_features.OLD.feature_B_resolver_1 import FeatureResolver
from f03_features.OLD.feature_B_cache_1 import cached_compute


class FeatureEngine:
    """
    DSL-driven production feature engine
    """

    def __init__(self, mode: str = "train"):
        self.mode = mode

    # ---------------------------------------------------------
    def compute(self, df: pd.DataFrame, spec_str: str) -> pd.DataFrame:
        return cached_compute(
            engine=self,
            df=df,
            spec_str=spec_str,
            mode=self.mode,
        )

    # ---------------------------------------------------------
    def _execute(self, df: pd.DataFrame, spec_str: str) -> pd.DataFrame:

        # 1. parse DSL
        parsed = parse_spec(spec_str)

        # 2. registry lookup
        spec = get_indicator(parsed.name, self.mode)

        if spec is None:
            raise ValueError(f"Unknown or unsupported indicator: {parsed.name}")

        # 3. resolve required columns
        resolver = FeatureResolver(list(df.columns))
        resolved = resolver.resolve(spec_str)

        if resolved["missing"]:
            raise ValueError(
                f"Missing columns for {parsed.name}: {resolved['missing']}"
            )

        # 4. BUILD ARGS (CRITICAL FIX)
        kwargs = dict(parsed.kwargs)

        # inject required columns automatically
        for col in spec.required_cols:
            tf = parsed.timeframe
            mapped = col if col.startswith("open") or col.startswith("high") \
                    or col.startswith("low") or col.startswith("close") \
                    or col.startswith("volume") else col

            if col in ["open", "high", "low", "close", "volume"]:
                kwargs.setdefault(col, df[col])

        # 5. special case: single-column indicators
        if "column" not in kwargs and spec.required_cols == ["close"]:
            kwargs["column"] = "close"

        # 6. execute
        result = spec.fn(df, **kwargs)

        # 7. strict contract
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"[ENGINE CONTRACT ERROR] {parsed.name} must return DataFrame "
                f"but got {type(result)}"
            )

        return result
    