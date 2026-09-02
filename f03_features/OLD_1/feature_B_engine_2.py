# f03_features/feature_B_engine.py
# Status: FINAL (registry + resolver + cache aligned)

from __future__ import annotations

from typing import List
import pandas as pd

from .feature_B_cache_1 import cached_compute
from ...f10_utils.functions.parser import parse_spec
from ..feature_B_registry_1 import (
    get_indicator,
    ALL_MODES,
    TRAIN_MODES,
    INCREMENTAL_MODES,
)

class FeatureEngine:
    """
    Execution layer only.
    Responsibilities:
        - validate mode
        - resolve indicator from registry
        - validate required columns
        - execute indicator
    Does NOT:
        - build datasets
        - perform schema resolution
        - manage cache
    """

    # ------------------------------------------------------------------
    def __init__(self, mode: str):
        if mode not in ALL_MODES:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame, spec_str: str):
        """
        Public entrypoint.
        """
        return cached_compute(
            self,
            df,
            spec_str,
            self.mode,
            extra=None,
        )

    # ------------------------------------------------------------------
    def compute_many(
        self,
        df: pd.DataFrame,
        specs: List[str],
    ) -> pd.DataFrame:

        out = pd.DataFrame(index=df.index)
        for spec_str in specs:
            parsed = parse_spec(spec_str)
            result = self.compute(df, spec_str)
            feature_name = self._build_feature_name(parsed)
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    out[f"{feature_name}:{col}"] = result[col]
            else:
                out[feature_name] = result
        return out

    # ------------------------------------------------------------------
    def _execute(
        self,
        df: pd.DataFrame,
        spec_str: str,
    ):

        parsed = parse_spec(spec_str)
        spec = get_indicator(parsed.name, self.mode)
        if spec is None:
            raise ValueError(
                f"Indicator '{parsed.name}' "
                f"is not available in mode '{self.mode}'"
            )
        self._validate_required_columns(
            df=df,
            required_cols=spec.required_cols,
            timeframe=parsed.timeframe,
            indicator_name=parsed.name,
        )
        if self.mode in TRAIN_MODES:
            return self._apply_batch(
                df,
                spec,
                parsed.args,
                parsed.kwargs,
            )
        if self.mode in INCREMENTAL_MODES:
            return self._apply_live(
                df,
                spec,
                parsed.args,
                parsed.kwargs,
            )
        raise ValueError(f"Unsupported execution mode: {self.mode}")

    # ------------------------------------------------------------------
    def _apply_batch(
        self,
        df: pd.DataFrame,
        spec,
        args,
        kwargs,
    ):
        return spec.fn(df, *args, **kwargs)

    # ------------------------------------------------------------------
    def _apply_live(
        self,
        df: pd.DataFrame,
        spec,
        args,
        kwargs,
    ):

        engine = spec.fn(*args, **kwargs)
        out = []
        for _, row in df.iterrows():
            value = engine.update(
                row.to_dict()
            )
            out.append(value)
        return pd.Series(
            out,
            index=df.index,
        )

    # ------------------------------------------------------------------
    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        required_cols,
        timeframe,
        indicator_name,
    ):

        tf = (timeframe or "").upper()
        for col in required_cols:
            mapped_col = (
                col
                if col.startswith(f"{tf}_")
                else f"{tf}_{col}"
            )
            if mapped_col not in df.columns:
                raise ValueError(
                    f"Missing required column "
                    f"'{mapped_col}' "
                    f"for indicator '{indicator_name}'"
                )

    # ------------------------------------------------------------------
    def _build_feature_name(self, parsed) -> str:
        tf = parsed.timeframe
        args_part = [str(x) for x in parsed.args]
        for k, v in parsed.kwargs.items():
            args_part.append(f"{k}={v}")

        args_text = ",".join(args_part)
        if args_text:
            return f"{parsed.name}({args_text})@{tf}"

        return f"{parsed.name}@{tf}"