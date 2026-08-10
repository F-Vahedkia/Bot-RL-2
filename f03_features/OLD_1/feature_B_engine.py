# f03_features/feature_B_engine.py
# Status: FINAL CONSISTENT ENGINE (DSL-aware unified execution)

from __future__ import annotations
from typing import List
import pandas as pd
from .feature_B_cache_1 import cached_compute
from ...f10_utils.parser import parse_spec
from ..feature_B_registry_1 import (
    get_indicator,
    ALL_MODES,
    INCREMENTAL_MODES,
    TRAIN_MODES
)

# =============================================================================
# Core Engine
# =============================================================================
class FeatureEngine:
    """
    Single-responsibility DSL engine:
    RULE:
        ALL execution MUST go through cached_compute
        cached_compute decides execution path internally
    """

    def __init__(self, mode: str):
        if mode not in ALL_MODES:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

    # -------------------------------------------------------------------------
    def _resolve_indicator(self, name: str, parsed):
        spec = get_indicator(name, self.mode)

        if spec is None:
            raise ValueError(
                f"Indicator '{name}' not available in mode '{self.mode}'"
            )
        return spec

    # -------------------------------------------------------------------------
    def _apply_batch(self, df: pd.DataFrame, spec, args, kwargs):
        return spec.fn(df, *args, **kwargs)

    # -------------------------------------------------------------------------
    def _apply_live(self, df: pd.DataFrame, spec, args, kwargs):
        engine = spec.fn(*args, **kwargs)

        out = []
        for _, row in df.iterrows():
            out.append(engine.update(row.to_dict()))

        return pd.Series(out, index=df.index)

    # -------------------------------------------------------------------------
    def compute(self, df: pd.DataFrame, spec_str: str):
        """
        SINGLE ENTRYPOINT (FINAL CONTRACT)
        """
        return cached_compute(
            self,
            df,
            spec_str,
            self.mode,
            extra=None
        )

    # -------------------------------------------------------------------------
    def compute_many(self, df: pd.DataFrame, specs: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for s in specs:
            parsed = parse_spec(s)
            res = self.compute(df, s)

            feature_name = self._build_feature_name(s, parsed)

            if isinstance(res, pd.DataFrame):
                for c in res.columns:
                    out[f"{feature_name}:{c}"] = res[c]
            else:
                out[feature_name] = res

        return out

    # =============================================================================
    # INTERNAL EXECUTION (ONLY USED BY CACHE)
    # =============================================================================
    def _execute(self, df: pd.DataFrame, spec_str: str):
        parsed = parse_spec(spec_str)

        spec = self._resolve_indicator(parsed.name, parsed)

        for col in spec.required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column '{col}' for {parsed.name}"
                )

        if self.mode in TRAIN_MODES:
            return self._apply_batch(df, spec, parsed.args, parsed.kwargs)

        if self.mode in INCREMENTAL_MODES:
            return self._apply_live(df, spec, parsed.args, parsed.kwargs)

        raise ValueError(f"Unsupported mode: {self.mode}")

    # -------------------------------------------------------------------------
    def _build_feature_name(self, spec_str: str, parsed) -> str:
        tf = parsed.timeframe or "base"

        args_part = [str(x) for x in parsed.args]

        for k, v in parsed.kwargs.items():
            args_part.append(f"{k}={v}")

        args_text = ",".join(args_part)

        return (
            f"{parsed.name}({args_text})@{tf}"
            if args_text
            else f"{parsed.name}@{tf}"
        )
    