from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from f03_features.feature_B_registry_1 import get_indicator, IndicatorSpec
from f10_utils.parser import parse_spec, ParsedSpec

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================
# ENGINE CORE (PURE FEATURE TRANSFORMER)
# ============================================================
class FeatureEngine:

    def __init__(self):
        self._live_cache: Dict[str, Any] = {}

    # --------------------------------------------------------
    def execute(
        self,
        df: pd.DataFrame,
        specs: List[str],
        mode: str = "train"
    ) -> pd.DataFrame:

        if df is None or df.empty:
            return df

        result_df = df.copy()

        parsed_specs: List[ParsedSpec] = []

        # 1. PARSE ALL SPECS
        for spec in specs:
            try:
                parsed_specs.append(parse_spec(spec))
            except Exception as e:
                logger.warning("Invalid spec skipped: %s | %s", spec, e)

        # 2. EXECUTE EACH SPEC
        for ps in parsed_specs:
            result_df = self._apply_spec(result_df, ps, mode)

        return result_df

    # --------------------------------------------------------
    def _apply_spec(
        self,
        df: pd.DataFrame,
        ps: ParsedSpec,
        mode: str
    ) -> pd.DataFrame:

        # 1. get registry entry
        spec: Optional[IndicatorSpec] = get_indicator(ps.name, mode)

        if spec is None:
            logger.warning("Indicator not found in registry: %s", ps.name)
            return df

        # 2. extract function
        fn = spec.fn

        try:
            # ------------------------------------------------
            # 3. BATCH MODE (TRAIN)
            # ------------------------------------------------
            if spec.is_batch_mode(mode):

                out = self._call_batch(fn, df, ps)

                if isinstance(out, pd.DataFrame):
                    df = self._merge(df, out)

                return df

            # ------------------------------------------------
            # 4. LIVE MODE (STATEFUL)
            # ------------------------------------------------
            print(ps.name, mode, spec,                            # for debug
                spec.is_batch_mode(mode) if spec else None,       # for debug
                spec.is_incremental_mode(mode) if spec else None  # for debug
            )            
            # -------------
            
            if spec.is_incremental_mode(mode):
                print("=====================")   # for debug
                df = self._apply_live(spec, df, ps)
                return df

        except Exception as e:
            logger.exception("Execution failed for %s: %s", ps.name, e)

        return df

    # --------------------------------------------------------
    def _call_batch(
        self,
        fn,
        df: pd.DataFrame,
        ps: ParsedSpec
    ) -> pd.DataFrame:

        # resolve args
        kwargs = dict(ps.kwargs)

        # IMPORTANT: inject positional args dynamically
        # (registry-driven, no hardcoding)

        try:
            return fn(df, *ps.args, **kwargs)
        except TypeError:
            # fallback: try keyword-only style
            return fn(df, **kwargs)

    # --------------------------------------------------------
    def _apply_live(
        self,
        spec: IndicatorSpec,
        df: pd.DataFrame,
        ps: ParsedSpec
    ) -> pd.DataFrame:
        print("====>  ENTER _apply_live")  # for debug
        key = ps.name

        # init stateful object once
        if key not in self._live_cache:
            print("ARGS =", ps.args)      # for debug
            print("KWARGS =", ps.kwargs)  # for debug
            try:                          # for debug
                self._live_cache[key] = spec.fn(*ps.args, **ps.kwargs)
            except Exception as e:        # for debug
                print("CTOR ERROR =", e)  # for debug
                raise                     # for debug


        obj = self._live_cache[key]
        print("CACHE CREATED")  # for debug
        print(type(obj))        # for debug

        results = []

        # streaming update row by row
        for _, row in df.iterrows():

            try:
                r = self._update_live(obj, row)
                print("RESULT =", r)  # for debug
                results.append(r)
            except Exception as e:
                print("LIVE ERROR =", e)  # for debug
                results.append(None)

        df[f"{ps.name}_live"] = results
        return df

    # --------------------------------------------------------
    def _update_live(self, obj: Any, row: pd.Series):

        # dynamic routing based on available update signature
        if hasattr(obj, "update"):

            try:
                # multi-input indicators
                return obj.update(*row.values[:3])
            except TypeError:
                return obj.update(row.get("close", None))

        return None

    # --------------------------------------------------------
    def _merge(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """
        Merge strategy:
        - avoid column collision
        - preserve index alignment
        """

        if out is None or out.empty:
            return df

        # prevent overwrite
        for col in out.columns:
            if col in df.columns:
                df[col + "__dup"] = out[col]
            else:
                df[col] = out[col]

        return df
    
    # --------------------------------------------------------    