# f03_features/engine.py

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from f10_utils.functions.parser import parse_spec
from f03_features.feature_B_registry_1 import get_indicator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Helpers
# ============================================================================

_PRICE_TOKENS = {"open", "high", "low", "close", "volume", "spread"}


def _spec_suffix(spec_text: str) -> str:
    """
    ema(close,20)@M5
      -> (close,20)@M5

    atr(14)@M5
      -> (14)@M5
    """
    idx = spec_text.find("(")
    if idx >= 0:
        return spec_text[idx:]
    return ""


def _column_name_for_tf(token: str, tf: str) -> str:
    """
    close -> M5_close
    high  -> M5_high
    """
    return f"{tf}_{token}"


def _rename_output_columns(
    out_df: pd.DataFrame,
    spec_text: str,
) -> pd.DataFrame:
    """
    Example:

    macd
    macd_signal
    macd_hist

    ->
    macd(12,26,9)@M1
    macd_signal(12,26,9)@M1
    macd_hist(12,26,9)@M1
    """

    suffix = _spec_suffix(spec_text)

    rename_map = {
        col: f"{col}{suffix}"
        for col in out_df.columns
    }

    return out_df.rename(columns=rename_map)


# ============================================================================
# Train Engine
# ============================================================================

class FeatureEngine:
    """
    Layer-3 Feature Engine

    Responsibilities:
        - parse feature specs
        - load processed data
        - execute registry batch indicators
        - build feature dataframe

    Non-responsibilities:
        - RL
        - Agent
        - Env
        - Reward
        - Observation
        - Trading
    """

    # ---------------------------------------------------------------------
    # ctor
    # ---------------------------------------------------------------------

    def __init__(
        self,
        cfg: Dict[str, Any],
        mode: str,
    ) -> None:

        self.cfg = cfg
        self.mode = mode

        if mode not in (
            "train",
            "optimize",
            "live",
            "paper",
            "shadow",
            "backtest",
            "replay",
            "eval",
        ):
            raise ValueError(f"unsupported mode: {mode}")

        self.feature_specs: List[str] = (
            ((cfg.get("features") or {}).get("indicators"))
            or []
        )

        self._parsed_specs = [
            parse_spec(spec)
            for spec in self.feature_specs
        ]

        self._live_indicators: Dict[str, Any] = {}

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------

    def build_train_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Input:
            processed dataframe

        Output:
            processed dataframe + indicator features
        """

        feature_frames: List[pd.DataFrame] = []

        for spec_text, parsed in zip(
            self.feature_specs,
            self._parsed_specs,
        ):

            try:
                feat = self._run_batch_indicator(
                    df=df,
                    spec_text=spec_text,
                    parsed=parsed,
                )

                if feat is not None and not feat.empty:
                    feature_frames.append(feat)

            except Exception as ex:
                logger.exception(
                    "feature failed: %s | %s",
                    spec_text,
                    ex,
                )

        if not feature_frames:
            return df.copy()

        features_df = pd.concat(
            feature_frames,
            axis=1,
        )

        return pd.concat(
            [df, features_df],
            axis=1,
        )

    # ---------------------------------------------------------------------
    # batch executor
    # ---------------------------------------------------------------------

    def _run_batch_indicator(
        self,
        df: pd.DataFrame,
        spec_text: str,
        parsed,
    ) -> pd.DataFrame:

        spec = get_indicator(
            parsed.name,
            mode="train",
        )

        if spec is None:
            logger.warning(
                "indicator not found in registry: %s",
                parsed.name,
            )
            return pd.DataFrame(index=df.index)

        fn = spec.fn

        kwargs = self._build_batch_kwargs(
            fn=fn,
            parsed=parsed,
        )

        out = fn(
            df=df,
            **kwargs,
        )

        if isinstance(out, pd.Series):
            out = out.to_frame()

        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"{parsed.name} returned "
                f"{type(out)}"
            )

        out = _rename_output_columns(
            out_df=out,
            spec_text=spec_text,
        )

        return out

    # ---------------------------------------------------------------------
    # kwargs resolver
    # ---------------------------------------------------------------------

    def _build_batch_kwargs(
        self,
        fn,
        parsed,
    ) -> Dict[str, Any]:

        tf = parsed.timeframe

        sig = inspect.signature(fn)

        params = sig.parameters

        kwargs: Dict[str, Any] = {}

        user_args = list(parsed.args)

        for pname in params:

            if pname == "df":
                continue

            # ----------------------------------------------------------
            # OHLC mappings
            # ----------------------------------------------------------

            if pname in (
                "column",
                "close_col",
                "price_col",
            ):
                kwargs[pname] = _column_name_for_tf(
                    "close",
                    tf,
                )
                continue

            if pname == "open_col":
                kwargs[pname] = _column_name_for_tf(
                    "open",
                    tf,
                )
                continue

            if pname == "high_col":
                kwargs[pname] = _column_name_for_tf(
                    "high",
                    tf,
                )
                continue

            if pname == "low_col":
                kwargs[pname] = _column_name_for_tf(
                    "low",
                    tf,
                )
                continue

            if pname == "volume_col":
                kwargs[pname] = _column_name_for_tf(
                    "volume",
                    tf,
                )
                continue

            # ----------------------------------------------------------
            # skip output names
            # ----------------------------------------------------------

            if (
                pname.endswith("_col")
                or pname.endswith("_prefix")
                or pname == "result_col"
            ):
                continue

            if pname == "add_para_to_names":
                kwargs[pname] = False
                continue

            # ----------------------------------------------------------
            # positional values from spec
            # ----------------------------------------------------------

            if user_args:

                value = user_args.pop(0)

                if (
                    isinstance(value, str)
                    and value.lower() in _PRICE_TOKENS
                ):
                    continue

                kwargs[pname] = value

        kwargs.update(parsed.kwargs)

        return kwargs

    # ---------------------------------------------------------------------
    # Live
    # ---------------------------------------------------------------------

    def create_live_indicators(self) -> None:

        self._live_indicators.clear()

        for spec_text, parsed in zip(
            self.feature_specs,
            self._parsed_specs,
        ):

            spec = get_indicator(
                parsed.name,
                mode="live",
            )

            if spec is None:
                logger.warning(
                    "live indicator missing: %s",
                    parsed.name,
                )
                continue

            cls = spec.fn

            ctor_kwargs = self._build_live_ctor_kwargs(
                cls,
                parsed,
            )

            try:

                self._live_indicators[
                    spec_text
                ] = cls(**ctor_kwargs)

            except Exception as ex:

                logger.exception(
                    "unable to create indicator %s : %s",
                    spec_text,
                    ex,
                )

    # ---------------------------------------------------------------------

    def _build_live_ctor_kwargs(
        self,
        cls,
        parsed,
    ) -> Dict[str, Any]:

        sig = inspect.signature(cls.__init__)

        kwargs: Dict[str, Any] = {}

        user_args = list(parsed.args)

        for pname in sig.parameters:

            if pname == "self":
                continue

            if user_args:

                value = user_args.pop(0)

                if (
                    isinstance(value, str)
                    and value.lower() in _PRICE_TOKENS
                ):
                    continue

                kwargs[pname] = value

        kwargs.update(parsed.kwargs)

        return kwargs

    # ---------------------------------------------------------------------

    def update_live_features(
        self,
        candle: Dict[str, float],
    ) -> Dict[str, float]:

        output: Dict[str, float] = {}

        for spec_text, indicator in self._live_indicators.items():

            try:

                value = self._update_one_indicator(
                    indicator,
                    candle,
                )

                self._store_live_result(
                    output,
                    spec_text,
                    value,
                )

            except Exception as ex:

                logger.exception(
                    "live update failed: %s | %s",
                    spec_text,
                    ex,
                )

        return output

    # ---------------------------------------------------------------------

    def _update_one_indicator(
        self,
        indicator,
        candle: Dict[str, float],
    ):

        sig = inspect.signature(
            indicator.update
        )

        args = []

        for pname in sig.parameters:

            args.append(
                candle[pname]
            )

        return indicator.update(*args)

    # ---------------------------------------------------------------------

    def _store_live_result(
        self,
        out: Dict[str, float],
        spec_text: str,
        value,
    ) -> None:

        if not isinstance(
            value,
            tuple,
        ):
            out[spec_text] = value
            return

        base_name = spec_text.split("(")[0]

        suffix = _spec_suffix(spec_text)

        if base_name == "macd":

            names = [
                f"macd{suffix}",
                f"macd_signal{suffix}",
                f"macd_hist{suffix}",
            ]

        elif base_name == "bollinger_bands":

            names = [
                f"bb_upper{suffix}",
                f"bb_middle{suffix}",
                f"bb_lower{suffix}",
                f"bb_width{suffix}",
                f"bb_percent{suffix}",
            ]

        elif base_name == "keltner_channel":

            names = [
                f"kc_upper{suffix}",
                f"kc_middle{suffix}",
                f"kc_lower{suffix}",
                f"kc_width{suffix}",
                f"kc_percent{suffix}",
            ]

        elif base_name == "stochastic":

            names = [
                f"stoch_k{suffix}",
                f"stoch_d{suffix}",
            ]

        elif base_name == "heikin_ashi":

            names = [
                f"ha_open{suffix}",
                f"ha_high{suffix}",
                f"ha_low{suffix}",
                f"ha_close{suffix}",
            ]

        elif base_name == "supertrend":

            names = [
                f"supertrend{suffix}",
                f"st_direction{suffix}",
            ]

        elif base_name == "aroon":

            names = [
                f"aroon_up{suffix}",
                f"aroon_down{suffix}",
                f"aroon_oscillator{suffix}",
            ]

        else:

            names = [
                f"{base_name}_{i}{suffix}"
                for i in range(len(value))
            ]

        for k, v in zip(names, value):
            out[k] = v

    # ---------------------------------------------------------------------

    def reset(self) -> None:

        for obj in self._live_indicators.values():

            if hasattr(obj, "reset"):
                obj.reset()

