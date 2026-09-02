
# f10_utils/config_completer.py
# Date reviewed:
#    1405/05/24-17:25 --> run result is ?????????????????

# Run: python -m f10_utils.config_completer

# =============================================================================
#  Imports
# =============================================================================
from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path
from f10_utils.config_loader import load_config
from f10_utils.functions.parse_warmups import get_warmup_from_config_allsyms
import logging

# =============================================================================
# Logger
# =============================================================================
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# Merge common symbols
# =============================================================================
def _apply_common_symbol_config(cfg: Dict) -> Dict:
    """
    Apply features.symbols.COMMON settings to all symbols listed in
    COMMON.symbols.

    COMMON.candles are appended to each target symbol's candles.
    COMMON.indicators are appended to each target symbol's indicators.

    The COMMON section itself is not included as a trading symbol.
    """

    symbols_cfg = cfg.get("features", {}).get("symbols", {})

    if not isinstance(symbols_cfg, dict):
        raise ValueError(
            "config.features.symbols must be a dictionary."
        )

    common_cfg = symbols_cfg.get("COMMON")

    if common_cfg is None:
        return cfg

    if not isinstance(common_cfg, dict):
        raise ValueError(
            "config.features.symbols.COMMON must be a dictionary."
        )

    common_symbols = common_cfg.get("symbols", [])

    if not isinstance(common_symbols, list):
        raise ValueError(
            "config.features.symbols.COMMON.symbols must be a list."
        )

    common_candles = common_cfg.get("candles", [])
    common_indicators = common_cfg.get("indicators", [])

    if not isinstance(common_candles, list):
        raise ValueError(
            "config.features.symbols.COMMON.candles must be a list."
        )

    if not isinstance(common_indicators, list):
        raise ValueError(
            "config.features.symbols.COMMON.indicators must be a list."
        )

    for symbol in common_symbols:

        if symbol not in symbols_cfg:
            raise ValueError(
                f"Symbol '{symbol}' is listed in COMMON.symbols "
                f"but does not exist in config.features.symbols."
            )
        symbol_cfg = symbols_cfg[symbol]
        if not isinstance(symbol_cfg, dict):
            raise ValueError(
                f"config.features.symbols.{symbol} must be a dictionary."
            )
        # -------------------------------------------------
        # candles
        # -------------------------------------------------
        symbol_candles = symbol_cfg.setdefault("candles", [])
        if not isinstance(symbol_candles, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.candles must be a list."
            )
        for candle in common_candles:
            if candle not in symbol_candles:
                symbol_candles.append(candle)

        # -------------------------------------------------
        # indicators
        # -------------------------------------------------
        symbol_indicators = symbol_cfg.setdefault("indicators", [])
        if not isinstance(symbol_indicators, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.indicators must be a list."
            )
        for indicator in common_indicators:
            if indicator not in symbol_indicators:
                symbol_indicators.append(indicator)

    return cfg


def _apply_common_symbol_config_2(cfg: Dict) -> Dict:
    """
    Expand the COMMON configuration into the selected symbols.

    COMMON.symbols:
        List of symbols that should receive the common configuration.

    COMMON.candles:
        Candles added to each selected symbol.

    COMMON.indicators:
        Indicators added to each selected symbol.

    A symbol does NOT need to have candles or indicators beforehand.
    Missing candles/indicators are created automatically.

    The COMMON section itself is not treated as a trading symbol.
    """

    # ------------------------------------------------------------------
    # Locate features.symbols
    # ------------------------------------------------------------------
    features_cfg = cfg.get("features", {})

    if not isinstance(features_cfg, dict):
        raise ValueError(
            "config.features must be a dictionary."
        )

    symbols_cfg = features_cfg.get("symbols", {})

    if not isinstance(symbols_cfg, dict):
        raise ValueError(
            "config.features.symbols must be a dictionary."
        )

    # ------------------------------------------------------------------
    # COMMON
    # ------------------------------------------------------------------
    common_cfg = symbols_cfg.get("COMMON")

    # No COMMON section -> nothing to do
    if common_cfg is None:
        return cfg

    if not isinstance(common_cfg, dict):
        raise ValueError(
            "config.features.symbols.COMMON must be a dictionary."
        )

    # ------------------------------------------------------------------
    # COMMON.symbols
    # ------------------------------------------------------------------
    common_symbols = common_cfg.get("symbols", [])

    if common_symbols is None:
        common_symbols = []

    if not isinstance(common_symbols, list):
        raise ValueError(
            "config.features.symbols.COMMON.symbols must be a list."
        )

    # ------------------------------------------------------------------
    # COMMON.candles
    # ------------------------------------------------------------------
    common_candles = common_cfg.get("candles", [])

    if common_candles is None:
        common_candles = []

    if not isinstance(common_candles, list):
        raise ValueError(
            "config.features.symbols.COMMON.candles must be a list."
        )

    # ------------------------------------------------------------------
    # COMMON.indicators
    # ------------------------------------------------------------------
    common_indicators = common_cfg.get("indicators", [])

    if common_indicators is None:
        common_indicators = []

    if not isinstance(common_indicators, list):
        raise ValueError(
            "config.features.symbols.COMMON.indicators must be a list."
        )

    # ------------------------------------------------------------------
    # Apply COMMON to every selected symbol
    # ------------------------------------------------------------------
    for symbol in common_symbols:

        # --------------------------------------------------------------
        # The symbol must exist as a real symbol in the configuration.
        # --------------------------------------------------------------
        if symbol not in symbols_cfg:
            raise ValueError(
                f"Symbol '{symbol}' is listed in "
                f"config.features.symbols.COMMON.symbols "
                f"but does not exist in config.features.symbols."
            )

        symbol_cfg = symbols_cfg[symbol]

        if not isinstance(symbol_cfg, dict):
            raise ValueError(
                f"config.features.symbols.{symbol} must be a dictionary."
            )

        # ==============================================================
        # candles
        # ==============================================================

        # The symbol may have NO candles at all.
        # In that case create an empty list first.
        symbol_candles = symbol_cfg.get("candles")

        if symbol_candles is None:
            symbol_candles = []
            symbol_cfg["candles"] = symbol_candles

        elif not isinstance(symbol_candles, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.candles "
                f"must be a list."
            )

        # Add COMMON candles.
        for candle in common_candles:
            if candle not in symbol_candles:
                symbol_candles.append(candle)

        # ==============================================================
        # indicators
        # ==============================================================

        # The symbol may have NO indicators at all.
        # In that case create an empty list first.
        symbol_indicators = symbol_cfg.get("indicators")

        if symbol_indicators is None:
            symbol_indicators = []
            symbol_cfg["indicators"] = symbol_indicators

        elif not isinstance(symbol_indicators, list):
            raise ValueError(
                f"config.features.symbols.{symbol}.indicators "
                f"must be a list."
            )

        # Add COMMON indicators.
        for indicator in common_indicators:
            if indicator not in symbol_indicators:
                symbol_indicators.append(indicator)

    # ------------------------------------------------------------------
    # Return the completed configuration
    # ------------------------------------------------------------------
    return cfg

# =============================================================================
# Main
# =============================================================================

def config_completer(path: Optional[Union[str, Path]] = None,
                    env_prefix: str = "BOT_",
                    enable_env_override: bool = True,
                    copy_: Literal["main", "shallow", "mutable-safe", "deep"] = "shallow"
                    ) -> Dict[str, Any]:
    """

    Some Data structures:
    __warmups_dicts = {
        "XAUUSD" : {'M1': 14,  'M5':9 , 'H1':14, 'D1':12},
        "EURUSD" : {'M5': 12, 'M30':14, 'D1':5 },
        "BITCOIN": {'H2': 16,  'D1':9 },
    }
    __symbols = ["XAUUSD", "EURUSD", "BITCOIN"]

    __timeframes_dict = {
        "XAUUSD" : ['M1',  'M5', 'H1', 'D1],
        "EURUSD" : ['M5', 'M30', 'D1'],
        "BITCOIN": ['H2',  'D1'],
    }

    __base_tfs_dict = {
        "XAUUSD" : 'M1' ,
        "EURUSD" : 'M5' ,
        "BITCOIN": 'M12',
    }
    """
    cfg = load_config(
        path=path,
        env_prefix=env_prefix,
        enable_env_override=enable_env_override,
        copy_=copy_,
        )

    if cfg is None:
        raise ValueError("Message from f10_utils/config_completer: cfg is None."
                         "Therefore can't initialize robot.")

    # -----------------------------------------------------
    # Apply COMMON symbol configuration
    # -----------------------------------------------------
    cfg = _apply_common_symbol_config_2(cfg)

    # -----------------------------------------------------
    # 1) Warmup Dicts
    # -----------------------------------------------------
    warmups_dicts: Dict[str, Dict[str, int]] = get_warmup_from_config_allsyms(cfg)
    warmups_dicts.pop("COMMON", None)

    # -----------------------------------------------------
    # solve symbols, timeframes_dict, base_tfs_dict
    # -----------------------------------------------------
    if warmups_dicts == {}:
        print("WARNING ! Message from f10_utils/config_completer: In config.features.symbols there is no any timeframe.")
        cfg["__warmups_dicts"] = {}
        cfg["__symbols"] = []
        cfg["__timeframes_dict"] = {}
    else:
        # 1) Warmup Dicts
        cfg["__warmups_dicts"] = warmups_dicts   # یک دیکشنری که حاوی چندین warmup_dict است

        # 2) --- symbols
        symbols: List[str] = list(warmups_dicts.keys())

        cfg["__symbols"] = symbols

        # 3) --- timeframes
        timeframes_dict = {sym: list(warmup.keys()) for sym, warmup in warmups_dicts.items()}
        cfg["__timeframes_dict"] = timeframes_dict

        # 4) --- base_tfs_dict
        base_tfs_dict = {sym: list(warmup.keys())[0] for sym, warmup in warmups_dicts.items()}
        """
        # بلوک زیر، همان کار سطر بالا را انجام میدهد.
        # ولی نمادهایی که دیکشنری وارم-آپ انها تهی است را بطور خودکار به حساب نمی آورد.
        """
        # base_tfs_dict = {}
        # for sym, warmup in warmups_dicts.items():
        #     if warmup.keys():
        #         base_tfs_dict[sym] = list(warmup.keys())[0]
        cfg["__base_tfs_dict"] = base_tfs_dict

    return cfg

# ===================================================================
# TESTER
# ===================================================================
def main():
    cfg = config_completer()

    print("\n", "="*3, "__warmups_dicts", "="* (60-len("__warmups_dicts")))
    print(cfg["__warmups_dicts"])

    print("\n", "="*3, "__symbols", "="* (60-len("__symbols")))
    print(cfg["__symbols"])

    print("\n", "="*3, "__timeframes_dict", "="* (60-len("__timeframes_dict")))
    print(cfg["__timeframes_dict"])

    print("\n", "="*3, "__base_tfs_dict", "="* (60-len("__base_tfs_dict")))
    print(cfg["__base_tfs_dict"])

    for sym in cfg["__symbols"]:
        print("\n", "="*3, {sym} , "="* (60-len(sym)-4))
        specs = list(cfg["features"]["symbols"][sym]["indicators"])
        for s in specs:
            print(s)
        print("\n", (cfg["features"]["symbols"][sym]["candles"]))
    print("\n")
# ===================================================================
if __name__ == "__main__":
    raise SystemExit(main())
