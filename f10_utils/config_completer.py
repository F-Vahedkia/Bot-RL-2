# f10_utils/config_completer.py
# Run: python -m f10_utils.config_completer
# -----------------------------------------------------------------------------
from typing import Dict, List
from f10_utils.config_loader import load_config
from f10_utils.parse_warmups import get_warmup_from_config_allsyms
import logging

# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------------------------------------------------
def config_completer(enable_env_override=True) -> Dict:
    """
    Some Data structures:
    warmups_dicts = {
        "XAUUSD" : {'M1': 14,  'M5':9 , 'H1':14, 'D1':12},
        "EURUSD" : {'M5': 12, 'M30':14, 'D1':5 },
        "BITCOIN": {'H2': 16,  'D1':9 },
    }
    symbols = ["XAUUSD", "EURUSD", "BITCOIN"]

    timeframes_dict = {
        "XAUUSD" : ['M1',  'M5', 'H1', 'D1],
        "EURUSD" : ['M5', 'M30', 'D1'],
        "BITCOIN": ['H2',  'D1'],
    }

    base_tfs_dict = {
        "XAUUSD" : 'M1' ,
        "EURUSD" : 'M5' ,
        "BITCOIN": 'M12',
    }
    """
    cfg = load_config(enable_env_override=enable_env_override)
    if cfg is None:
        raise ValueError("Message for f10_utils/config_completer: cfg is None. Therefore can't initialize robot.")

    # 1) Warmup Dicts
    warmups_dicts: Dict[str, Dict[str, int]] = get_warmup_from_config_allsyms(cfg)
    
    if warmups_dicts == {}:
        print("WARNING ! Message for f10_utils/config_completer: In config.features.symbols there is no any timeframe.")
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

# ===============================================
# TESTER
# ===============================================
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

# ===============================================

if __name__ == "__main__":
    raise SystemExit(main())