# -*- coding: utf-8 -*-
# f04_features/price_action/registry.py
# Status in (Bot-RL-2H): Completed
"""
Price Action — Registry & Orchestration (Extended)
==================================================
رجیستری و ارکستریشن پرایس‌اکشن با ۸ ماژول:

- market_structure.build_market_structure
- zones.build_zones
- imbalance.build_imbalance_liquidity
- mtf_context.build_mtf_context
- confluence.build_confluence
- regime.build_regime
- breakouts.build_breakouts
- microchannels.build_microchannels
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional
import pandas as pd

from f04_features.price_action import (
    market_structure as _ms,
    zones as _zn,
    imbalance as _im,
    mtf_context as _mtf,
    confluence as _cf,
    regime as _rg,
    breakouts as _bo,
    microchannels as _mc,
)

# ------------------------------------------------------------
# Public: لیست نام‌ها و دسترسی به بیلدرها
# ------------------------------------------------------------
def list_pa_features() -> List[str]:
    return [
        "pa_market_structure",
        "pa_zones",
        "pa_imbalance",
        "pa_mtf_context",
        "pa_confluence",
        "pa_regime",
        "pa_breakouts",
        "pa_microchannels",
    ]


def get_pa_builder(name: str) -> Callable:
    mapping: Dict[str, Callable] = {
        "pa_market_structure": _ms.build_market_structure,
        "pa_zones": _zn.build_zones,
        "pa_imbalance": _im.build_imbalance_liquidity,
        "pa_mtf_context": _mtf.build_mtf_context,
        "pa_confluence": _cf.build_confluence,
        "pa_regime": _rg.build_regime,
        "pa_breakouts": _bo.build_breakouts,
        "pa_microchannels": _mc.build_microchannels,
    }
    if name not in mapping:
        raise KeyError(f"Unknown PA feature builder: {name}")
    return mapping[name]


# ------------------------------------------------------------
# Orchestrator: ساخت یک‌جای بستهٔ پرایس‌اکشن
# ------------------------------------------------------------
def build_all_price_action_features(
    df_base: pd.DataFrame,
    *,
    df_higher: Optional[pd.DataFrame] = None,
    anti_lookahead: bool = True,
    # پارامترهای اختیاری عبوری به زیرماژول‌ها:
    market_structure_kwargs: Optional[dict] = None,
    zones_kwargs: Optional[dict] = None,
    imbalance_kwargs: Optional[dict] = None,
    mtf_context_kwargs: Optional[dict] = None,
    confluence_kwargs: Optional[dict] = None,
    regime_kwargs: Optional[dict] = None,
    breakouts_kwargs: Optional[dict] = None,
    microchannels_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    ترتیب اجرای پیشنهادی:
      1) Market Structure
      2) Regime Detector
      3) Zones (SD/OB)
      4) Imbalance & Liquidity (FVG, Sweeps)
      5) Breakouts (وابسته به رنج/کانال rolling)
      6) Micro-channels
      7) MTF Context (با df_higher اختیاری)
      8) Confluence (تجمیع نهایی)
    """
    ms_kw = dict(lookback=3)
    if market_structure_kwargs:
        ms_kw.update(market_structure_kwargs)

    rg_kw = dict(anti_lookahead=anti_lookahead)
    if regime_kwargs:
        rg_kw.update(regime_kwargs)

    zn_kw = dict(anti_lookahead=anti_lookahead)
    if zones_kwargs:
        zn_kw.update(zones_kwargs)

    im_kw = dict(anti_lookahead=anti_lookahead)
    if imbalance_kwargs:
        im_kw.update(imbalance_kwargs)

    bo_kw = dict(anti_lookahead=anti_lookahead)
    if breakouts_kwargs:
        bo_kw.update(breakouts_kwargs)

    mc_kw = dict(anti_lookahead=anti_lookahead)
    if microchannels_kwargs:
        mc_kw.update(microchannels_kwargs)

    mtf_kw = dict(
        df_higher=df_higher,
        anti_lookahead=anti_lookahead,
        base_close_col="close",
        higher_close_col="close",
    )
    if mtf_context_kwargs:
        mtf_kw.update(mtf_context_kwargs)

    cf_kw = dict(
        anti_lookahead=anti_lookahead,
        strong_entry_threshold=0.7,
        filter_threshold=0.35,
    )
    if confluence_kwargs:
        cf_kw.update(confluence_kwargs)

    out = df_base.copy()

    out = _ms.build_market_structure(out, **ms_kw)
    out = _rg.build_regime(out, **rg_kw)
    out = _zn.build_zones(out, **zn_kw)
    out = _im.build_imbalance_liquidity(out, **im_kw)
    out = _bo.build_breakouts(out, **bo_kw)
    out = _mc.build_microchannels(out, **mc_kw)
    out = _mtf.build_mtf_context(out, **mtf_kw)
    out = _cf.build_confluence(out, **cf_kw)

    return out
