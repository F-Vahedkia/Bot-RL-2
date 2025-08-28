# f04_features/indicators/registry.py
# -*- coding: utf-8 -*-
"""رجیستری یکپارچه: core + extras + volume + patterns + levels + divergences"""
from __future__ import annotations
from typing import Dict, Callable

Registry = Dict[str, Callable]

def build_registry() -> Registry:
    reg: Registry = {}
    # core
    from .core import registry as core_reg
    reg.update(core_reg())
    # extras
    from .extras_trend import registry as trend_reg
    reg.update(trend_reg())
    from .extras_channel import registry as ch_reg
    reg.update(ch_reg())
    # volume
    from .volume import registry as vol_reg
    reg.update(vol_reg())
    # patterns
    from .patterns import registry as pat_reg
    reg.update(pat_reg())
    # levels
    from .levels import registry as lvl_reg
    reg.update(lvl_reg())
    # divergences
    from .divergences import registry as div_reg
    reg.update(div_reg())
    return reg