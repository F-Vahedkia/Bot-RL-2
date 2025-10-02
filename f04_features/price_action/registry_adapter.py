# -*- coding: utf-8 -*-
# f04_features/price_action/registry_adapter.py
# Status in (Bot-RL-2H): Completed

"""
Price Action — Registry Adapter
===============================
آداپتر به‌روزشده برای ۸ ماژول پرایس‌اکشن.

این ماژول بدون دستکاری رجیستری اندیکاتورهای موجود، یک آداپتر ارائه می‌کند تا
بیلدرهای پرایس‌اکشن را به رجیستری مرکزی (mapping name -> builder) اضافه کنید.

API:
- get_price_action_registry_v2() -> dict[str, callable]
- register_price_action_to_indicators_registry(registry_dict: dict) -> dict

نکات:
- هیچ وابستگی بیرون از f01..f14 ندارد.
- اگر کلیدها قبلاً در رجیستری وجود داشته باشند، با پیشوند جلوگیری از تداخل انجام می‌دهد.
"""

from __future__ import annotations
from typing import Dict, Callable

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

# نگاشت استاندارد نام→بیلدر برای PA
# نام‌ها را ساده و شفاف نگه می‌داریم؛ در صورت نیاز می‌توانند در DSL نیز استفاده شوند.

_PA_REGISTRY_V2: Dict[str, Callable] = {
    "pa_market_structure": _ms.build_market_structure,
    "pa_zones": _zn.build_zones,
    "pa_imbalance": _im.build_imbalance_liquidity,
    "pa_mtf_context": _mtf.build_mtf_context,
    "pa_confluence": _cf.build_confluence,
    "pa_regime": _rg.build_regime,
    "pa_breakouts": _bo.build_breakouts,
    "pa_microchannels": _mc.build_microchannels,
}

# برگرداندن نگاشت کامل بیلدرهای پرایس‌اکشن (نسخه v2).

def get_price_action_registry_v2() -> Dict[str, Callable]:
    return dict(_PA_REGISTRY_V2)

def register_price_action_to_indicators_registry(registry_dict: Dict[str, Callable]) -> Dict[str, Callable]:
    if registry_dict is None or not isinstance(registry_dict, dict):
        raise TypeError("registry_dict must be a dict mapping from str to callable.")
    for k, fn in _PA_REGISTRY_V2.items():
        if k in registry_dict and registry_dict.get(k) is not fn:
            alt = f"pa__{k}"
            if alt not in registry_dict:
                registry_dict[alt] = fn
        else:
            registry_dict.setdefault(k, fn)
    return registry_dict
