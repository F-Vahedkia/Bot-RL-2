# -*- coding: utf-8 -*-
# f09_execution/broker_adapter_mt5.py   (آداپتر بروکر MT5 - اسکلت)
# Status in (Bot-RL-2): Completed

"""
آداپتر بروکر:
- MT5Broker: قابل‌استفاده در صورت فراهم‌بودن اتصال واقعی (این‌جا فقط اسکلت لاگ‌محور).
- NoOpBroker: برای اجراهای غیر زنده (dry/semi-live) که صرفاً لاگ می‌دهد.

نکته: مطابق قوانین پروژه، از حدس دربارهٔ API های خارجی خودداری می‌کنیم؛
در صورت نیاز به اتصال واقعی، همین اسکلت باید با کُد MT5 موجود شما جایگزین/تکمیل شود.
"""

from __future__ import annotations
import logging

LOGGER = logging.getLogger("broker")


class NoOpBroker:
    """آداپتر No-Op برای اجراهای بدون بروکر (صرفاً لاگ)."""
    def buy(self, symbol: str, lot: float) -> None: LOGGER.info("[NOOP] BUY %s lot=%.3f", symbol, lot)
    def sell(self, symbol: str, lot: float) -> None: LOGGER.info("[NOOP] SELL %s lot=%.3f", symbol, lot)
    def close_long(self, symbol: str) -> None: LOGGER.info("[NOOP] CLOSE LONG %s", symbol)
    def close_short(self, symbol: str) -> None: LOGGER.info("[NOOP] CLOSE SHORT %s", symbol)
    
    # Optional helpers for risk sizing (returning None by design)
    def get_account_equity(self) -> float | None:
        return None

    def get_pip_value_per_lot(self, symbol: str) -> float | None:
        return None


class MT5Broker:
    """
    اسکلت آداپتر واقعی MT5 (بدون حدس دربارهٔ جزئیات).
    اگر ماژول/اتصال MT5 در پروژهٔ شما موجود است، بدنهٔ متدها را با فراخوان‌های واقعی جایگزین کنید.
    """
    def __init__(self) -> None:
        LOGGER.info("MT5Broker initialized (skeleton).")

    def buy(self, symbol: str, lot: float) -> None:
        LOGGER.info("BUY %s lot=%.3f (skeleton)", symbol, lot)

    def sell(self, symbol: str, lot: float) -> None:
        LOGGER.info("SELL %s lot=%.3f (skeleton)", symbol, lot)

    def close_long(self, symbol: str) -> None:
        LOGGER.info("CLOSE LONG %s (skeleton)", symbol)

    def close_short(self, symbol: str) -> None:
        LOGGER.info("CLOSE SHORT %s (skeleton)", symbol)

    # Optional helpers; replace bodies with real MT5 calls if available in your project
    def get_account_equity(self) -> float | None:
        LOGGER.info("get_account_equity() not implemented in skeleton.")
        return None

    def get_pip_value_per_lot(self, symbol: str) -> float | None:
        LOGGER.info("get_pip_value_per_lot(%s) not implemented in skeleton.", symbol)
        return None

