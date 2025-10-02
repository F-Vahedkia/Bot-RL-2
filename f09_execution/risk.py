# -*- coding: utf-8 -*-
# f09_execution/risk.py
# Status in (Bot-RL-2): Completed

"""
ماژول مدیریت ریسک و سایز پوزیشن (سطح حرفه‌ای)
- بدون هرگونه حدس: فقط از ساختار موجود در config.yaml (کلید risk/position_sizing و risk_per_trade) استفاده می‌کند.
- هیچ وابستگی به بروکر/MT5 ندارد؛ صرفاً محاسبات عددی/گِیت‌های ریسک را انجام می‌دهد.
- خروجی‌ها به «لات» (Lots) هستند؛ تبدیل ارزش هر پیپ/تیک باید توسط Caller (مثلاً آداپتر بروکر یا لایهٔ اجرا) تامین شود.
- پیام‌های runtime انگلیسی؛ توضیحات فارسی.

نحوهٔ استفادهٔ معمول:
    from f10_utils.config_loader import load_config
    from f09_execution.risk import RiskManager

    cfg = load_config("f01_config/config.yaml", enable_env_override=True)
    rm = RiskManager.from_config(cfg)

    # فرض: این مقادیر را از لایهٔ اجرا به‌صورت عددی می‌دهید:
    equity_usd = 10000.0
    atr_price = 1.25    # ATR برحسب «واحد قیمت» (نه پیپ)
    pip_value_per_lot = 1.0  # ارزش هر پیپ برای 1 لات (USD/pip/lot)، باید توسط Caller داده شود.
    price_point = 0.01       # اندازهٔ یک پیپ/پوینت (برای تبدیل قیمت→پیپ)، اگر لازم شد.

    stop_dist_price = rm.stop_distance_from_atr(atr_price)  # اگر use_atr_for_sl=True باشد
    lot = rm.size_fixed_fractional(
        equity_usd=equity_usd,
        stop_distance_price=stop_dist_price,
        pip_value_per_lot=pip_value_per_lot,
        price_point=price_point
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import logging
import math

LOGGER = logging.getLogger("risk")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


# ===============================
# خواندن پیکربندی ریسک از cfg
# ===============================

@dataclass
class PositionSizingCfg:
    """پیکربندی سایز پوزیشن (از cfg['risk']['position_sizing'])."""
    method: str = "fixed_fractional"
    min_lot: float = 0.01
    max_lot: float = 10.0
    lot_step: float = 0.01
    use_atr_for_sl: bool = True
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.0
    breakeven_enabled: bool = False
    breakeven_r_multiple: float = 1.0
    trailing_enabled: bool = False
    trailing_type: str = "chandelier"
    trailing_atr_mult: float = 3.0
    trailing_atr_period: int = 22


@dataclass
class RiskCfg:
    """پیکربندی کلی ریسک (از cfg['risk'])."""
    risk_per_trade: float = 0.01                # نسبت ریسک هر معامله نسبت به equity (مثلاً 0.01 = 1%)
    max_daily_loss_pct: float = 0.03            # حداکثر زیان روزانه نسبت به equity
    max_total_drawdown_pct: float = 0.20        # حداکثر افت سرمایه نسبت به اوج
    position_limits: Dict[str, Any] = None      # شامل: max_open_positions, max_positions_per_symbol, ...
    position_sizing: PositionSizingCfg = field(default_factory=PositionSizingCfg)


class RiskManager:
    """
    هستهٔ محاسبات ریسک/سایز پوزیشن.
    - وابستگی به ATR فقط در حد «عدد ATR برحسب واحد قیمت» است و توسط Caller داده می‌شود.
    - تبدیل قیمت به پیپ/پوینت (price_point) نیز توسط Caller مشخص می‌شود.
    """

    def __init__(self, rcfg: RiskCfg) -> None:
        self.rcfg = rcfg

    # -------- سازنده از cfg --------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RiskManager":
        r = (cfg.get("risk") or {})
        ps = (r.get("position_sizing") or {})

        # breakeven
        be = ps.get("breakeven") or {}
        trailing = ps.get("trailing") or {}

        ps_cfg = PositionSizingCfg(
            method=str(ps.get("method", "fixed_fractional")),
            min_lot=float(ps.get("min_lot", 0.01)),
            max_lot=float(ps.get("max_lot", 10.0)),
            lot_step=float(ps.get("lot_step", 0.01)),
            use_atr_for_sl=bool(ps.get("use_atr_for_sl", True)),
            sl_atr_mult=float(ps.get("sl_atr_mult", 1.5)),
            tp_atr_mult=float(ps.get("tp_atr_mult", 2.0)),
            breakeven_enabled=bool(be.get("enabled", False)),
            breakeven_r_multiple=float(be.get("at_r_multiple", 1.0)),
            trailing_enabled=bool(trailing.get("enabled", False)),
            trailing_type=str(trailing.get("type", "chandelier")),
            trailing_atr_mult=float(trailing.get("atr_mult", 3.0)),
            trailing_atr_period=int(trailing.get("atr_period", 22)),
        )

        rcfg = RiskCfg(
            risk_per_trade=float(r.get("risk_per_trade", 0.01)),
            max_daily_loss_pct=float(r.get("max_daily_loss_pct", 0.03)),
            max_total_drawdown_pct=float(r.get("max_total_drawdown_pct", 0.20)),
            position_limits=(r.get("position_limits") or {}),
            position_sizing=ps_cfg,
        )
        return cls(rcfg)

    # -------- ابزارهای کمکی داخلی --------
    @staticmethod
    def _round_step(x: float, step: float, min_x: float, max_x: float) -> float:
        """گرد کردن مقدار x به نزدیک‌ترین گام step و بستن در بازه [min_x, max_x]."""
        if step <= 0:
            return max(min_x, min(max_x, x))
        k = round(x / step)
        y = k * step
        if y < min_x:
            y = min_x
        if y > max_x:
            y = max_x
        return float(f"{y:.8f}")

    # ===============================
    # 1) محاسبهٔ حد ضرر بر مبنای ATR
    # ===============================
    def stop_distance_from_atr(self, atr_price: float) -> float:
        """
        محاسبهٔ فاصلهٔ حدضرر برحسب «واحد قیمت» از روی ATR.
        - اگر use_atr_for_sl=False باشد، برمی‌گرداند 0 (Caller باید روش دیگری بدهد).
        - atr_price باید برحسب «همان واحد قیمتی» باشد که قیمت‌ها هستند (نه پیپ).
        """
        ps = self.rcfg.position_sizing
        if not ps.use_atr_for_sl:
            return 0.0
        if atr_price is None or atr_price <= 0:
            LOGGER.warning("ATR value is invalid for stop calculation. Returning 0.")
            return 0.0
        return float(ps.sl_atr_mult * atr_price)

    # ===============================
    # 2) سایز پوزیشن (Fixed Fractional)
    # ===============================
    def size_fixed_fractional(
        self,
        *,
        equity_usd: float,
        stop_distance_price: float,
        pip_value_per_lot: float,
        price_point: Optional[float] = None,
    ) -> float:
        """
        محاسبهٔ سایز پوزیشن به «لات» با روش fixed_fractional:
            lot = (equity * risk_per_trade) / (stop_pips * pip_value_per_lot)

        - equity_usd: حقوق صاحبان سهام فعلی (USD)
        - stop_distance_price: فاصلهٔ حدضرر برحسب «واحد قیمت» (مثلاً اگر قیمت XAUUSD دو رقم اعشار باشد، 0.50 یعنی 50 پوینت)
        - pip_value_per_lot: ارزش هر پیپ برای 1 لات (USD/pip/lot). این مقدار باید توسط Caller داده شود.
        - price_point: اندازهٔ یک پوینت/پیپ برحسب «واحد قیمت» (مثلاً 0.01). اگر None باشد، فرض می‌کنیم stop_distance_price خود برحسب «پیپ» است.

        نکته: اگر هرکدام از ورودی‌ها نامعتبر باشد، امن‌ترین گزینه برگشت به min_lot است.
        """
        ps = self.rcfg.position_sizing
        risk_usd = max(0.0, float(self.rcfg.risk_per_trade) * float(equity_usd))

        # تبدیل فاصلهٔ SL به پیپ (اگر price_point داده شده باشد)
        if stop_distance_price <= 0 or pip_value_per_lot <= 0 or risk_usd <= 0:
            LOGGER.warning("Invalid inputs for sizing (stop/pip_value/equity). Falling back to min_lot.")
            return ps.min_lot

        if price_point and price_point > 0:
            stop_pips = float(stop_distance_price / price_point)
        else:
            # فرض: stop_distance_price از قبل «برحسب پیپ» است
            stop_pips = float(stop_distance_price)

        if stop_pips <= 0:
            LOGGER.warning("Computed stop_pips <= 0. Falling back to min_lot.")
            return ps.min_lot

        try:
            raw_lot = risk_usd / (stop_pips * float(pip_value_per_lot))
        except ZeroDivisionError:
            LOGGER.warning("Division by zero in sizing. Falling back to min_lot.")
            return ps.min_lot

        lot = self._round_step(raw_lot, ps.lot_step, ps.min_lot, ps.max_lot)
        return lot

    # ===============================
    # 3) گِیت‌های ریسک (روزانه/کل)
    # ===============================
    def daily_loss_breached(self, equity_start_day: float, equity_now: float) -> bool:
        """
        آیا زیان روزانه از آستانهٔ مجاز عبور کرده است؟
        آستانه: max_daily_loss_pct × equity_start_day
        """
        try:
            if equity_start_day <= 0:
                return False
            dd = (equity_now - equity_start_day) / equity_start_day
            return dd <= -abs(self.rcfg.max_daily_loss_pct)
        except Exception:
            return False

    def total_drawdown_breached(self, equity_peak: float, equity_now: float) -> bool:
        """
        آیا افت سرمایهٔ کل از آستانهٔ مجاز عبور کرده است؟
        آستانه: max_total_drawdown_pct × equity_peak
        """
        try:
            if equity_peak <= 0:
                return False
            dd = (equity_now - equity_peak) / equity_peak
            return dd <= -abs(self.rcfg.max_total_drawdown_pct)
        except Exception:
            return False

    # ===============================
    # 4) Breakeven / Trailing (سیگنال‌های مدیریتی)
    # ===============================
    def breakeven_trigger(self, r_multiple: float) -> bool:
        """
        آیا شرایط انتقال SL به نقطهٔ سر‌به‌سر (Breakeven) برقرار است؟
        - نیاز است Caller مقدار r_multiple (سود جاری تقسیم بر ریسک اولیه) را بدهد.
        """
        ps = self.rcfg.position_sizing
        return bool(ps.breakeven_enabled and r_multiple >= ps.breakeven_r_multiple)

    def trailing_offset_from_atr(self, atr_price: float) -> float:
        """
        فاصلهٔ تریلینگ‌استاپ بر مبنای ATR (chandelier یا …).
        - خروجی برحسب «واحد قیمت».
        """
        ps = self.rcfg.position_sizing
        if not ps.trailing_enabled:
            return 0.0
        if atr_price is None or atr_price <= 0:
            return 0.0
        # فعلاً فقط نوع "chandelier" تعریف شده (طبق cfg موجود)
        return float(ps.trailing_atr_mult * atr_price)

    # ===============================
    # 5) کمکی‌ها برای اکوسیستم اجرا
    # ===============================
    def cap_lot(self, lot: float) -> float:
        """اعمال محدودیت min/max/step روی lot."""
        ps = self.rcfg.position_sizing
        return self._round_step(float(lot), ps.lot_step, ps.min_lot, ps.max_lot)

    def canary_volume(self, lot: float, volume_multiplier: Optional[float]) -> float:
        """
        اعمال چندبرابرساز «کاناری» (از cfg['executor']['canary_deployment']['volume_multiplier']).
        اگر None/نامعتبر بود، همان lot برگردانده می‌شود.
        """
        try:
            vm = float(volume_multiplier)
            if vm <= 0:
                return lot
            return self.cap_lot(lot * vm)
        except Exception:
            return lot

    # برای آینده: می‌توان تبدیل «price→pips» و «pip_value_per_lot» را با ورودی‌های آداپتر بروکر یکپارچه کرد.
