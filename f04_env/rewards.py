## f04_env/rewards.py


# -*- coding: utf-8 -*-
"""
تابع‌های پاداش (Reward) قابل انتخاب از config
- بدون لوک‌اِهد: پاداش‌ها با استفاده از تغییر قیمت بین t و t+1 محاسبه می‌شوند.
- امکان نرمال‌سازی با ATR یا z-score.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class RewardConfig:
    mode: str = "pnl"           # "pnl" | "logret" | "atr_norm"
    atr_period: int = 14         # برای atr_norm
    cost_spread_pts: float = 0.0 # هزینهٔ اسپرد به پوینت
    cost_commission_per_lot: float = 0.0  # کارمزد ثابت (برای سادگی)
    cost_slippage_pts: float = 0.0        # اسلیپیج تخمینی
    point_value: float = 0.01   # ارزش هر پوینت به واحد قیمت (XAUUSD≈0.01)

# --- کمکی‌ها ---

def _price_to_return(close: pd.Series) -> pd.Series:
    """بازده ساده بین t→t+1 (نسبت قیمت)."""
    return close.pct_change().fillna(0.0).astype("float32")

def _log_return(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).fillna(0.0).astype("float32")

def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().astype("float32")

# --- توابع پاداش ---

def reward_pnl(position: int, ret_t1: float, cfg: RewardConfig) -> float:
    """
    # پاداش PnL ساده (سهمی از بازده) منهای هزینه‌ها.
    gross = position * ret_t1
    # هزینه‌ها را ساده تخمین می‌زنیم (spread + slippage تبدیل شده به بازده نسبی)
    spread_cost = (cfg.cost_spread_pts * cfg.point_value)
    slippage_cost = (cfg.cost_slippage_pts * cfg.point_value)
    commission_cost = cfg.cost_commission_per_lot  # ثابت به‌ازای هر معامله (می‌توان پویا کرد)
    net = gross - (spread_cost + slippage_cost + commission_cost)
    return float(net)
    """
    """پاداش PnL ساده بر پایهٔ بازده نسبی؛ هزینه‌ها در Env کسر می‌شوند (A1/A2)."""
    gross = float(position) * float(ret_t1)
    return float(gross)


def reward_logret(position: int, logret_t1: float, cfg: RewardConfig) -> float:
    return float(position) * float(logret_t1)

def reward_atr_norm(position: int, ret_t1: float, atr_t: float, cfg: RewardConfig) -> float:
    denom = max(atr_t, 1e-8)
    return float(position) * float(ret_t1) / float(denom)

# --- انتخاب‌گر ---

def build_reward_fn(mode: str, series: Dict[str, pd.Series], cfg: RewardConfig) -> Callable[[int, int], float]:
    """
    بر اساس mode و سری‌های موردنیاز (close/logret/ret/atr)، یک closure برمی‌گرداند
    که ورودی‌اش (position_t, t_index) است و reward_t را می‌دهد.
    """
    mode = (mode or "pnl").lower()

    if mode == "logret":
        logret = series.get("logret")
        assert logret is not None, "logret series required"
        def _fn(pos: int, t: int) -> float:
            return reward_logret(pos, float(logret.iloc[t]), cfg)
        return _fn

    if mode == "atr_norm":
        ret = series.get("ret")
        atr = series.get("atr")
        assert ret is not None and atr is not None, "ret & atr series required"
        def _fn(pos: int, t: int) -> float:
            return reward_atr_norm(pos, float(ret.iloc[t]), float(atr.iloc[t]), cfg)
        return _fn

    # پیش‌فرض: pnl
    ret = series.get("ret")
    assert ret is not None, "ret series required"
    def _fn(pos: int, t: int) -> float:
        return reward_pnl(pos, float(ret.iloc[t]), cfg)
    return _fn