# -*- coding: utf-8 -*-
"""
Reward functions — Bot-RL-2 (Phase E)
- محاسبهٔ پاداش گام‌به‌گام بدون لوک‌اِهد
- نرمال‌سازی با ADR (در صورت در دسترس بودن)
- هزینهٔ تراکنش/اسلیپیج اختیاری

توضیح:
- reward_t = position_{t} * return_{t+1}  (بدون لوک‌اِهد: اکشن در t اعمال و در t+1 محقق می‌شود)
- return_{t+1} به‌صورت log-return یا simple-return قابل محاسبه است.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RewardConfig:
    """پیکربندی پاداش."""
    use_log_return: bool = True
    trans_cost_per_turn: float = 0.0     # هزینهٔ تراکنش به‌ازای تغییر پوزیشن (مثلاً 0.0001)
    slip_per_turn: float = 0.0           # اسلیپیج تخمینی برحسب نسبت (مثلاً 0.00005)
    adr_floor: float = 1e-6              # کف برای جلوگیری از تقسیم بر صفر
    adr_scale: float = 1.0               # ضریب اضافی مقیاس‌گذاری پس از ADR-normalization


def _next_return(close: np.ndarray, use_log: bool) -> np.ndarray:
    """محاسبهٔ بازده یک-گام-آینده بر اساس close. اندازه: len-1 (برای هم‌ترازی با t → t+1)."""
    if use_log:
        r = np.diff(np.log(close))
    else:
        r = np.diff(close) / close[:-1]
    return r


def compute_step_reward(
    close: pd.Series,
    position_t: pd.Series,
    cfg: RewardConfig = RewardConfig(),
    adr_series: Optional[pd.Series] = None,
) -> pd.Series:
    """
    reward_t = position_t * ret_{t+1} - costs
    - position_t: پوزیشن اعمال‌شده در زمان t (پس از اکشن)
    - ret_{t+1}: بازده از t به t+1 (بدون لوک‌اِهد)
    - adr_series: اگر موجود باشد، برای نرمال‌سازی استفاده می‌شود (ret / ADR)
    خروجی: سری با طول len(close)-1 هم‌تراز با ایندکس t (نه t+1).
    """
    close_np = close.to_numpy(dtype=float)
    ret_fwd = _next_return(close_np, use_log=cfg.use_log_return)  # len-1

    # تریم position_t به len-1 و هم‌ترازی با t
    pos_np = position_t.to_numpy(dtype=float)[: len(ret_fwd)]

    # نرمال‌سازی با ADR در صورت وجود
    if adr_series is not None:
        adr_np = adr_series.to_numpy(dtype=float)[: len(ret_fwd)]
        scale = np.where(np.isfinite(adr_np) & (np.abs(adr_np) > cfg.adr_floor), adr_np, 1.0)
        ret_fwd = ret_fwd / scale
        ret_fwd = ret_fwd * float(cfg.adr_scale)

    # هزینهٔ تراکنش/اسلیپیج (برحسب تغییر پوزیشن)
    pos_prev = np.concatenate([[0.0], pos_np[:-1]])
    turns = np.abs(pos_np - pos_prev)
    costs = turns * (float(cfg.trans_cost_per_turn) + float(cfg.slip_per_turn))

    rew = pos_np * ret_fwd - costs
    return pd.Series(rew, index=close.index[: len(rew)], name="reward")
