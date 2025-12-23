# f04_env/base_env.py
# -*- coding: utf-8 -*-
"""
BaseTradingEnv: اسکلت پایهٔ محیط معاملاتی
- بدون وابستگی اجباری به gym/gymnasium؛ اگر نصب بود، spaces را مقداردهی می‌کنیم.
- مدیریت حالت عمومی (reset/step) در کلاس پایه انجام نمی‌شود؛
  کلاس‌های مشتق (مثل TradingEnv) آن را پیاده‌سازی می‌کنند.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym  # اولویت با gymnasium
    from gymnasium import spaces
except Exception:
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except Exception:
        gym = None  # type: ignore
        spaces = None  # type: ignore


import numpy as np

@dataclass
class StepResult:
    """نتیجهٔ یک گام از محیط (سازگار با Gym/Gymnasium)."""
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]

class BaseTradingEnv(gym.Env if gym else object):
    ...

    """
    پایهٔ محیط معاملاتی.
    کلاس‌های مشتق باید متدهای زیر را پیاده‌سازی کنند:
    - reset(seed: Optional[int] = None) -> np.ndarray | tuple(obs, info)
    - step(action: int | float | np.ndarray) -> StepResult | tuple(obs, reward, terminated, truncated, info)
    - render() (اختیاری)
    """
    metadata = {"render.modes": ["human"], "name": "Bot-RL-1 TradingEnv"}

    # اگر gym در دسترس بود، این فیلدها را در کلاس مشتق مقداردهی کنید
    observation_space = None
    action_space = None

    def seed(self, seed: Optional[int] = None) -> None:
        """تنظیم seed برای تکرارپذیری."""
        if seed is not None:
            np.random.seed(seed)

    # نگاشت سادهٔ اکشن‌های گسسته به پوزیشن (-1,0,+1)
    @staticmethod
    def action_to_position(action: int) -> int:
        # اگر آرایه بود، عنصر اول را بگیر
        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(action[0])
        # فقط کلیپ؛ نگاشت گسسته در آداپتر SB3 انجام می‌شود
        return int(np.clip(action, -1, 1))
