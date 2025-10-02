# -*- coding: utf-8 -*-
# f05_agent/env.py
# Status in (Bot-RL-2): Completed
"""
MTFTradingEnv — Bot-RL-2 (Phase E)
- محیط گام‌به‌گام برای تعامل RL با FeatureDataset
- state_t: پنجرهٔ [t-window+1 .. t] از فیچرها (+ قیمت‌ها)
- action_t ∈ {-1, 0, +1}  (short/flat/long)  ← قابل توسعه به سایزدهی پیوسته
- position_t: پس از اعمال action_t بروزرسانی می‌شود
- reward_t: بر اساس حرکت قیمت t→t+1 و پوزیشنِ t (بدون لوک‌اِهد) + نرمال‌سازی/هزینه

توجه:
- این کلاس وابسته به هیچ فریم‌ورک RL نیست؛ صرفاً یک API تمیز فراهم می‌کند.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from f05_agent.dataset import FeatureDataset, DatasetConfig
from f05_agent.reward import compute_step_reward, RewardConfig


@dataclass
class EnvConfig:
    """پیکربندی محیط."""
    window: int = 64
    use_prices_in_state: bool = True
    allow_hold: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)


class MTFTradingEnv:
    """
    محیط آموزش/ارزیابی برای RL با تضمین عدم لوک‌اِهد.
    """

    def __init__(self, ds: FeatureDataset, cfg: EnvConfig = EnvConfig()):
        self.ds = ds
        self.cfg = cfg

        # وضعیت داخلی
        self.t: Optional[int] = None
        self.position: float = 0.0

        # کش ADR برای استفاده در پاداش
        self._adr_series = None
        if ds.adr_cols:
            # ساده‌ترین انتخاب: اولین ستون ADR موجود
            self._adr_series = ds.df[ds.adr_cols[0]]

    # ====================== API ======================

    def reset(self, t0: Optional[int] = None) -> np.ndarray:
        """
        ریست محیط به t0 (یا شروع معتبر).
        خروجی: state_t
        """
        self.position = 0.0
        self.t = self.ds.t_min if t0 is None else int(t0)
        if self.t < self.ds.t_min or self.t > self.ds.t_max:
            raise IndexError(f"reset t0 out of range: {self.t} not in [{self.ds.t_min}, {self.ds.t_max}]")
        return self.ds.get_state(self.t, use_prices=self.cfg.use_prices_in_state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        اعمال اکشن و حرکت یک گام:
        - action ∈ {-1, 0, +1}
        - position_t = action  (می‌توان بعدها به منطق position sizing گسترش داد)
        - reward_t بر حسب t→t+1
        """
        if self.t is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # اعمال اکشن → پوزیشن در t
        a = int(action)
        if a not in (-1, 0, 1):
            raise ValueError("action must be -1, 0, or +1")
        self.position = float(a)

        # محاسبهٔ reward_t (بدون لوک‌اِهد)
        # position_t: سری ثابت با مقدار current position تا انتهای داده (فقط برای محاسبهٔ محلی نیاز است)
        pos_series = pd.Series(self.position, index=self.ds.df.index)
        rew_series = compute_step_reward(
            close=self.ds.df["close"],
            position_t=pos_series,
            cfg=self.cfg.reward,
            adr_series=self._adr_series,
        )
        # reward در t هم‌تراز با ایندکس t است
        reward_t = float(rew_series.iloc[self.t])

        # حرکت زمان
        done = (self.t >= self.ds.t_max)
        if not done:
            self.t += 1
            next_state = self.ds.get_state(self.t, use_prices=self.cfg.use_prices_in_state)
        else:
            next_state = np.zeros((self.cfg.window, len(self.ds.price_cols) + len(self.ds.feature_cols)), dtype=float)

        info = {
            "t": self.t,
            "position": self.position,
            "adr_col": (self.ds.adr_cols[0] if self.ds.adr_cols else None),
        }
        return next_state, reward_t, done, info
