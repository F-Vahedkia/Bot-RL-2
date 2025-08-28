## f03_env/trading_env.py

# -*- coding: utf-8 -*-
"""
TradingEnv: محیط RL مبتنی بر داده‌های M1 + فیچرهای MTF
- اکشن گسسته: {-1, 0, +1} = Short/Flat/Long (امکان توسعه به پیوسته/سایز پوزیشن)
- پاداش قابل انتخاب (pnl/logret/atr_norm) از config
- بدون لوک‌اِهد: ورودی‌ها قبلاً shift شده‌اند (data_handler + indicators)
- Split زمانی train/val/test از config
- نرمال‌سازی بر اساس آمار train (scaler ذخیره/بارگذاری)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np
import pandas as pd

from f10_utils.config_loader import load_config
from .base_env import BaseTradingEnv, StepResult
from .rewards import RewardConfig, build_reward_fn
from .utils import (
    _project_root,
    paths_from_cfg,
    read_processed,
    time_slices,
    slice_df_by_range,
    FeatureSelect,
    infer_observation_columns,
    fit_scaler,
    save_scaler,
    load_scaler,
)

logger = logging.getLogger(__name__)

@dataclass
class EnvConfig:
    symbol: str
    base_tf: str = "M1"
    window_size: int = 128
    normalize: bool = True
    features_whitelist: Optional[List[str]] = None
    features_blacklist: Optional[List[str]] = None
    use_ohlc: bool = True
    use_volume: bool = True
    use_spread: bool = False
    reward_mode: str = "pnl"  # pnl | logret | atr_norm

class TradingEnv(BaseTradingEnv):
    """محیط معاملاتی M1 + فیچرهای MTF برای RL."""
    def __init__(self, cfg: Dict[str, Any], env_cfg: EnvConfig):
        super().__init__()
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.paths = paths_from_cfg(cfg)

        # بارگذاری دیتای پردازش‌شده + فیچرها
        self.df_full = read_processed(env_cfg.symbol, env_cfg.base_tf, cfg, fmt="parquet")

        # انتخاب ستون‌های مشاهده
        sel = FeatureSelect(
            include_ohlc=env_cfg.use_ohlc,
            include_volume=env_cfg.use_volume,
            include_spread=env_cfg.use_spread,
            whitelist=env_cfg.features_whitelist,
            blacklist=env_cfg.features_blacklist,
        )
        self.obs_cols = infer_observation_columns(self.df_full, sel)
        assert len(self.obs_cols) > 0, "هیچ ستون مشاهده‌ای پیدا نشد؛ config را بررسی کنید."

        # Split زمانی (train/val/test)
        self.slices = time_slices(cfg)
        if not self.slices:
            # اگر در کانفیگ split تعریف نشده، کل داده train فرض می‌شود
            t0, t1 = self.df_full.index[0], self.df_full.index[-1]
            self.slices = {"train": (t0, t1)}

        # نرمال‌سازی (fit روی train)
        self.scaler_tag = f"{env_cfg.symbol}_{env_cfg.base_tf}_v1"
        self.scaler = load_scaler(self.paths["cache"], self.scaler_tag)
        if env_cfg.normalize and (self.scaler is None):
            tr_a, tr_b = self.slices.get("train", (self.df_full.index[0], self.df_full.index[-1]))
            df_tr = self.df_full.loc[(self.df_full.index >= tr_a) & (self.df_full.index <= tr_b), self.obs_cols]
            self.scaler = fit_scaler(df_tr, self.obs_cols)
            save_scaler(self.scaler, self.paths["cache"], self.scaler_tag)

        # سری‌های بازده/ATR برای پاداش
        close = self.df_full[f"{env_cfg.base_tf}__close"] if f"{env_cfg.base_tf}__close" in self.df_full.columns else self.df_full["M1_close"]
        self.ret = close.pct_change().fillna(0.0).astype("float32")
        self.logret = np.log(close / close.shift(1)).fillna(0.0).astype("float32")
        # ATR را اگر داریم از فیچرها بخوانیم، وگرنه ساده محاسبه کنیم
        atr_col = f"{env_cfg.base_tf}__atr_14"
        if atr_col in self.df_full.columns:
            self.atr = self.df_full[atr_col].astype("float32")
        else:
            high = self.df_full.get(f"{env_cfg.base_tf}__high") or self.df_full.filter(like="_high").iloc[:, 0]
            low = self.df_full.get(f"{env_cfg.base_tf}__low") or self.df_full.filter(like="_low").iloc[:, 0]
            prev_close = close.shift(1)
            tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
            self.atr = tr.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean().astype("float32")

        # ساخت تابع پاداش
        rcfg = RewardConfig(
            mode=self.env_cfg.reward_mode,
            cost_spread_pts=float(((self.cfg.get("trading") or {}).get("costs") or {}).get("spread_pts", 0)),
            cost_commission_per_lot=float(((self.cfg.get("trading") or {}).get("costs") or {}).get("commission_per_lot", 0.0)),
            cost_slippage_pts=float(((self.cfg.get("trading") or {}).get("costs") or {}).get("slippage_pts", 0)),
            point_value=float(((self.cfg.get("trading") or {}).get("costs") or {}).get("point_value", 0.01)),
        )
        series = {"ret": self.ret, "logret": self.logret, "atr": self.atr}
        self.reward_fn = build_reward_fn(rcfg.mode, series, rcfg)

        # فضای اکشن/مشاهده (اگر gym نصب است)
        try:
            from gymnasium import spaces  # type: ignore
            self.action_space = spaces.Discrete(3)  # {-1,0,+1}
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(env_cfg.window_size, len(self.obs_cols)), dtype=np.float32)
        except Exception:
            pass

        # حالت داخلی
        self._t0: int = 0
        self._t1: int = 0
        self._t: int = 0
        self._pos: int = 0
        self._done_hard: bool = False
        self._last_reward: float = 0.0
        self._current_df: pd.DataFrame = self.df_full

    # --- کمکی‌ها ---
    def _make_obs(self, t: int) -> np.ndarray:
        ws = self.env_cfg.window_size
        lo = max(0, t - ws + 1)
        frame = self._current_df.iloc[lo:t+1][self.obs_cols]

        if self.env_cfg.normalize and self.scaler is not None and len(frame) > 0:
            frame = self.scaler.transform(frame)

        obs = np.zeros((ws, len(self.obs_cols)), dtype=np.float32)
        if len(frame) > 0:
            obs[-len(frame):, :] = frame.astype("float32").values  # فقط وقتی طول>0
        return obs

    def _reset_range(self, split: str = "train") -> None:
        # 1) برش طبق split
        if split in self.slices:
            a, b = self.slices[split]
            df = self.df_full.loc[(self.df_full.index >= a) & (self.df_full.index <= b)].copy()
        else:
            df = self.df_full.copy()

        # 2) اگر خالی شد، به کل دیتاست برگرد
        if df.empty:
            logger.warning("Selected split '%s' is empty. Falling back to full dataset.", split)
            df = self.df_full.copy()

        # 3) مرتب‌سازی زمانی و ست‌کردن وضعیت داخلی
        df = df.sort_index()
        if df.empty:
            # این حالت یعنی دیتای پردازش‌شده نداریم؛ به‌جای کرش، پیام شفاف بده
            raise RuntimeError(
                "Current dataframe is empty after split/fallback. "
                "Check processed data and your config.env.split ranges."
            )

        self._current_df = df
        self._t0 = 0
        self._t1 = len(self._current_df) - 1

    # --- API استاندارد ---
    def reset(self, seed: Optional[int] = None, split: str = "train") -> Tuple[np.ndarray, Dict[str, Any]]:
        super().seed(seed)
        self._reset_range(split)
        ws = int(self.env_cfg.window_size)

        # اگر طول بازه کمتر از window باشد، از آخرین سطر موجود شروع کن
        self._t = min(self._t0 + ws, self._t1)
        if self._t < 0:
            self._t = 0

        self._pos = 0
        self._done_hard = False
        self._last_reward = 0.0
        obs = self._make_obs(self._t)
        info = {"t": int(self._t), "pos": int(self._pos), "split": split, "len": int(self._t1 + 1)}
        return obs, info

    def step(self, action: int) -> StepResult:
        if self._done_hard:
            # وقتی تمام شد، دیگر قدم‌زدن ممکن نیست
            return StepResult(self._make_obs(self._t), 0.0, True, False, {"msg": "episode done"})

        # نگاشت اکشن به پوزیشن
        new_pos = BaseTradingEnv.action_to_position(action)  # -1/0/+1

        # محاسبهٔ پاداش در t → t+1
        # توجه: self.ret/logret/atr از کل df_full هستند؛ ایندکس محلی را به ایندکس سراسری نگاشت می‌کنیم
        idx_global = self._current_df.index[self._t]
        t_global = int(self.df_full.index.get_indexer([idx_global])[0])

        reward = 0.0
        if t_global + 1 <= len(self.df_full) - 1:
            reward = self.reward_fn(self._pos, t_global + 1)  # پاداش بر اساس پوزیشنِ فعلی در گام گذشته

        # به‌روزرسانی پوزیشن: سیاست ساده → فوراً به new_pos تغییر می‌دهیم
        self._pos = int(new_pos)

        # حرکت به گام بعد
        self._t += 1
        terminated = (self._t >= self._t1)
        truncated = False
        self._last_reward = float(reward)

        obs = self._make_obs(self._t)
        info = {"t": int(self._t), "pos": int(self._pos), "reward": float(reward)}
        if terminated:
            self._done_hard = True
        return StepResult(obs, float(reward), bool(terminated), bool(truncated), info)

    # رندرِ متنی ساده (اختیاری)
    def render(self) -> None:
        print(f"t={self._t} pos={self._pos} last_reward={self._last_reward:.6f}")

# --- CLI تست دود ---

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Smoke test for TradingEnv")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c", "--config", default=str(_project_root()/"f01_config"/"config.yaml"))
    p.add_argument("--base-tf", default=None)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--split", default="train", choices=["train","val","test"])
    p.add_argument("--reward", default="pnl", choices=["pnl","logret","atr_norm"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)
    base_tf = (args.base_tf or (cfg.get("features") or {}).get("base_timeframe", "M1")).upper()

    env_cfg = EnvConfig(
        symbol=args.symbol,
        base_tf=base_tf,
        window_size=int(args.window),
        normalize=bool(args.normalize),
        reward_mode=str(args.reward).lower(),
    )

    env = TradingEnv(cfg, env_cfg)
    obs, info = env.reset(split=args.split)

    total_r = 0.0
    for i in range(int(args.steps)):
        a = np.random.choice([-1, 0, +1])  # سیاست تصادفی برای تست دود
        step = env.step(a)
        total_r += step.reward
        if step.terminated or step.truncated:
            break
    print(f"SmokeTest finished: steps={i+1}, total_reward={total_r:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())