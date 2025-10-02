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
from f06_news.filter import NewsGate

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
    def __init__(self, cfg: Dict[str, Any], 
                 env_cfg: EnvConfig, 
                 news_gate: Optional[NewsGate] = None):
        """
        سازندهٔ محیط معاملاتی.
        - بارگذاری دیتای پردازش‌شده (Base TF)
        - انتخاب ستون‌های مشاهده (OHLC/Volume/Spread + فیچرها) با امکان whitelist/blacklist
        - اعمال splitهای زمانی (train/val/test) از config
        - نرمال‌سازی (fit روی train → ذخیره در cache، سپس load و استفاده)
        - آماده‌سازی سری‌های پاداش (ret/logret/atr) و ساخت تابع پاداش
        - تعریف فضاهای observation/action در صورت وجود gymnasium/gym
        """
        super().__init__()
        self.cfg = cfg
        self.env_cfg = env_cfg
        # --- NEW: keep gate + default risk scale ---
        self.news_gate: Optional[NewsGate] = news_gate
        self._news_risk_scale: float = 1.0  # Persian: ضریب کاهش ریسک در حالت reduce
        
        # -----------------------------
        # ۱) مسیرها و بارگذاری دیتای پردازش‌شده
        # -----------------------------
        self.paths = paths_from_cfg(cfg)  # {'processed': ..., 'cache': ...}
        # فرمت را به عهدهٔ util بگذاریم (parquet اگر باشد، وگرنه csv)
        self.df_full = read_processed(
            symbol=env_cfg.symbol,
            base_tf=env_cfg.base_tf,
            cfg=cfg,
            fmt=None,
        )

        # اطمینان از وجود ایندکس زمانی و مرتب‌سازی (util همین کار را می‌کند؛ صرفاً دفاعی)
        self.df_full = self.df_full.sort_index()
        if self.df_full.index.name != "time":
            raise ValueError("Processed DataFrame must be indexed by 'time'.")

        # -----------------------------------------
        # ۲) انتخاب ستون‌های مشاهده (Observation)
        # -----------------------------------------
        sel = FeatureSelect(
            include_ohlc=env_cfg.use_ohlc,
            include_volume=env_cfg.use_volume,
            include_spread=env_cfg.use_spread,
            whitelist=env_cfg.features_whitelist,
            blacklist=env_cfg.features_blacklist,
        )
        self.obs_cols = infer_observation_columns(self.df_full, sel)
        assert len(self.obs_cols) > 0, "هیچ ستون مشاهده‌ای پیدا نشد؛ تنظیمات features/whitelist/blacklist را بررسی کنید."

        # -----------------------------------------
        # ۳) split زمانی (train/val/test) از config
        # -----------------------------------------
        self.slices = time_slices(cfg)
        if not self.slices:
            # اگر در کانفیگ split تعریف نشده، کل داده train فرض می‌شود
            t0, t1 = self.df_full.index[0], self.df_full.index[-1]
            self.slices = {"train": (t0, t1)}

        # -----------------------------------------
        # ۴) نرمال‌سازی: fit روی train → ذخیره/بارگذاری
        # -----------------------------------------
        self.scaler = None
        self.scaler_tag = f"{env_cfg.symbol}_{env_cfg.base_tf}_v1"
        if env_cfg.normalize:
            loaded = load_scaler(self.paths["cache"], self.scaler_tag)
            if loaded is None:
                # fit روی بازهٔ train
                a, b = self.slices["train"]
                df_train = slice_df_by_range(self.df_full[self.obs_cols], a, b)
                if df_train.empty:
                    # اگر بازهٔ train خالی شد، از کل دیتاست fit می‌کنیم (حالت دفاعی)
                    df_train = self.df_full[self.obs_cols]
                self.scaler = fit_scaler(df_train, self.obs_cols)
                save_scaler(self.scaler, self.paths["cache"], self.scaler_tag)
            else:
                self.scaler = loaded

        # -------------------------------------------------
        # ۵) سری‌های کمکی برای پاداش: بازده، لگ‌ریت و ATR
        #     (بدون look-ahead؛ در step از اندیس t+1 خوانده می‌شود)
        # -------------------------------------------------
        tf = env_cfg.base_tf.upper()

        # انتخاب مقاومِ ستون‌های OHLC برای TF پایه (در برابر یک/دوداش‌خط زیرخط)
        def _pick(field: str) -> str:
            cand1 = f"{tf}_{field}"      # مثل M1_close
            cand2 = f"{tf}__{field}"     # مثل M1__close (اگر جایی این نامگذاری باشد)
            if cand1 in self.df_full.columns: return cand1
            if cand2 in self.df_full.columns: return cand2
            # fallback: نخستین ستونی که به _field ختم می‌شود (برای ایمن‌سازی)
            for c in self.df_full.columns:
                if c.lower().endswith(f"_{field}"):
                    return c
            raise KeyError(f"ستون {field} برای TF {tf} یافت نشد.")

        close_col = _pick("close")
        high_col  = _pick("high")
        low_col   = _pick("low")

        close = self.df_full[close_col].astype("float32")
        # pct_change در اندیس i بازده از i-1→i است؛ در step با t+1 می‌خوانیم تا بازده t→t+1 باشد.
        self.ret = close.pct_change().fillna(0.0).astype("float32")
        self.logret = np.log(close / close.shift(1)).fillna(0.0).astype("float32")

        # اگر ATR از قبل (به‌صورت فیچر) موجود بود، همان را بردار؛ وگرنه محاسبهٔ ساده
        atr_feat = f"{tf}__atr_14"
        if atr_feat in self.df_full.columns:
            self.atr = self.df_full[atr_feat].astype("float32")
        else:
            prev_close = close.shift(1)
            tr = np.maximum.reduce([
                (self.df_full[high_col] - self.df_full[low_col]).abs().values,
                (self.df_full[high_col] - prev_close).abs().values,
                (self.df_full[low_col]  - prev_close).abs().values,
            ])
            tr = pd.Series(tr, index=self.df_full.index)
            self.atr = tr.ewm(alpha=1.0/14, adjust=False, min_periods=14).mean().astype("float32")

        # -----------------------------
        # ۶) ساخت تابع پاداش
        # -----------------------------
        tr_cfg = (cfg.get("trading") or {})
        costs = (tr_cfg.get("costs") or {})
        rcfg = RewardConfig(
            mode=env_cfg.reward_mode,  # 'pnl' | 'logret' | 'atr_norm'
            cost_spread_pts=float(costs.get("spread_pts", 0.0)),
            cost_commission_per_lot=float(costs.get("commission_per_lot", 0.0)),
            cost_slippage_pts=float(costs.get("slippage_pts", 0.0)),
            point_value=float(costs.get("point_value", 0.01)),
        )
        series = {"ret": self.ret, "logret": self.logret, "atr": self.atr}
        self.reward_fn = build_reward_fn(rcfg.mode, series, rcfg)

        # --------------------------------
        # ۷) فضاهای مشاهده/اکشن (اختیاری)
        # --------------------------------
        try:
            from gymnasium import spaces  # type: ignore
            self.action_space = spaces.Discrete(3)  # {-1, 0, +1}
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(int(env_cfg.window_size), len(self.obs_cols)),
                dtype=np.float32,
            )
        except Exception:
            # اگر gym/gymnasium نصب نبود، فقط فیلدها مقداردهی نمی‌شوند
            pass

        # -----------------------------
        # ۸) وضعیت داخلی اولیهٔ اپیزود
        # -----------------------------
        self._current_df = self.df_full  # در reset بر اساس split برش می‌دهیم
        self._t0 = 0
        self._t1 = len(self._current_df) - 1
        self._t = 0
        self._pos = 0               # -1/0/+1
        self._last_reward = 0.0
        self._done_hard = False

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

    def _news_status(self, ts) -> Dict[str, Any]:
        """وضعیت گیت خبری در لحظهٔ ts (UTC)."""
        if self.news_gate is None:
            return {"freeze": False, "reduce_risk": False, "reason": "no_gate", "events": []}
        try:
            ts = pd.to_datetime(ts, utc=True)
            return self.news_gate.status(ts)
        except Exception as ex:
            # Persian: اگر گِیت خطا داد، محیط را از کار نینداز
            import logging
            logging.warning("News gate error: %s", ex)
            return {"freeze": False, "reduce_risk": False, "reason": "error", "events": []}

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
        #--- بلوک NewsGate
        st = {"freeze": False, "reduce_risk": False, "reason": "no_gate", "events": []}
        if getattr(self, "news_gate", None) is not None:
            ts_utc = pd.to_datetime(idx_global, utc=True)
            st = self.news_gate.status(ts_utc)

            # اگر فریز است، پوزیشن جدید را همان قبلی نگه دار (HOLD اجباری)
            if st.get("freeze") and new_pos != self._pos:
                new_pos = self._pos

            # اگر کاهش ریسک است، ضریب ریسک داخلی را ست کن (در جای محاسبه‌ی سایز استفاده کن)
            self._news_risk_scale = st.get("reduce_scale", 0.5) if st.get("reduce_risk") else 1.0
        else:
            self._news_risk_scale = 1.0

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

        # ATR در همان تایم جاری (به واحد قیمت)
        atr_t = float(self.atr.iloc[t_global]) if (0 <= t_global < len(self.atr)) else float("nan")

        obs = self._make_obs(self._t)
        info = {
            "t": int(self._t),
            "pos": int(self._pos),
            "reward": float(reward),
            "atr_price": atr_t,      # ← برای RiskManager/Executor
        }
        info["news_gate"] = st


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

    from f06_news.integration import make_news_gate
    gate = make_news_gate(cfg, symbol=args.symbol)
    env = TradingEnv(cfg, env_cfg, news_gate=gate)

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