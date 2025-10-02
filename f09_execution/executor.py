# -*- coding: utf-8 -*-
# f09_execution/executor.py   (اجرای زنده/نیمه‌زنده پالیسی روی Env)
# Status in (Bot-RL-2): Completed

"""
اجرای زنده/نیمه‌زنده پالیسی روی Env
- فقط از APIهای موجود پروژه استفاده می‌کند (بدون حدس).
- از cfg «در حافظه» استفاده می‌شود؛ هیچ نوشتن دیسکی وسط اجرا انجام نمی‌شود.
- اگر --live فعال باشد و آداپتر بروکر در دسترس نباشد، اجرا متوقف می‌شود.

پیام‌های runtime انگلیسی هستند.
توضیحات فارسی صرفاً برای توسعه‌دهندگان داخل پروژه است.
"""

from __future__ import annotations

import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Optional

from f10_utils.config_loader import load_config
from f03_env.utils import paths_from_cfg
from f03_env.trading_env import TradingEnv, EnvConfig  # موجود در پروژه
from f07_training.train import transform_obs, load_model  # موجود در پروژه
from f09_execution.risk import RiskManager

LOGGER = logging.getLogger("executor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live/Semi-live executor")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("-c", "--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default=None, help="override base timeframe, e.g. M5")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--reward", default="pnl")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "live"])
    p.add_argument("--live", action="store_true", help="enable broker execution (requires adapter)")
    p.add_argument("--model-path", default=None, help="optional absolute model path")
    return p.parse_args()


def _resolve_model_path(cfg: dict, symbol: str, base_tf: str, explicit: Optional[str]) -> Optional[Path]:
    """
    تعیین مسیر مدل برای اجرا:
      - اگر صریح داده شده باشد → همان
      - در غیر این صورت: ابتدا prod سپس staging
    """
    paths = paths_from_cfg(cfg)
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    prod = paths["models_dir"] / "prod" / f"{symbol}_{base_tf}_reinforce_linear.npz"
    if prod.exists():
        return prod
    stg = paths["models_dir"] / "staging" / f"{symbol}_{base_tf}_reinforce_linear.npz"
    if stg.exists():
        return stg
    return None


def _load_broker_adapter(live: bool):
    """
    تلاش برای لود آداپتر بروکر.
    - اگر --live False باشد: No-OpAdapter برمی‌گرداند (صرفاً برای سازگاری).
    - اگر --live True و ماژول در دسترس نباشد: خطا می‌دهیم و اجرا را قطع می‌کنیم.
    """
    if not live:
        from f09_execution.broker_adapter_mt5 import NoOpBroker as BA  # موجود در همین پوشه
        LOGGER.info("Broker execution disabled; using No-Op broker adapter.")
        return BA()
    try:
        from f09_execution.broker_adapter_mt5 import MT5Broker as BA  # اگر MT5 آماده باشد
        b = BA()
        LOGGER.info("Broker adapter initialized.")
        return b
    except Exception as e:
        LOGGER.error("Live broker adapter is required but not available: %s", e)
        raise


def _env_from_cfg(cfg: dict, symbol: str, base_tf: Optional[str], window: int, reward: str, normalize: bool) -> TradingEnv:
    """ساخت Env از روی cfg در حافظه (بدون خواندن/نوشتن دیسک)."""
    base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    base_tf_final = (base_tf or base_tf_cfg).upper()
    env_cfg = EnvConfig(
        symbol=symbol,
        base_tf=base_tf_final,
        window_size=window,
        reward_mode=reward,
        normalize=bool(normalize),
    )
    return TradingEnv(cfg=cfg, env_cfg=env_cfg)


def _act_and_execute(broker, prev_pos: int, action: int, symbol: str, lot: float) -> int:
    """
    نگاشت اکشن به اردرهای واقعی:
      -1 → Short, 0 → Flat, +1 → Long
    در صورت تغییر وضعیت، ابتدا پوزیشن قبلی بسته می‌شود.
    """
    # حالت قبلی را به حالت فعلی تبدیل می‌کنیم
    if action not in (-1, 0, 1):
        return prev_pos

    if action == prev_pos:
        return prev_pos

    # بستن هر وضعیت قبلی
    if prev_pos == 1:
        broker.close_long(symbol)
    elif prev_pos == -1:
        broker.close_short(symbol)

    # باز کردن وضعیت جدید
    if action == 1:
        broker.buy(symbol, lot)
    elif action == -1:
        broker.sell(symbol, lot)
    # اگر صفر باشد، فلت می‌مانیم
    return action


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)

    # کم‌حرف کردن لاگ‌ها در اجرا
    logging.getLogger("hparam_search").setLevel(logging.WARNING)
    logging.getLogger("self_optimizer").setLevel(logging.WARNING)
    logging.getLogger("backtest_api").setLevel(logging.WARNING)

    # Overrideهای CLI روی cfg در حافظه
    (cfg.setdefault("env", {}))["normalize"] = bool(args.normalize)
    if args.base_tf:
        (cfg.setdefault("features", {}))["base_timeframe"] = str(args.base_tf).upper()

    env = _env_from_cfg(cfg, args.symbol, args.base_tf, args.window, args.reward, args.normalize)

    base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    base_tf_final = (args.base_tf or base_tf_cfg).upper()
    model_path = _resolve_model_path(cfg, args.symbol, base_tf_final, args.model_path)

    policy = None
    if model_path is None:
        LOGGER.warning("No model found (prod/staging). Will act randomly.")
    else:
        try:
            policy = load_model(model_path)
            LOGGER.info("Model loaded: %s", model_path)
        except Exception as e:
            LOGGER.error("Failed to load model: %s", e)
            policy = None

    broker = _load_broker_adapter(args.live)

    # --- Risk wiring ---
    rm = RiskManager.from_config(cfg)
    lot_base = float(((cfg.get("execution") or {}).get("lot") or 0.10))  # fallback when dynamic inputs unavailable


    obs, info = env.reset(split=args.split)
    total_r, steps, pos = 0.0, 0, 0

    while steps < args.steps:
        x = transform_obs(obs, mode="mean", last_k=16).astype(np.float32)
        if policy is None:
            a = int(np.random.randint(0, 3))  # {0,1,2}
        else:
            a, _ = policy.act(x, greedy=True)
        step = env.step(a - 1)
        obs = step.observation
        r = float(step.reward)
        term = bool(step.terminated)
        trunc = bool(step.truncated)



        total_r += r
        steps += 1

        # --- Dynamic position sizing (if inputs are available) ---
        lot_eff = lot_base
        try:
            # ورودی‌های اختیاری: اگر آداپتر/محیط تامین کند، سایز دینامیک می‌شود
            equity = broker.get_account_equity() if hasattr(broker, "get_account_equity") else None
            pip_val = broker.get_pip_value_per_lot(args.symbol) if hasattr(broker, "get_pip_value_per_lot") else None

            # تلاش برای دریافت ATR به «واحد قیمت»؛ فقط اگر محیط چنین عددی را در info فراهم کند
            atr_price = None
            if isinstance(info, dict):
                atr_price = info.get("atr_price") or info.get("atr")  # اگر موجود نباشد، None می‌ماند

            if equity and pip_val and atr_price:
                stop_dist = rm.stop_distance_from_atr(float(atr_price))
                if stop_dist > 0.0:
                    lot_dyn = rm.size_fixed_fractional(
                        equity_usd=float(equity),
                        stop_distance_price=float(stop_dist),
                        pip_value_per_lot=float(pip_val),
                        # اگر price_point لازم باشد، از cfg قابل برداشت است؛ در غیر این صورت None یعنی stop برحسب «پیپ»
                        price_point=(cfg.get("execution") or {}).get("price_point")
                    )
                    lot_eff = rm.cap_lot(lot_dyn)
                else:
                    lot_eff = rm.cap_lot(lot_base)
            else:
                # اگر ورودی‌های دینامیک موجود نبود، از lot پایه استفاده می‌کنیم
                lot_eff = rm.cap_lot(lot_base)
        except Exception:
            # در هر حالت خطا، امن‌ترین حالت: استفاده از lot پایه
            lot_eff = rm.cap_lot(lot_base)

        # اجرای بروکری با سایز نهایی
        pos = _act_and_execute(broker, pos, a - 1, args.symbol, lot_eff)




        if term or trunc:
            LOGGER.info("Episode terminated: steps=%d total_reward=%.6f", steps, total_r)
            break

    LOGGER.info("Execution finished: steps=%d total_reward=%.6f avg_per_step=%.6f", steps, total_r, (total_r / steps if steps else 0.0))
    # بستن پوزیشن باز در انتها (محافظه‌کارانه)
    if pos == 1:
        broker.close_long(args.symbol)
    elif pos == -1:
        broker.close_short(args.symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
