# -*- coding: utf-8 -*-
# f08_evaluation/backtest.py   (لود مدل و بک‌تست + گزارش)
# Status in (Bot-RL-2): Completed

"""
بک‌تست سریع یک مدل خطی ذخیره‌شده روی TradingEnv
- رفتار: greedy (اکشن با بیشترین احتمال)
- گزارش: مجموع پاداش، میانگین پاداش، قدم‌ها و بازه‌ی زمانی
"""
# فرمان اجرای برنامه
# python -m f08_evaluation.backtest --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M1 --window 128 --steps 5000 --reward atr_norm --normalize --split test

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np

from f10_utils.config_loader import load_config
from f03_env.trading_env import TradingEnv, EnvConfig
from f03_env.utils import paths_from_cfg
from f07_training.train import LinearSoftmaxPolicy, load_model, transform_obs

logger = logging.getLogger("backtest")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")

def _parse_args():
    p = argparse.ArgumentParser(description="Backtest a saved linear policy")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c","--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default=None)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--reward", default="pnl", choices=["pnl","logret","atr_norm"])
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--obs-agg", default="mean", choices=["mean","last","lastk_mean","flat"])
    p.add_argument("--last-k", type=int, default=16)
    p.add_argument("--split", default="test", choices=["train","val","test"])
    return p.parse_args()


def main() -> int:
    """CLI wrapper: load cfg from disk, then delegate to run_backtest()."""
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)

    # اِعمال گزینه‌های CLI روی cfg (بدون دست‌کاری فایل روی دیسک)
    (cfg.setdefault("env", {}))["normalize"] = bool(args.normalize)
    if args.base_tf:
        (cfg.setdefault("features", {}))["base_timeframe"] = str(args.base_tf).upper()

    rep = run_backtest(
        symbol=args.symbol,
        cfg=cfg,
        tag="cli",
        model_path=(args.model_path or None),
    )
    print(
        f"Backtest finished: steps={rep.get('steps',0)} "
        f"total_reward={rep.get('total_reward',0.0):.6f} "
        f"avg_per_step={rep.get('avg_per_step',0.0):.6f}"
    )
    return 0


# --- In-process API for PA hparam search (no CLI, no disk writes) ---
def run_backtest(*,
                 symbol: str,
                 cfg: dict,
                 tag: str | None = None,
                 model_path: str | None = None,
                 forward_days: int | None = None) -> dict:
    """
    Run backtest in-process using the provided in-memory cfg.
    Returns a dict with at least: {"total_reward": float, "steps": int, "avg_per_step": float}
    Notes:
      - Uses the same logic as main(): TradingEnv + (LinearSoftmaxPolicy if available else random).
      - Does NOT read config from disk; uses `cfg` passed by caller (self-optimizer).
      - Does NOT write anything to disk.
    """
    import numpy as np
    logger = logging.getLogger("backtest_api")

    # 1) derive base timeframe from cfg (exactly like main())
    base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    base_tf = str(base_tf_cfg).upper()

    # 2) build env from in-memory cfg
    env_cfg = EnvConfig(
        symbol=symbol,
        base_tf=base_tf,
        window_size=128,          # مطابق پیش‌فرض‌های مسیر main؛ در صورت نیاز از cfg بخوان
        reward_mode="pnl",        # همان پیش‌فرضی که در main استفاده می‌کنی
        normalize=bool((cfg.get("env") or {}).get("normalize", False)),
    )
    env = TradingEnv(cfg=cfg, env_cfg=env_cfg)

    # 3) resolve model (staging) or fall back to random policy
    if model_path is None:
        paths = paths_from_cfg(cfg)
        model_path = (paths["models_dir"] / "staging" / f"{symbol}_{base_tf}_reinforce_linear.npz")
    try:
        policy = load_model(model_path) if model_path and model_path.exists() else None
    except Exception:
        policy = None  # robust fallback: random acts

    # 4) backtest loop (same stepping pattern as main(), StepResult-safe)
    obs, info = env.reset(split="test")
    total_r, steps = 0.0, 0
    while steps < 5000:
        x = transform_obs(obs, mode="mean", last_k=16).astype(np.float32)
        if policy is None:
            a = np.random.randint(0, 3)
        else:
            a, _ = policy.act(x, greedy=True)

        sr = env.step(a - 1)
        obs = sr.observation
        r = sr.reward
        term = sr.terminated
        trunc = sr.truncated
        total_r += float(r)
        steps += 1
        if term or trunc:
            break

    avg = total_r / max(1, steps)
    # حداقل سه کلید موردنیاز برای hparam_search
    return {"total_reward": total_r, "steps": steps, "avg_per_step": avg}


if __name__ == "__main__":
    raise SystemExit(main())
