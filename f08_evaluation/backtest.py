# f08_evaluation/backtest.py   (لود مدل و بک‌تست + گزارش)
# -*- coding: utf-8 -*-
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
    args = _parse_args()
    cfg = load_config(args.config, enable_env_override=True)
    paths = paths_from_cfg(cfg)
    base_tf_cfg = (cfg.get("features") or {}).get("base_timeframe", "M5")
    base_tf = (args.base_tf or base_tf_cfg).upper()

    env_cfg = EnvConfig(
        symbol=args.symbol,
        base_tf=base_tf,
        window_size=args.window,
        #max_steps=args.steps,
        reward_mode=args.reward,
        normalize=bool(args.normalize),
    )
    env = TradingEnv(cfg=cfg, env_cfg=env_cfg)

    # یافتن مدل
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = paths["models_dir"] / "staging" / f"{args.symbol}_{base_tf}_reinforce_linear.npz"
    if not model_path.exists():
        logger.warning("Model not found at %s. Running random baseline.", model_path)
        policy = None
    else:
        policy = load_model(model_path)

    # اجرا
    obs, info = env.reset(split=args.split)
    total_r = 0.0; steps = 0
    while steps < args.steps:
        x = transform_obs(obs, mode=args.obs_agg, last_k=args.last_k).astype(np.float32)
        if policy is None:
            a = np.random.randint(0, 3)  # 0,1,2
        else:
            a, _ = policy.act(x, greedy=True)
        obs, r, term, trunc, inf = env.step(a-1)
        total_r += float(r); steps += 1
        if term or trunc: break

    logger.info("Backtest finished: steps=%d, total_reward=%.6f, avg_per_step=%.6f",
                steps, total_r, (total_r/steps if steps else 0.0))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
