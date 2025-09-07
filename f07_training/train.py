# f07_training/train.py   (آموزش REINFORCE سبک با خط‌مشی خطی + CLI)
# -*- coding: utf-8 -*-
"""
آموزش ساده با REINFORCE (خط‌مشی خطی softmax) بدون وابستگی سنگین
- مشاهده: از Env می‌آید (window x features). با گزینه‌ی agg به بردار تبدیل می‌شود.
- مدل: W (d x A) و b (A). d = ابعاد مشاهده‌ پس از تبدیل، A=3.
- ذخیره مدل: f12_models/staging/<symbol>_<tf>_reinforce_linear.npz
"""
# فرمان اجرای برنامه
# python -m f07_training.train --symbol XAUUSD -c .\f01_config\config.yaml --base-tf M1 --window 128 --steps 2048 --episodes 20 --reward atr_norm --normalize --obs-agg mean

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from f10_utils.config_loader import load_config
from f03_env.trading_env import TradingEnv, EnvConfig
from f03_env.utils import paths_from_cfg

logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S")

# ----------------------------
# ابزارهای تبدیل مشاهده
# ----------------------------
def transform_obs(obs: np.ndarray, mode: str = "mean", last_k: int = 16) -> np.ndarray:
    """
    تبدیل observation شکل (T,F) به بردار 1D:
    - mean: میانگین روی محور زمان
    - last: فقط آخرین فریم
    - lastk_mean: میانگین آخرین k فریم
    - flat: تخت کردن کل پنجره
    """
    if obs.ndim == 1:
        return obs
    if mode == "mean":
        return obs.mean(axis=0)
    if mode == "last":
        return obs[-1]
    if mode == "lastk_mean":
        k = min(last_k, obs.shape[0])
        return obs[-k:].mean(axis=0)
    if mode == "flat":
        return obs.reshape(-1)
    raise ValueError(f"Unknown obs transform mode: {mode}")

# ----------------------------
# مدل خطی Softmax
# ----------------------------
class LinearSoftmaxPolicy:
    """خط‌مشی خطی با softmax؛ خروجی 3 عمل {-1,0,1}"""
    def __init__(self, in_dim: int, n_actions: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.01, size=(in_dim, n_actions)).astype(np.float32)
        self.b = np.zeros((n_actions,), dtype=np.float32)

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W + self.b  # (A,)

    def act(self, x: np.ndarray, greedy: bool = False, rng: np.random.Generator | None = None) -> Tuple[int, np.ndarray]:
        z = self.forward_logits(x)
        z = z - z.max()  # numerical stability
        probs = np.exp(z) / np.exp(z).sum()
        if greedy:
            a = int(np.argmax(probs))
        else:
            rng = rng or np.random.default_rng()
            a = int(rng.choice(len(probs), p=probs))
        return a, probs

    def grad_logpi(self, x: np.ndarray, a: int, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """∇_(W,b) log π(a|x) برای softmax خطی"""
        onehot = np.zeros_like(probs); onehot[a] = 1.0
        diff = (onehot - probs).astype(np.float32)  # (A,)
        gW = np.outer(x, diff)  # (D,A)
        gb = diff
        return gW, gb

# ----------------------------
# آموزش REINFORCE
# ----------------------------
def reinforce_train(env: TradingEnv,
                    episodes: int = 50,
                    steps_per_ep: int = 2048,
                    gamma: float = 0.99,
                    lr: float = 1e-3,
                    obs_mode: str = "mean",
                    obs_last_k: int = 16,
                    seed: int = 0) -> LinearSoftmaxPolicy:

    # نمونه‌ای از observation برای تعیین in_dim
    obs0, info = env.reset(split="train")
    x0 = transform_obs(obs0, mode=obs_mode, last_k=obs_last_k)
    in_dim = int(x0.shape[-1])
    policy = LinearSoftmaxPolicy(in_dim=in_dim, n_actions=3, seed=seed)
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        obs, info = env.reset(split="train")
        X, A, R = [], [], []
        ep_ret = 0.0

        for t in range(steps_per_ep):
            x = transform_obs(obs, mode=obs_mode, last_k=obs_last_k).astype(np.float32)
            a, probs = policy.act(x, greedy=False, rng=rng)
            obs, r, terminated, truncated, info = env.step(a-1)  # نگاشت {0,1,2}→{-1,0,+1}
            X.append(x); A.append((a, probs)); R.append(r)
            ep_ret += float(r)
            if terminated or truncated:
                break

        # محاسبهٔ بازگشت تخفیف‌یافته و گرادیان‌ها
        G = 0.0
        gW_acc = np.zeros_like(policy.W)
        gb_acc = np.zeros_like(policy.b)
        for t in reversed(range(len(R))):
            G = float(R[t]) + gamma * G
            a, probs = A[t]
            x = X[t]
            gW, gb = policy.grad_logpi(x, a, probs)
            gW_acc += (G * gW)
            gb_acc += (G * gb)

        # گرادیان صعودی
        policy.W += lr * gW_acc / max(1, len(R))
        policy.b += lr * gb_acc / max(1, len(R))

        logger.info("Episode %d/%d | steps=%d | return=%.6f",
                    ep+1, episodes, len(R), ep_ret)

    return policy

# ----------------------------
# ذخیره/لود مدل
# ----------------------------
def save_model(policy: LinearSoftmaxPolicy, outdir: Path, tag: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{tag}.npz"
    np.savez(path, W=policy.W, b=policy.b)
    logger.info("Model saved: %s", path)
    return path

def load_model(path: Path) -> LinearSoftmaxPolicy:
    dat = np.load(path)
    W = dat["W"]; b = dat["b"]
    pi = LinearSoftmaxPolicy(in_dim=W.shape[0], n_actions=W.shape[1])
    pi.W[:] = W; pi.b[:] = b
    return pi

# ----------------------------
# CLI
# ----------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Train linear REINFORCE policy on TradingEnv")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c","--config", default="f01_config/config.yaml")
    p.add_argument("--base-tf", default=None)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--obs-agg", default="mean", choices=["mean","last","lastk_mean","flat"])
    p.add_argument("--last-k", type=int, default=16)
    p.add_argument("--reward", default="pnl", choices=["pnl","logret","atr_norm"])
    p.add_argument("--normalize", action="store_true")
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

    policy = reinforce_train(env,
                             episodes=args.episodes,
                             steps_per_ep=args.steps,
                             gamma=args.gamma,
                             lr=args.lr,
                             obs_mode=args.obs_agg,
                             obs_last_k=args.last_k)

    models_dir = (paths["models_dir"] / "staging")
    tag = f"{args.symbol}_{base_tf}_reinforce_linear"
    save_model(policy, models_dir, tag)
    logger.info("Training finished.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
