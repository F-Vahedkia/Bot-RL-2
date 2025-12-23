# -*- coding: utf-8 -*-
"""
Quick check Phase E:
- بارگذاری Feature Storeِ تولیدشده در Phase D
- ساخت FeatureDataset و MTFTradingEnv
- اجرای چند گام با اکشن‌های ساده (random) و چاپ میانگین پاداش
"""
# روش اجرای برنامه:
# python .\run_phaseE_quickcheck.py

from __future__ import annotations
import numpy as np
from pathlib import Path
from f05_envexe_core.dataset_deleted import FeatureDataset, DatasetConfig
from f05_envexe_core.env_deleted import MTFTradingEnv, EnvConfig
from f05_envexe_core.reward_deleted import RewardConfig

DATA = Path("./f15_testcheck/_reports/feature_store/SYNTH_2022Q1_12x8_ADV.parquet")
META = Path("./f15_testcheck/_reports/feature_store/SYNTH_2022Q1_12x8_ADV.meta.csv")

def main():
    ds = FeatureDataset(data_path=DATA, meta_path=META, cfg=DatasetConfig(window=64))
    env = MTFTradingEnv(ds, cfg=EnvConfig(window=64, reward=RewardConfig(use_log_return=True, trans_cost_per_turn=0.0)))

    s = env.reset()  # initial state
    rews = []
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = int(rng.integers(-1, 2))  # random action in {-1,0,1}
        s, r, done, info = env.step(a)
        rews.append(r)
        if done:
            break

    print("Random policy — mean reward:", float(np.mean(rews)))

if __name__ == "__main__":
    main()
