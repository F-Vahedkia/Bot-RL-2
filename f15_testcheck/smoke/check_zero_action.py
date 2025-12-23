from pathlib import Path
import numpy as np, pandas as pd
from f05_envexe_core.dataset_deleted import FeatureDataset, DatasetConfig
from f05_envexe_core.env_deleted import MTFTradingEnv, EnvConfig
from f05_envexe_core.reward_deleted import RewardConfig

DATA=Path("./f15_testcheck/_reports/feature_store/SYNTH_2022Q1_12x8_ADV.parquet")
META=Path("./f15_testcheck/_reports/feature_store/SYNTH_2022Q1_12x8_ADV.meta.csv")
ds=FeatureDataset(DATA, META, DatasetConfig(window=64)); env=MTFTradingEnv(ds, EnvConfig(window=64, reward=RewardConfig()))
s=env.reset(); rews=[] 
for _ in range(500):
    s,r,done,info=env.step(0); rews.append(r)
    if done: break
print("mean reward (all flat):", float(np.mean(rews)))
