from pathlib import Path
import numpy as np, pandas as pd
from f05_agent.dataset import FeatureDataset, DatasetConfig
from f05_agent.env import MTFTradingEnv, EnvConfig
from f05_agent.reward import RewardConfig

DATA=Path("./feature_store/SYNTH_2022Q1_12x8_ADV.parquet"); META=Path("./feature_store/SYNTH_2022Q1_12x8_ADV.meta.csv")
ds=FeatureDataset(DATA, META, DatasetConfig(window=64)); env=MTFTradingEnv(ds, EnvConfig(window=64, reward=RewardConfig()))
s=env.reset(); rews=[]
for _ in range(500):
    s,r,done,info=env.step(0); rews.append(r)
    if done: break
print("mean reward (all flat):", float(np.mean(rews)))
