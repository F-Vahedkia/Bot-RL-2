# run_spec_phaseC.py
# Example of using f04_features indicators engine to compute features for Phase C

import numpy as np, pandas as pd
from f04_features.indicators import engine

idx = pd.date_range("2022-01-01", periods=200, freq="min", tz="UTC")
rng = np.random.default_rng(0); close = pd.Series(np.cumsum(rng.normal(0,0.2,200))+100, index=idx)
open = close.shift(1).fillna(close); high = np.maximum(open, close) + rng.random(200)*0.1; low = np.minimum(open, close) - rng.random(200)*0.1; vol = pd.Series(rng.integers(100,1000,200), index=idx)
df = pd.DataFrame({"open": open, "high": high, "low": low, "close": close, "volume": vol})
specs = ["sma(20)@M1","rsi(14)@M1","macd()@M1"]
out = engine.apply(df, specs)
out.to_csv(r"E:\Bot-RL-2\phaseC_sample.csv")
print("Cols:", list(out.columns)[-12:]); print("Shape:", out.shape)
