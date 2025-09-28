import numpy as np, pandas as pd
from f04_features.indicators.registry import get_indicator_v2
idx = pd.date_range("2022-01-01", periods=200, freq="min", tz="UTC")
rng = np.random.default_rng(0)
close = pd.Series(np.cumsum(rng.normal(0,0.2,200))+100, index=idx)
open_ = close.shift(1).fillna(close); high = np.maximum(open_, close)+0.1; low = np.minimum(open_, close)-0.1
df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":close})
ado = get_indicator_v2("adr_distance_to_open")
print(sorted(ado(df, window=14).keys()))
