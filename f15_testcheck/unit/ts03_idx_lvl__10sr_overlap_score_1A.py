# f15_testcheck/unit/ts03_idx_lvl__10sr_overlap_score_1A.py
# Run: python -m f15_testcheck.unit.ts03_idx_lvl__10sr_overlap_score_1A

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from f03_features.indicators.levels import sr_overlap_score

_PATH = "f16_test_results/"
# --- بارگذاری داده -------------------------------------------------
data = pd.read_csv("f02_data/raw/XAUUSD/M1.csv")
df = data[-400:].copy()
df["time"] = pd.to_datetime(df["time"], utc=True)
df.set_index("time", inplace=True)

if not {"high", "low", "close"}.issubset(df.columns):
    raise ValueError("Data must contain high, low, close")

df.to_csv(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1A_data.csv")

# --- ساخت سطوح داینامیک (rolling pivots ساده) ----------------------
rolling_high = df["high"].rolling(window=100).max()
rolling_low = df["low"].rolling(window=100).min()

# حذف NaN اولیه
sr_levels = np.unique(
    np.concatenate([
        rolling_high.dropna().values[-20:],   # آخرین 20 مقاومت
        rolling_low.dropna().values[-20:],    # آخرین 20 حمایت
    ])
)

# --- وزن‌دهی: فرض کنیم کفها (حمایت ها) قوی‌ترند ---
sr_weights = []
for lv in sr_levels:
    if lv in rolling_low.values:
        sr_weights.append(1.2)
    else:
        sr_weights.append(1.0)

sr_weights = np.array(sr_weights)
temp = pd.DataFrame({"sr_levels":sr_levels, "sr_weights":sr_weights})
temp.to_csv(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1A_Levels_Weights.csv")

# --- تست حساسیت tol_pct --------------------------------------------
tol_list = [0.001, 0.002, 0.003, 0.005]

for tol_pct in tol_list:

    scores = []

    for price in df["close"].values:
        score = sr_overlap_score(
            price=price,
            sr_levels=sr_levels,
            tol_pct=tol_pct,
            sr_weights=sr_weights,
        )
        scores.append(score)

    col_name = f"sr_overlap_{int(tol_pct*10000)}"
    df[col_name] = scores

    # ذخیره CSV جدا
    df[["close", col_name]].to_csv(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1A_{tol_pct}.csv")

    # رسم هیستوگرام
    plt.figure()
    plt.hist(df[col_name], bins=50)
    plt.title(f"SR Overlap Score Distribution (tol={tol_pct})")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{_PATH}ts03_idx_lvl__10sr_overlap_score_1A_hist_tol_{tol_pct}.png")
    plt.show()

    print(f"✅ tol={tol_pct} completed.")

print("🎯 All sensitivity tests finished.")
print(df[[c for c in df.columns if 'sr_overlap_' in c]].describe())

print("----------------------------------------")
print("Added 6 test result files to main project root:")
print("✅   ts03_idx_lvl__10sr_overlap_score_1A_data.csv")
print("✅   ts03_idx_lvl__10sr_overlap_score_1A_Levels_Weights.csv")
print("✅   ts03_idx_lvl__10sr_overlap_score_1A_001.csv")
print("✅   ts03_idx_lvl__10sr_overlap_score_1A_002.csv")
print("✅   ts03_idx_lvl__10sr_overlap_score_1A_003.csv")
print("✅   ts03_idx_lvl__10sr_overlap_score_1A_005.csv")
print("----------------------------------------")
