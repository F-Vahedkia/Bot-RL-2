# f03_features/observation_B_builder.py
"""
ObservationBuilder
مسئول:
    - انتخاب featureها از engine output
    - alignment بین timeframe ها
    - ساخت ماتریس نهایی (RL-ready)
    - اعمال whitelist / blacklist / shift / normalization
"""
from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_B_graph import FeatureGraph


# ============================================================
# Observation Builder
# ============================================================

class ObservationBuilder:  #(version-2)

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

        self.shift = config.get("features", {}).get("shift_features_by", 0)
        self.drop_na = config.get("features", {}).get("drop_na_head", True)

        self.whitelist = set(config.get("env", {}).get("features_whitelist", []))
        self.blacklist = set(config.get("env", {}).get("features_blacklist", []))

    # --------------------------------------------------------
    def build_deleted(self, dataset: MTFDataset, graph: FeatureGraph) -> pd.DataFrame:

        # -------------------------------------------------------
        # مرحله اول:
        # ساخت Observation روی اندیس تایم‌فریم پایه
        # -------------------------------------------------------
        base_df = dataset.get(dataset.base_tf)
        obs = pd.DataFrame(index=base_df.index)

        # 1. collect feature columns from graph
        feature_cols = [n.raw for n in graph.all_nodes()]
        print("=" * 60)           # for debug new
        print("FEATURE COLS")     # for debug new
        for c in feature_cols:    # for debug new
            print(repr(c))        # for debug new
        print("=" * 60)           # for debug new

        # for c in feature_cols:                 # for debug
        #     print("GRAPH:", repr(c), len(c))   # for debug
        # for c in df.columns:                   # for debug
        #     print("DF   :", repr(c), len(c))   # for debug

        # # 2. whitelist filter (strict match OR substring)
        # # این قسمت حذف نشود. فقط موقتاً کامنت شده است
        # if self.whitelist:
        #     feature_cols = [
        #         c for c in feature_cols
        #         if any(w in c for w in self.whitelist)
        #     ]

        # # 3. blacklist filter (support wildcard *)
        # # این قسمت حذف نشود. فقط موقتاً کامنت شده است
        # if self.blacklist:
        #     feature_cols = [
        #         c for c in feature_cols
        #         if not any(b.replace("*", "") in c for b in self.blacklist)
        #     ]

        # # ---------------- for debug
        # print("whitelist =", self.whitelist)                  # for debug
        # print("blacklist =", self.blacklist)                  # for debug
        # print("feature_cols after filters =", feature_cols)   # for debug
        # # ---------------- for debug
        # print("GRAPH TYPES:")                 # for debug
        # for g in feature_cols:                # for debug
        #     print(type(g), repr(g), len(g))   # for debug

        # print("\nDF TYPES:")                      # for debug
        # for d in df.columns:                      # for debug
        #     print(type(d), repr(d), len(str(d)))  # for debug
        # # ---------------- for debug
        # print("\n========== EXACT COMPARE ==========")        # for debug
        # for g in feature_cols:                                # for debug
        #     for d in df.columns:                              # for debug
        #         # if len(g) == len(d):                          # for debug
        #         print("----------------------------")     # for debug
        #         print("GRAPH :", repr(g))                 # for debug
        #         print("DF    :", repr(d))                 # for debug
        #         print("==    :", g == d)                  # for debug
        #         print("GRAPH ORD:", [ord(x) for x in g])  # for debug
        #         print("DF    ORD:", [ord(x) for x in d])  # for debug
        # print("==================================\n")         # for debug

        # -------------------------------------------------------
        # مرحله دوم:
        # پیدا کردن Featureها در تمام تایم‌فریم‌ها
        # -------------------------------------------------------
        matched = []
        for tf in dataset.timeframes:
            df = dataset.get(tf)

            for c in feature_cols:                            # for debug
                print(repr(c), "IN DF =", c in df.columns)    # for debug

            cols = [c for c in feature_cols if c in df.columns]

            print("-" * 60)
            print("TIMEFRAME:", tf)
            print("COLUMNS  :", df.columns.tolist())
            print("MATCHED  :", cols)

            if cols:
                matched.append((tf, cols))

        print("=" * 60)                      # for debug
        print("FINAL MATCHED =", matched)    # for debug
        print("=" * 60)                      # for debug
 
        return obs

    # --------------------------------------------------------
    def build(
        self,
        dataset: MTFDataset,
        graph: FeatureGraph,
    ) -> pd.DataFrame:

        if dataset.aligned is None:
            raise ValueError("dataset.aligned is not built.")

        obs = dataset.aligned.copy()

        feature_cols = [n.raw for n in graph.all_nodes()]

        cols = [c for c in feature_cols if c in obs.columns]

        obs = obs[cols]

        if self.shift:
            obs = obs.shift(self.shift)

        if self.drop_na:
            obs = obs.dropna()

        return obs
    
    # --------------------------------------------------------
    def build_numpy(self, dataset: MTFDataset, graph: FeatureGraph):     # << === در بدنه خودم دو سطر موفتی را نوشته ام

        import numpy as np
        base_df = dataset.get(dataset.base_tf) # Temporary
        df = base_df.copy()                    # Temporary
        return self.build(dataset, graph).to_numpy(dtype=float)

