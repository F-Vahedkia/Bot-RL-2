# f03_features/test_feature_engine_B_batch.py
# Run: python -m f03_features.test_feature_engine_B_batch
"""

Integration test:
config -> DSL specs -> parser -> registry -> engine -> dataframe

هدف:
تست کامل مسیر batch feature generation
بدون کدنویسی logic اضافی داخل تستر
"""

from __future__ import annotations
import traceback
import pandas as pd
from f10_utils.config_loader import load_config
from f03_features.OLD.feature_B_engine_1 import FeatureEngine

# =============================================================================
# Helpers
# =============================================================================

def _extract_indicator_specs(cfg: dict) -> list[str]:

    features = cfg.get("features", {})
    indicators = features.get("indicators", [])

    if not isinstance(indicators, list):
        raise TypeError(
            "config['features']['indicators'] "
            "must be a list"
        )

    return indicators


# =============================================================================
# Main Test
# =============================================================================

def main():

    print("=" * 80)
    print("FEATURE ENGINE BATCH INTEGRATION TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # load config
    # -------------------------------------------------------------------------
    cfg = load_config()

    specs = _extract_indicator_specs(cfg)

    print(f"[INFO] indicators count: {len(specs)}")

    if len(specs) == 0:
        raise ValueError("No indicators found in config")

    # -------------------------------------------------------------------------
    # load dataset
    # -------------------------------------------------------------------------
    data_path = cfg["data"]["train_path"]

    print(f"[INFO] loading data: {data_path}")

    df = pd.read_csv(data_path)

    # normalize columns
    df.columns = [c.lower() for c in df.columns]

    # optional datetime index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)

    print(f"[INFO] dataframe shape: {df.shape}")

    # -------------------------------------------------------------------------
    # create engine
    # -------------------------------------------------------------------------
    engine = FeatureEngine(mode="train")

    # -------------------------------------------------------------------------
    # compute all indicators
    # -------------------------------------------------------------------------
    print()
    print("[INFO] computing indicators...")
    print()

    features_df = engine.compute_many(df, specs)

    # -------------------------------------------------------------------------
    # result
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(features_df.tail())

    print()
    print("[INFO] output shape:", features_df.shape)
    print("[INFO] columns:")
    for c in features_df.columns:
        print(" -", c)

    print()
    print("[SUCCESS] integration test passed")


# =============================================================================
# Entry
# =============================================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print()
        print("=" * 80)
        print("TEST FAILED")
        print("=" * 80)

        print(type(e).__name__, ":", e)

        traceback.print_exc()

        raise

