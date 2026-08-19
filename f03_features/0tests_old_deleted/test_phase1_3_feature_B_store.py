# Run: python -m pytest -q f03_features/0tests/test_phase1_3_feature_B_store.py
import pandas as pd
import numpy as np
from f03_features.feature_B_store import FeatureStoreV2


def test_build_merges_features_correctly():
    df = pd.DataFrame({"open": [1, 2, 3]})
    features = pd.DataFrame({"__indicator.sma@M1__x": [10, 20, 30]})

    store = FeatureStoreV2()
    out = store.build(df, features)

    assert "__indicator.sma@M1__x" in out.columns
    assert out["__indicator.sma@M1__x"].tolist() == [10, 20, 30]


def test_legacy_column_normalization():
    df = pd.DataFrame({"open": [1]})
    features = pd.DataFrame({"__macd@M1__x": [5]})

    store = FeatureStoreV2()
    out = store.build(df, features)

    assert "__indicator.macd@M1__x" in out.columns


def test_extract_metadata_runs():
    df = pd.DataFrame({
        "__indicator.sma@M1__x": [np.nan, 1, 2, 3]
    })

    store = FeatureStoreV2()
    meta = store.extract_metadata(df)

    assert len(meta) == 1
    assert meta.iloc[0]["indicator"] == "sma"


def test_save_parquet_and_meta(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    meta = pd.DataFrame([{"column": "x"}])

    store = FeatureStoreV2()
    paths = store.save(df, meta, tmp_path, "test_data")

    assert "data" in paths
    assert "meta_csv" in paths
    assert "meta_json" in paths
