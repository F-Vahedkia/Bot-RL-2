# f03_features/feature_B_store.py
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# from f03_features.feature_B_registry_1 import list_indicators
from f10_utils.config_loader import load_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# Feature Naming Schema (Future-Proof)
# =============================================================================
_FEATURE_COL_RE = re.compile(
    r"^__(?P<domain>[a-zA-Z0-9_]+)\.(?P<ind>[a-zA-Z0-9_]+)@(?P<tf>[A-Z]\d+)__(?P<key>[a-zA-Z0-9_]+)$"
)

_LEGACY_COL_RE = re.compile(
    r"^__(?P<ind>[a-zA-Z0-9_]+)@(?P<tf>[A-Z]\d+)__(?P<key>[a-zA-Z0-9_]+)$"
)

# =============================================================================
# Metadata
# =============================================================================
@dataclass
class FeatureMeta:
    column: str
    domain: str
    indicator: str
    tf: str
    key: str
    dtype: str
    first_valid_ts: Optional[str]
    warmup: int
    coverage_ratio: float

# =============================================================================
# Schema Normalization
# =============================================================================
def _normalize_column(col: str) -> str:
    """
    Convert legacy schema → new schema
    __macd@M1__x → __indicator.macd@M1__x
    """
    m = _LEGACY_COL_RE.match(col)
    if not m:
        return col

    ind = m.group("ind")
    tf = m.group("tf")
    key = m.group("key")

    return f"__indicator.{ind}@{tf}__{key}"


def _parse_column(col: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Returns: (domain, indicator, tf, key)
    """
    m = _FEATURE_COL_RE.match(col)
    if not m:
        return None

    return (
        m.group("domain"),
        m.group("ind"),
        m.group("tf"),
        m.group("key"),
    )

# =============================================================================
# Core Feature Store
# =============================================================================
class FeatureStoreV2:

    def __init__(self, config=None):
        self.cfg = config or load_config()

    # -------------------------------------------------------------------------
    def build(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Merge raw + features (post-engine step)
        """
        df = df.copy()
        features = features.copy()

        features.columns = [_normalize_column(c) for c in features.columns]

        for col in features.columns:
            df[col] = features[col]

        return df

    # -------------------------------------------------------------------------
    def extract_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fully domain-aware metadata extraction
        """
        rows: List[FeatureMeta] = []

        for col in df.columns:

            if not col.startswith("__"):
                continue

            parsed = _parse_column(col)

            if parsed is None:
                continue

            domain, ind, tf, key = parsed
            s = df[col]

            fv = s.first_valid_index()
            warmup = int((~np.isfinite(s.to_numpy(dtype=float))).sum())

            cov = (
                float(np.isfinite(s.to_numpy(dtype=float)).sum() / len(s))
                if len(s) > 0
                else 0.0
            )

            rows.append(
                FeatureMeta(
                    column=col,
                    domain=domain,
                    indicator=ind,
                    tf=tf,
                    key=key,
                    dtype=str(s.dtype),
                    first_valid_ts=None if fv is None else str(fv),
                    warmup=warmup,
                    coverage_ratio=cov,
                )
            )

        return pd.DataFrame([asdict(r) for r in rows])

    # -------------------------------------------------------------------------
    def save(
        self,
        df: pd.DataFrame,
        meta: pd.DataFrame,
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ) -> Dict[str, str]:

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        if fmt == "parquet":
            data_path = out / f"{name}.parquet"
            df.to_parquet(data_path, index=True)
            paths["data"] = str(data_path)

        elif fmt == "csv":
            data_path = out / f"{name}.csv"
            df.to_csv(data_path)
            paths["data"] = str(data_path)

        else:
            raise ValueError("Unsupported format")

        meta_csv = out / f"{name}.meta.csv"
        meta_json = out / f"{name}.meta.json"

        meta.to_csv(meta_csv, index=False)

        with meta_json.open("w", encoding="utf-8") as f:
            json.dump(meta.to_dict(orient="records"), f, indent=2)

        paths["meta_csv"] = str(meta_csv)
        paths["meta_json"] = str(meta_json)

        return paths

# =============================================================================
# Functional API (clean)
# =============================================================================
def build_feature_store(
    df: pd.DataFrame,
    features: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    fmt: str = "parquet",
    config=None,
) -> Dict[str, str]:

    store = FeatureStoreV2(config)

    merged = store.build(df, features)
    meta = store.extract_metadata(merged)

    return store.save(merged, meta, out_dir, name, fmt)
