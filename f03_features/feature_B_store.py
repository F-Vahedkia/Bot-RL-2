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

from f02_data.mtf_dataset import MTFDataset
from f10_utils.config_loader import load_config
from f10_utils.constants import _TF_MINUTES

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
@dataclass  # <= قدیمی
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
# Merging
# =============================================================================
def align_to_base_old1(
    dataset: MTFDataset,
    *,
    candles: list[str]
) -> pd.DataFrame:
    """
    تمام تایم‌فریم‌ها را بدون Look-Ahead
    روی تایم‌فریم پایه همتراز می‌کند.

    Alignment بر اساس close_time انجام می‌شود.
    """
    symbol = dataset.symbol
    base_tf = dataset.base_tf

    base = dataset.get(base_tf).copy()

    candles = {tf.upper() for tf in candles}

    # -------------------------------------------
    # آماده سازی تایم‌فریم پایه
    # -------------------------------------------

    base["__open_time__"] = base.index

    base["__close_time__"] = (base.index + pd.Timedelta(_TF_MINUTES[base_tf], unit="min"))

    base = base.set_index("__close_time__")
    base.index.name = "__close_time__"

    merged = base

    # -------------------------------------------
    # Merge سایر تایم‌فریم‌ها
    # -------------------------------------------

    ordered = sorted(
        dataset.timeframes,
        key=lambda tf: _TF_MINUTES[tf],
    )

    for tf in ordered:
        if tf == base_tf:
            continue
        df = dataset.get(tf).copy()
        # --------------------------- patch-3
        price_cols = []
        feature_cols = []
        for c in df.columns:
            prefix = f"{symbol}_{tf}_"
            if c.startswith(prefix):
                
                suffix = c[len(prefix):]
                if suffix in {"open", "high", "low", "close", "volume", "spread", "open_time"}:
                    price_cols.append(c)
                else:
                    feature_cols.append(c)

            else:
                feature_cols.append(c)
        #---------------------------- patch-3
        #---------------------------- patch-4
        cols_to_merge = feature_cols.copy()

        if tf in candles:
            cols_to_merge.extend(price_cols)
        #---------------------------- patch-4
        if not cols_to_merge:
            continue
        
        # -------------------------------------------
        # آماده سازی دیتافریم کمکی
        # -------------------------------------------

        df["__open_time__"] = df.index
        df["__close_time__"] = (df.index + pd.Timedelta(_TF_MINUTES[tf], unit="min"))

        df = df.set_index("__close_time__")
        df.index.name = "__close_time__"

        # فقط ستون‌هایی که واقعاً باید مرج شوند
        right = df[cols_to_merge].reset_index()

        left = merged.reset_index()

        merged = pd.merge_asof(
            left.sort_values("__close_time__"),
            right.sort_values("__close_time__"),
            on="__close_time__",
            direction="backward",
        )

        merged = merged.set_index("__close_time__")

    return merged


def align_to_base(
    dataset: MTFDataset,
    *,
    candles: list[str] | None = None,
) -> pd.DataFrame:
    """
    Align all timeframe dataframes onto the base timeframe.

    Rules
    -----
    - No look-ahead (merge_asof backward)
    - Base index remains unchanged.
    - Higher TF candle values remain constant until next candle closes.
    - Feature columns are preserved.
    """

    symbol = dataset.symbol
    base_tf = dataset.base_tf

    base = dataset.get(base_tf).copy()

    base = base.copy()
    base["__close_time__"] = (base.index + pd.Timedelta(minutes=_TF_MINUTES[base_tf]))
    base = (
        base
        .reset_index(names="__open_time__")
        .sort_values("__close_time__")
    )

    merged = base

    ordered_tfs = sorted(
        dataset.timeframes,
        key=lambda tf: _TF_MINUTES[tf]
    )

    for tf in ordered_tfs:
        if tf == base_tf:
            continue
        df = dataset.get(tf)
        if df is None or df.empty:
            continue

        right = df.copy()
        right["__close_time__"] = (right.index + pd.Timedelta(minutes=_TF_MINUTES[tf]))
        right = (
            right
            .reset_index(names="__open_time__")
            .sort_values("__close_time__")
        )

        merged = pd.merge_asof(
            merged,
            right,
            on="__close_time__",
            direction="backward",
            allow_exact_matches=True,
        )

    merged = merged.set_index("__open_time__")

    merged.index.name = base.index.name

    if "__close_time__" in merged.columns:
        merged.drop(columns="__close_time__", inplace=True)

    return merged

# =============================================================================
# Feature Store V2 (MTFDataset based)
# =============================================================================

class FeatureStoreV2:
    """
    Persistence layer مخصوص MTFDataset.

    مسئولیت‌ها:
        • استخراج Metadata
        • ذخیره Dataset
        • بارگذاری Dataset
        • بدون هیچ وابستگی به Registry
        • بدون وابستگی به FeatureEngine
    """

    # ------------------------------------------------------------------
    def __init__(self, config=None):
        self.cfg = config or load_config()

    # ------------------------------------------------------------------
    def extract_metadata(
        self,
        dataset: MTFDataset,
    ) -> pd.DataFrame:

        rows: List[FeatureMeta] = []

        for tf, df in dataset.frames.items():
            if df is None or df.empty:
                continue

            for col in df.columns:
                s = df[col]
                fv = s.first_valid_index()
                try:
                    arr = s.to_numpy(dtype=float)
                    valid = np.isfinite(arr)
                    warmup = int((~valid).sum())
                    coverage = float(valid.sum() / len(arr)) if len(arr) else 0.0
                except Exception:
                    warmup = int(s.isna().sum())
                    coverage = float(s.notna().sum() / len(s)) if len(s) else 0.0

                rows.append(
                    FeatureMeta(
                        timeframe=tf,
                        column=col,
                        dtype=str(s.dtype),
                        first_valid_ts=None if fv is None else str(fv),
                        warmup=warmup,
                        coverage_ratio=coverage,
                    )
                )

        return pd.DataFrame(asdict(r) for r in rows)

    # -------------------------------------------------------------------------
    def _rename_columns(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """
        Rename all columns to a globally unique schema.

        Result:
            EURUSD_M1_open
            EURUSD_M1_close
            EURUSD_M1_sma(period=20)@M1
            EURUSD_H1_macd(...)@H1::hist
        """

        df = df.copy()

        prefix = f"{symbol}_{timeframe}_"

        rename = {}

        for col in df.columns:

            if col.startswith(prefix):
                continue

            rename[col] = prefix + col

        return df.rename(columns=rename)

    # -------------------------------------------------------------------------
    def build(
        self,
        dataset: MTFDataset,
        features: MTFDataset,
    ) -> MTFDataset:
        """
        Merge FeatureEngine output into raw dataset.
        خروجی:
            1) out.frames  -> دیتاست چندتایم‌فریمی (بدون همترازی)
            2) out.aligned -> دیتافریم همتراز شده مخصوص RL
        """
        out = dataset.copy()
        symbol = dataset.symbol

        # -------------------------------------------------------
        # مرحله اول:
        # ساخت دیتاست چندتایم‌فریمی با ستون‌های یکتا
        # -------------------------------------------------------
        for tf in out.timeframes:

            raw_df = self._rename_columns(out.get(tf), symbol, tf)

            feature_df = self._rename_columns(features.get(tf), symbol,tf)

            feature_cols = [
                c for c in feature_df.columns
                if c not in raw_df.columns
            ]
            raw_df = raw_df.join(feature_df[feature_cols], how="left")

            out.replace(tf, raw_df)

        # -------------------------------------------------------
        # Debug
        # -------------------------------------------------------
        print("=" * 60)
        for tf in out.timeframes:
            df = out.get(tf)
            print(tf)
            print(df.columns.tolist())
            print("-" * 60)

        # -------------------------------------------------------
        # مرحله دوم:
        # ساخت دیتافریم همتراز شده
        # -------------------------------------------------------
        candles = self.cfg["features"]["symbols"][symbol]["candles"]

        aligned = align_to_base(
            out,
            candles=candles,
        )

        aligned.attrs["symbol"] = symbol
        aligned.attrs["base_tf"] = out.base_tf

        # دیتافریم نهایی مخصوص ObservationBuilder
        out.aligned = aligned

        return out 
    
    # -------------------------------------------------------------------------
    def extract_metadata(
        self,
        dataset: MTFDataset,
    ) -> Dict[str, pd.DataFrame]:
        """
        Metadata هر تایم‌فریم را جداگانه استخراج می‌کند.
        """

        result: Dict[str, pd.DataFrame] = {}

        for tf in dataset.timeframes:

            df = dataset.get(tf)

            rows: List[FeatureMeta] = []

            for col in df.columns:

                if not col.startswith("__"):
                    continue

                parsed = _parse_column(col)

                if parsed is None:
                    continue

                domain, ind, _, key = parsed

                s = df[col]

                fv = s.first_valid_index()

                warmup = int(
                    (~np.isfinite(s.to_numpy(dtype=float))).sum()
                )

                coverage = (
                    float(
                        np.isfinite(
                            s.to_numpy(dtype=float)
                        ).sum()
                        / len(s)
                    )
                    if len(s)
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
                        coverage_ratio=coverage,
                    )
                )

            result[tf] = pd.DataFrame(
                [asdict(r) for r in rows]
            )

        return result

    # -------------------------------------------------------------------------
    def save(
        self,
        dataset: MTFDataset,
        metadata: Dict[str, pd.DataFrame],
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ) -> Dict[str, Dict[str, str]]:
        """
        ذخیره مستقل هر تایم‌فریم.
        """

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Dict[str, str]] = {}

        for tf in dataset.timeframes:

            tf_dir = out / tf
            tf_dir.mkdir(exist_ok=True)

            df = dataset.get(tf)
            meta = metadata.get(tf, pd.DataFrame())

            tf_paths: Dict[str, str] = {}

            if fmt == "parquet":

                data_path = tf_dir / f"{name}.parquet"
                df.to_parquet(data_path, index=True)

            elif fmt == "csv":

                data_path = tf_dir / f"{name}.csv"
                df.to_csv(data_path)

            else:
                raise ValueError("Unsupported format")

            tf_paths["data"] = str(data_path)

            meta_csv = tf_dir / f"{name}.meta.csv"
            meta_json = tf_dir / f"{name}.meta.json"

            meta.to_csv(meta_csv, index=False)

            with meta_json.open("w", encoding="utf-8") as f:
                json.dump(
                    meta.to_dict(orient="records"),
                    f,
                    indent=2,
                )

            tf_paths["meta_csv"] = str(meta_csv)
            tf_paths["meta_json"] = str(meta_json)

            paths[tf] = tf_paths

        return paths


# =============================================================================
# Functional API
# =============================================================================
def build_feature_store(
    dataset: MTFDataset,
    features: MTFDataset,
    out_dir: str | Path,
    name: str,
    fmt: str = "parquet",
    config=None,
) -> Dict[str, Dict[str, str]]:

    store = FeatureStoreV2(config)

    merged = store.build(dataset, features)

    meta = store.extract_metadata(merged)

    return store.save(
        merged,
        meta,
        out_dir,
        name,
        fmt,
    )






'''
    def _rename_columns_old1(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:

        df = df.copy()

        prefix = f"{symbol}_{timeframe}_"

        rename = {}
        for col in df.columns:
            rename[col] = prefix + col

        df.rename(columns=rename, inplace=True)

        return df
    
    # ---------------------------------
    def _rename_columns_old2(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> pd.DataFrame:

        df = df.copy()

        prefix = f"{symbol}_{timeframe}_"

        PRICE_COLUMNS = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
            "open_time",
        }

        rename = {}

        for col in df.columns:

            if col in PRICE_COLUMNS:
                rename[col] = prefix + col

        df.rename(columns=rename, inplace=True)

        return df

    # ---------------------------------
    def _rename_columns_old3(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        mode: str = "none",
    ) -> pd.DataFrame:
        """
        Rename dataframe columns.

        mode
        ----
        none  : روی تمام ستون‌ها پیشوند می‌گذارد.
        ohlc  : فقط روی OHLCV و open_time پیشوند می‌گذارد.
        specs : فقط روی ستون‌های Feature (غیر OHLCV) پیشوند می‌گذارد.
        """

        df = df.copy()

        prefix = f"{symbol}_{timeframe}_"

        ohlc_cols = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "spread",
            "open_time",
        }

        rename = {}

        for col in df.columns:

            if mode == "none":
                rename[col] = prefix + col

            elif mode == "ohlc":
                if col in ohlc_cols:
                    rename[col] = prefix + col

            elif mode == "specs":
                if col not in ohlc_cols:
                    rename[col] = prefix + col

            else:
                raise ValueError(f"Unknown rename mode: {mode}")

        return df.rename(columns=rename)

'''

