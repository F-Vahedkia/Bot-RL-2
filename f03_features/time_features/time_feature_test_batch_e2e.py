# f03_features/time_features/time_feature_test_batch_e2e.py
# Date reviewed:
#    1405/05/27-08:30 --> run result is OK.

# Run: python -m f03_features.time_features.time_feature_test_batch_e2e.py

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
import yaml
from f02_data.mtf_dataset import MTFDataset
from f03_features.time_features.time_feature_engine import (
    apply_time_features,
)
from f03_features.time_features.time_feature_registry import (
    get_time_feature,
)
from f10_utils.config_completer import config_completer
from f10_utils.config_path_funcs import full_file_path
# =============================================================================
# Paths
# =============================================================================
DATA_DIR = Path(
    r"E:\Bot-RL-2\f02_data\raw\BITCOIN"
)
SYMBOL = "BITCOIN"

# =============================================================================
# Load config
# =============================================================================
def load_time_feature_config(cfg) -> list[str]:
    cfg = config_completer(enable_env_override=True)
    features = (
        cfg["features"]
        ["time_features"]
        ["symbols"]
        [SYMBOL]
    )
    if not isinstance(features, list):
        raise TypeError(
            "BITCOIN time_features configuration must be a list."
        )
    return features

# =============================================================================
# Discover BITCOIN timeframe files
# =============================================================================
def discover_data_files(cfg, timeframes: list) -> dict[str, Path]:
    """
    Discover files such as:
        BITCOIN_M1.parquet
        BITCOIN_M5.parquet
        BITCOIN_H1.parquet
    If both parquet and csv exist for the same timeframe,
    parquet is selected.
    """
    raw_dir_path = Path(cfg["paths"]["raw_dir"])
    files: dict[str, Path] = {}
    for tf in timeframes:
        path = full_file_path(raw_dir_path, SYMBOL, tf, "parquet")
        if path.exists():
            files[tf] = path
        elif Path(path.replace(".parquet", ".csv")).exists():
            files[tf] = Path(path.replace(".parquet", ".csv"))

    if not files:
        raise FileNotFoundError(
            f"No {SYMBOL}_*.parquet or {SYMBOL}_*.csv files "
            f"found in:\n{raw_dir_path / SYMBOL}"
        )
    return dict(sorted(files.items()))


# =============================================================================
# Load OHLCVS dataframe
# =============================================================================
def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0)

    else:
        raise ValueError(
            f"Unsupported data format: {path}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"DataFrame index is not DatetimeIndex:\n{path}"
        )

    if df.index.tz is None:
        raise ValueError(
            f"DataFrame index is timezone-naive:\n{path}"
        )

    # Project contract: timestamps must be UTC.
    if str(df.index.tz) != "UTC":
        raise ValueError(
            f"DataFrame index is not UTC:\n"
            f"{path}\n"
            f"Detected timezone: {df.index.tz}"
        )

    return df


# =============================================================================
# Validate generated columns
# =============================================================================
def validate_output_columns(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    requested_features: list[str],
) -> None:

    for feature_name in requested_features:
        spec = get_time_feature(feature_name)

        for column in spec.output_columns:
            if column not in df_after.columns:
                raise AssertionError(
                    f"Missing output column '{column}' "
                    f"for time feature '{feature_name}'."
                )

    if len(df_before) != len(df_after):
        raise AssertionError(
            "Time-feature engine changed dataframe row count."
        )

    if not df_before.index.equals(df_after.index):
        raise AssertionError(
            "Time-feature engine changed dataframe index."
        )


# =============================================================================
# Main End-to-End Test
# =============================================================================
def main() -> int:

    # -------------------------------------------------------------------------
    # 0. Print header
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("TIME FEATURES END-TO-END TEST")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load configuration
    # -------------------------------------------------------------------------
    cfg = config_completer(enable_env_override=True)
    requested_features = load_time_feature_config(cfg)

    print(f"\nSymbol: {SYMBOL}")
    print(f"Configured features: {len(requested_features)}")
    print(
        "Features:",
        ", ".join(requested_features),
    )

    # -------------------------------------------------------------------------
    # 2. Discover data
    # -------------------------------------------------------------------------
    timeframes = cfg["__timeframes_dict"][SYMBOL]
    data_files = discover_data_files(cfg, timeframes)

    print("\nDiscovered data files:")

    for tf, path in data_files.items():
        print(f"  {tf:<5} -> {path.name}")

    # -------------------------------------------------------------------------
    # 3. Test every discovered timeframe
    # -------------------------------------------------------------------------
    tested = 0
    for timeframe, path in data_files.items():

        print("\n" + "-" * 80)
        print(f"Testing {SYMBOL} / {timeframe}")
        print(f"File: {path.name}")
        print("-" * 80)

        # ---------------------------------------------------------------------
        # Load real data
        # ---------------------------------------------------------------------
        df = load_dataframe(path)

        print(f"Rows before: {len(df):,}")
        print(f"Columns before: {list(df.columns)}")
        print(f"Index timezone: {df.index.tz}")

        # ---------------------------------------------------------------------
        # Build MTFDataset
        # ---------------------------------------------------------------------
        dataset = MTFDataset(
            symbol=SYMBOL,
            base_tf=timeframe,
        )
        dataset.add(
            timeframe,
            df.copy(),
        )

        # ---------------------------------------------------------------------
        # Execute the actual engine
        # ---------------------------------------------------------------------
        result = apply_time_features(
            dataset=dataset,
            time_feature_config={
                SYMBOL: requested_features,
            },
        )

        # ---------------------------------------------------------------------
        # Retrieve result
        # ---------------------------------------------------------------------
        result_df = result.get(timeframe)

        # ---------------------------------------------------------------------
        # Validate
        # ---------------------------------------------------------------------
        validate_output_columns(
            df_before=df,
            df_after=result_df,
            requested_features=requested_features,
        )

        print(f"Rows after:  {len(result_df):,}")
        print(f"Columns after: {len(result_df.columns)}")
        print("\nGenerated time-feature columns:")

        new_columns = [
            c
            for c in result_df.columns
            if c not in df.columns
        ]
        for column in new_columns:
            print(f"  + {column}")
        print(f"\nPASS: {SYMBOL}/{timeframe}")
        tested += 1

    # -------------------------------------------------------------------------
    # 4. Final result
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(
        f"END-TO-END TEST PASSED — "
        f"{tested} timeframe(s) tested."
    )
    print("=" * 80)
    return 0

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())
    