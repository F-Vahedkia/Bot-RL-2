# f04_features/indicators/__init__.py
# -*- coding: utf-8 -*-
"""
Entry اصلی پکیج اندیکاتورها + CLI
- خواندن processed از data_handler
- اعمال Specها بر اساس config.features.indicators
- ذخیره به صورت augment یا features_only
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import pandas as pd

from f10_utils.config_loader import load_config
from .registry import build_registry
from .engine import IndicatorEngine, EngineConfig

logger = logging.getLogger(__name__)
#logger.addHandler(logging.NullHandler())

# مسیرها از config

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _paths_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Path]:
    paths = cfg.get("paths", {}) or {}
    processed = _project_root() / (paths.get("processed_dir") or "data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    return {"processed_dir": processed}

# IO

def _read_processed(symbol: str, base_tf: str, cfg: Dict[str, Any], fmt_hint: Optional[str] = None) -> pd.DataFrame:
    """
    خواندن دیتای «پردازش‌شده» از خروجی data_handler برای نماد و تایم‌فریم پایه.
    رفتارها:
      - اگر fmt_hint مشخص نباشد، اولویت با Parquet و سپس CSV است.
      - اگر fmt_hint مشخص باشد (parquet/csv)، همان را تلاش می‌کنیم و در صورت نبود، به فرمت دیگر fallback می‌کنیم.
      - اگر فایل augment قبلی داشته باشد (ستون‌هایی با الگوی TF__feature)، آن ستون‌ها حذف می‌شوند تا در اجرای مجدد
        حالت idempotent داشته باشیم و خطای overlap در join رخ ندهد.
      - ایندکس زمانی “time” تضمین می‌شود، داده براساس زمان مرتب و دابل‌کی‌ها حذف می‌شوند.
    """
    processed_dir = _paths_from_cfg(cfg)["processed_dir"]
    sym_dir = processed_dir / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)

    # نام فایل‌ها بر اساس base_tf
    base_name = base_tf.upper()
    pq_path = sym_dir / f"{base_name}.parquet"
    csv_path = sym_dir / f"{base_name}.csv"

    def _read_parquet(p: Path) -> pd.DataFrame:
        if not p.exists():
            raise FileNotFoundError(str(p))
        df = pd.read_parquet(p)
        # اگر ستون time وجود دارد ولی ایندکس نیست، ایندکس زمانی را ست کن
        if "time" in df.columns and df.index.name != "time":
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.set_index("time")
        return df

    def _read_csv(p: Path) -> pd.DataFrame:
        if not p.exists():
            raise FileNotFoundError(str(p))
        # CSV باید ستون time داشته باشد
        return pd.read_csv(p, parse_dates=["time"], index_col="time")

    # انتخاب فرمت خواندن
    df: Optional[pd.DataFrame] = None
    if fmt_hint:
        fmt = fmt_hint.lower().strip()
        if fmt == "parquet":
            try:
                df = _read_parquet(pq_path)
            except FileNotFoundError:
                # fallback به CSV
                df = _read_csv(csv_path)
        elif fmt == "csv":
            try:
                df = _read_csv(csv_path)
            except FileNotFoundError:
                # fallback به Parquet
                df = _read_parquet(pq_path)
        else:
            raise ValueError(f"Unsupported fmt_hint: {fmt_hint}")
    else:
        # پیش‌فرض: اول Parquet، سپس CSV
        if pq_path.exists():
            df = _read_parquet(pq_path)
        elif csv_path.exists():
            df = _read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Processed file not found: {pq_path} or {csv_path}")

    # تضمین ایندکس زمانی
    if df.index.name != "time":
        # اگر ایندکس زمانی نیست اما ستون time داریم:
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.set_index("time")
        else:
            # اگر نه ایندکس و نه ستون time داریم، این وضعیت با پایپ‌لاین ما سازگار نیست
            raise ValueError("Processed DataFrame must have a time index or a 'time' column.")

    # مرتب‌سازی زمانی و حذف ایندکس‌های تکراری (آخرین مورد را نگه می‌داریم)
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # --- حذف ستون‌های فیچرِ قبلی (idempotent augment) ---
    # هر ستونی که شامل '__' باشد (مثل 'M1__rsi_14') فیچر قبلی است؛
    # برای اجرای مجدد augment آن‌ها را حذف می‌کنیم تا تداخل نام پیش نیاید.
    base_cols = [c for c in df.columns if "__" not in c]
    df = df.loc[:, base_cols].copy()

    return df


def _write_output(df: pd.DataFrame, symbol: str, base_tf: str, cfg: Dict[str, Any], fmt: str, out_mode: str) -> Path:
    fmt = fmt.lower()
    out_dir = _paths_from_cfg(cfg)["processed_dir"] / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / (f"{base_tf.upper()}.features" if out_mode == "features_only" else f"{base_tf.upper()}")
    out = base.with_suffix(".parquet" if fmt == "parquet" else ".csv")
    if fmt == "parquet":
        try:
            df.to_parquet(out)
        except Exception as ex:
            logger.warning("ذخیرهٔ پارکت ناموفق بود (%s). به CSV برمی‌گردیم.", ex)
            out = base.with_suffix(".csv"); df.to_csv(out); return out
        return out
    else:
        df.to_csv(out); return out

# CLI

def _setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl,
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")
    logger.setLevel(lvl)
    logger.propagate = True

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="تولید فیچرهای اندیکاتوری (MTF)")
    p.add_argument("--symbol", required=True)
    p.add_argument("-c","--config", default=str(_project_root()/"f01_config"/"config.yaml"))
    p.add_argument("--base-tf", default=None)
    p.add_argument("--timeframes", nargs="*", default=None)
    p.add_argument("--out-mode", default="augment", choices=["augment","features_only"])
    p.add_argument("--format", default=None, choices=["csv","parquet"])
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)
    cfg = load_config(args.config, enable_env_override=True)

    feat_cfg = (cfg.get("features") or {})
    base_tf_cfg = str(feat_cfg.get("base_timeframe", "M5")).upper()
    base_tf = (args.base_tf or base_tf_cfg).upper()

    specs_cfg: List[str] = list((feat_cfg.get("indicators") or []))
    if not specs_cfg:
        logger.warning("هیچ Spec در config.features.indicators تعریف نشده است.")

    eng_cfg = EngineConfig(
        specs=specs_cfg,
        timeframes=[tf.upper() for tf in (args.timeframes or [])] if args.timeframes else None,
        shift_all=int(feat_cfg.get("shift_features_by", 1)),
        drop_na_head=bool(feat_cfg.get("drop_na_head", True)),
    )

    registry = build_registry()
    engine = IndicatorEngine(eng_cfg, registry)

    fmt_hint = args.format
    base_df = _read_processed(args.symbol, base_tf, cfg, fmt_hint=fmt_hint)

    feats = engine.apply_all(base_df, base_tf_fallback=base_tf)

    if args.out_mode == "features_only":
        out = _write_output(feats, args.symbol, base_tf, cfg, fmt=(args.format or "parquet"), out_mode="features_only")
        logger.info("فایل فیچرها ذخیره شد: %s (rows=%d, cols=%d)", out, len(feats), len(feats.columns))
    else:
        merged = base_df.join(feats, how="left")
        out = _write_output(merged, args.symbol, base_tf, cfg, fmt=(args.format or "parquet"), out_mode="augment")
        logger.info("دیتای پردازش‌شده + فیچرها ذخیره شد: %s (rows=%d, cols=%d)", out, len(merged), len(merged.columns))
    return 0

def _strip_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    حذف ستون‌های فیچر که قبلاً به فایل augment اضافه شده‌اند (الگوی TF__name).
    این کار اجرای مجدد را idempotent می‌کند و از تداخل نام ستون‌ها جلوگیری می‌کند.
    """
    base_cols = [c for c in df.columns if "__" not in c]
    return df.loc[:, base_cols].copy()

if __name__ == "__main__":
    raise SystemExit(main())


