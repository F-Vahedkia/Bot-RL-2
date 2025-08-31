# -*- coding: utf-8 -*-
# f15_scripts/news_build_cache.py
"""
Build/refresh local calendar cache — messages in English; Persian comments.

Examples To Run:
python -m f15_scripts.news_build_cache `
  --urls "https://example.com/thisweek.csv" `
  --out-dir f02_data/news `
  -c f01_config/config.yaml
"""
from __future__ import annotations

import argparse, logging, pathlib
import pandas as pd

try:
    from f10_utils.config_loader import load_config as _load_cfg
except Exception:
    _load_cfg = None

from f06_news.providers.forexfactory_http import ForexFactoryHTTPProvider
from f06_news.normalizer import normalize_forexfactory_csv, concat_and_dedupe
from f06_news.dataset import news_dir_path, save_cache

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser(description="Download and normalize calendar CSV(s), then build Parquet cache.")
    p.add_argument("--urls", nargs="*", default=None, help="List of CSV URLs to download.")
    p.add_argument("--csvs", nargs="*", default=None, help="Local CSV paths (optional).")
    p.add_argument("--out-dir", default=None, help="Output news dir (default: from config or f02_data/news).")
    p.add_argument("-c", "--config", default=None, help="Path to config.yaml (optional).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level)

    cfg = _load_cfg(args.config, enable_env_override=True) if (args.config and _load_cfg) else {}
    news_dir = pathlib.Path(args.out_dir) if args.out_dir else news_dir_path(cfg)
    raw_dir = news_dir / "raw"; raw_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[pathlib.Path] = []

    # 1) Download URLs
    if args.urls:
        provider = ForexFactoryHTTPProvider(urls=list(args.urls), dest_dir=raw_dir)
        saved_paths.extend(provider.fetch())

    # 2) Add local CSVs
    if args.csvs:
        for c in args.csvs:
            p = pathlib.Path(c)
            if p.exists():
                saved_paths.append(p)

    if not saved_paths:
        logging.error("No input (URLs/CSVs). Nothing to do.")
        return 2

    # 3) Normalize all CSVs (FF)
    frames = []
    for p in saved_paths:
        try:
            logging.info("Normalizing: %s", p)
            frames.append(normalize_forexfactory_csv(str(p)))
        except Exception as ex:
            logging.warning("Normalization failed for %s (%s). Skipped.", p, ex)

    if not frames:
        logging.error("No valid frames after normalization.")
        return 3

    # 4) Concat + dedupe + save Parquet cache
    df = concat_and_dedupe(frames)
    out = save_cache(df, news_dir)
    logging.info("Saved cache: %s | rows=%d", out, len(df))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
