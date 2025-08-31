# -*- coding: utf-8 -*-
# f15_scripts/news_refresh_daily.py
"""
Daily refresh wrapper for calendar cache — messages in English; Persian comments.

Examples:
  python -m f15_scripts.news_refresh_daily ^
    --urls "https://example.com/thisweek.csv" ^
    -c f01_config/config.yaml
"""
from __future__ import annotations

import argparse, logging
from f15_scripts.news_build_cache import main as build_cache_main  # Persian: از اسکریپت اصلی استفاده مجدد

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                        datefmt="%H:%M:%S")

def parse_args():
    p = argparse.ArgumentParser(description="Daily refresh for news calendar cache.")
    p.add_argument("--urls", nargs="*", default=None, help="List of CSV URLs to download.")
    p.add_argument("--csvs", nargs="*", default=None, help="Local CSVs (optional).")
    p.add_argument("-c", "--config", default="f01_config/config.yaml", help="Config path.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    _setup_logging(args.log_level)
    # Persian: همهٔ آرگومان‌ها را به news_build_cache پاس بده
    return build_cache_main()

if __name__ == "__main__":
    raise SystemExit(main())
