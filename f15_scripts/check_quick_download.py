# -*- coding: utf-8 -*-
#!/usr/bin/env python
r"""
scripts/check_quick_download.py
تست سریع دانلود داده از MT5 برای یک نماد/تایم‌فریم (پیش‌فرض: XAUUSD/M5، 1000 کندل)

کارها:
1) بارگذاری کانفیگ (با امکان Override از ENV و .env)
2) ساخت MT5DataLoader
3) ساخت یک طرح دانلود (symbol/timeframe/lookback)
4) اجرای دانلود و ذخیره در data/raw/<SYMBOL>/<TF>.csv|parquet
5) چاپ خلاصه‌ی خروجی و متادیتا (rows و زمان آخرین کندل)

روش اجرا (از ریشه‌ی ریپو):
    python .\f15_scripts\check_quick_download.py -c .\f01_config\config.yaml --symbol XAUUSD --tf M5 --lookback 1000 --format csv
اگر آرگومان‌ها را ندهید، مقادیر پیش‌فرض استفاده می‌شود.
"""
 
from __future__ import annotations
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import timezone

import pandas as pd

# ---------------------------------------------------------
# کشف ریشه‌ی ریپو و آماده‌سازی sys.path برای ایمپورت ماژول‌ها
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = None
for cand in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (cand / "f10_utils").exists() and (cand / "f02_data").exists():
        PROJECT_ROOT = cand
        break
if PROJECT_ROOT is None:
    PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# ایمپورت ماژول‌های پروژه
# ---------------------------------------------------------
from f10_utils.config_loader import load_config
from f02_data.mt5_data_loader import MT5DataLoader, _project_root  # استفاده از همان منطق مسیردهی

# ---------------------------------------------------------
# تنظیم لاگ
# ---------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

log = logging.getLogger("test_quick_download")


def resolve_raw_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    """
    سازگار با منطق MT5DataLoader: paths.raw_dir یا data/raw را برمی‌گرداند.
    """
    paths = cfg.get("paths", {}) or {}
    raw = paths.get("raw_dir") or (Path(paths.get("data_dir", "data")) / "raw")
    return _project_root() / raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="تست سریع دانلود داده از MT5 (یک نماد/تایم‌فریم).")
    p.add_argument("-c", "--config", type=str, default=str(PROJECT_ROOT / "f01_config" / "config.yaml"),
                   help="مسیر فایل کانفیگ (پیش‌فرض: f01_config/config.yaml)")
    p.add_argument("--symbol", type=str, default="XAUUSD", help="نماد (پیش‌فرض: XAUUSD)")
    p.add_argument("--tf", "--timeframe", dest="timeframe", type=str, default="M5",
                   help="تایم‌فریم (مثال: M1/M5/M15/M30/H1/H4؛ پیش‌فرض: M5)")
    p.add_argument("--lookback", type=int, default=1000, help="تعداد کندل‌های انتهایی برای دریافت (پیش‌فرض: 1000)")
    p.add_argument("--format", type=str, default=None, choices=["csv", "parquet"],
                   help="فرمت ذخیره‌سازی؛ اگر ندهید از config استفاده می‌شود.")
    p.add_argument("--log-level", type=str, default="INFO", help="سطح لاگ: DEBUG/INFO/WARN/ERROR")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    log.info("شروع تست سریع دانلود…")

    try:
        # 1) بارگذاری کانفیگ (ENV فعال)
        cfg = load_config(args.config, enable_env_override=True)
        if args.format:
            cfg.setdefault("download_defaults", {})
            cfg["download_defaults"]["save_format"] = args.format

        # 2) ساخت لودر و طرح
        loader = MT5DataLoader(cfg=cfg)
        plans = loader.build_plan(
            symbols=[args.symbol],
            timeframes=[args.timeframe],
            lookback_bars=args.lookback,
        )

        # 3) اجرا
        results = loader.run(plans)
        if not results:
            log.error("هیچ نتیجه‌ای برنگشت. طرح/اتصال را بررسی کنید.")
            return 2

        # 4) گزارش نتیجه
        ok = [r for r in results if "error" not in r]
        bad = [r for r in results if "error" in r]
        for r in ok:
            log.info("موفق: %s %s → rows_written=+%s total=%s file=%s",
                     r["symbol"], r["timeframe"], r["rows_written"], r["rows_total"], r["file"])
        for r in bad:
            log.error("ناموفق: %s %s → %s", r["symbol"], r["timeframe"], r["error"])

        # 5) اگر موفق بود، چند خط آخر داده را نشان بدهیم
        if ok:
            out_path = Path(ok[0]["file"])
            fmt = out_path.suffix.lower()
            if fmt == ".parquet":
                try:
                    df = pd.read_parquet(out_path)
                except Exception:
                    # اگر موتور parquet نصب نبود، نسخه‌ی CSV کنار آن را می‌خوانیم
                    csv_path = out_path.with_suffix(".csv")
                    df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time") if csv_path.exists() else None
            else:
                df = pd.read_csv(out_path, parse_dates=["time"], index_col="time")

            if df is not None and not df.empty:
                last_ts = df.index.max().tz_convert(timezone.utc) if getattr(df.index, "tz", None) else df.index.max()
                log.info("آخرین کندل ذخیره‌شده: %s | تعداد کل ردیف‌ها: %d", last_ts, len(df))
                # نمایش 5 ردیف آخر برای کنترل چشمی
                tail = df.tail(5)
                with pd.option_context("display.max_columns", None, "display.width", 120):
                    print("\n----- آخرین 5 کندل -----\n", tail)
            else:
                log.warning("فایل خروجی خالی است یا خوانده نشد: %s", out_path)

        return 0 if ok and not bad else 2

    except Exception as ex:
        log.exception("خطا در اجرای تست: %s", ex)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
