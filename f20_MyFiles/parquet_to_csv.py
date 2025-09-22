#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
اسکریپت تبدیل پارکت به CSV (نمایش 10 سطر اول با هدر ستون‌ها)

آموزش و توضیح:
- این برنامه یک فایل Parquet را می‌خواند و یک فایل CSV خروجی می‌سازد که شامل
  "نام ستون‌ها (هدر)" و "۱۰ ردیف اول داده" است.
- ورودی/خروجی را می‌توان از طریق خط فرمان (CLI) تعیین کرد؛ در غیر این‌صورت
  از مقادیر پیش‌فرض داخل کد استفاده می‌شود.
- در صورت بروز خطا (نبود فایل، مشکل در خواندن پارکت، مجوز نوشتن و ...)،
  پیام خطا به انگلیسی در ترمینال چاپ می‌شود.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

data_line_number = 40
# ========start======================================================================
# تنظیمات پیش‌فرض داخلی (در صورت ندادن آرگومان CLI از این‌ها استفاده می‌شود)
DEFAULT_INPUT_PARQUET = Path("E:/Bot-RL-2/f02_data/processed/XAUUSD/H1.parquet")      # مسیر پیش‌فرض ورودی پارکت
DEFAULT_OUTPUT_CSV    = Path("E:/Bot-RL-2/f20_MyFiles/XAUUSD_H1.csv")   # مسیر پیش‌فرض خروجی CSV
# ========end========================================================================


# ========start======================================================================
# تابع کمکی: پارس کردن آرگومان‌های CLI
# ورودی: آرگومان‌های خط فرمان
# خروجی: نام‌فضای آرگومان‌ها شامل مسیر ورودی/خروجی
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Parquet file to a CSV containing header and first 10 rows."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to input Parquet file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output CSV file"
    )
    return parser.parse_args()
# ========end========================================================================


# ========start======================================================================
# تابع اصلی اجرای برنامه
# ورودی: هیچ (از CLI و/یا مقادیر پیش‌فرض استفاده می‌کند)
# خروجی: وضعیت خروج (0 موفق، 1 خطا)
def main() -> int:
    args = parse_args()

    # تعیین مسیرهای نهایی با اولویت: CLI > پیش‌فرض
    in_path = Path(args.input) if args.input else DEFAULT_INPUT_PARQUET
    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT_CSV

    print(f"[INFO] Input Parquet: {in_path}")
    print(f"[INFO] Output CSV  : {out_path}")

    try:
        if not in_path.exists():
            print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
            return 1

        # خواندن پارکت (نیازمند pyarrow یا fastparquet نصب‌شده)
        df = pd.read_parquet(in_path)

        # گرفتن 10 ردیف اول
        head10 = df.head(data_line_number)

        # نوشتن CSV با هدر ستون‌ها (index حذف می‌شود)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        head10.to_csv(out_path, index=False)

        print(f"[OK] CSV written with header and first 10 rows -> {out_path}")
        return 0

    except Exception as e:
        print(f"[ERROR] Failed to process: {e}", file=sys.stderr)
        return 1
# ========end========================================================================


if __name__ == "__main__":
    sys.exit(main())
