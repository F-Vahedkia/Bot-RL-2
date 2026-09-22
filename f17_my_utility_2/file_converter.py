# f17_my_utility_2/file_converter.py
# Run: python -m f17_my_utility_2.file_converter
"""
کلاس تبدیل و خواندن فایل‌های CSV و Parquet
"""
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Literal


class FileConverter:
    """تبدیل بین فرمت‌های CSV و Parquet + خواندن سطرهای آخر Parquet"""

    @staticmethod
    def csv_to_parquet(file_path: str) -> None:
        """خواندن CSV و ذخیره به صورت Parquet در همان آدرس"""
        path = Path(file_path)
        df = pd.read_csv(path)
        df.to_parquet(path.with_suffix(".parquet"))
        print(f"✅ Transformed: {path.name} → {path.stem}.parquet")

    @staticmethod
    def parquet_to_csv(file_path: str) -> None:
        """خواندن Parquet و ذخیره به صورت CSV در همان آدرس"""
        path = Path(file_path)
        df = pd.read_parquet(path)
        df.to_csv(path.with_suffix(".csv"), index=False)
        print(f"✅ Transformed: {path.name} → {path.stem}.csv")

    @staticmethod
    def read_last_n_rows(
        parquet_path: str,
        n: int,
        output_format: Literal["csv", "parquet"],
        output_path: str = None
    ) -> None:
        """
        استخراج n سطر آخر از فایل Parquet بدون خواندن کل فایل
        و ذخیره به فرمت csv یا parquet
        """
        path = Path(parquet_path)
        
        # خواندن فقط اطلاعات متادیتا برای تعداد کل ردیف‌ها
        parquet_file = pq.ParquetFile(path)
        total_rows = parquet_file.metadata.num_rows
        
        # خواندن فقط n سطر آخر
        start_row = max(0, total_rows - n)
        table = parquet_file.read().slice(start_row, n)
        df = table.to_pandas()
        
        # ذخیره در فرمت خواسته شده
        if output_path is None:
            output_path = path.parent / f"{path.stem}_last_{n}.{output_format}"
        else:
            output_path = Path(output_path)
        
        if output_format == "csv":
            df.to_csv(output_path, index=False)
        else:
            df.to_parquet(output_path)
        
        print(f"✅ {n} last rows stored: {output_path}")


# ========== مثال استفاده =====================================================
if __name__ == "__main__":
    # تبدیل CSV به Parquet
    # FileConverter.csv_to_parquet("f02_data/raw/XAUUSD_I/D1.csv")
    # FileConverter.csv_to_parquet("f02_data/raw/XAUUSD_I/H4.csv")
    # FileConverter.csv_to_parquet("f02_data/raw/XAUUSD_I/M1.csv")
    # FileConverter.csv_to_parquet("f02_data/raw/XAUUSD_I/M5.csv")
    # FileConverter.csv_to_parquet("f02_data/raw/XAUUSD_I/M30.csv")
    # FileConverter.csv_to_parquet("f02_data/raw/XAUUSD_I/W1.csv")
    
    # تبدیل Parquet به CSV
    # FileConverter.parquet_to_csv("data.parquet")
    
    # خواندن 10 سطر آخر و ذخیره به صورت CSV
    # FileConverter.read_last_n_rows("data.parquet", 10, "csv")
    
    # خواندن 10 سطر آخر و ذخیره به صورت Parquet
    # FileConverter.read_last_n_rows("f02_data/raw/XAUUSD_I/D1.parquet", 10, "parquet")

    # تبدیل Parquet به CSV
    FileConverter.parquet_to_csv("f02_data/raw/BITCOIN/M1.parquet")
    
