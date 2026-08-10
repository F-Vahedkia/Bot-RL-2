import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, List, Iterator
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# تابع کمکی برای خواندن کارآمد آخرین سطرهای فایل csv بدون load نمودن کل فایل
# =============================================================================
def _read_last_n_rows_csv(
    csv_path: Path,
    last_n_rows: int,
    parse_dates: List[str],
    index_col: str
) -> pd.DataFrame:
    """
    خواندن فقط آخرین N سطر از فایل CSV با استفاده از chunksize و بافر.
    حافظه مصرفی ثابت (مستقل از حجم فایل).
    """
    if last_n_rows <= 0:
        return pd.DataFrame()

    # از deque با حداکثر طول last_n_rows استفاده می‌کنیم تا فقط آخرین ردیف‌ها نگهداری شوند
    from collections import deque
    buffer = deque(maxlen=last_n_rows)

    # خواندن فایل به صورت تکه‌تکه (chunk)
    chunk_iter = pd.read_csv(
        csv_path,
        parse_dates=parse_dates,
        index_col=index_col,
        chunksize=10000  # اندازهٔ تکه قابل تنظیم است
    )

    for chunk in chunk_iter:
        # هر تکه را به دیکشنری رکوردها تبدیل و به بافر اضافه کن
        # (توجه: تبدیل به دیکشنری می‌تواند حافظه را موقتاً افزایش دهد، اما بهتر از نگه‌داری کل دیتافریم است)
        for record in chunk.to_dict('records'):
            buffer.append(record)

    # اگر بافر خالی است
    if not buffer:
        return pd.DataFrame()

    # تبدیل بافر به دیتافریم نهایی (فقط شامل آخرین N ردیف)
    df_final = pd.DataFrame(list(buffer))
    # بازگرداندن ایندکس و اطمینان از نوع datetime برای ستون time
    if index_col in df_final.columns:
        df_final.set_index(index_col, inplace=True)
    if parse_dates:
        for col in parse_dates:
            if col in df_final.columns:
                df_final[col] = pd.to_datetime(df_final[col])
    return df_final

# =============================================================================
# تابع کمکی برای خواندن کارآمد آخرین سطرهای فایل parquet بدون load نمودن کل فایل
# =============================================================================
def _read_last_n_rows_parquet(
    file_path: Path,
    last_n_rows: int,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if last_n_rows < 0:
        raise ValueError("last_n_rows cannot be negative")
    if last_n_rows == 0:
        return pd.DataFrame()

    parquet_file = pq.ParquetFile(file_path)
    total_rows = parquet_file.metadata.num_rows
    if total_rows == 0:
        return pd.DataFrame()

    if total_rows <= last_n_rows:
        return parquet_file.read(columns=columns).to_pandas()

    start_row = total_rows - last_n_rows
    rows_so_far = 0
    first_rg_idx = None
    for i in range(parquet_file.num_row_groups):
        rg_rows = parquet_file.metadata.row_group(i).num_rows
        if rows_so_far + rg_rows > start_row:
            first_rg_idx = i
            break
        rows_so_far += rg_rows

    row_groups_to_read = list(range(first_rg_idx, parquet_file.num_row_groups))
    table = parquet_file.read_row_groups(row_groups_to_read, columns=columns)
    df = table.to_pandas()

    offset_in_first_rg = start_row - rows_so_far
    if offset_in_first_rg > 0:
        df = df.iloc[offset_in_first_rg:]

    if len(df) > last_n_rows:
        df = df.iloc[-last_n_rows:]

    return df


# =============================================================================
# Main File
# =============================================================================
def read_raw_df(path: Path, last_n_rows: int) -> pd.DataFrame:
    """
    تلاش برای خواندن parquet و در صورت خطا/نبود، فرمت CSV را با روش chunked امتحان می‌کند.
    دیتا فریم محصول را نرمال می‌کند (با فرض وجود normalize_df).
    """
    # اعتبارسنجی last_n_rows
    path = Path(path)
    if last_n_rows < 0:
        raise ValueError("last_n_rows cannot be negative")

    # اگر پسوند پارکت است و فایل وجود دارد، اولویت با پارکت
    if path.suffix.lower() == ".parquet" and path.exists():
        try:
            df = _read_last_n_rows_parquet(path, last_n_rows)
            return df
        except Exception as ex:
            logger.warning("Failed to read Parquet (%s). Switching to CSV.", ex)
            # fall through to CSV

    # خواندن از CSV (با وجود یا بدون وجود پارکت، در صورت خطا یا فایل csv مستقیم)
    csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    if csv_path.exists():
        try:
            df = _read_last_n_rows_csv(csv_path, last_n_rows, parse_dates=["time"], index_col="time")
            return df
        except Exception as ex:
            logger.error("Failed to read CSV (%s): %s", csv_path, ex)
            return pd.DataFrame()

    # اگر هیچکدام نبود
    return pd.DataFrame()


# =============================================================================
# Test
# =============================================================================
# Run: python -m f02_data.loading_functions
def test():
    from datetime import datetime


    # --- Load data ----------------------------------------------------- 1
    t1 = datetime.now()
    data = pd.read_csv("f02_data/raw/XAUUSD_I/M1.csv")
    data_ = data[-100_000:]
    t2 = datetime.now()
    # ---------------------------
    elapsed = round((t2 - t1).total_seconds(), 3)

    print(f"Time taken to load total_csv and slice: {elapsed} seconds, length_df:{len(data)}")

    # ------------------------------------------------------------------- 2
    t1 = datetime.now()
    data = pd.read_parquet("f02_data/raw/XAUUSD_I/M1.parquet")
    data_ = data[-100_000:]
    t2 = datetime.now()
    # ---------------------------
    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to load total_parquet and slice: {elapsed} seconds, length_df:{len(data)}")

    # ------------------------------------------------------------------- 3
    t1 = datetime.now()
    data = read_raw_df("f02_data/raw/XAUUSD_I/M1.csv", 100_000)
    t2 = datetime.now()
    # ---------------------------
    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to load sliced_csv: {elapsed} seconds, length_df:{len(data)}")

    # ------------------------------------------------------------------- 4
    t1 = datetime.now()
    data = read_raw_df("f02_data/raw/XAUUSD_I/M1.parquet", 100_000)
    t2 = datetime.now()
    # ---------------------------
    elapsed = round((t2 - t1).total_seconds(), 3)
    print(f"Time taken to load sliced_parquet: {elapsed} seconds, length_df:{len(data)}")




if __name__ == "__main__":
    raise SystemExit(test())