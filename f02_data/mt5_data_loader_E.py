# f02_data/mt5_data_loader_E.py
# Last reviewed at 1405-04-12

r"""
Data Loader برای MT5 (Bot-RL-1)
# =======================================================================================
وظایف:
    - خواندن پیکربندی دانلود از config (symbols, timeframes, lookback_bars, batch_size, save_format)
    - دریافت داده‌ی OHLCV از MT5 از طریق MT5Connector
    - ذخیره‌ی داده به CSV/Parquet با ساختار پوشه‌ای استاندارد در data/raw
        بصورت: data/raw/<SYMBOL>/<TF>.(csv|parquet)
    - تکراری‌ها را حذف و ایندکس زمانی را مرتب می‌کند،
    - خلاصهٔ اجرای دانلود را گزارش می‌دهد و متادیتا می‌نویسد،
    - CLI دارد تا با یک فرمان اجرا شود.
# =======================================================================================
پیش‌نیاز:
    - pandas (اجباری)، (اختیاری) pyarrow یا fastparquet برای Parquet
# =======================================================================================
# اجرای از طریق فراخوانی مستقیم این فایل، سبب میشود که داده های جدید در ریشه پروژه ذخیره شوند

نمونه اجرا (از ریشه‌ی ریپو):
python -m f02_data.mt5_data_loader_E `
    --config f01_config/config.yaml  `
    --symbols XAUUSD_i               `
    --timeframes M1 M2 M4 M20 H1 H4  `
    --lookback 10000000              `
    --format parquet                 `
    --log-level DEBUG

python -m f02_data.mt5_data_loader_E `
    -c .\f01_config\config.yaml      `
    --symbols XAUUSD_i    `
    --timeframes D1 `
    --format csv

python -m f02_data.mt5_data_loader_E

اگر آرگومان‌ها را ندهید، از مقادیر بخش download_defaults در config استفاده می‌شود.
# =======================================================================================
نکات:
    - با config.yaml فعلی سازگار است (paths.raw_dir, download_defaults.*, mt5_credentials).
    - اگر pyarrow/fastparquet نداشتی، format: csv بگذار یا اجازه بده به csv برگردد.
    - فایل متادیتای JSON کنار هر فایل داده نوشته می‌شود تا در گزارش/مانیتورینگ سریع به‌کار رود.
    - برای بازهٔ تاریخی از --date-from/--date-to استفاده کن؛
        در غیر این صورت از lookback یا مقدار پیش‌فرض کانفیگ می‌گیرد.
"""
# =======================================================================================
# Imports & Logger
# =======================================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import re
import pandas as pd
import json
import logging
import argparse
from dateutil import parser

# ------------------ Importing Internal Modules -----------------------------------------
from f10_utils.config_loader import load_config, ConfigLoader
from f10_utils.config_path_funcs import project_root, resolve_raw_dir, full_file_path
from f02_data.mt5_connector import MT5Connector

# -------------------- Logger for this module -------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =======================================================================================
# ساختار برنامه و کمکی‌ها 
# =======================================================================================
# ------------------------------------------------------------------- OK=
@dataclass
class DownloadPlan:
    """طرح دانلود برای یک جفت ارز (نماد/تایم‌فریم)."""
    symbol: str
    timeframe: str
    # یکی از lookback_bars یا (date_from, date_to) باید مشخص باشد
    lookback_bars: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime|str] = None
    range_policy: Optional[str] = None  # could be one of:   "min" | "max" | "date" | "count"

# ------------------------------------------------------------------- OK=
def _parse_dt(value, input_tz, output_tz=None):

    if value is None or value == "":
        return None

    if output_tz is None:
        output_tz = input_tz

    try:
        if isinstance(input_tz, str):
            input_tz = ZoneInfo(input_tz)

        if isinstance(output_tz, str):
            output_tz = ZoneInfo(output_tz)

    except ZoneInfoNotFoundError as e:
        raise ValueError(f"Invalid timezone: {e}")

    # ---------- datetime ----------
    if isinstance(value, datetime):

        if value.tzinfo is None:
            value = value.replace(tzinfo=input_tz)

        return value.astimezone(output_tz)

    # ---------- date ----------
    if isinstance(value, date):

        dt = datetime.combine(value, datetime.min.time())
        dt = dt.replace(tzinfo=input_tz)
        return dt.astimezone(output_tz)

    # ---------- unix timestamp ----------
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=output_tz)

    # ---------- string ----------
    if isinstance(value, str):

        value = value.strip()

        if value.lower() == "now":
            return datetime.now(output_tz)

        dt = datetime.fromisoformat(value)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=input_tz)

        return dt.astimezone(output_tz)

    raise TypeError(f"Unsupported type: {type(value)}")

# ------------------------------------------------------------------- OK=
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """ نرمال‌سازی دیتافریم دریافتی از متاتریدر
    - تنظیم ایندکس زمانی در UTC      <<<=== این تنظیم کامنت شد. این تنظیم کار را خراب کرده بود.
    - انتخاب ستون‌های استاندارد
    - تبدیل نوع‌ها و مرتب‌سازی
    - حذف رکوردهای تکراری بر اساس ایندکس
    """
    
    logger.debug('starting "normalize_df" function')

    # -- 1 -- بررسی وجود و خالی نبودن دیتافریم ورودی ---------------------------------
    if df is None or df.empty:
        logger.debug('df is None or empty')
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"],
                            index=pd.DatetimeIndex([], name="time")
                            )
    cols = list(df.columns)
    logger.debug(f'columns of df at end of part 1 is : {cols}')

    # -- 2 -- معرفی ستون time به عنوان اندکس زمانی دیتافریم --------------------------
    if "time" in cols:
        df["time"] = pd.to_datetime(df["time"]) #, utc=True)
        df.set_index("time", inplace=True)
    logger.debug(f'columns of df at end of part 2 is : {cols}')

    # -- 3 -- مرتب‌سازی بر اساس زمان ---------------------------------------------------
    df.sort_index(inplace=True)

    # -- 4 -- ساخت لیست نام ستونهای موجود و مورد نیاز به عنوان فیلتر ----------------
    keep = [c for c in ["open", "high", "low", "close", "volume", "spread"] if c in cols]

    # -- 5 -- انتخاب ستونهای مشخصی از دیتافریم، طبق لیست قسمت قبل  ------------------
    if keep:
        df = df[keep].copy()
    
    # -- 6 -- معقول سازی نوع دادع ها ----------------------------------------------------
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # float64 با NaN
    
    for col in ["volume", "spread"]:
        if col in df.columns:
            # Int64 از NaN پشتیبانی میکند (برای RL حیاتی است)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    # -- 7 -- معرفی نام ایندکس ----------------------------------------------------------
    df.index.name = "time"

    # -- 8 -- حذف رکوردهای با ایندکس تکراری ---------------------------------------------
    df = df[~df.index.duplicated(keep="last")]

    return df

# ------------------------------------------------------------------- OK=
def _append_or_write(df_new: pd.DataFrame, out_path: Path, fmt: str) -> Tuple[int, int]:
    fmt = fmt.lower().replace(" ", "")
    
    # --- بخش خواندن فایل موجود (با مدیریت خطای کامل برای هر دو فرمت) ---
    if out_path.exists():
        df_old = pd.DataFrame()
        
        if fmt == "parquet":
            try:
                df_old = pd.read_parquet(out_path)
            except Exception:
                # اگر پارکت خراب است، سعی کن CSV هم‌نام را بخوان
                csv_fallback = out_path.with_suffix(".csv")
                if csv_fallback.exists():
                    try:
                        df_old = pd.read_csv(csv_fallback, parse_dates=["time"], index_col="time")
                    except Exception:
                        df_old = pd.DataFrame()  # اگر CSV هم خراب بود، از خالی شروع کن
                else:
                    df_old = pd.DataFrame()
        else:  # fmt == "csv"
            try:
                df_old = pd.read_csv(out_path, parse_dates=["time"], index_col="time")
            except Exception:
                # اگر CSV خراب است، از خالی شروع کن (مشابه رفتار پارکت)
                df_old = pd.DataFrame()
        
        df_old = normalize_df(df_old)
        before = len(df_old)

        parts = [x for x in (df_old, df_new) if x is not None and not x.empty]
        if parts:
            df_all = pd.concat(parts, axis=0)
        else:
            df_all = pd.DataFrame(columns=["open","high","low","close","volume","spread"],
                                  index=pd.DatetimeIndex([], name="time"))
        df_all = normalize_df(df_all)
    else:
        before = 0
        df_all = normalize_df(df_new)

    # --- بخش ذخیره سازی (با پاکسازی فایل خراب قبلی در صورت لزوم) ---
    if fmt == "parquet":
        try:
            df_all.to_parquet(out_path)
        except Exception:
            # اگر پارکت ذخیره نشد، فایل پارکت قبلی (اگر خراب است) را حذف کن تا باعث سردرگمی نشود
            if out_path.exists():
                out_path.unlink()  # حذف فایل پارکت خراب/ناقص
            csv_path = out_path.with_suffix(".csv")
            df_all.to_csv(csv_path)
            return (before, len(df_all))
    else:  # fmt == "csv"
        df_all.to_csv(out_path)
    
    return (before, len(df_all))

# ------------------------------------------------------------------- OK=
def _write_metadata(raw_dir: Path, symbol: str, timeframe: str, rows: int, columns: List, fmt: str) -> Path:
    """
    متادیتا (فایل JSON) را کنار داده ذخیره می‌کند تا برنامه‌های دیگر بتوانند سریع گزارش بگیرند.
    """
    meta = {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "rows": int(rows),
        "columns": list(columns),
        "format": fmt.lower(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = raw_dir / symbol / f"{timeframe.upper()}.meta.json"
    
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path

# ------------------------------------------------------------------- OK= new 050315
def _fetch_candles(connector: MT5Connector, p: DownloadPlan) -> pd.DataFrame:
    """ دریافت داده‌های کندلی از متاتریدر بر اساس طرح دانلود (DownloadPlan) """

    df = pd.DataFrame()

    if not p.symbol or not p.timeframe:
        raise ValueError("symbol and timeframe must be non-empty strings.")
    
    # logger.debug(f"symbol        for _fetch_candles is {p.symbol}")
    # logger.debug(f"timeframe     for _fetch_candles is {p.timeframe}")
    # logger.debug(f"lookback_bars for _fetch_candles is {p.lookback_bars}")
    # logger.debug(f"date_from     for _fetch_candles is {p.date_from}")
    # logger.debug(f"date_to       for _fetch_candles is {p.date_to}")
    # logger.debug(f"range_policy  for _fetch_candles is {p.range_policy}")

    if p.date_from and p.date_to and p.lookback_bars:
        df = connector.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)
        
        policy = str(p.range_policy).lower()    

        if   policy=="min" and len(df)> p.lookback_bars:
            df = df[:p.lookback_bars]
        elif policy=="min" and len(df)<=p.lookback_bars:
            pass
        elif policy=="max" and len(df)>=p.lookback_bars:
            pass
        elif policy=="max" and len(df)< p.lookback_bars:
            df = connector.get_candles_num(p.symbol, p.timeframe, p.lookback_bars)
        elif policy=="date":
            # df = connector.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)
            pass
        elif policy=="count":
            df = connector.get_candles_num(p.symbol, p.timeframe, p.lookback_bars)
        else:
            logger.warning("Check policy, date_from, date_to, lookback_bars.")

    elif p.date_from and p.date_to and not p.lookback_bars:
        df = connector.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)

    elif not p.date_from and not p.date_to and p.lookback_bars:
        df = connector.get_candles_num(p.symbol, p.timeframe, p.lookback_bars)
    
    else:  # if every 3 ones is None or empty
        logger.error(
            f"Invalid DownloadPlan for {p.symbol} {p.timeframe}: "
            "either (lookback_bars) or (date_from and date_to) must be provided. Skipping..."
        )
    
    # logger.debug(f"type of DF is {type(df)}")
    # logger.debug(f"Columns of DF are: {df.columns}")
    # logger.debug(f"Len of DF is: {len(df)}")
    # logger.debug(f"{df.head(5)}")
    return df            


# =======================================================================================
# هسته‌ی دانلود برای حالت batch
# =======================================================================================
class MT5DataLoader_batch:
    """لودر داده‌ی MT5. با MT5Connector کار می‌کند و داده‌ها را در data/raw ذخیره می‌کند """
    # ---------------------------------------------------------------
    # سازنده 
    # --------------------------------------------------------------- OK=
    def __init__(self, 
                 cfg:       Optional[Dict[str, Any]] = None,
                 connector: Optional[MT5Connector]   = None,
                 ) -> None:
        # -- 1 -- config, raw_dir -----------------------------------
        self.cfg: Dict[str, Any] = cfg or load_config()
        self.raw_dir: Path = resolve_raw_dir(self.cfg)

        # -- 2 -- download defaults ---------------------------------
        # گزینه‌های دانلود از config.download_defaults خوانده میشود 
        dl = (self.cfg.get("download_defaults") or {})

        self.default_symbols:    List[str] = list(dl.get("symbols") or [])
        self.default_timeframes: List[str] = list(dl.get("timeframes") or [])
        self.default_lookback:   int       = int (dl.get("lookback_bars") or 5_000_000)

        # -- 3 -- broker_timezone -----------------------------------
        project_cfg = self.cfg.get("project")
        if not project_cfg:
            raise ValueError("'project' key not found in config !")
            
        self.broker_timezone = project_cfg.get("broker_timezone")
        if not self.broker_timezone:
            raise ValueError("'broker_timezone' key not found in 'project' key !")


        # -- 4 -- broker_date_from & broker_date_to -----------------   # <= گیت ورودی از config
        temp = dl.get("broker_date_from")
        self.date_from: datetime = _parse_dt(temp, "UTC")

        temp = dl.get("broker_date_to")
        self.date_to: datetime = _parse_dt(temp, "UTC")

        # logger.info(f"fake time: (1) broker_date_form = {self.date_from},   broker_date_to = {self.date_to}")   # for debug


        # -- 5 -- policy, batch_size, save_format -------------------
        self.range_policy:  str = str(dl.get("range_policy")).lower()

        # self.default_batch: int = int(dl.get("batch_size") or 50)          # NOT USED
        self.save_format:   str = str(dl.get("save_format", "csv")).lower()
        
        if self.save_format not in ("csv", "parquet"):
            logger.warning("save_format is unknown; falling back to csv.")
            self.save_format = "csv"

        # -- 6 -- save_at_utc_time ----------------------------------
        self.save_at_utc_time = bool(dl["save_at_utc_time"])
        
        # -- 7 -- connection to mt5 ---------------------------------
        self.conn = connector or MT5Connector(config=self.cfg)

    # ---------------------------------------------------------------
    # ساخت طرح دانلود
    # --------------------------------------------------------------- OK=
    def build_plan(self,
                   symbols: Optional[Iterable[str]] = None,
                   timeframes: Optional[Iterable[str]] = None,
                   lookback_bars: Optional[int] = None,
                   date_from: Optional[datetime] = None,    # <= گیت ورودی از متد عمومی بیلد
                   date_to: Optional[datetime] = None,      # <= گیت ورودی از متد عمومی بیلد
                   range_policy: Optional[str] = None,
                   ) -> List[DownloadPlan]:
        """
        این تابع بر اساس آرگومان‌ها یا پیش‌فرض‌های کانفیگ، لیست DownloadPlan تولید می‌کند.
        """
        # -- 1 -- Standardize date_from & date_to -------------------
        date_from = _parse_dt(date_from, "UTC")
        date_to   = _parse_dt(date_to  , "UTC")
        # logger.info(f"fake time: (2) date_form = {self.date_from},   date_to = {self.date_to}")   # for debug

        # -- 2 -- Assigning and fallbacks ---------------------------
        syms     = list(symbols      ) if symbols                   else self.default_symbols
        tfs      = list(timeframes   ) if timeframes                else self.default_timeframes
        lb       = int (lookback_bars) if lookback_bars is not None else self.default_lookback
        dt_from  =      date_from      if date_from     is not None else self.date_from
        dt_to    =      date_to        if date_to       is not None else self.date_to
        rng_plcy = str (range_policy ) if range_policy  is not None else self.range_policy

        # -- 3 -- Checking symbols & TFs ----------------------------
        if not syms or not tfs:
            raise ValueError("symbols/timeframes are empty. Set them in config or arguments.")
            
        # -- 4 -- Checking dt_from, dt_to  --------------------------
        if rng_plcy == "date":
            lb = None
            if (dt_from >= dt_to) and dt_to.lower() != "now":
                raise ValueError("check date_farom and date_to in config.")
        elif rng_plcy == "count":
            if (lb<= 0):
                raise ValueError("check lookback_bars in config.")
            dt_from = None; dt_to = None
        elif rng_plcy == "max" or rng_plcy == "min":
            pass

        # logger.info(f"fake time: (3) date_form = {self.date_from},   date_to = {self.date_to}")   # for debug
    
        # -- 5 -- Wrapping plans ------------------------------------
        plans: List[DownloadPlan] = []
        for s in syms:
            for tf in tfs:
                # if date_from and date_to:
                #     plans.append(DownloadPlan(symbol=s, timeframe=tf, date_from=dt_from, date_to=dt_to))
                # else:
                #     plans.append(DownloadPlan(symbol=s, timeframe=tf, lookback_bars=lb))
                plans.append(DownloadPlan(
                    symbol=s, timeframe=tf, lookback_bars=lb,
                    date_from=dt_from, date_to=dt_to,
                    range_policy=rng_plcy)
                )
        return plans

    # ---------------------------------------------------------------
    # اجرای طرح دانلود برای حالت batch
    # --------------------------------------------------------------- OK=
    def run(self,
            plans: List[DownloadPlan],
            ) -> List[Dict[str, Any]]:
        """
        طرح را اجرا می‌کند و خلاصه‌ی هر کار را برمی‌گرداند.
        """
        # -- 1 -- اتصال به متاتریدر ---------------------------------
        if not self.conn.initialize():
            raise RuntimeError("Unable to connect to MT5. Check the credentials/terminal.")

        results: List[Dict[str, Any]] = []  # (Future) شاید بهتر باشد که این متغیر، یک دیکشنری از دیکشنری ها باشد
        # -- 2 -- حلقه روی تمام پلان های دانلود ---------------------
        for p in plans:
            try:
                # -- L1 -- fetching candles -------------------------
                df = _fetch_candles(self.conn, p)
                logger.debug(f'if df after "_fetch_candles" is a pd.DataFrame: {isinstance(df, pd.DataFrame)}')

                df.index = df.index.tz_localize(None)                    # *** درست نمودن زمان داده های خام دانلود شده
                df.index = df.index.tz_localize(self.broker_timezone)    # *** درست نمودن زمان داده های خام دانلود شده
                if self.save_at_utc_time:                                # *** درست نمودن زمان داده های خام دانلود شده
                    df.index = df.index.tz_convert("UTC")                # *** درست نمودن زمان داده های خام دانلود شده

                # -- L2 -- Normalizing DataFrame --------------------
                df = normalize_df(df)
                logger.debug(f'if df after "normalize_df" is a pd.DataFrame: {isinstance(df, pd.DataFrame)}')

                # -- L3 -- Logging requested and returned candles ---
                req = int(p.lookback_bars or self.default_lookback)
                if df is None or df.empty:
                    logger.info("TF=%s | requested=%d | returned=0",
                                p.timeframe, req )
                else:
                    logger.info("TF=%s | requested=%d | returned=%d | range=%s → %s",
                                p.timeframe, req, len(df), df.index.min(), df.index.max() )
                
                # -- L4 -- Writing downloaded dataframes to files ---
                out_path = full_file_path(self.raw_dir, p.symbol, p.timeframe, self.save_format)
                before, after = _append_or_write(df, out_path, self.save_format)

                _write_metadata(self.raw_dir, p.symbol, p.timeframe, after, df.columns, self.save_format)

                # -- L5 -- Logging saved files & sizes --------------
                logger.info("Saved: %s %s → %s (rows: +%d / total %d)",
                            p.symbol, p.timeframe, out_path, after - before, after)

                # -- L6 -- Wrapping list of dictionaries ------------
                results.append({
                    "symbol": p.symbol,    #.upper(),
                    "timeframe": p.timeframe.upper(),
                    "rows_written": after - before,
                    "rows_total": after,
                    "file": str(out_path),
                    "dataframe": df,
                })

            except Exception as ex:
                # -- L7 -- Exception handling -----------------------
                logger.exception("Error downloading/saving %s %s: %s", p.symbol, p.timeframe, ex)
                results.append({
                    "symbol": p.symbol,     #.upper(),
                    "timeframe": p.timeframe.upper(),
                    "error": str(ex),
                })

        # -- 3 -- disconnetcing mt5 ---------------------------------
        self.conn.shutdown()

        return results      


# =============================================================================
# CLI
# =============================================================================
# ------------------------------------------------------------------- OK=
def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        # format="%(asctime)s | %(levelname)-8s | %(filename)s | %(lineno)d : %(funcName)s | %(message)s",
        format="%(asctime)s | %(levelname)-6s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ------------------------------------------------------------------- OK=
def _parse_args() -> argparse.Namespace:
    path = str(project_root() / "f01_config" / "config.yaml")
    parser = argparse.ArgumentParser(description="Download data from MT5 and save to f02_data/raw (CSV/Parquet).")
    
    parser.add_argument("-c","--config", type=str, default=path, help="Config file path (default: f01_config/config.yaml)")
    parser.add_argument("--symbols", nargs="*", default=None, help="List of symbols (Example: XAUUSD EURUSD)")
    parser.add_argument("--timeframes", nargs="*", default=None, help="List of time frames (example: M5 H1)")
    parser.add_argument("--lookback", type=int, default=None, help="Number of closing candles to receive") #تعداد کندلهای انتهایی
    parser.add_argument("--brk_date-from", type=str, default=None, help="Start of interval (ISO 8601 like 2024-01-01T00:00:00Z)")
    parser.add_argument("--brk_date-to", type=str, default=None, help="End of interval (ISO8601)")
    parser.add_argument("--format", type=str, default=None, choices=["csv", "parquet"], help="Storage format")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    
    return parser.parse_args()

# ------------------------------------------------------------------- OK=
def main() -> int:
    # --1 -- استخراج مقادیر از خط فرمان 
    args = _parse_args()

    # -- 2 -- ساخت لاگر و تعیین سطح آن، همراه با تعیین فرمت و فرمت زمان 
    _setup_logging(args.log_level)

    # -- 3 -- بارگذاری کانفیگ (با ENV Override فعال)
    cfg = load_config(args.config, enable_env_override=True)

    # -- 4 -- اوور راید موقتی فرمت بر روی کانفیگ
    # اگر کاربر فرمت را در CLI تعیین کرد، آن را در cfg منعکس کنیم (Override موقتی) 
    #setdefault راه کوتاهی است برای «دریافت مقدار یا ایجاد/قرار دادن مقدار پیش‌فرض در صورت نبودن» — خواندن + نوشتن هم‌زمان 
    if args.format:
        # cfg.setdefault("download_defaults", {})
        # cfg["download_defaults"]["save_format"] = args.format
        cfg.setdefault("download_defaults", {})["save_format"] = args.format
    # -- 5 -- ساخت کانکتور
    my_connector = MT5Connector(config=cfg)

    # -- 6 -- ساخت بچ لودر
    loader = MT5DataLoader_batch(cfg=cfg, connector=my_connector)


    # -- 7 -- ساخت پلان های دانلود
    # if args.brk_date_from is not None:
    #     date_from = _parse_dt(args.brk_date_from, ZoneInfo("Europe/Athens"))   # <= گیت ورودی از CLI
    # if args.brk_date_to is not None:
    #     date_to   = _parse_dt(args.brk_date_to  , ZoneInfo("Europe/Athens"))   # <= گیت ورودی از CLI

    plans = loader.build_plan(
        symbols=args.symbols,
        timeframes=args.timeframes,
        lookback_bars=args.lookback,
        date_from=args.brk_date_from,
        date_to=args.brk_date_to,
        # range_policy="count",
    )

    # -- 8 -- اجرای لودر و دریافت نتیجه دانلودها
    results = loader.run(plans)

    # -- 9 -- نوشتن نام ستونهای محصول
    for res in results:
        symbol = res.get("symbol")
        tf = res.get("timeframe")
        df = res.get("dataframe")  # اگر کلید موجود نباشد None برمی‌گرداند

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df.to_csv(f"__{symbol}_{tf}.csv")

    # -- 10 -- گزارش خلاصه 
    ok  = [r for r in results if "error" not in r]
    bad = [r for r in results if "error"     in r]

    logger.info("Summary: Successful %d | Error %d", len(ok), len(bad))
    if bad:
        for r in bad:
            logger.error("Failed: %s %s → %s", r.get("symbol"), r.get("timeframe"), r.get("error"))
        return 2
    return 0

# ------------------------------------------------------------------- OK=
# اجرای از طریق فراخوانی مستقیم این فایل، سبب میشود که داده های جدید در ریشه پروژه ذخیره شوند
if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =============================================================================
""" Func Names                                 Used in Functions: ...
                            1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
1  DownloadPlan            --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --
2  resolve_raw_dir        --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --
3  full_file_path         --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
4  normalize_df           --  --  --  --  ok  --  --  --  --  --  --  ok  --  --  --  --  --
5  _append_or_write        --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
6  _write_metadata         --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
7  _floor_to_last_closed   --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --
8  _lookback_to_range      --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  Commented
9  MT5DataLoader           --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
10 __init__                --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --
11 build_plan              --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
12 run                     --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
13 _parse_args             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
14 _setup_logging          --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
15 _parse_dt               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
16 main                    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
17 (Global code)           -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
"""
