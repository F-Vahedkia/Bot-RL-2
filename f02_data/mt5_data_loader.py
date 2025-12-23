# -*- coding: utf-8 -*-
# f02_data/mt5_data_loader.py
# Status in (Bot-RL-2): Completed at 1404-09-10

r"""
Data Loader برای MT5 (Bot-RL-1)
# =====================================================================================
وظایف:
- خواندن پیکربندی دانلود از config (symbols, timeframes, lookback_bars, batch_size, save_format)
- دریافت داده‌ی OHLCV از MT5 از طریق MT5Connector
- ذخیره‌ی داده به CSV/Parquet با ساختار پوشه‌ای استاندارد در data/raw
      بصورت: data/raw/<SYMBOL>/<TF>.(csv|parquet)
- تکراری‌ها را حذف و ایندکس زمانی را مرتب می‌کند،
- خلاصهٔ اجرای دانلود را گزارش می‌دهد و متادیتا می‌نویسد،
- CLI دارد تا با یک فرمان اجرا شود.
# =====================================================================================
پیش‌نیاز:
- pandas (اجباری)، (اختیاری) pyarrow یا fastparquet برای Parquet
# =====================================================================================
نمونه اجرا (از ریشه‌ی ریپو):
python -m f02_data.mt5_data_loader   `
    --config f01_config/config.yaml  `
    --symbols XAUUSD                 `
    --timeframes M1                  `
    --lookback 10000                 `
    --format csv

python -m f02_data.mt5_data_loader  `
    -c .\f01_config\config.yaml     `
    --symbols XAUUSD EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD  `
    --timeframes M1 M5 M15 M30 H1 H4 D1 W1 `
    --format parquet

اگر آرگومان‌ها را ندهید، از مقادیر بخش download_defaults در config استفاده می‌شود.
# =====================================================================================
نکات:
- با config.yaml فعلی سازگار است (paths.raw_dir, download_defaults.*, mt5_credentials).
- اگر pyarrow/fastparquet نداشتی، format: csv بگذار یا اجازه بده به csv برگردد.
- فایل متادیتای JSON کنار هر فایل داده نوشته می‌شود تا در گزارش/مانیتورینگ سریع به‌کار رود.
- برای بازهٔ تاریخی از --date-from/--date-to استفاده کن؛
      در غیر این صورت از lookback یا مقدار پیش‌فرض کانفیگ می‌گیرد.

"""
# =============================================================================
# Imports & Logger
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import logging
import argparse
import pandas as pd
from dateutil import parser

# ------------------ Importing Internal Modules ------------------
from f10_utils.config_loader import load_config, ConfigLoader, _project_root
from f02_data.mt5_connector import MT5Connector

# -------------------- Logger for this module --------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =============================================================================
# نگاشت دقیقه هر تایم‌فریم (برای تبدیل lookback به بازهٔ زمانی امن) 
# =============================================================================
_TF_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
    "H1": 60, "H2": 120, "H3": 180, "H4": 240,
    "D1": 1440, "W1": 10080, "MN1": 43200,
}
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

# =============================================================================
# ساختار برنامه و کمکی‌ها 
# =============================================================================
# ------------------------------------------------------------------- OK
@dataclass
class DownloadPlan:
    """طرح دانلود برای یک جفت (نماد/تایم‌فریم)."""
    symbol: str
    timeframe: str
    # یکی از lookback_bars یا (date_from, date_to) باید مشخص باشد
    lookback_bars: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime|str] = None
    range_policy: Optional[str] = None  # could be one of:   "min" | "max" | "date" | "count"

# ------------------------------------------------------------------- OK
def _resolve_raw_dir(cfg: Dict[str, Any]) -> Path:
    """مسیر خروجی داده‌ی خام را از config استخراج می‌کند."""
    paths = cfg.get("paths", {}) or {}
    raw = paths.get("raw_dir") or (Path(paths.get("data_dir", "data")) / "raw")
    raw_path = _project_root() / raw
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path

# ------------------------------------------------------------------- OK
def _full_file_path(raw_dir: Path, symbol: str, timeframe: str, fmt: str) -> Path:
    """مسیر فایل خروجی را بر اساس (نماد/تایم‌فریم/فرمت) می‌سازد."""
    sym_dir = raw_dir / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if fmt.lower() == "parquet" else ".csv"
    return sym_dir / f"{timeframe.upper()}{ext}"

# ------------------------------------------------------------------- OK
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """: نرمال‌سازی DataFrame دریافتی از MT5 
    - تنظیم ایندکس زمانی در UTC
    - انتخاب ستون‌های استاندارد
    - تبدیل نوع‌ها و مرتب‌سازی
    - حذف رکوردهای تکراری بر اساس ایندکس
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "spread"],
                            index=pd.DatetimeIndex([], name="time")
                            )

    cols = list(df.columns)
    
    if "time" in cols:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
    # ------------------------------------------ مرتب‌سازی بر اساس زمان ---------- 
    df.sort_index(inplace=True)

    # ---------- فقط ستون‌های کلیدی را نگه داریم (در صورت موجود بودن) ---------- 
    keep = [c for c in ["open", "high", "low", "close", "volume", "spread"] if c in cols]
    # -------------------------------- فقط ستون‌های کلیدی را نگه داریم ---------- 
    if keep:
        df = df[keep]
    
    # --------------------------------------- نوع داده‌ها را معقول کنیم ---------- 
    # Deletd at 040924
    # for col in ["open", "high", "low", "close"]:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors="coerce")
    # for col in ["volume", "spread"]:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    
    # Added at 040924
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["volume", "spread"]:
        if col in df.columns:
            # df.loc[:, col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            # تبدیل به numeric با coerce، سپس به nullable Int64
            s = pd.to_numeric(df[col], errors="coerce")
           
            # کپی کامل دیتافریم قبل از جایگزینی ستون
            df = df.copy()
            df[col] = pd.Series(s, dtype="Int64")


    df.index.name = "time"
    # ---------------------------------- حذف رکوردهای با ایندکس تکراری ---------- 
    df = df[~df.index.duplicated(keep="last")]
    return df

# ------------------------------------------------------------------- OK
def _append_or_write(df_new: pd.DataFrame, out_path: Path, fmt: str) -> Tuple[int, int]:
    """
    داده‌ی جدید را به فایل موجود اضافه می‌کند (در صورت وجود) و تکراری‌ها را حذف می‌کند.
    خروجی: (تعداد_رکورد_قبل، تعداد_رکورد_بعد از ادغام)
    """
    fmt = fmt.lower().replace(" ","")
    if out_path.exists():
        
        # --- Read exist file ------------------- start
        if fmt == "parquet":
            try:
                df_old = pd.read_parquet(out_path)
            except Exception:
                # اگر موتور parquet نصب نیست یا فایل مشکل دارد، به CSV fallback می‌کنیم
                if out_path.with_suffix(".csv").exists():
                    df_old = pd.read_csv(out_path.with_suffix(".csv"), parse_dates=["time"], index_col="time")
                else:
                    df_old = pd.DataFrame()
        else: # fmt == "csv"
            df_old = pd.read_csv(out_path, parse_dates=["time"], index_col="time")
        # --- Read exist file ------------------- end

        df_old = _normalize_df(df_old)
        before = len(df_old)

        # حذف DFهای خالی برای جلوگیری از FutureWarning و dtype-ambiguity 
        parts = [x for x in (df_old, df_new) if x is not None and not x.empty]
        if parts:
            df_all = pd.concat(parts, axis=0)
        else:
            df_all = pd.DataFrame(columns=["open","high","low","close","volume","spread"], 
                                  index=pd.DatetimeIndex([], name="time")
                                  )
        df_all = _normalize_df(df_all)
    else:  # out_path is not exist => then: before = 0
        before = 0
        df_all = _normalize_df(df_new)

    # Upto here "df_all" is calculated, now, it's time to store it
    if fmt == "parquet":
        try:
            df_all.to_parquet(out_path)
        except Exception:
            # اگر pyarrow/fastparquet در دسترس نیست، CSV ذخیره کن
            csv_path = out_path.with_suffix(".csv")
            df_all.to_csv(csv_path)
            return (before, len(df_all))
    else: # fmt == "csv"
        df_all.to_csv(out_path)
    return (before, len(df_all))

# ------------------------------------------------------------------- OK
def _write_metadata(raw_dir: Path, symbol: str, timeframe: str, rows: int, columns: List, fmt: str) -> Path:
    """
    متادیتا (فایل JSON) را کنار داده ذخیره می‌کند تا برنامه‌های دیگر بتوانند سریع گزارش بگیرند.
    """
    meta = {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "rows": int(rows),
        "columns": list(columns),
        "format": fmt.lower(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = raw_dir / symbol.upper() / f"{timeframe.upper()}.meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path

# ------------------------------------------------------------------- OK
def _floor_to_last_closed(tf: str, now_utc: datetime) -> datetime:
    # محاسبه‌ی زمان آخرین کندل بسته‌شده برای هر TF (UTC floor) 
    # Return the latest *closed* candle time at or before now_utc.

    tf = tf.upper().replace(" ","")   # tf = tf.upper().strip()

    if tf.startswith("M"):
        mins = int(tf[1:]) if tf[1:].isdigit() else _TF_MINUTES.get(tf, 1)
        m = (now_utc.minute // mins) * mins
        #return now_utc.replace(second=0, microsecond=0, minute=0) + timedelta(minutes=m)
        return now_utc.replace(second=0, microsecond=0, minute=m)
    
    if tf.startswith("H"):
        hrs = int(tf[1:]) if tf[1:].isdigit() else 1
        h = (now_utc.hour // hrs) * hrs
        return now_utc.replace(second=0, microsecond=0, minute=0, hour=h)
    
    if tf == "D1":
        return now_utc.replace(second=0, microsecond=0, minute=0, hour=0)
    
    if tf == "W1":
        # هفته‌ی MT5 معمولاً از دوشنبه/یکشنبه به UTC گره می‌خورد؛
        # ساده: یکشنبه 00:00 اخیر (UTC). (در صورت نیاز بعداً قابل تطبیق است)
        dow = now_utc.weekday()  # Mon=0..Sun=6
        # برو به یکشنبه‌ی اخیر
        delta = (dow + 1) % 7
        last_sun = now_utc - timedelta(days=delta)
        return last_sun.replace(second=0, microsecond=0, minute=0, hour=0)
    
    # پیش‌فرض امن: کفِ ساعت
    return now_utc.replace(second=0, microsecond=0, minute=0)

# ------------------------------------------------------------------- (OK)
# به نظر می آید اصلاً نیازی به این تابع نباشد. 
def _lookback_to_range(tf: str, lookback: int) -> tuple[datetime, datetime]:
    # نگاشت lookback به بازه با سقف‌های منطقی و فلورِ date_to 

    tf = tf.upper().strip()
    now_utc = datetime.now(timezone.utc)
    date_to = _floor_to_last_closed(tf, now_utc)

    # English: reasonable caps per TF to avoid extremely huge ranges.
    # (قابل تنظیم در آینده)
    caps = {
        "M1" : 2_000_000, "M2" : 1_800_000, "M3" : 1_600_000, "M5" : 1_500_000,
        "M10": 1_200_000, "M15": 1_000_000, "M30":   800_000,
        "H1" :   600_000, "H2" :   500_000, "H4" :   400_000, "H12":   300_000,
        "D1" :   100_000, "W1" :    10_000, "MN1":   2_000
    }
    lb = int(lookback)
    cap = caps.get(tf, 365 * 200)  # روزانه/پیش‌فرض
    lb = min(lb, cap)

    # English: convert lookback bars → timedelta by TF
    MIN_MAP = {"M1": 1, "M5": 5, "M15": 15, "M30": 30}
    HRS_MAP = {"H1": 1, "H4": 4}
    try:
        if tf in MIN_MAP:
            delta = timedelta(minutes=MIN_MAP[tf] * lb)
        elif tf in HRS_MAP:
            delta = timedelta(hours=HRS_MAP[tf] * lb)
        elif tf == "D1":
            delta = timedelta(days=lb)
        elif tf == "W1":
            delta = timedelta(weeks=lb)
        else:
            delta = timedelta(days=min(lb, 365 * 200))
    except OverflowError:
        return _EPOCH_UTC, date_to

    date_from = date_to - delta
    if date_from < _EPOCH_UTC:
        date_from = _EPOCH_UTC
    # English: ensure strict ordering
    if date_from >= date_to:
        date_from = date_to - timedelta(days=1)
    return date_from, date_to

# =============================================================================
# هسته‌ی دانلود 
# =============================================================================

class MT5DataLoader:
    """لودر داده‌ی MT5. با MT5Connector کار می‌کند و داده‌ها را در data/raw ذخیره می‌کند 
    """
    # -------------------------------------------
    # سازنده 
    # ------------------------------------------- OK
    def __init__(self, 
                 cfg: Optional[Dict[str, Any]] = None,
                 connector: Optional[MT5Connector] = None
                 ) -> None:
        # کانفیگ 
        self.cfg: Dict[str, Any] = cfg or load_config()
        self.raw_dir: Path = _resolve_raw_dir(self.cfg)

        # گزینه‌های دانلود از config.env.download_defaults خوانده میشود 
        dl = ((self.cfg.get("env") or {}).get("download_defaults") or {})

        self.default_symbols: List[str] = list(dl.get("symbols") or [])
        self.default_timeframes: List[str] = list(dl.get("timeframes") or [])
        self.default_lookback: int = int(dl.get("lookback_bars"))

        # --- date_from ------------------------- start
        temp = dl.get("date_from")
        self.date_from: datetime = temp if isinstance(temp, datetime) else datetime.fromisoformat(str(temp))
        # --- date_to --------------------------- start
        temp: Any = dl.get("date_to")
        if str(temp).lower() == "now":
            self.date_to = datetime.now()
        else:
            self.date_to: datetime = temp if isinstance(temp, datetime) else datetime.fromisoformat(str(temp))
        # --------------------------------------- end

        self.range_policy: str = str(dl.get("range_policy", "max")).lower()
        self.default_batch: int = int(dl.get("batch_size"))
        self.save_format: str = str(dl.get("save_format", "csv")).lower()

        if self.save_format not in ("csv", "parquet"):
            logger.warning("save_format is unknown; falling back to csv.")
            self.save_format = "csv"

        # اتصال MT5
        self.conn = connector or MT5Connector(config=self.cfg)

    # -------------------------------------------
    # ساخت طرح دانلود 
    # ------------------------------------------- OK
    def build_plan(self,
                   symbols: Optional[Iterable[str]] = None,
                   timeframes: Optional[Iterable[str]] = None,
                   lookback_bars: Optional[int] = None,
                   date_from: Optional[datetime] = None,
                   date_to: Optional[datetime] = None,
                   range_policy: Optional[str] = None,
                   ) -> List[DownloadPlan]:
        """
        بر اساس آرگومان‌ها یا پیش‌فرض‌های کانفیگ، لیست DownloadPlan تولید می‌کند.
        """
        # --- Assigning and fallbacks -----------
        syms = list(symbols) if symbols else self.default_symbols
        tfs = list(timeframes) if timeframes else self.default_timeframes
        lb = int(lookback_bars) if lookback_bars is not None else self.default_lookback
        dt_from = datetime(date_from) if date_from is not None else self.date_from
        dt_to = datetime(date_to) if date_to is not None else self.date_to
        rng_plcy = str(range_policy) if range_policy is not None else self.range_policy

        # --- Checking symbols & TFs ------------
        if not syms or not tfs:
            raise ValueError("symbols/timeframes are empty. Set them in config or arguments.")
        # --- Checking lb, date_from, date_to ---
        
        if (lb<= 0):
            if (dt_from >= dt_to):
                                       # در این حالت هر دو مشکل دارند و قابل استفاده نیستند 
                raise ValueError("check lookback_bars, date_farom and date_to in config.")
            else: # (dt_from <= dt_to)
                lb = None              # در این حالت فقط از بازه زمانی استفاده میشود 
        else: # lb>0
            if (dt_from >= dt_to):
                dt_from = dt_to = None # در این حالت فقط از lookback استفاده میشود 
            
        # --- Wrapping plans --------------------
        plans: List[DownloadPlan] = []
        for s in syms:
            for tf in tfs:
                # if date_from and date_to:
                #     plans.append(DownloadPlan(symbol=s, timeframe=tf, date_from=dt_from, date_to=dt_to))
                # else:
                #     plans.append(DownloadPlan(symbol=s, timeframe=tf, lookback_bars=lb))
                plans.append(DownloadPlan(symbol=s, timeframe=tf, lookback_bars=lb,
                                          date_from=dt_from, date_to=dt_to,
                                          range_policy=rng_plcy
                                          ))
        return plans

    # -------------------------------------------
    # اجرای طرح دانلود 
    # ------------------------------------------- OK
    def run(self, plans: List[DownloadPlan]) -> List[Dict[str, Any]]:
        """
        طرح را اجرا می‌کند و خلاصه‌ی هر کار را برمی‌گرداند.
        """
        # اتصال 
        if not self.conn.initialize():
            raise RuntimeError("Unable to connect to MT5. Check the credentials/terminal.")

        results: List[Dict[str, Any]] = []
        for p in plans:
            try:
                # --- checking range_policy and get candles ------------------- start
                df = pd.DataFrame()
                
                if p.date_from and p.date_to and p.lookback_bars:                   
                    df = self.conn.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)   
                    policy = str(p.range_policy).lower()    

                    if   policy=="min" and len(df)> p.lookback_bars:
                        df = df[:p.lookback_bars]
                    elif policy=="min" and len(df)<=p.lookback_bars:
                        pass
                    elif policy=="max" and len(df)>=p.lookback_bars:
                        pass
                    elif policy=="max" and len(df)< p.lookback_bars:
                        df = self.conn.get_candles_num(p.symbol, p.timeframe, p.lookback_bars)
                    elif policy=="date":
                        df = self.conn.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)
                    elif policy=="count":
                        df = self.conn.get_candles_num(p.symbol, p.timeframe, p.lookback_bars)
                    else:
                        logger.warning("From ==> mt5_data_loader/run ==> Check policy, date_from, date_to, lookback_bars.")

                elif p.date_from and p.date_to and p.lookback_bars is None:
                    df = self.conn.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)
                
                elif p.date_from is None and p.date_to is None and p.lookback_bars:
                    df = self.conn.get_candles_num(p.symbol, p.timeframe, p.lookback_bars)
                else:  # if every 3 ones is None
                    logger.warning("From ==> mt5_data_loader/run ==> Check date_from, date_to, lookback_bars.")
                    # df = self.conn.get_candles_num(p.symbol, p.timeframe, 1) # read only 1 candle
                
                # --- Normalizing DataFrame -----------------------------------
                df = _normalize_df(df)

                # --- Logging requested and returned candles ------------------
                req = int(p.lookback_bars or self.default_lookback)
                if df is None or df.empty:
                    logger.info("TF=%s | requested=%d | returned=0",
                                p.timeframe, req )
                else:
                    logger.info("TF=%s | requested=%d | returned=%d | range=%s → %s",
                                p.timeframe, req, len(df), df.index.min(), df.index.max() )
                
                # --- Writing downloaded dataframes to files ------------------
                out_path = _full_file_path(self.raw_dir, p.symbol, p.timeframe, self.save_format)
                before, after = _append_or_write(df, out_path, self.save_format)
                _write_metadata(self.raw_dir, p.symbol, p.timeframe, after, df.columns, self.save_format)

                # --- Logging saved files & sizes -----------------------------
                logger.info("Saved: %s %s → %s (rows: +%d / total %d)",
                            p.symbol, p.timeframe, out_path, after - before, after)

                # --- Wrapping list of dictionaries ---------------------------
                results.append({
                    "symbol": p.symbol.upper(),
                    "timeframe": p.timeframe.upper(),
                    "rows_written": after - before,
                    "rows_total": after,
                    "file": str(out_path),
                })

            except Exception as ex:
                logger.exception("Error downloading/saving %s %s: %s", p.symbol, p.timeframe, ex)
                results.append({
                    "symbol": p.symbol.upper(),
                    "timeframe": p.timeframe.upper(),
                    "error": str(ex),
                })

        # قطع اتصال
        self.conn.shutdown()
        return results

# =============================================================================
# CLI
# =============================================================================
# ------------------------------------------------------------------- OK
def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ------------------------------------------------------------------- OK
def _parse_args() -> argparse.Namespace:
    path = str(_project_root() / "f01_config" / "config.yaml")
    parser = argparse.ArgumentParser(description="Download data from MT5 and save to f02_data/raw (CSV/Parquet).")
    
    parser.add_argument("-c","--config", type=str, default=path, help="Config file path (default: f01_config/config.yaml)")
    parser.add_argument("--symbols", nargs="*", default=None, help="List of symbols (Example: XAUUSD EURUSD)")
    parser.add_argument("--timeframes", nargs="*", default=None, help="List of time frames (example: M5 H1)")
    parser.add_argument("--lookback", type=int, default=None, help="Number of closing candles to receive") #تعداد کندلهای انتهایی
    parser.add_argument("--date-from", type=str, default=None, help="Start of interval (ISO 8601 like 2024-01-01T00:00:00Z)")
    parser.add_argument("--date-to", type=str, default=None, help="End of interval (ISO8601)")
    parser.add_argument("--format", type=str, default=None, choices=["csv", "parquet"], help="Storage format")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    
    return parser.parse_args()

# ------------------------------------------------------------------- OK
def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # پشتیبانی ساده از Z و بدون منطقه زمانی => همه را به UTC تبدیل می‌کنیم
    try:
        # سطر زیر: رشته ISO-8601 را به شیء datetime تبدیل میکند 
        # اگر رشته شامل افست زمانی باشد، خروجی tz-aware خواهد بود؛ در غیر این‌صورت tz-naive است 
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        raise ValueError(f"Invalid date format: {s} (example: 2024-01-01T00:00:00Z)")

# ------------------------------------------------------------------- OK
def main() -> int:
    # استخراج مقادیر از خط فرمان 
    args = _parse_args()

    # ساخت لاگر و تعیین سطح آن، همراه با تعیین فرمت و فرمت زمان 
    _setup_logging(args.log_level)

    # بارگذاری کانفیگ (با ENV Override فعال)
    cfg = load_config(args.config, enable_env_override=True)

    # اگر کاربر فرمت را در CLI تعیین کرد، آن را در cfg منعکس کنیم (Override موقتی) 
    #setdefault راه کوتاهی است برای «دریافت مقدار یا ایجاد/قرار دادن مقدار پیش‌فرض در صورت نبودن» — خواندن + نوشتن هم‌زمان 
    if args.format:
        # cfg.setdefault("download_defaults", {})
        # cfg["env"]["download_defaults"]["save_format"] = args.format
        cfg.setdefault("env", {}).setdefault("download_defaults", {})["save_format"] = args.format

    # ساخت لودر و طرح
    loader = MT5DataLoader(cfg=cfg)

    date_from = _parse_dt(args.date_from)
    date_to = _parse_dt(args.date_to)
    plans = loader.build_plan(
        symbols=args.symbols,
        timeframes=args.timeframes,
        lookback_bars=args.lookback,
        date_from=date_from,
        date_to=date_to
    )

    # اجرا 
    results = loader.run(plans)

    # گزارش خلاصه 
    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    logger.info("Summary: Successful %d | Error %d", len(ok), len(bad))
    if bad:
        for r in bad:
            logger.error("Failed: %s %s → %s", r.get("symbol"), r.get("timeframe"), r.get("error"))
        return 2
    return 0

# ------------------------------------------------------------------- OK
if __name__ == "__main__":
    raise SystemExit(main())

# =============================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =============================================================================
""" Func Names                                 Used in Functions: ...
                            1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
1  DownloadPlan            --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --
2  _resolve_raw_dir        --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --
3  _full_file_path         --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
4  _normalize_df           --  --  --  --  ok  --  --  --  --  --  --  ok  --  --  --  --  --
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
