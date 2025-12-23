# -*- coding: utf-8 -*-
# f02_data/data_handler.py
# Status in (Bot-RL-2): Completed at 1404-09-15
r"""
DataHandler (Bot-RL-2)
----------------------
هدف:
- دادهٔ خام هر تایم‌فریم را از data/raw/<SYMBOL>/<TF>.(csv|parquet) می‌خوانَد،
- همه را روی یک شبکهٔ زمانی پایه (base_tf) هم‌خط می‌کند (با merge_asof/ffill)،
- برای هر تایم‌فریم ستون‌ها را با پیشوندِ خودِ تایم‌فریم می‌سازد (مثلاً M5_close, H1_close)،
- ویژگی‌های زمانی (hour/day/session + نسخهٔ نرمال/چرخه‌ای) را اضافه می‌کند
      (طبق تنظیمات features.time_features در کانفیگ)،
- خروجی را در data/processed/<SYMBOL>/<base_tf>.(csv|parquet) ذخیره می‌کند و متادیتا می‌نویسد،
- CLI(Command Line Interface) دارد تا با یک فرمان اجرا شود.

نکته:
- برای دقت بیشتر، بهتر است حداقل base_tf را از داده‌های خام دانلود کرده باشید (با check_quick_download.py).

طراحی:
- برای هر تایم‌فریم، ستون‌ها با پیشوند همان تایم‌فریم ساخته می‌شوند (مثال: M5_close, H1_close).
- شبکه‌ی زمانی پایه از خودِ دیتای base_tf ساخته می‌شود (Index = زمانِ UTC).
- ادغام سایر تایم‌فریم‌ها با merge_asof (جهت left) و روش ffill انجام می‌شود
  تا در بازه‌ی بین کندل‌ها مقدار «آخرین کندل بسته‌شده» آن تایم‌فریم تکرار شود.

تنظیمات مورد استفاده از کانفیگ:
- paths.raw_dir / paths.processed_dir (یا data/raw و data/processed پیش‌فرض)
- features.time_features: add_hour_of_day, add_day_of_week, add_session_flags, normalize_time
- sessions: {asia,london,newyork}.start_utc / end_utc  (برای ساخت فلگ‌های سشن)
- project.timezone (پیش‌فرض UTC)
- download_defaults.timeframes  (در صورت ندادن timeframes به CLI)

- فرمان اجرای قدیمی برنامه
# python -m f02_data.data_handler --symbol XAUUSD --base-tf M5 --timeframes M5 M30 H1 -c .\f01_config\config.yaml --format parquet

- فرمان اجرا جدید برنامه 
-  بدون base_tf ، یعنی براساس آنچه در کانفیگ داده شده:
python -m f02_data.data_handler  `
    -c .\f01_config\config.yaml  `
    --symbol XAUUSD              `
    --timeframes M1 M5 M30 H1    `
    --format parquet

- فرمان اجرای جدید برنامه
-  همراه با base_tf
python -m f02_data.data_handler    `
    -c .\f01_config\config.yaml    `
    --symbol XAUUSD                `
    --base-tf H1                   `
    --timeframes H1 H4 D1 W1       `
    --format parquet

- فرمان اجرای جدید برنامه
-  همراه با base_tf
python -m f02_data.data_handler    `
    -c .\f01_config\config.yaml    `
    --symbol GBPUSD                `
    --base-tf W1                   `
    --timeframes W1                `
    --format parquet
"""

# =============================================================================
# Imports & Logger
# ============================================================================= OK
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pathlib import Path
from datetime import datetime, time, timezone
import logging
import numpy as np
import pandas as pd

# ------------------ Importing Internal Modules ---------------------
from f02_data.mt5_data_loader import _resolve_raw_dir, _full_file_path, _normalize_df
from f10_utils.config_loader import load_config, _project_root

# -------------------- Logger for this module -----------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# کمکی‌های مسیر و IO 
# ============================================================================= OK
# ساخت آدرس کامل دسترسی به پوشه processed و ساخت همان پوشه 
def _resolve_process_dir(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) or {}
    proc = paths.get("processed_dir") or (Path(paths.get("data_dir", "data")) / "processed")
    p = _project_root() / proc
    p.mkdir(parents=True, exist_ok=True)
    return p

# ---------------------------
# _resolve_raw_dir  imported from f02_data.mt5_data_loader
# _full_file_path   imported from f02_data.mt5_data_loader

# =============================================================================
# نرمال‌سازی دیتافریم‌های خام 
# ============================================================================= OK
# _normalize_df   imported from f02_data.mt5_data_loader
# ---------------------------

def _read_raw_df(path: Path) -> pd.DataFrame:
    """تلاش برای خواندن parquet؛ در صورت خطا/نبود، CSV را امتحان می‌کند."""
    if path.suffix.lower() == ".parquet" and path.exists():
        try:
            return _normalize_df(pd.read_parquet(path))
        except Exception as ex:
            logger.warning("Failed to read Parquet (%s). Switching to CSV.", ex)

    csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    if csv_path.exists():
        return _normalize_df(pd.read_csv(csv_path, parse_dates=["time"], index_col="time"))

    # اگر هیچکدام نبود، دیتافریم خالی
    return _normalize_df(pd.DataFrame())

# =============================================================================
# ویژگی‌های زمانی و سشن‌ها 
# =============================================================================

# ------------------------------------------------------------------- OK
def _parse_hhmm(s: str) -> time:
    """رشته 'HH:MM' به time (UTC)."""
    h, m = s.strip().split(":")
    return time(int(h), int(m), tzinfo=timezone.utc)

# ------------------------------------------------------------------- OK
def _in_utc_range(t: datetime, start: time, end: time) -> bool:
    """
    بررسی تعلق زمان t (UTC) به بازه start..end (UTC).
    بازه‌های پیچیده مثل عبور از نیمه‌شب را هم پوشش می‌دهد.
    """
    t_utc = t.astimezone(timezone.utc)
    tt = t_utc.timetz()  # فقط جزء زمانی (ساعت:دقیقه:ثانیه با اطلاعات منطقه زمانی) را در متغیر tt قرار میدهد 
    if start <= end:
        return start <= tt <= end
    # عبور از نیمه‌شب: مثلاً 21:00 تا 06:00
    return (tt >= start) or (tt <= end)

# ------------------------------------------------------------------- OK
def _add_session_flags(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """افزودن ستون‌های بولی برای سشن‌های asia/london/newyork طبق کانفیگ."""
    sessions = ((cfg.get("env") or {}).get("sessions")) or {}
    
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    idx_times = out.index.tz_convert("UTC")

    for name in ["asia", "london", "newyork"]:
        s = sessions.get(name)
        if not s:
            continue
        start = _parse_hhmm(str(s.get("start_utc", "00:00")))
        end = _parse_hhmm(str(s.get("end_utc", "23:59.9999")))
        #out[f"session_{name}"] = out.index.map(lambda ts: _in_utc_range(ts, start, end)).astype("bool")
        # سطر بالا قدیمی است و با آن برنامه جواب داده است 
        out[f"session_{name}"] = [ _in_utc_range(ts, start, end) for ts in idx_times ]
    return out

# ------------------------------------------------------------------- OK
def _add_time_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    افزودن hour_of_day, day_of_week و نگاشت چرخه‌ای (sin/cos)؛
    و فلگ‌های سشن‌ها در صورت فعال بودن.
    """
    tf_cfg = ((cfg.get("features") or {}).get("time_features") or {})
    add_hour = bool(tf_cfg.get("add_hour_of_day", True))
    add_dow = bool(tf_cfg.get("add_day_of_week", True))
    add_sessions = bool(tf_cfg.get("add_session_flags", True))
    do_norm = bool(tf_cfg.get("normalize_time", True))

    out = df.copy()

    if add_hour:
        out["hour_of_day"] = out.index.map(lambda x: x.astimezone(timezone.utc).hour).astype("int64")
        if do_norm:
            # نگاشت چرخه‌ای ساعت (۰..۲π)
            rad = 2.0 * np.pi * out["hour_of_day"] / 24.0
            out["hour_sin"] = np.sin(rad)
            out["hour_cos"] = np.cos(rad)

    if add_dow:
        out["day_of_week"] = out.index.map(lambda x: x.astimezone(timezone.utc).weekday()).astype("int64")
        if do_norm:
            # نگاشت چرخه‌ای روز هفته (۰..۲π بر ۷)
            rad = 2.0 * np.pi * out["day_of_week"] / 7.0
            out["dow_sin"] = np.sin(rad)
            out["dow_cos"] = np.cos(rad)

    if add_sessions:
        out = _add_session_flags(out, cfg)

    return out


# =============================================================================
# هم‌خط‌سازی چند-تایم‌فریم روی شبکه‌ی base_tf 
# =============================================================================
# ------------------------------------------------------------------- OK
def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """به همه‌ی ستون‌ها پیشوند اضافه می‌کند (برای تمایز تایم‌فریم‌ها)."""
    df2 = df.copy()
    df2.columns = [f"{prefix}_{c}" for c in df2.columns]
    return df2

# ------------------------------------------------------------------- OK
def _merge_on_base(base_df: pd.DataFrame, other_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    ادغام DataFrame کمکی روی شبکه‌ی base_df با merge_asof (ffill).
    - فرض: ایندکس هر دو UTC و مرتب است.
    """
    if other_df.empty:
        # اگر دیتای کمکی خالی بود، فقط base را برگردان
        return base_df

    # merge_asof روی ستون زمان؛ بنابراین index را به ستون تبدیل می‌کنیم
    b = base_df.copy()
    o = _prefix_columns(other_df, prefix).copy()

    b.index.name = "time"# نام ستون ایندکس را برابر با time قرار میدهد 
    b = b.reset_index()  # ایندکس فعلی دیتافریم b را به یک ستون معمولی تبدیل می‌کند و یک ایندکس عددی جدید 0,1,2,… می‌سازد 
    o.index.name = "time"
    o = o.reset_index()

    merged = pd.merge_asof(
        b.sort_values("time"),
        o.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True)
    
    merged.set_index("time", inplace=True)   # برعکس متد reset_index عمل میکند 
    merged.index = pd.to_datetime(merged.index, utc=True)
    return merged


# =============================================================================
# کلاس اصلی DataHandler 
# =============================================================================

# ------------------------------------------------------------------- OK
@dataclass
class BuildParams:
    symbol: str 
    base_tf: str
    timeframes: Optional[List[str]] = None            # اگر None، از config.download_defaults.timeframes استفاده می‌شود 
    format_: Literal["csv", "parquet"] = "parquet"    # csv | parquet

# ------------------------------------------------------------------- OK
class DataHandler:
    """
    سازنده‌ی دیتاست پردازش‌شده‌ی چند-تایم‌فریم برای آموزش/بک‌تست/اجرا.
    """
    # -------------------------------------------
    # سازنده 
    # ------------------------------------------- OK
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        # --- Setting config ------------------------------------------
        self.cfg: Dict[str, Any] = cfg or load_config()
        
        # --- Setting directories -------------------------------------
        self.raw_dir: Path = _resolve_raw_dir(self.cfg)
        self.proc_dir: Path = _resolve_process_dir(self.cfg)

        # -------------- پیش‌فرض تایم‌فریم‌ها (اگر کاربر مشخص نکند) --- 
        dl = ((self.cfg.get("env") or {}).get("download_defaults") or {})
        self.default_timeframes: List[str] = list(dl.get("timeframes") or [])

        # ------------------------------ فرمت ذخیره (اگر CLI ندهد) --- 
        self.save_format: str = str(dl.get("save_format", "parquet")).lower()
        if self.save_format not in ("csv", "parquet"):
            self.save_format = "parquet"
        
        # --- Default base_tf from config (features.base_timeframe) ---
        feat = (self.cfg.get("features") or {})
        self.default_base_tf: str = str(feat.get("base_timeframe", "M5")).upper()        

    # -------------------------------------------
    # بارگذاری یک تایم‌فریم خام 
    # ------------------------------------------- OK
    def _load_raw(self, symbol: str, timeframe: str, fmt: str = "parquet") -> pd.DataFrame:
        path = _full_file_path(self.raw_dir, symbol, timeframe, fmt=fmt)
        df = _read_raw_df(path)
        if df.empty:
            logger.warning("Raw data %s/%s not found or empty: %s", symbol, timeframe, path)
        return df

    # -------------------------------------------
    # ساخت دیتاست 
    # ------------------------------------------- OK
    def build(self, params: BuildParams) -> pd.DataFrame:
        """
        دیتاست نهایی را می‌سازد:
        1) base_df را از خام می‌سازد (ستون‌ها با پیشوند base_tf)
        2) سایر تایم‌فریم‌ها را با merge_asof به آن می‌چسباند
        3) ویژگی‌های زمانی را اضافه می‌کند
        """
        symbol = params.symbol.upper()
        base_tf = params.base_tf.upper()
        tfs: List[str] = list(params.timeframes) if params.timeframes else self.default_timeframes
        # اطمینان: base_tf در لیست باشد (اول لیست)
        if base_tf not in tfs:
            tfs = [base_tf] + tfs

        # 1) بارگذاری base
        base_raw = self._load_raw(symbol, base_tf, params.format_)
        if base_raw.empty:
            raise FileNotFoundError(f"Raw data for {symbol}/{base_tf} is not available. Please download first with check_quick_download.")
        base_prefixed = _prefix_columns(base_raw, base_tf)

        # 2) ادغام سایر تایم‌فریم‌ها
        merged = base_prefixed.copy()
        for tf in tfs:
            tf = tf.upper()
            if tf == base_tf:
                continue
            odf = self._load_raw(symbol, tf, params.format_)   # odf: other df
            if odf.empty:
                logger.warning("Skipping %s: Raw data not available.", tf)
                continue
            merged = _merge_on_base(merged, odf, prefix=tf)

        # 3) ویژگی‌های زمانی
        merged = _add_time_features(merged, self.cfg)
    
        # ----------------------------------------------------------- start part-4
        # 4) ساخت پرچم‌های کیفیت (QC) روی دیتافریم 
        # تشخیص «گپ زمانی بزرگ»، «اسپایک قیمتی بزرگ» و «ثابت/بدون تغییر بودن قیمت» 
        
        # تبدیل ایندکس (که باید زمان باشد) به یک سری و بدست آوردن اختلاف زمانی بین هر ردیف و ردیف قبلی. اولین ردیف NaN است 
        dts = merged.index.to_series().diff()
        
        # آستانهٔ گپ را تعریف می‌کند: میانهٔ فواصل زمانی ضربدر ۵. 
        # اگر اصلاً فاصله‌ای وجود نداشته باشد (یا همه NaT باشند) آستانه صفر می‌شود. 
        gap_thr = dts.median()*5 if len(dts.dropna()) else pd.Timedelta(0)

        # ستون بولی qc_gap ایجاد می‌کند که برای هر ردیف نشان می‌دهد آیا فاصلهٔ زمانی از ردیف قبلی بزرگ‌تر از آستانه هست یا نه 
        merged["qc_gap"] = dts > gap_thr

        # انتخاب ستون قیمتی: اگر ستون close وجود داشته باشد از آن استفاده می‌کند، 
        # در غیر این صورت اولین ستونی که نامش با _close تمام می‌شود را برمی‌دارد؛ اگر هیچ‌کدام نباشد c = None 
        c = "close" if "close" in merged.columns else next((x for x in merged.columns if x.endswith("_close")), None)

        # اگر ستون قیمت (c) موجود باشد، تغییر نسبی قیمت بین ردیف فعلی و قبلی را محاسبه می‌کند (pct_change())، 
        # قدر مطلق می‌گیرد و اگر بیشتر از ۵٪ بود آن ردیف را به‌عنوان «اسپایک» علامت می‌زند. 
        # اگر ستونی پیدا نشده باشد، کل ستون qc_spike با False پر می‌شود (پاندا آن را broadcast می‌کند) 
        merged["qc_spike"] = (merged[c].pct_change().abs() > 0.05) if c else False

        #اگر ستون قیمت موجود باشد، اختلاف مطلق بین قیمت فعلی و قبلی را محاسبه می‌کند و 
        # اگر برابر صفر باشد (یعنی قیمت هیچ‌تغییری نکرده)، آن ردیف را «stale» علامت می‌زند. 
        # در صورت نبودن ستون قیمت، qc_stale همه‌اش False خواهد شد. 
        merged["qc_stale"] = (merged[c].diff().abs().eq(0))        if c else False
        # ----------------------------------------------------------- end part-4

        # 5) QC summary (log counts of flags across all TFs)

        # سطر زیر نام ستون‌های QC که واقعاً در merged وجود دارند را جمع می‌کند.
        qc_cols = [k for k in ["qc_gap","qc_spike","qc_stale"] if k in merged.columns]
        if qc_cols:
            logger.info("QC flags (counts): %s", {c: int(merged[c].sum()) for c in qc_cols})

        # 6) پاکسازی نهایی: حذف رکوردهای دارای NaN کامل (در صورت نیاز می‌توان سفت‌گیرانه‌تر کرد)
        merged.sort_index(inplace=True)
        # تبدیل ایندکس از naive به UTC-aware 
        if merged.index.tz is None:
            merged.index = merged.index.tz_localize(timezone.utc)
        # حذف ردیفهای با اندکس تکراری و نگهداشتن فقط آخرین ردیف 
        merged = merged[~merged.index.duplicated(keep="last")]

        # برای نگهداشتن کانتکست آخرین بیلد
        self._last_build_context = {
            "symbol": symbol,
            "base_tf": base_tf,
            "timeframes": list(tfs),
            "rows": int(len(merged)),
            "columns": list(merged.columns),
        }

        return merged

    # -------------------------------------------
    # ذخیره‌سازی 
    # ------------------------------------------- OK
    def save(self, df: pd.DataFrame, symbol: str, base_tf: str, fmt: Optional[str] = None) -> Path:
        fmt = (fmt or self.save_format or "parquet").lower()
        out = _full_file_path(self.proc_dir, symbol, base_tf, fmt)  # برای ذخیره فایل پروسس شده 
        
        if fmt == "parquet":
            try:
                df.to_parquet(out)
            except Exception as ex:
                logger.warning("Parquet save failed (%s). Reverting to CSV.", ex)
                out = out.with_suffix(".csv")
                df.to_csv(out); fmt = "csv"
        else: 
            df.to_csv(out); fmt = "csv"

        # متادیتا
        # meta = {
        #     "symbol": symbol.upper(),
        #     "base_timeframe": base_tf.upper(),
        #     "rows": int(len(df)),
        #     "columns": list(df.columns),
        #     "format": fmt.lower(),
        #     "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        # }
        # meta_path = out.with_suffix(".meta.json")
        # meta_path.write_text(
        #     pd.Series(meta).to_json(force_ascii=False, indent=2),
        #     encoding="utf-8"
        # )

        # ========== Manifest برای تکرارپذیری
        manifest = {
            "symbol": symbol.upper(),             # meta["symbol"],
            "base_timeframe": base_tf.upper(),    # meta["base_timeframe"],
            "rows": int(len(df)),                 # meta["rows"],
            "columns": list(df.columns),          # meta["columns"],
            "format": fmt.lower(),                # meta["format"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),    # meta["updated_at_utc"],
            "config_version": (self.cfg.get("version") or "unknown"),
            "timeframes_used": (self._last_build_context or {}).get("timeframes", []),
            "features": {
                "time_features": (self.cfg.get("features", {}) or {}).get("time_features", {}),
            },
        }
        manifest_path = out.with_suffix(".manifest.json")
        manifest_path.write_text(
            pd.Series(manifest).to_json(force_ascii=False, indent=2),
            encoding="utf-8"
        )
        # ========== 

        logger.info("Processed data saved: %s (rows=%d, cols=%d)", out, len(df), len(df.columns))
        return out

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
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Create a processed multi-timeframe dataset from raw MT5 data.")
    p.add_argument("-c", "--config", type=str, default=str(_project_root() / "f01_config" / "config.yaml"),
                   help="Path to the config file (default: f01_config/config.yaml)")
    p.add_argument("--symbol", type=str, required=True, help="Symbol (example: XAUUSD)")
    p.add_argument("--base-tf", type=str, default=None, help="Base timeframe")
    p.add_argument("--timeframes", nargs="*", default=None,
                   help="Timeframes to use. If not provided, config.download_defaults.timeframes will be used.")
    p.add_argument("--format", type=str, default=None, choices=["csv", "parquet"],
                   help="Output format (default: from config)")
    p.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    return p.parse_args()

# ------------------------------------------------------------------- OK
def main() -> int:
    # استخراج مقادیر از خط فرمان 
    args = _parse_args()

    # ساخت لاگر و تعیین سطح آن، همراه با تعیین فرمت و فرمت زمان 
    _setup_logging(args.log_level)

    # بارگذاری کانفیگ با ENV Override
    cfg = load_config(args.config, enable_env_override=True)

    # ساخت هندلر
    handler = DataHandler(cfg=cfg)

    # اگر فرمت خروجی CLI داده شد، override شود
    if args.format:
        handler.save_format = args.format

    # ساخت دیتاست
    base_tf = (args.base_tf or handler.default_base_tf or "M5").upper()
    format_=(args.format or handler.save_format or "parquet").lower()
    params = BuildParams(
        symbol=args.symbol,
        base_tf=base_tf,
        timeframes=args.timeframes,
        # format_="parquet",   # In the future, this line should delete.
        format_=format_,
    )
    df = handler.build(params)
    # out = handler.save(df, symbol=args.symbol, base_tf=base_tf, fmt=handler.save_format)  # In the future, this line should delete.
    out = handler.save(df, symbol=args.symbol, base_tf=base_tf, fmt=format_)

    logger.info("Done. Output: %s", out)
    return 0

# ------------------------------------------------------------------- OK
if __name__ == "__main__":
    raise SystemExit(main())

# =============================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =============================================================================
""" Func Names                                 Used in Functions: ...
                            1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
1  _resolve_process_dir    --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --
2  _read_raw_df            --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --
3  _parse_hhmm             --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
4  _in_utc_range           --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
5  _add_session_flags      --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --
6  _add_time_features      --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
7  _prefix_columns         --  --  --  --  --  --  --  ok  --  --  --  --  ok  --  --  --  --  --
8  _merge_on_base          --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
9  BuildParams             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
10 DataHandler             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
11 __init__                --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --
12 _load_raw               --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --
13 build                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
14 save                    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
15 _setup_logging          --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
16 _parse_args             --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
17 main                    --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
18 (Global code)           -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
"""