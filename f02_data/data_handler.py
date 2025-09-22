# -*- coding: utf-8 -*-
# f02_data/data_handler.py
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
    --symbol XAUUSD                `
    --base-tf W1                   `
    --timeframes W1                `
    --format parquet
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple
from pathlib import Path
from datetime import datetime, time, timezone
import logging
import numpy as np
import pandas as pd

# ماژول‌های داخلی پروژه
from f10_utils.config_loader import load_config

# --- لاگر ماژول ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# کمکی‌های مسیر و IO
# ============================================================================

def _project_root() -> Path:
    """ریشه‌ی ریپو را از موقعیت فایل حدس می‌زنیم: f02_data/.. → ریشه."""
    return Path(__file__).resolve().parents[1]

def _raw_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) or {}
    raw = paths.get("raw_dir") or (Path(paths.get("data_dir", "data")) / "raw")
    p = _project_root() / raw
    p.mkdir(parents=True, exist_ok=True)
    return p

def _processed_dir_from_cfg(cfg: Dict[str, Any]) -> Path:
    paths = cfg.get("paths", {}) or {}
    proc = paths.get("processed_dir") or (Path(paths.get("data_dir", "data")) / "processed")
    p = _project_root() / proc
    p.mkdir(parents=True, exist_ok=True)
    return p

def _raw_file_path(raw_dir: Path, symbol: str, timeframe: str, prefer_parquet: bool = True) -> Path:
    """مسیر فایل خام را برای نماد/تایم‌فریم تعیین می‌کند (parquet یا csv)."""
    sym_dir = raw_dir / symbol.upper()
    parquet = sym_dir / f"{timeframe.upper()}.parquet"
    csv = sym_dir / f"{timeframe.upper()}.csv"
    if prefer_parquet and parquet.exists():
        return parquet
    if csv.exists():
        return csv
    return parquet  # مسیر پیش‌فرض (ممکن است وجود نداشته باشد)

def _processed_file_path(proc_dir: Path, symbol: str, base_tf: str, fmt: str) -> Path:
    sym_dir = proc_dir / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if fmt.lower() == "parquet" else ".csv"
    return sym_dir / f"{base_tf.upper()}{ext}"


# ============================================================================
# نرمال‌سازی دیتافریم‌های خام
# ============================================================================

def _normalize_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - ایندکس زمانی UTC
    - مرتب‌سازی زمانی
    - ستون‌های استاندارد
    - حذف رکوردهای تکراری بر اساس ایندکس
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "tick_volume", "spread"],
                            index=pd.DatetimeIndex([], name="time"))

    cols = list(df.columns)
    if "time" in cols:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True, drop=True)
    df.sort_index(inplace=True)

    # فقط ستون‌های کلیدی را نگه داریم (در صورت موجود بودن)
    keep = [c for c in ["open", "high", "low", "close", "tick_volume", "spread"] if c in df.columns]
    if keep:
        df = df[keep]

    # نوع‌ها
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["tick_volume", "spread"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # حذف رکوردهای با ایندکس تکراری
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "time"
    return df

def _read_raw_df(path: Path) -> pd.DataFrame:
    """تلاش برای خواندن parquet؛ در صورت خطا/نبود، CSV را امتحان می‌کند."""
    if path.suffix.lower() == ".parquet" and path.exists():
        try:
            return _normalize_raw_df(pd.read_parquet(path))
        except Exception as ex:
            logger.warning("Failed to read Parquet (%s). Switching to CSV.", ex)

    csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    if csv_path.exists():
        return _normalize_raw_df(pd.read_csv(csv_path, parse_dates=["time"], index_col="time"))

    # اگر هیچکدام نبود، دیتافریم خالی
    return _normalize_raw_df(pd.DataFrame())


# ============================================================================
# ویژگی‌های زمانی و سشن‌ها
# ============================================================================

def _parse_hhmm(s: str) -> time:
    """رشته 'HH:MM' به time (UTC)."""
    h, m = s.strip().split(":")
    return time(int(h), int(m), tzinfo=timezone.utc)

def _in_utc_range(t: datetime, start: time, end: time) -> bool:
    """
    بررسی تعلق زمان t (UTC) به بازه start..end (UTC).
    بازه‌های پیچیده مثل عبور از نیمه‌شب را هم پوشش می‌دهد.
    """
    t_utc = t.astimezone(timezone.utc)
    tt = t_utc.timetz()
    if start <= end:
        return start <= tt <= end
    # عبور از نیمه‌شب: مثلاً 21:00 تا 06:00
    return (tt >= start) or (tt <= end)

def _add_session_flags(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """افزودن ستون‌های بولی برای سشن‌های asia/london/newyork طبق کانفیگ."""
    sessions = cfg.get("sessions", {}) or {}
    out = df.copy()
    for name in ["asia", "london", "newyork"]:
        s = sessions.get(name)
        if not s:
            continue
        start = _parse_hhmm(str(s.get("start_utc", "00:00")))
        end = _parse_hhmm(str(s.get("end_utc", "23:59")))
        out[f"session_{name}"] = out.index.map(lambda ts: _in_utc_range(ts, start, end)).astype("bool")
    return out

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


# ============================================================================
# هم‌خط‌سازی چند-تایم‌فریم روی شبکه‌ی base_tf
# ============================================================================

def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """به همه‌ی ستون‌ها پیشوند اضافه می‌کند (برای تمایز تایم‌فریم‌ها)."""
    df2 = df.copy()
    df2.columns = [f"{prefix}_{c}" for c in df2.columns]
    return df2

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

    b = b.reset_index().rename(columns={"time": "time"})
    o = o.reset_index().rename(columns={"time": "time"})

    merged = pd.merge_asof(
        b.sort_values("time"),
        o.sort_values("time"),
        on="time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged.set_index("time", inplace=True)
    merged.index = pd.to_datetime(merged.index, utc=True)
    return merged


# ============================================================================
# کلاس اصلی DataHandler
# ============================================================================

@dataclass
class BuildParams:
    symbol: str
    base_tf: str
    timeframes: Optional[List[str]] = None     # اگر None، از config.download_defaults.timeframes استفاده می‌شود
    prefer_parquet: bool = True

class DataHandler:
    """
    سازنده‌ی دیتاست پردازش‌شده‌ی چند-تایم‌فریم برای آموزش/بک‌تست/اجرا.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        # Setting config
        self.cfg: Dict[str, Any] = cfg or load_config()
        
        # Setting directories
        self.raw_dir: Path = _raw_dir_from_cfg(self.cfg)
        self.proc_dir: Path = _processed_dir_from_cfg(self.cfg)
        
        # پیش‌فرض تایم‌فریم‌ها (اگر کاربر مشخص نکند)
        dl = self.cfg.get("download_defaults", {}) or {}
        self.default_timeframes: List[str] = list(dl.get("timeframes") or [])

        # فرمت ذخیره (اگر CLI ندهد)
        self.save_format: str = str(dl.get("save_format", "parquet")).lower()
        if self.save_format not in ("csv", "parquet"):
            self.save_format = "csv"
        
        # Default base_tf from config (features.base_timeframe)
        feat = (self.cfg.get("features") or {})
        self.default_base_tf: str = str(feat.get("base_timeframe", "M5")).upper()        


    # -----------------------
    # بارگذاری یک تایم‌فریم خام
    # -----------------------
    def _load_raw(self, symbol: str, timeframe: str, prefer_parquet: bool = True) -> pd.DataFrame:
        path = _raw_file_path(self.raw_dir, symbol, timeframe, prefer_parquet=prefer_parquet)
        df = _read_raw_df(path)
        if df.empty:
            #logger.warning("داده‌ی خام %s/%s یافت نشد یا خالی بود: %s", symbol, timeframe, path)
            logger.warning("Raw data %s/%s not found or empty: %s", symbol, timeframe, path)
        return df

    # -----------------------
    # ساخت دیتاست
    # -----------------------
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
        base_raw = self._load_raw(symbol, base_tf, params.prefer_parquet)
        if base_raw.empty:
            raise FileNotFoundError(f"Raw data for {symbol}/{base_tf} is not available. Please download first with check_quick_download.")
        base_prefixed = _prefix_columns(base_raw, base_tf)

        # 2) ادغام سایر تایم‌فریم‌ها
        merged = base_prefixed.copy()
        for tf in tfs:
            tf = tf.upper()
            if tf == base_tf:
                continue
            odf = self._load_raw(symbol, tf, params.prefer_parquet)
            if odf.empty:
                logger.warning("Skipping %s: Raw data not available.", tf)
                continue
            merged = _merge_on_base(merged, odf, prefix=tf)

        # 3) ویژگی‌های زمانی
        merged = _add_time_features(merged, self.cfg)

        # 4) پاکسازی نهایی: حذف رکوردهای دارای NaN کامل (در صورت نیاز می‌توان سفت‌گیرانه‌تر کرد)
        merged.sort_index(inplace=True)
        if merged.index.tz is None:
            merged.index = merged.index.tz_localize(timezone.utc)
        merged = merged[~merged.index.duplicated(keep="last")]

        return merged

    # -----------------------
    # ذخیره‌سازی
    # -----------------------
    def save(self, df: pd.DataFrame, symbol: str, base_tf: str, fmt: Optional[str] = None) -> Path:
        fmt = (fmt or self.save_format or "parquet").lower()
        out = _processed_file_path(self.proc_dir, symbol, base_tf, fmt)
        if fmt == "parquet":
            try:
                df.to_parquet(out)
            except Exception as ex:
                logger.warning("Parquet save failed (%s). Reverting to CSV.", ex)
                out = out.with_suffix(".csv")
                df.to_csv(out)
        else:
            df.to_csv(out)
        # متادیتا
        meta = {
            "symbol": symbol.upper(),
            "base_timeframe": base_tf.upper(),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = out.with_suffix(".meta.json")
        meta_path.write_text(
            pd.Series(meta).to_json(force_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info("Processed data saved: %s (rows=%d, cols=%d)", out, len(df), len(df.columns))
        return out


# ============================================================================
# CLI
# ============================================================================

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="ساخت دیتاست پردازش‌شدهٔ چند-تایم‌فریم از داده‌های خام MT5.")
    p.add_argument("-c", "--config", type=str, default=str(_project_root() / "f01_config" / "config.yaml"),
                   help="مسیر فایل کانفیگ (پیش‌فرض: f01_config/config.yaml)")
    p.add_argument("--symbol", type=str, required=True, help="نماد (مثال: XAUUSD)")
    p.add_argument("--base-tf", type=str, default=None, help="تایم‌فریم پایه")
    p.add_argument("--timeframes", nargs="*", default=None,
                   help="تایم‌فریم‌های مورد استفاده. اگر ندهید از config.download_defaults.timeframes استفاده می‌شود.")
    p.add_argument("--format", type=str, default=None, choices=["csv", "parquet"],
                   help="فرمت خروجی (پیش‌فرض: از config)")
    p.add_argument("--log-level", type=str, default="INFO", help="سطح لاگ: DEBUG/INFO/WARN/ERROR")
    return p.parse_args()

def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)

    # بارگذاری کانفیگ با ENV Override
    cfg = load_config(args.config, enable_env_override=True)

    # ساخت هندلر
    handler = DataHandler(cfg=cfg)

    # اگر فرمت خروجی CLI داده شد، override شود
    if args.format:
        handler.save_format = args.format

    # ساخت دیتاست
    # base_tf=args.base_tf,  # قدیمی
    base_tf = (args.base_tf or handler.default_base_tf or "M5").upper()   # جدید
    params = BuildParams(
        symbol=args.symbol,
        base_tf=base_tf,
        timeframes=args.timeframes,
        prefer_parquet=True,
    )
    df = handler.build(params)
    out = handler.save(df, symbol=args.symbol, base_tf=base_tf, fmt=handler.save_format)

    logger.info("Done. Output: %s", out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
