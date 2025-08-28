# -*- coding: utf-8 -*-
# f02_data/mt5_data_loader.py

r"""
Data Loader برای MT5 (Bot-RL-1)
--------------------------------
وظایف:
- خواندن پیکربندی دانلود از config (symbols, timeframes, lookback_bars, batch_size, save_format)
- دریافت داده‌ی OHLCV از MT5 از طریق MT5Connector
- ذخیره‌ی داده به CSV/Parquet با ساختار پوشه‌ای استاندارد در data/raw
      بصورت: data/raw/<SYMBOL>/<TF>.(csv|parquet)
- تکراری‌ها را حذف و ایندکس زمانی را مرتب می‌کند،
- خلاصهٔ اجرای دانلود را گزارش می‌دهد و متادیتا می‌نویسد،
- CLI دارد تا با یک فرمان اجرا شود.

پیش‌نیاز:
- pandas (اجباری)، (اختیاری) pyarrow یا fastparquet برای Parquet

نمونه اجرا (از ریشه‌ی ریپو):
    python -m f02_data.mt5_data_loader \
        --config f01_config/config.yaml \
        --symbols XAUUSD EURUSD \
        --timeframes M5 H1 \
        --lookback 5000 \
        --format csv

اگر آرگومان‌ها را ندهید، از مقادیر بخش download_defaults در config استفاده می‌شود.


نکات:
- با config.yaml فعلی سازگار است (paths.raw_dir, download_defaults.*, mt5_credentials).
- اگر pyarrow/fastparquet نداشتی، format: csv بگذار یا اجازه بده به csv برگردد.
- فایل متادیتای JSON کنار هر فایل داده نوشته می‌شود تا در گزارش/مانیتورینگ سریع به‌کار رود.
- برای بازهٔ تاریخی از --date-from/--date-to استفاده کن؛
      در غیر این صورت از lookback یا مقدار پیش‌فرض کانفیگ می‌گیرد.

- فرمان اجرای برنامه
# python -m f02_data.mt5_data_loader -c .\f01_config\config.yaml --symbols XAUUSD --timeframes M1 M5 M30 H1 --lookback 1000 --format parquet
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import logging
import argparse

import pandas as pd

# ماژول‌های داخلی پروژه
from f10_utils.config_loader import load_config, ConfigLoader
from f02_data.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =====================================================================
# نگاشت دقیقه هر تایم‌فریم (برای تبدیل lookback→بازهٔ زمانی امن)
# =====================================================================
_TF_MINUTES = {
    "M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
    "H1": 60, "H2": 120, "H3": 180, "H4": 240,
    "D1": 1440, "W1": 10080, "MN1": 43200,
}

def _lookback_to_range(tf: str, lookback_bars: int) -> tuple[datetime, datetime]:
    """
    ورودی: تایم‌فریم و تعداد کندل موردنیاز
    خروجی: date_from/date_to امن برای copy_rates_range
    - بازه را طوری می‌سازیم که حداکثر همان تعداد کندلِ بسته‌شده برگردد
    - اگر موجودی کمتر باشد، خودِ MT5 کمتر برمی‌گرداند (بنابراین هرگز هنگ نمی‌کنیم)
    """
    tfu = str(tf).upper().strip()
    minutes = _TF_MINUTES.get(tfu)
    if not minutes:
        raise ValueError(f"Unsupported timeframe for lookback: {tf}")
    now_utc = datetime.now(timezone.utc)
    # کمی حاشیهٔ ۱ کندل اضافه می‌کنیم تا لبِ مرز کم نیاید
    delta_min = minutes * (int(lookback_bars) + 1)
    date_from = now_utc - timedelta(minutes=delta_min)
    date_to = now_utc
    return date_from, date_to

# =====================================================================
# ساختار برنامه و کمکی‌ها
# =====================================================================

@dataclass
class DownloadPlan:
    """طرح دانلود برای یک جفت (نماد/تایم‌فریم)."""
    symbol: str
    timeframe: str
    # یکی از lookback_bars یا (date_from, date_to) باید مشخص باشد
    lookback_bars: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]  # f02_data/.. → ریشه‌ی ریپو


def _resolve_raw_dir(cfg: Dict[str, Any]) -> Path:
    """مسیر خروجی داده‌ی خام را از config استخراج می‌کند."""
    paths = cfg.get("paths", {}) or {}
    raw = paths.get("raw_dir") or (Path(paths.get("data_dir", "data")) / "raw")
    raw_path = _project_root() / raw
    raw_path.mkdir(parents=True, exist_ok=True)
    return raw_path


def _output_path_for(raw_dir: Path, symbol: str, timeframe: str, fmt: str) -> Path:
    """مسیر فایل خروجی را بر اساس نماد/تایم‌فریم/فرمت می‌سازد."""
    sym_dir = raw_dir / symbol.upper()
    sym_dir.mkdir(parents=True, exist_ok=True)
    ext = ".parquet" if fmt.lower() == "parquet" else ".csv"
    return sym_dir / f"{timeframe.upper()}{ext}"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    نرمال‌سازی DataFrame دریافتی از MT5:
    - تنظیم ایندکس زمانی در UTC
    - انتخاب ستون‌های استاندارد
    - تبدیل نوع‌ها و مرتب‌سازی
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "tick_volume", "spread"], index=pd.DatetimeIndex([], name="time"))

    cols = list(df.columns)
    # برخی ترمینال‌ها ستون real_volume هم دارند؛ در خروجی نگه نمی‌داریم ولی مشکلی هم نیست
    keep = [c for c in ["open", "high", "low", "close", "tick_volume", "spread"] if c in cols]
    if "time" in cols:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.set_index("time", inplace=True)
    # مرتب‌سازی بر اساس زمان
    df.sort_index(inplace=True)
    # فقط ستون‌های کلیدی را نگه داریم
    df = df[keep] if keep else df
    # نوع داده‌ها را معقول کنیم
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tick_volume", "spread"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df.index.name = "time"
    # حذف رکوردهای با ایندکس تکراری
    df = df[~df.index.duplicated(keep="last")]
    return df


def _append_or_write(df_new: pd.DataFrame, out_path: Path, fmt: str) -> Tuple[int, int]:
    """
    داده‌ی جدید را به فایل موجود اضافه می‌کند (در صورت وجود) و تکراری‌ها را حذف می‌کند.
    خروجی: (تعداد_رکورد_قبل، تعداد_رکورد_بعد از ادغام)
    """
    fmt = fmt.lower()
    if out_path.exists():
        if fmt == "parquet":
            try:
                df_old = pd.read_parquet(out_path)
            except Exception:
                # اگر موتور parquet نصب نیست یا فایل مشکل دارد، به CSV fallback می‌کنیم
                df_old = pd.read_csv(out_path.with_suffix(".csv"), parse_dates=["time"], index_col="time") if out_path.with_suffix(".csv").exists() else pd.DataFrame()
        else:
            df_old = pd.read_csv(out_path, parse_dates=["time"], index_col="time")
        df_old = _normalize_df(df_old)
        before = len(df_old)
        df_all = pd.concat([df_old, df_new], axis=0)
        df_all = _normalize_df(df_all)
    else:
        before = 0
        df_all = _normalize_df(df_new)

    # ذخیره
    if fmt == "parquet":
        try:
            df_all.to_parquet(out_path)
        except Exception:
            # اگر pyarrow/fastparquet در دسترس نیست، CSV ذخیره کن
            csv_path = out_path.with_suffix(".csv")
            df_all.to_csv(csv_path)
            return (before, len(df_all))
    else:
        df_all.to_csv(out_path)
    return (before, len(df_all))


def _write_metadata(raw_dir: Path, symbol: str, timeframe: str, rows: int, fmt: str) -> Path:
    """
    متادیتا (فایل JSON) را کنار داده ذخیره می‌کند تا برنامه‌های دیگر بتوانند سریع گزارش بگیرند.
    """
    meta = {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "rows": int(rows),
        "format": fmt.lower(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = raw_dir / symbol.upper() / f"{timeframe.upper()}.meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


# =====================================================================
# هسته‌ی دانلود
# =====================================================================

class MT5DataLoader:
    """
    لودر داده‌ی MT5. با MT5Connector کار می‌کند و داده‌ها را در data/raw ذخیره می‌کند.
    """

    def __init__(self,
                 cfg: Optional[Dict[str, Any]] = None,
                 connector: Optional[MT5Connector] = None) -> None:
        # کانفیگ
        self.cfg: Dict[str, Any] = cfg or load_config()
        self.raw_dir: Path = _resolve_raw_dir(self.cfg)

        # گزینه‌های دانلود از config
        dl = self.cfg.get("download_defaults", {}) or {}
        self.default_symbols: List[str] = list(dl.get("symbols") or [])
        self.default_timeframes: List[str] = list(dl.get("timeframes") or [])
        self.default_lookback: int = int(dl.get("lookback_bars", 5000))
        self.default_batch: int = int(dl.get("batch_size", 5000))
        self.save_format: str = str(dl.get("save_format", "csv")).lower()
        if self.save_format not in ("csv", "parquet"):
            logger.warning("save_format is unknown; falling back to csv.")
            self.save_format = "csv"

        # اتصال MT5
        self.conn = connector or MT5Connector(config=self.cfg)

    # -----------------------------
    # ساخت طرح دانلود
    # -----------------------------
    def build_plan(self,
                   symbols: Optional[Iterable[str]] = None,
                   timeframes: Optional[Iterable[str]] = None,
                   lookback_bars: Optional[int] = None,
                   date_from: Optional[datetime] = None,
                   date_to: Optional[datetime] = None) -> List[DownloadPlan]:
        """
        بر اساس آرگومان‌ها یا پیش‌فرض‌های کانفیگ، لیست DownloadPlan تولید می‌کند.
        """
        syms = list(symbols) if symbols else self.default_symbols
        tfs = list(timeframes) if timeframes else self.default_timeframes
        lb = int(lookback_bars) if lookback_bars is not None else self.default_lookback

        if not syms or not tfs:
            raise ValueError("symbols/timeframes are empty. Set them in config or arguments.")

        plans: List[DownloadPlan] = []
        for s in syms:
            for tf in tfs:
                if date_from and date_to:
                    plans.append(DownloadPlan(symbol=s, timeframe=tf, date_from=date_from, date_to=date_to))
                else:
                    plans.append(DownloadPlan(symbol=s, timeframe=tf, lookback_bars=lb))
        return plans

    # -----------------------------
    # اجرای طرح دانلود
    # -----------------------------
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
                if p.date_from and p.date_to:
                    df = self.conn.get_candles_range(p.symbol, p.timeframe, p.date_from, p.date_to)
                else:
                    # به‌جای copy_from_pos، بازهٔ امن می‌سازیم تا هرگز بیشتر از موجودی نخواهیم
                    df_range_from, df_range_to = _lookback_to_range(p.timeframe, int(p.lookback_bars or self.default_lookback))
                    df = self.conn.get_candles_range(p.symbol, p.timeframe, df_range_from, df_range_to)
                    # اگر بیشتر از درخواست برگشت (به ندرت)، برشِ انتهایی می‌گیریم
                    if p.lookback_bars and len(df) > int(p.lookback_bars):
                        df = df.tail(int(p.lookback_bars))

                df = _normalize_df(df)

                # لاگ شفافِ درخواست/بازگشتی
                req = int(p.lookback_bars or self.default_lookback)
                if df is None or df.empty:
                    logger.info("TF=%s | requested=%d | returned=0", p.timeframe, req)
                else:
                    logger.info(
                        "TF=%s | requested=%d | returned=%d | range=%s → %s",
                        p.timeframe, req, len(df), df.index.min(), df.index.max()
                    )

                out_path = _output_path_for(self.raw_dir, p.symbol, p.timeframe, self.save_format)
                before, after = _append_or_write(df, out_path, self.save_format)
                _write_metadata(self.raw_dir, p.symbol, p.timeframe, after, self.save_format)

                results.append({
                    "symbol": p.symbol.upper(),
                    "timeframe": p.timeframe.upper(),
                    "rows_written": after - before,
                    "rows_total": after,
                    "file": str(out_path),
                })
                logger.info("Saved: %s %s → %s (rows: +%d / total %d)",
                            p.symbol, p.timeframe, out_path, after - before, after)
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


# =====================================================================
# CLI
# =====================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download data from MT5 and save to f02_data/raw (CSV/Parquet).")
    parser.add_argument("-c", "--config", type=str, default=str(_project_root() / "f01_config" / "config.yaml"),
                        help="Config file path (default: f01_config/config.yaml)")
    parser.add_argument("--symbols", nargs="*", default=None, help="List of symbols (Example: XAUUSD EURUSD)")
    parser.add_argument("--timeframes", nargs="*", default=None, help="List of time frames (example: M5 H1)")
    parser.add_argument("--lookback", type=int, default=None, help="Number of closing candles to receive") #تعداد کندلهای انتهایی
    parser.add_argument("--date-from", type=str, default=None, help="Start of interval (ISO 8601 like 2024-01-01T00:00:00Z)")
    parser.add_argument("--date-to", type=str, default=None, help="End of interval (ISO8601)")
    parser.add_argument("--format", type=str, default=None, choices=["csv", "parquet"], help="Storage format")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    return parser.parse_args()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    # پشتیبانی ساده از Z و بدون منطقه زمانی → همه را به UTC تبدیل می‌کنیم
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        raise ValueError(f"Invalid date format: {s} (example: 2024-01-01T00:00:00Z)")

def main() -> int:
    args = _parse_args()
    _setup_logging(args.log_level)

    # بارگذاری کانفیگ (با ENV Override فعال)
    cfg = load_config(args.config, enable_env_override=True)

    # اگر کاربر فرمت را در CLI تعیین کرد، آن را در cfg منعکس کنیم (Override موقتی)
    if args.format:
        cfg.setdefault("download_defaults", {})
        cfg["download_defaults"]["save_format"] = args.format

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


if __name__ == "__main__":
    raise SystemExit(main())
