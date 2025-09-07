# -*- coding: utf-8 -*-
# f02_data/mt5_connector.py
"""
MT5Connector (نسخهٔ حرفه‌ای برای Bot-RL-1)
-------------------------------------------
قابلیت‌ها:
- اتصال ایمن به MetaTrader5 با پشتیبانی از مسیر ترمینال (terminal path) و retry با backoff سبک.
- ورود (login) با کرِدها از کانفیگ (یا ENV)، و بررسی سلامت واقعی اتصال با mt5.account_info().
- متدهای عمومی: initialize(), ensure_connection(), shutdown(), get_candles(), get_candles_range().
- انتخاب/اشتراک نمادها (symbol_select) بر مبنای کانفیگ (symbols) به‌صورت اختیاری.
- نگاشت جامع تایم‌فریم‌ها (M1,M5,M15,M30,H1,H4,D1,W1,MN1 + معادل‌های کوچک‌نویسی).
- گزارش سلامت (health_check) شامل: اتصال، مجازبودن معامله، ارز حساب، سرور، وضعیت نمادهای کلیدی.
- سازگاری با DataLoader و DataHandler موجود پروژه (امضاها و خصوصیت‌های مورد انتظار).
- Context manager برای استفادهٔ امن با with.
- لاگ‌گذاری کامل و خوانا؛ کدنویسی تمیز و آمادهٔ گسترش.

پیش‌نیاز: pip install MetaTrader5
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List   #, Union, Tuple
from dataclasses import dataclass  #, field
import time
#import math
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# --- لاگر ماژول ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- تلاش برای import MT5 ---
try:
    import MetaTrader5 as mt5  # type: ignore
    _HAS_MT5 = True
except Exception as ex:        # pragma: no cover
    mt5 = None                 # type: ignore
    _HAS_MT5 = False
    logger.error("MetaTrader5 package not available: %s", ex)

# --- لودر کانفیگ ---
try:
    from f10_utils.config_loader import load_config
except Exception:
    load_config = None  # type: ignore
    logger.warning("Could not import load_config from f10_utils.config_loader; please ensure it exists.")

# =====================================================================================
# ساختار تنظیمات Connector (اختیاری، برای خوانایی و توسعه‌پذیری)
# ===================================================================================== OK

@dataclass
class MT5Credentials:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None
    terminal_path: Optional[str] = None  # مسیر نصب ترمینال (متاتریدر) - اختیاری

@dataclass
class MT5ConnectorOptions:
    max_retries: int = 5 #60             # تعداد تلاش‌ها برای initialize/login
    retry_delay: float = 5.0             # فاصله بین تلاش‌ها (ثانیه)
    backoff_multiplier: float = 1.0      # ضریب backoff (می‌توان 1.2 گذاشت)
    auto_select_symbols: bool = True     # نمادهای کانفیگ را auto-select کند
    check_trade_allowed: bool = False    # در health_check بررسی trade_allowed را هم انجام دهد
    symbol_activation_timeout: float = 4  # مکث کوتاه پس از انتخاب نماد (ثانیه)
    ensure_on_each_call: bool = True     # قبل از هر fetch یک ensure_connection بزن

# =====================================================================================
# نگاشت تایم‌فریم‌ها
# ===================================================================================== OK

_TF_MAP = {
    # دقیقه
    "M1" : "M1" , "1" : "M1" , "1m" : "M1" , "m1" : "M1" ,
    "M2" : "M2" , "2" : "M2" , "2m" : "M2" , "m2" : "M2" ,
    "M3" : "M3" , "3" : "M3" , "3m" : "M3" , "m3" : "M3" ,
    "M5" : "M5" , "5" : "M5" , "5m" : "M5" , "m5" : "M5" ,
    "M10": "M10", "10": "M10", "10m": "M10", "m10": "M10",
    "M15": "M15", "15": "M15", "15m": "M15", "m15": "M15",
    "M30": "M30", "30": "M30", "30m": "M30", "m30": "M30",
    # ساعت
    "H1": "H1", "1h": "H1", "h1": "H1",
    "H2": "H2", "2h": "H2", "h2": "H2",
    "H3": "H3", "3h": "H3", "h3": "H3",
    "H4": "H4", "4h": "H4", "h4": "H4",
    # روز/هفته/ماه
    "D1": "D1", "1d": "D1", "d1": "D1",
    "W1": "W1", "1w": "W1", "w1": "W1",
    "MN1": "MN1", "1mn": "MN1", "mn1": "MN1",
}

def _to_mt5_timeframe(tf: str) -> int:
    """
    نگاشت رشتهٔ تایم‌فریم به ثابت متناظر در MetaTrader5.
    اگر نامعتبر باشد ValueError می‌دهد.
    """
    if not _HAS_MT5:
        raise RuntimeError("MetaTrader5 package not available")

    key = str(tf).strip().upper()
    if key not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe key: {tf}")

    # تبدیل کلید به ثابت mt5.TIMEFRAME_*
    name = _TF_MAP[key]
    return getattr(mt5, f"TIMEFRAME_{name}")

# =====================================================================================
# Connector
# =====================================================================================

class MT5Connector:
    """
    Connector عمومی برای اتصال به MetaTrader5.

    - با کانفیگ `f01_config/config.yaml` کار می‌کند (از طریق load_config()).
    - خواص مورد نیاز سایر بخش‌ها:
        * self.connected  (bool)
        * initialize(), ensure_connection(), shutdown()
        * get_candles(symbol, timeframe, num_candles)
        * get_candles_range(symbol, timeframe, date_from, date_to)
    """

    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 credentials: Optional[MT5Credentials] = None,
                 options: Optional[MT5ConnectorOptions] = None):
        """
        اگر config ندهید، از load_config() استفاده می‌شود.
        اگر credentials ندهید، از config['mt5_credentials'] خوانده می‌شود.
        """
        if not _HAS_MT5:
            raise RuntimeError("MetaTrader5 package not installed or failed to import.")

        self.cfg: Dict[str, Any] = config or (load_config() if callable(load_config) else {})
        self.creds: MT5Credentials = credentials or self._read_credentials_from_config(self.cfg)
        self.opts: MT5ConnectorOptions = options or self._read_options_from_config(self.cfg)

        self.connected: bool = False
        self._last_init_error: Optional[str] = None

    # ---------------------------
    # helpers برای ساخت تنظیمات
    # ---------------------------
    @staticmethod
    def _read_credentials_from_config(cfg: Dict[str, Any]) -> MT5Credentials:
        mt5c = (cfg.get("mt5_credentials") or {}) if isinstance(cfg, dict) else {}
        login = mt5c.get("login")
        # تلاش برای تبدیل login به int (در صورت رشته بودن)
        try:
            login = int(login) if login is not None else None
        except Exception:
            pass
        return MT5Credentials(
            login=login,
            password=mt5c.get("password"),
            server=mt5c.get("server"),
            terminal_path=mt5c.get("terminal_path") or mt5c.get("path") or None,
        )

    @staticmethod
    def _read_options_from_config(cfg: Dict[str, Any]) -> MT5ConnectorOptions:
        dd = (cfg.get("download_defaults") or {}) if isinstance(cfg, dict) else {}
        init_opts = dd.get("initialize_retry", {}) if isinstance(dd, dict) else {}
        return MT5ConnectorOptions(
            max_retries=int(init_opts.get("max_retries", 60)),
            retry_delay=float(init_opts.get("retry_delay", 5)),
            backoff_multiplier=float(init_opts.get("backoff_multiplier", 1.0)),
            auto_select_symbols=bool(dd.get("auto_select_symbols", True)),
            check_trade_allowed=bool(dd.get("check_trade_allowed", False)),
            symbol_activation_timeout=float(dd.get("symbol_activation_timeout", 0.5)),
            ensure_on_each_call=True,
        )

    # ---------------------------
    # اتصال/لاگین
    # ---------------------------
    def initialize(self,
                   max_retries: Optional[int] = None,
                   retry_delay: Optional[float] = None,
                   backoff_multiplier: Optional[float] = None) -> bool:
        """
        اتصال به ترمینال MetaTrader5 و ورود به حساب کاربری.
        پارامترها اگر None باشند از self.opts خوانده می‌شوند.
        """
        if not _HAS_MT5:
            raise RuntimeError("MetaTrader5 package not available")

        max_retries = int(max_retries or self.opts.max_retries)
        retry_delay = float(retry_delay or self.opts.retry_delay)
        backoff_multiplier = float(backoff_multiplier or self.opts.backoff_multiplier or 1.0)

        self._last_init_error = None

        # مسیر ترمینال (اختیاری) — برای مثال نصب Alpari/MetaTrader 5
        init_kwargs: Dict[str, Any] = {}
        if self.creds.terminal_path:
            init_kwargs["path"] = str(self.creds.terminal_path)

        # حلقهٔ تلاش‌ها
        delay = retry_delay
        for attempt in range(1, max_retries + 1):
            try:
                # initialize
                ok_init = bool(mt5.initialize(**init_kwargs)) if init_kwargs else bool(mt5.initialize())
                if not ok_init:
                    self._last_init_error = "initialize_failed"
                    logger.error("MT5 initialize failed (attempt %d/%d)", attempt, max_retries)
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue

                # login
                if self.creds.login is None or not self.creds.password or not self.creds.server:
                    self._last_init_error = "missing_credentials"
                    logger.critical("MT5 credentials incomplete: login/password/server required.")
                    mt5.shutdown()
                    return False

                authorized = bool(mt5.login(login=int(self.creds.login),
                                            password=str(self.creds.password),
                                            server=str(self.creds.server)))
                if not authorized:
                    self._last_init_error = "login_failed"
                    logger.error("MT5 login failed (attempt %d/%d)", attempt, max_retries)
                    mt5.shutdown()
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue

                # sanity check: account_info
                acct = mt5.account_info()
                if acct is None:
                    self._last_init_error = "no_account_info"
                    logger.error("MT5 login returned True but account_info() is None.")
                    mt5.shutdown()
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue

                self.connected = True
                logger.info("MT5 connected: login=%s server=%s name=%s",
                            getattr(acct, "login", None), getattr(acct, "server", None), getattr(acct, "name", None))

                # auto-select symbols (اختیاری)
                if self.opts.auto_select_symbols:
                    self._auto_select_symbols()

                return True

            except Exception as ex:
                self._last_init_error = f"exception:{ex}"
                logger.exception("Exception during MT5 initialize/login (attempt %d/%d): %s", attempt, max_retries, ex)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(delay)
                delay *= backoff_multiplier

        logger.critical("All attempts to connect/login to MT5 failed. last_error=%s", self._last_init_error)
        self.connected = False
        return False

    # ---------------------------
    # اطمینان از اتصال
    # ---------------------------
    def ensure_connection(self) -> bool:
        """
        بررسی زنده‌بودن اتصال با mt5.account_info() و در صورت نیاز تلاش به reconnect.
        """
        if not _HAS_MT5:
            return False
        try:
            acct = mt5.account_info()
            if acct is not None:
                self.connected = True
                return True
        except Exception:
            pass

        logger.warning("MT5 not connected — attempting re-initialize()")
        ok = False
        try:
            ok = self.initialize()
        except Exception as ex:
            logger.exception("initialize() raised during ensure_connection: %s", ex)
            ok = False

        self.connected = bool(ok)
        if ok:
            logger.info("MT5 reconnected successfully.")
        else:
            logger.error("MT5: unable to re-establish connection.")
        return ok

    # ---------------------------
    # خاموش‌کردن اتصال
    # ---------------------------
    def shutdown(self) -> None:
        """قطع اتصال از MT5 و بروزرسانی وضعیت داخلی."""
        if not _HAS_MT5:
            return
        try:
            mt5.shutdown()
        except Exception as ex:
            logger.exception("Error while shutting down MT5: %s", ex)
        finally:
            self.connected = False
            logger.info("MT5 shutdown complete.")

    # ---------------------------
    # انتخاب/اشتراک نمادها
    # ---------------------------
    def _auto_select_symbols_old1(self) -> None:
        """
        نمادهایی که در کانفیگ آمده‌اند را در ترمینال انتخاب (subscribe) می‌کند تا copy_rates_* کار کند.
        """
        symbols: List[str] = []
        try:
            # تلاش به خواندن از چند جای رایج در کانفیگ
            data = self.cfg.get("data_fetch_defaults", {}) or {}
            symbols = list(self.cfg.get("symbols", []) or data.get("symbols", []) or [])
        except Exception:
            symbols = []

        if not symbols:
            return

        for sym in symbols:
            try:
                if mt5.symbol_select(sym, True):
                    logger.debug("Symbol selected: %s", sym)
                    time.sleep(self.opts.symbol_activation_timeout)
                else:
                    logger.warning("Failed to select symbol: %s", sym)
            except Exception:
                logger.exception("Exception while selecting symbol: %s", sym)

    def _auto_select_symbols(self) -> None:
        """
        نمادهایی که در کانفیگ آمده‌اند را در ترمینال انتخاب (subscribe) می‌کند تا copy_rates_* کار کند.
        """
        # توضیح فارسی: ابتدا از download_defaults.symbols بخوان؛ اگر نبود از مسیرهای قدیمی‌تر.
        symbols: List[str] = []
        try:
            dd = (self.cfg.get("download_defaults") or {})
            symbols = list(dd.get("symbols") or [])
            if not symbols:
                data = self.cfg.get("data_fetch_defaults", {}) or {}
                symbols = list(self.cfg.get("symbols", []) or data.get("symbols", []) or [])
        except Exception:
            symbols = []

        if not symbols:
            return

        for sym in symbols:
            try:
                ok = mt5.symbol_select(sym, True)
                if ok:
                    logger.debug("Symbol selected: %s", sym)  # English log
                    time.sleep(self.opts.symbol_activation_timeout)
                else:
                    logger.warning("Failed to select symbol: %s", sym)
            except Exception:
                logger.exception("Exception while selecting symbol: %s", sym)

    # ---------------------------
    # دریافت کندل‌ها (آخرین/رنج)
    # ---------------------------
    def get_candles(self, symbol: str, timeframe: str, num_candles: int = 1000) -> "pd.DataFrame":
        """
        دریافت آخرین num_candles کندل برای نماد/تایم‌فریم.
        خروجی: DataFrame با index=datetime و ستون‌های: open, high, low, close, tick_volume, spread
        """

        if self.opts.ensure_on_each_call:
            if not self.ensure_connection():
                raise ConnectionError("Cannot connect to MT5")

        tf = _to_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(num_candles))
        if rates is None or len(rates) == 0:
            logger.warning("No rates for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def get_candles_range(self, symbol: str, timeframe: str,
                          date_from: datetime, date_to: datetime) -> "pd.DataFrame":
        """
        دریافت کندل‌ها بین دو تاریخ [inclusive].
        date_from / date_to باید datetime (ترجیحاً timezone-aware در UTC) باشند.
        """

        if self.opts.ensure_on_each_call:
            if not self.ensure_connection():
                raise ConnectionError("Cannot connect to MT5")

        tf = _to_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            logger.warning("No rates (range) for %s %s", symbol, timeframe)
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ---------------------------
    # Health & Diagnostics
    # ---------------------------
    def health_check(self, sample_symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        گزارش سریع سلامت اتصال و حساب.
        sample_symbol: اگر داده شود، یک symbol_info_tick می‌گیرد تا تاخیر/دسترسی سنجیده شود.
        """
        info: Dict[str, Any] = {
            "connected": False,
            "login": None,
            "server": None,
            "name": None,
            "currency": None,
            "trade_allowed": None,
            "sample_symbol_ok": None,
            "last_init_error": self._last_init_error,
        }

        if not _HAS_MT5:
            info["connected"] = False
            return info

        try:
            acct = mt5.account_info()
            info["connected"] = acct is not None
            if acct is not None:
                info["login"] = getattr(acct, "login", None)
                info["server"] = getattr(acct, "server", None)
                info["name"] = getattr(acct, "name", None)
                info["currency"] = getattr(acct, "currency", None)
                if self.opts.check_trade_allowed:
                    info["trade_allowed"] = bool(getattr(acct, "trade_allowed", False))
        except Exception as ex:
            info["connected"] = False
            info["last_error"] = f"account_info_exception:{ex}"
            return info

        if sample_symbol:
            try:
                t0 = time.perf_counter()
                tick = mt5.symbol_info_tick(sample_symbol)
                dt = time.perf_counter() - t0
                info["sample_symbol_ok"] = tick is not None
                info["latency_ms"] = round(dt * 1000.0, 2)
            except Exception as ex:
                info["sample_symbol_ok"] = False
                info["last_error"] = f"tick_exception:{ex}"

        return info

    # ---------------------------
    # Context Manager
    # ---------------------------
    def __enter__(self) -> "MT5Connector":
        ok = self.initialize()
        if not ok:
            raise RuntimeError(f"MT5Connector failed to initialize: {self._last_init_error}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


# =====================================================================================
# نمونهٔ اجرا (اختیاری)
# =====================================================================================
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    conn = MT5Connector()
    if conn.initialize():
        hc = conn.health_check(sample_symbol="XAUUSD")
        logger.info("Health: %s", hc)
        df = conn.get_candles("XAUUSD", "M5", 500)
        logger.info("Fetched bars: %s", len(df))
        conn.shutdown()
