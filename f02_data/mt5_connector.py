# -*- coding: utf-8 -*-
# f02_data/mt5_connector.py
# Status in (Bot-RL-2): Completed at 1404-09-05

"""MT5Connector (نسخهٔ حرفه‌ای برای Bot-RL-2) 
=====================================================================================
قابلیت‌ها:
- اتصال ایمن به MetaTrader5 با پشتیبانی از مسیر ترمینال (terminal path) و retry با backoff سبک.
- ورود (login) با کرِدها از کانفیگ (یا ENV)، و بررسی سلامت واقعی اتصال با mt5.account_info().
- متدهای عمومی: initialize(), ensure_connection(), shutdown(), get_candles_num(), get_candles_range().
- انتخاب/اشتراک نمادها (symbol_select) بر مبنای کانفیگ (symbols) به‌صورت اختیاری.
- نگاشت جامع تایم‌فریم‌ها (M1,M5,M15,M30,H1,H4,D1,W1,MN1 + معادل‌های کوچک‌نویسی).
- گزارش سلامت (health_check) شامل: اتصال، مجازبودن معامله، ارز حساب، سرور، وضعیت نمادهای کلیدی.
- سازگاری با DataLoader و DataHandler موجود پروژه (امضاها و خصوصیت‌های مورد انتظار).
- Context manager برای استفادهٔ امن با with.

پیش‌نیاز: pip install MetaTrader5
"""

# =====================================================================================
# Imports & Logger
# =====================================================================================
from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time
from datetime import datetime, timezone
import pandas as pd
import logging

# -------------------- Logger for this module -----------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -------------------- Trying to import MT5 -------------------------
try:
    import MetaTrader5 as mt5  # type: ignore
    _HAS_MT5 = True
except Exception as ex:        # pragma: no cover
    mt5 = None                 # type: ignore
    _HAS_MT5 = False
    logger.error("MetaTrader5 package not available: %s", ex)

# -------------------- Trying to import config_loader ---------------
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
    max_retries: int = 10                 #60  # تعداد تلاش‌ها برای initialize/login 
    retry_delay: float = 5.0              # فاصله بین تلاش‌ها (ثانیه) 
    backoff_multiplier: float = 1.0       # ضریب backoff (می‌توان 1.2 گذاشت) 
    auto_select_symbols: bool = True      # نمادهای کانفیگ را auto-select کند 
    check_trade_allowed: bool = False     # در health_check بررسی trade_allowed را هم انجام دهد 
    symbol_activation_timeout: float = 4  # مکث کوتاه پس از انتخاب نماد (ثانیه) 
    ensure_on_each_call: bool = True      # قبل از هر fetch یک ensure_connection بزن 

# =====================================================================================
# نگاشت تایم‌فریم‌ها 
# ===================================================================================== OK

_TF_MAP = {
    # دقیقه
    "M1" : "M1" , "1M" : "M1" , "1" : "M1" , "1m" : "M1" , "m1" : "M1" ,
    "M2" : "M2" , "2M" : "M2" , "2" : "M2" , "2m" : "M2" , "m2" : "M2" ,
    "M3" : "M3" , "3M" : "M3" , "3" : "M3" , "3m" : "M3" , "m3" : "M3" ,
    "M5" : "M5" , "5M" : "M5" , "5" : "M5" , "5m" : "M5" , "m5" : "M5" ,
    "M10": "M10", "10M": "M10", "10": "M10", "10m": "M10", "m10": "M10",
    "M15": "M15", "15M": "M15", "15": "M15", "15m": "M15", "m15": "M15",
    "M30": "M30", "30M": "M30", "30": "M30", "30m": "M30", "m30": "M30",
    # ساعت
    "H1" : "H1" , "1H"  : "H1"  , "1h" : "H1" , "h1" : "H1" ,
    "H2" : "H2" , "2H"  : "H2"  , "2h" : "H2" , "h2" : "H2" ,
    "H3" : "H3" , "3H"  : "H3"  , "3h" : "H3" , "h3" : "H3" ,
    "H4" : "H4" , "4H"  : "H4"  , "4h" : "H4" , "h4" : "H4" ,
    "H12": "H12", "12H" : "H12" , "12h": "H12", "h12": "H12",
    # روز/هفته/ماه
    "D1": "D1", "1D" : "D1" , "1d" : "D1", "d1" : "D1", 
    "W1": "W1", "1W" : "W1" , "1w" : "W1", "w1" : "W1",
    "MN1": "MN1", "1MN": "MN1", "1mn": "MN1", "mn1": "MN1",
}

def _to_mt5_timeframe(tf: str) -> int:
    """ نگاشت رشتهٔ تایم‌فریم به ثابت متناظر در MetaTrader5 
    اگر نامعتبر باشد ValueError می‌دهد
    """
    if not _HAS_MT5:
        raise RuntimeError("MetaTrader5 package not available")

    key = str(tf).replace(" ","").upper()  # key = str(tf).strip().upper()
    if key not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe key: {tf}")

    # تبدیل کلید به ثابت mt5.TIMEFRAME_*
    name = _TF_MAP[key]
    return getattr(mt5, f"TIMEFRAME_{name}")

# =====================================================================================
# Connector
# =====================================================================================

class MT5Connector:
    """Connector عمومی برای اتصال به MetaTrader5 
    - با کانفیگ `f01_config/config.yaml` کار می‌کند (از طریق load_config())
    - خواص مورد نیاز سایر بخش‌ها:
        * self.connected  (bool)
        * initialize(), ensure_connection(), shutdown()
        * get_candles_num(symbol, timeframe, num_candles_from, num_candles_to)
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
        
    # ---------------------------------------------------------------
    # helpers برای ساخت تنظیمات  
    # --------------------------------------------------------------- OK
    @staticmethod
    def _read_credentials_from_config(cfg: Dict[str, Any]) -> MT5Credentials:
        mt5c = (((cfg.get("connection") or {}).get("mt5_credentials")) or {}) if isinstance(cfg, dict) else {}
        login = mt5c.get("login")
        # تلاش برای تبدیل login به int (در صورت رشته بودن)
        try:
            login = int(login) if login is not None else None
        except Exception:
            pass
        return MT5Credentials(
            login = login,
            password = mt5c.get("password"),
            server = mt5c.get("server"),
            terminal_path = mt5c.get("terminal_path") or mt5c.get("path") or None,
        )

    @staticmethod
    def _read_options_from_config(cfg: Dict[str, Any]) -> MT5ConnectorOptions:
        init_opts = (((cfg.get("connection") or {}).get("initialize_retry")) or {}) if isinstance(cfg, dict) else {}
        return MT5ConnectorOptions(
            max_retries = int(init_opts.get("max_retries", 10)),
            retry_delay = float(init_opts.get("retry_delay", 5)),
            backoff_multiplier = float(init_opts.get("backoff_multiplier", 1.0)),
            auto_select_symbols = bool(init_opts.get("auto_select_symbols", True)),
            check_trade_allowed = bool(init_opts.get("check_trade_allowed", False)),
            symbol_activation_timeout = float(init_opts.get("symbol_activation_timeout", 4.0)),
            ensure_on_each_call = bool(init_opts.get("ensure_on_each_call", True)),
        )

    # ---------------------------------------------------------------
    # اتصال/لاگین 
    # --------------------------------------------------------------- OK
    def initialize(self,
                   max_retries: Optional[int] = None,
                   retry_delay: Optional[float] = None,
                   backoff_multiplier: Optional[float] = None) -> bool:
        """اتصال به ترمینال MetaTrader5 و ورود به حساب کاربری. 
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

        #--- شروع حلقهٔ تلاش‌ها --- 
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
                    continue  #اگر اتصال برقرار نشد، شمارنده بعدی حلقه اجرا می‌شود 

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
                    continue  #اگر لاگین انجام نشد، شمارنده بعدی حلقه اجرا می‌شود 

                # sanity check: account_info
                acct = mt5.account_info()
                
                if acct is None:
                    self._last_init_error = mt5.last_error()  # "no_account_info"
                    logger.error("MT5 login returned True but account_info() is None.")
                    mt5.shutdown()
                    time.sleep(delay)
                    delay *= backoff_multiplier
                    continue  #اگر اطلاعات اکانت وجود ندارد، شمارنده بعدی حلقه اجرا می‌شود 

                #در صورتی به این نقطه میرسیم که اتصال و لاگین موفق بوده باشد و account_info موجود باشد 
                self.connected = True
                logger.info("MT5 connected: login=%s server=%s name=%s",
                            getattr(acct, "login", None), getattr(acct, "server", None), getattr(acct, "name", None))
                
                # auto-select symbols (اختیاری)
                if self.opts.auto_select_symbols:
                    self._auto_select_symbols()   # نمادها را در market watch فعال می‌کند 

                return True  # موفقیت‌آمیز بودن اتصال و لاگین 

            except Exception as ex:
                self._last_init_error = f"exception:{ex}"
                logger.exception("Exception during MT5 initialize/login (attempt %d/%d): %s", attempt, max_retries, ex)
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(delay)
                delay *= backoff_multiplier
            # --- پایان حلقه تلاش‌ها ---
        
        # اگر به اینجا رسیدیم، همه تلاش‌ها ناموفق بوده‌اند
        logger.critical("All attempts to connect/login to MT5 failed. last_error=%s", self._last_init_error)
        self.connected = False
        
        return False
    
    # ---------------------------------------------------------------
    # اطمینان از اتصال 
    # --------------------------------------------------------------- OK
    def ensure_connection(self) -> bool:
        """
        بررسی زنده‌بودن اتصال با mt5.account_info() و در صورت نیاز تلاش به reconnect
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
            logger.exception("re-initialize() failed with exception during ensure_connection: %s", ex)
            ok = False

        self.connected = bool(ok)
        if ok:
            logger.info("MT5 reconnected successfully.")
        else:
            logger.error("MT5: unable to re-establish connection.")
        return ok

    # ---------------------------------------------------------------
    # خاموش‌کردن اتصال 
    # --------------------------------------------------------------- OK
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

    # ---------------------------------------------------------------
    # انتخاب/اشتراک نمادها 
    # --------------------------------------------------------------- OK
    def _auto_select_symbols(self) -> None:
        """
        نمادهایی که در کانفیگ آمده‌اند را در ترمینال انتخاب (subscribe) می‌کند تا copy_rates_* کار کند.
        ابتدا از env.download_defaults.symbols بخوان؛
        اگر نبود از مسیرهای قدیمی‌تر.
        در واقع نمادها را مطابق با کانفیگ در پنجره Market Watch فعال می‌کند.
        """
        symbols: List[str] = []
        try:
            dd = (((self.cfg.get("env") or {}).get("download_defaults")) or {})
            symbols = list(dd.get("symbols") or [])
            # سه سطر زیر در تاریخ 1404-09-04 حذف شدند 
            #if not symbols:
            #    data = self.cfg.get("data_fetch_defaults", {}) or {}
            #    symbols = list(self.cfg.get("symbols", []) or data.get("symbols", []) or [])
        except Exception:
            symbols = []

        if not symbols:
            logger.info("No symbols to auto-select from config.")
            return

        for sym in symbols:
            try:
                ok = mt5.symbol_select(sym, True)    # سمبل را در پنجره Market Watch فعال می‌کند 
                if ok:
                    logger.debug("Symbol selected: %s", sym)
                    time.sleep(self.opts.symbol_activation_timeout)  # مکث کوتاه پس از انتخاب نماد (ثانیه) 
                else:
                    logger.warning("Failed to select symbol: %s", sym)
            except Exception:
                logger.exception("Exception while selecting symbol: %s", sym)

    # ---------------------------------------------------------------
    # دریافت کندل‌ها (آخرین/رنج) 
    # --------------------------------------------------------------- OK- 2 func.s
    def get_candles_num(self, symbol: str, timeframe: str, num_candles: int = 1000) -> "pd.DataFrame":
        """
        دریافت آخرین کندل‌ها برای یک نماد و تایم‌فریم مشخص.

        پارامترها:
        - symbol: نام نماد (مثلاً "EURUSD")
        - timeframe: تایم‌فریم به صورت string (مثلاً "M1", "H1")
        - num_candles: تعداد آخرین کندل‌هایی که باید دریافت شوند (پیش‌فرض 1000)

        خروجی:
        - DataFrame با index از نوع datetime (UTC)
        - ستون‌ها:
            - open, high, low, close
            - volume: ستون حجم، که می‌تواند از real_volume یا tick_volume استخراج شود
            - spread
        - اگر داده‌ای دریافت نشود، DataFrame خالی برمی‌گردد

        توضیحات:
        - ستون volume همیشه با نام یکنواخت 'volume' بازگردانده می‌شود
        - داده‌ها بر اساس زمان مرتب شده‌اند
        - اگر opts.ensure_on_each_call فعال باشد، قبل از دریافت داده، اتصال به MT5 بررسی می‌شود
        """
        if self.opts.ensure_on_each_call:  # اگر true باشد، یعنی باید قبل از هر fetch یکبار ensure_connection را اجرا کنیم 
            if not self.ensure_connection():
                raise ConnectionError("Cannot connect to MT5")

        tf = _to_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(num_candles))
        if rates is None or len(rates) == 0:
            logger.warning("No rates for %s %s", symbol, timeframe)
            return pd.DataFrame()  # یک DataFrame خالی برمی‌گرداند 

        df = pd.DataFrame(rates)
        # ---Determining volume_col ----------------------- Start Add 040924
        volume_col = next(
            (c for c in ("tick_volume", "real_volume")
            if c in df.columns and df[c].notna().any() and (df[c] != 0).any()),
            None
            )
        # --- Constructing columns names ------------------
        cols = ["time", "open", "high", "low", "close"]
        if volume_col is not None:
            cols.append(volume_col)
        cols.append("spread")
        df = df[cols]
        # --- Renaming volume column ----------------------
        if volume_col is not None and volume_col != "volume":
            df.rename(columns={volume_col: "volume"}, inplace=True)
        # ------------------------------------------------- end  Add 040924

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    def get_candles_range(self, symbol: str, timeframe: str,
                          date_from: datetime, date_to: datetime) -> "pd.DataFrame":
        """
        دریافت کندل‌ها برای یک نماد و تایم‌فریم مشخص بین دو تاریخ مشخص [inclusive].

        پارامترها:
        - symbol: نام نماد (مثلاً "EURUSD")
        - timeframe: تایم‌فریم به صورت string (مثلاً "M1", "H1")
        - date_from: تاریخ شروع (datetime، ترجیحاً timezone-aware در UTC)
        - date_to: تاریخ پایان (datetime، ترجیحاً timezone-aware در UTC)

        خروجی:
        - DataFrame با index از نوع datetime (UTC)
        - ستون‌ها:
            - open, high, low, close
            - volume: ستون حجم، که می‌تواند از real_volume یا tick_volume استخراج شود
            - spread
        - اگر داده‌ای دریافت نشود، DataFrame خالی برمی‌گردد

        توضیحات:
        - ستون volume همیشه با نام یکنواخت 'volume' بازگردانده می‌شود
        - داده‌ها بر اساس زمان مرتب شده‌اند
        - اگر opts.ensure_on_each_call فعال باشد، قبل از دریافت داده، اتصال به MT5 بررسی می‌شود
        """
        if self.opts.ensure_on_each_call:  # اگر true باشد، یعنی باید قبل از هر fetch یکبار ensure_connection را اجرا کنیم 
            if not self.ensure_connection():
                raise ConnectionError("Cannot connect to MT5")

        tf = _to_mt5_timeframe(timeframe)
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            logger.warning("No rates (range) for %s %s", symbol, timeframe)
            return pd.DataFrame()  # یک DataFrame خالی برمی‌گرداند 

        df = pd.DataFrame(rates)
        # ---Determining volume_col ----------------------- Start Add 040924
        volume_col = next(
            (c for c in ("tick_volume", "real_volume")
            if c in df.columns and df[c].notna().any() and (df[c] != 0).any()),
            None
            )
        # --- Constructing columns names ------------------
        cols = ["time", "open", "high", "low", "close"]
        if volume_col is not None:
            cols.append(volume_col)
        cols.append("spread")
        df = df[cols]
        # --- Renaming volume column ----------------------
        if volume_col is not None and volume_col != "volume":
            df.rename(columns={volume_col: "volume"}, inplace=True)
        # ------------------------------------------------- end  Add 040924

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ---------------------------------------------------------------
    # Health & Diagnostics
    # --------------------------------------------------------------- OK
    def health_check(self, sample_symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        گزارش سریع سلامت اتصال و حساب.
        sample_symbol: اگر داده شود، یک symbol_info_tick می‌گیرد تا تاخیر/دسترسی سنجیده شود.
        """
        info: Dict[str, Any] = {
            "connected": False,   # 11/16 (find in this madule)
            "login": None,        # 25/35
            "server": None,       # 10/20
            "name": None,        #  6/13
            "currency": None,           #  1/8
            "trade_allowed": None,            #  5/9
            "sample_symbol_ok": None,               # 1/3
            "last_init_error": self._last_init_error,  # 9/11
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
            info["last_init_error"] = f"account_info_exception:{ex}"
            return info
        """
        بلوک زیر یک تست سریعِ دسترسی و تاخیر برای sample_symbol انجام می‌دهد
        و نتیجهٔ موفقیت/عدم موفقیت و مقدار لاتنسی را در info می‌گذارد
        تا health_check بتواند گزارش دهد.
        """
        if sample_symbol:
            try:
                t0 = time.perf_counter()
                tick = mt5.symbol_info_tick(sample_symbol)  # گرفتن آخرین تیک برای نماد نمونه از متاتریدر 
                dt = time.perf_counter() - t0
                info["sample_symbol_ok"] = tick is not None  # مقداری بولی دارد که نشان می‌دهد آیا تیک با موفقیت گرفته شده یا نه
                info["latency_ms"] = round(dt * 1000.0, 2)
            except Exception as ex:
                info["sample_symbol_ok"] = False
                info["last_init_error"] = f"tick_exception:{ex}"

        return info

    # ---------------------------------------------------------------
    # Context Manager
    # --------------------------------------------------------------- OK
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

    # ---------------------------------------------------------------
    # Symbol specs snapshot (for overlay)  — aligns with project style
    # --------------------------------------------------------------- OK
    def get_symbol_specs(self, symbols: list[str]) -> dict:
        """دریافت مشخصات نمادها از MT5 و بازگرداندن ساختاری مناسب برای ذخیره در overlay 
        خروجی:
            {
                "meta": {"as_of": "...Z", "account_currency": "...", "login": 12345, "server": "..."},
                "symbol_specs": {
                    ...,
                    "XAUUSD": {
                        "digits": 2,
                        "point": 0.01,
                        "trade_tick_value": 1.0,
                        "trade_tick_size": 0.01,
                        "contract_size": 100.0,
                        "volume_min": 0.01,
                        "volume_step": 0.01,
                        "volume_max": 100.0,
                        "stops_level": 50,
                        "pip_value_per_lot": 0.01,    
                        "raw": { ... }               
                    },
                    ...,
              }
            }
        """
        
        try:
            #  بررسی زنده‌بودن اتصال با mt5.account_info() و در صورت نیاز تلاش به reconnect 
            self.ensure_connection()
        except Exception as ex:  # pragma: no cover
            logger.error("Could not ensure MT5 connection: %s", ex)
            return {"meta": {"connected": False}, "symbol_specs": {}}

        # تلاش برای گرفتن اطلاعات حساب (ممکن است None باشد) 
        acct = None
        try:
            acct = mt5.account_info() if _HAS_MT5 else None
        except Exception:
            acct = None

        meta = {
            # as_of = زمان معتبر بودن داده در لحظهٔ گرفته شدن snapshot است 
            "as_of": datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
            "connected": bool(acct is not None),
            "account_currency": getattr(acct, "currency", None) if acct else None,
            "login": getattr(acct, "login", None) if acct else None,
            "server": getattr(acct, "server", None) if acct else None,
        }

        specs: dict[str, dict] = {}
        if not symbols:
            logger.warning("No symbols provided to get_symbol_specs().")
            return {"meta": meta, "symbol_specs": specs}
        
        for sym in symbols:
            # --------------------- گرفتن اطلاعات سمبل ---------------------
            si = None
            try:
                si = mt5.symbol_info(sym) if _HAS_MT5 else None
                if si is None and _HAS_MT5:
                    mt5.symbol_select(sym, True)   # سمبل را در پنجره Market Watch فعال می‌کند 
                    si = mt5.symbol_info(sym)      # تلاش مجدد برای گرفتن info 
            except Exception as ex:
                logger.exception("Error fetching symbol_info for %s: %s", sym, ex)
                si = None

            if si is None:
                logger.warning("Symbol not available or info missing: %s", sym)
                continue
            # --------------------- ساخت آیتم مشخصات ---------------------
            try:
                item = {
                    "digits": getattr(si, "digits", None),
                    "point": getattr(si, "point", None),
                    "trade_tick_value": getattr(si, "trade_tick_value", None),
                    "trade_tick_size": getattr(si, "trade_tick_size", None),
                    "contract_size": getattr(si, "trade_contract_size", None) or getattr(si, "contract_size", None),
                    "volume_min": getattr(si, "volume_min", None),
                    "volume_step": getattr(si, "volume_step", None),
                    "volume_max": getattr(si, "volume_max", None),
                    "stops_level": getattr(si, "stops_level", None),
                }

                # =========================
                # --- ذخیرهٔ نسخهٔ خام بازگشتی از MT5 برای دیباگ/آرشیو ---
                # اگر object دارای _asdict باشد (مثلاً namedtuple) از آن استفاده می‌کنیم، در غیر اینصورت
                # تلاش می‌کنیم فیلدهای عمومی را استخراج کنیم.
                raw_spec = None
                try:
                    if hasattr(si, "_asdict"):
                        raw_spec = dict(si._asdict())
                    else:
                        # محافظه‌کارانه: فقط ویژگی‌های عمومی و غیر متدها را بگیر
                        raw_spec = {k: getattr(si, k, None) for k in dir(si) if not k.startswith("_") and not callable(getattr(si, k, None))}
                except Exception:
                    raw_spec = None
                if raw_spec is not None:
                    item["raw"] = raw_spec

                # --- محاسبهٔ ایمن pip_value_per_lot ---
                # تعاریف: pip := 10 * point  (مطابق منطق موجود در snapshot script قبلی)
                try:
                    p = float(item.get("point") or 0)
                    tv = float(item.get("trade_tick_value") or 0)
                    ts = float(item.get("trade_tick_size") or 1)
                    # بررسی اینکه مقادیر معنادار و غیر صفر باشند
                    if p > 0 and tv > 0 and ts > 0:
                        item["pip_value_per_lot"] = round((tv / ts) * (10.0 * p), 6)   # pip := 10*point
                except Exception as ex:
                    # اگر خطایی در تبدیل/محاسبه بود، لاگ کن ولی اجرای تابع را ادامه بده
                    logger.debug("Could not compute pip_value_per_lot for %s: %s", sym, ex)
                # =========================
            
            except Exception as ex:
                logger.exception("Failed to build specs for %s: %s", sym, ex)
                continue # اگر مشکلی در ساخت آیتم بود، به سمبل بعدی می‌رود 

            specs[sym] = item

        return {"meta": meta, "symbol_specs": specs}


# =====================================================================================
# نمونهٔ اجرا (اختیاری) 
# =====================================================================================
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    conn = MT5Connector()
    if conn.initialize():
        hc = conn.health_check(sample_symbol="XAUUSD")
        logger.info("Health: %s", hc)
        df = conn.get_candles_num("XAUUSD", "M5", 500)
        logger.info("Fetched bars: %s", len(df))
        conn.shutdown()

# =====================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =====================================================================================
""" Func Names                                                Used in Functions: ...
                                   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
1  MT5Credentials                 --  --  --  --  ok  ok  --  --  --  --  --  --  --  --  --  --  --  --
2  MT5ConnectorOptions            --  --  --  --  ok  --  ok  --  --  --  --  --  --  --  --  --  --  --
3  _to_mt5_timeframe              --  --  --  --  --  --  --  --  --  --  --  ok  ok  --  --  --  --  --
4  MT5Connector                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
5  __init__                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
6  _read_credentials_from_config  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
7  _read_options_from_config      --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --
8  initialize                     --  --  --  --  --  --  --  --  ok  --  --  --  --  ok  --  --  --  ok 
9  ensure_connection              --  --  --  --  --  --  --  --  --  --  --  ok  ok  --  --  --  ok  -- 
10 shutdown                       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  ok 
11 _auto_select_symbols           --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
12 get_candles_nmu                --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
13 get_candles_range  (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
14 health_check                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
15 __enter__          (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
16 __exit__           (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
17 get_symbol_specs   (NOT USED)  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
18 (Global code)                  -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
"""

# =====================================================================================
# نمونه‌های استفاده از توابع MT5 (برای توسعه‌دهندگان) 
# =====================================================================================
"""
mt5.initialize(path, login=LOGIN, password="PASSWORD", server="SERVER", timeout=TIMEOUT, portable=False)
mt5.shutdown()
mt5.login(login, password="PASSWORD", server="SERVER", timeout=TIMEOUT)
mt5.account_info()
mt5.symbol_select(symbol, enable=None)
mt5.copy_rates_from_pos(symbol, timeframe, start_pos, count)
mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
mt5.symbol_info_tick(symbol)
mt5.symbol_info(symbol)
"""

# =====================================================================================
# نمونهٔ خروجی mt5.account_info() (برای توسعه‌دهندگان) 
# =====================================================================================
""" 
mt5.account_info() sample output:
    {
    'login': 52623142,
    'trade_mode': 0,
    'leverage': 100,
    'limit_orders': 10000,
    'margin_so_mode': 0,
    'trade_allowed': True,
    'trade_expert': True,
    'margin_mode': 2,
    'currency_digits': 2,
    'fifo_close': False,
    'balance': 1200.0,
    'credit': 0.0,
    'profit': 0.0,
    'equity': 1200.0,
    'margin': 0.0,
    'margin_free': 1200.0,
    'margin_level': 0.0,
    'margin_so_call': 80.0,
    'margin_so_so': 50.0,
    'margin_initial': 0.0,
    'margin_maintenance': 0.0,
    'assets': 0.0,
    'liabilities': 0.0,
    'commission_blocked': 0.0,
    'name': 'Farhad Vahedkia',
    'server': 'Alpari-MT5-Demo',
    'currency': 'USD',
    'company': 'Alpari'
    }
"""