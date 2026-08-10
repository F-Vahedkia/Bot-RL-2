# f02_data/data_handler_F_3.py
# Last reviewed at 1405-04-12

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
python -m f02_data.data_handler_E `
    -c .\f01_config\config.yaml `
    --symbol XAUUSD_i           `
    --base-tf M1                `
    --timeframes M1 M5 M30 H1  H4 D1    `
    --format csv

- فرمان اجرا جدید برنامه 
-  بدون base_tf ، یعنی براساس آنچه در کانفیگ داده شده:
python -m f02_data.data_handler_E  `
    -c .\f01_config\config_0_1.yaml  `
    --symbol XAUUSD_i            `
    --timeframes M10 M30 H1      `
    --format parquet

- فرمان اجرای جدید برنامه
-  همراه با base_tf
python -m f02_data.data_handler_E    `
    -c .\f01_config\config.yaml    `
    --symbol XAUUSD_i              `
    --base-tf M1                   `
    --timeframes H1 H4 D1 W1      `
    --format csv

- فرمان اجرای جدید برنامه
-  همراه با base_tf
python -m f02_data.data_handler_E    `
    -c .\f01_config\config.yaml    `
    --symbol XAUUSD_i              `
    --base-tf H4                   `
    --timeframes H4 D1             `
    --format csv
"""

# f02_data/data_handler_F_3.py
# =======================================================================================
# Imports & Logger
# ======================================================================================= OK=
# f02_data/data_handler_F_2_3.py
from __future__ import annotations
# from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from pathlib import Path
from datetime import datetime, time, timezone
import logging
# import numpy as np
import pandas as pd
# import pyarrow.parquet as pq

# ------------------ Importing Internal Modules -----------------------------------------
from f02_data.data_handler_Helpers import check_tfs    #, prefix_columns

from f02_data.mt5_data_loader_E import normalize_df
from f02_data.market_data_engine.event_bus_2 import EventBus          # ➕ Version-D
from f10_utils.config_path_funcs import project_root, resolve_raw_dir, resolve_process_dir, full_file_path
from f10_utils.constants import _TF_MINUTES, _TF_MAP
from f02_data.mtf_dataset import MTFDataset

# -------------------- Logger for this module -------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# =======================================================================================
# 1-- دانلود و نرمال‌سازی دیتافریم‌های خام
# =======================================================================================
def _read_raw_df(
    path: Path,
    mode: Literal["number", "time"] = "number",      # can be: "number" or "time"
    start_lastrows: Optional[int] = None,               # contain 
    end_lastrows: Optional[int] = None,                 # NOT contain
    start_time: Optional[Union[str, datetime]] = None,     # contain
    end_time: Optional[Union[str, datetime]] = None,       # NOT contain
    columns: Optional[List[str]] = None,
    broker_timezone: str = "UTC",
) -> pd.DataFrame:
    logger.debug("==== start _read_raw_df ========================")  # for debug
    logger.debug(f"path = {path}")                                    # for debug
    logger.debug(f"mode = {mode}")                                    # for debug
    logger.debug(f"start_lastrows = {start_lastrows}")                # for debug
    logger.debug(f"end_lastrows = {end_lastrows}")                    # for debug
    logger.debug(f"start_time = {start_time}")                        # for debug
    logger.debug(f"end_time = {end_time}")                            # for debug
    logger.debug("==== end _read_raw_df ==========================")  # for debug
    """
    *** تا قبل از این تابع تمام داده های ذخیره شده روی هارد naive هستند.
    *** در انتهای این تابع ابتدا منطقه زمانی بروکر به داده ها نسبت داده میشود،
    *** سپس منطقه زمانی آنها به UTC تبدیل میشود.
    *** به این ترتیب از ابتدای dataHandler ربات بر اساس داده های UTC کار میکند

    1) تلاش برای خواندن parquet و در صورت خطا/نبود، فرمت CSV را امتحان می‌کند.
    2) مقدار mode اجباراً باید تعیین شده باشد. وگرنه با خطا مواجه میشود.
    3) مقادیر start, end میتوانند تعریف نشوند و None باشند.
    4) دیتا فریم محصول را همانند نرمال سازی موجود در فایل mt5_data_loader.py نرمال میکند
    5) به دیتافریم نهایی شیفت زمانی داده میشود تا داده ها مطابق با utc بشوند.
    """
    # ===============================================================
    # اعمال شرطهای کنترلی برای انتهای بازه ها
    # ===============================================================
    end_lastrows_is_none = False
    end_time_is_none = False
    if end_lastrows is None: end_lastrows_is_none = True
    if end_time is None: end_time_is_none = True

    df = pd.DataFrame()
    # ===============================================================
    # Read parquet file
    # ===============================================================
    readparquet = False
    if path.suffix.lower() == ".parquet" and path.exists():
        try:
            df = pd.read_parquet(path)
            logger.debug(f" -----> type of index = {type(df.index[0])}")
            readparquet = True 
        except Exception as ex:
            logger.warning("Failed to read Parquet (%s). Switching to CSV.", ex)

    # ===============================================================
    # Read csv file
    # ===============================================================
    if readparquet == False:
        csv_path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
            except Exception as ex:
                logger.warning("Failed to read CSV (%s). Return None", ex)

    # ===============================================================
    # بررسی و تبدیل ایندکس به DatetimeIndex و هم‌سوسازی با UTC
    # ===============================================================
    if broker_timezone is None: broker_timezone="UTC"
    
    if isinstance(df.index, pd.DatetimeIndex):           # اگر اندکس از نوع زمانی است
        if df.index.tz is not None:                # اگر در اندکس، منطقه زمانی وجود دارد
            if str(df.index.tz) != "UTC":    # اگر منطقه زمانی برابر با UTC نیست
                df.index = df.index.tz_convert("UTC")   # منطقه زمانی را به UTC تبدیل کن
            else:                            # در غیر اینصورت
                pass                         # بیخیال عبور کن
        else:   # df.index.tz is None              # اگر منطقه زمانی ندارد
            df.index = df.index.tz_localize(broker_timezone)  # به داده ها منطقه زمانی بروکر را نسبت بده
            df.index = df.index.tz_convert("UTC")             # زمان را توسط تبدیل منطقه زمانی، به زمان UTC تبدیل کن
    
    elif not isinstance(df.index, pd.DatetimeIndex):     # اگر اندکس از نوع زمانی نیست
        if 'time' in df.columns:                   #  اگر ستون time در لیست ستونها وجود دارد
            df['time'] = pd.to_datetime(df['time'], tz=broker_timezone)   # ستون time را به نوع زمان با منطقه زمانی بروکر تبدیل کن
            df = df.set_index('time')              # سپس همان ستون time را اندکس دیتافریم قرار بده
            df.index = df.index.tz_convert('UTC')  # منطقه زمانی را به UTC تبدیل کن 
        else:                                      # اگر ستون time وجود ندارد، خطا بده
            raise ValueError("DataFrame index is not DatetimeIndex and no 'time' column found.")    
    
    # ===============================================================
    # Check & outputs
    # ===============================================================
    if len(df) == 0:
        selected_rows = pd.DataFrame()
    else:
        # --- modes ------------------------------------------------- start
        if mode == "number":
            if start_lastrows is None: start_lastrows = len(df)
            if end_lastrows is None: end_lastrows  = 0

            if (start_lastrows > 0) and (end_lastrows >= 0) and (start_lastrows > end_lastrows):
                first_row = max(0, len(df) - start_lastrows)
                last_row  = max(0, len(df) -   end_lastrows)
                # ---/-start-----------
                if end_lastrows_is_none:
                    selected_rows = df[first_row:]
                else:
                    selected_rows = df[first_row:last_row]
                # ---/-end-------------
            elif start_lastrows == 0:
                selected_rows = pd.DataFrame()
            elif (start_lastrows < 0) or (end_lastrows < 0):
                raise ValueError("'start_lastrows'/'end_lastrows' cannot be negative")
        
        elif mode == "time":
            first_time = df.index[0] if start_time is None else pd.to_datetime(start_time, utc=True) # old time
            last_time =  pd.Timestamp.now(tz='UTC') if end_time is None else pd.to_datetime(end_time, utc=True)  # new time
            
            if first_time < last_time:
                start = max(first_time, df.index[0])
                # end = min(last_time, df.index[-1])
                end = last_time
                # ---/-start-----------
                if end_time_is_none:
                    myfilter = (df.index >= start)
                else:
                    myfilter = (df.index >= start) & (df.index < end)   # <<<<<<<<<<==========
                # ---/-end-------------
                selected_rows = df[myfilter]
            else:
                logger.warning("Start_time is bigger than end_time! Return empty dataframe")
                selected_rows = pd.DataFrame()
                
        else:  # mode not in ["number", "time"]
            raise ValueError("mode can be: 'number' or 'time'. Check mode")
        # --- modes ------------------------------------------------- end

    # ===============================================================
    # columns
    # ===============================================================
    if columns is None:
        cols = selected_rows.columns
    if columns is not None:
        cols = [c for c in columns if c in selected_rows.columns]
    result = normalize_df(selected_rows[cols])

    logger.debug(f"len of result of _read_raw_df = {len(result)}")

    # --- output ----------------------------------------------------
    return result

# =======================================================================================
# 2-- (پنجره انتخاب دادهای خام) برای پردازش
# ======================================================================================= OK=
def _get_range_by_timeframe(
    timeframe: str,           # اجباری
    from_last_n: int,         # اجباری
    to_last_n: int,           # اجباری
    base_time: Optional[Union[str, pd.Timestamp]] = None  # اختیاری
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    محاسبه بازه زمانی بر اساس تعداد کندل‌های کامل و بسته‌شده از تایم‌فریم tf1.
    شروع هفته = دوشنبه ساعت 00:00:00 UTC
    پایان هفته = یکشنبه ساعت 23:59:59.999999 UTC

    پارامترها:
        from_last_n: تعداد کندل‌های tf1 به عقب برای شروع بازه (بزرگتر)
        to_last_n: تعداد کندل‌های tf1 به عقب برای پایان بازه (کوچکتر)
        tf1: تایم‌فریم مبنا (مثلاً 'W1' برای هفته، 'D1' برای روز)
        base_time: زمان مبنا (پیش‌فرض: زمان حال UTC)
    
    خروجی:
        (start_time, end_time) به صورت pandas Timestamp با timezone UTC
    
    # ---------------------------------------------------------
    # برای استفاده از این تابع
    # ---------------------------------------------------------
    import pandas as pd

    # روش ۱: استفاده از pd.Timestamp
    ts1 = pd.Timestamp("2026-02-03 04:05:06")

    # روش ۲: استفاده از pd.to_datetime
    ts2 = pd.to_datetime("2026-02-03 04:05:06")

    # تنظیم timezone به UTC (توصیه می‌شود)
    ts3 = pd.Timestamp("2026-02-03 04:05:06", tz='UTC')
    ts4 = pd.to_datetime("2026-02-03 04:05:06", utc=True)

    # ---------------------------
    from datetime import datetime

    dt = datetime(2026, 2, 3, 4, 5, 6)
    ts = pd.Timestamp(dt)  # تبدیل می‌شود
    # ---------------------------------------------------------"""        

    # =======================================
    # Internal Func-1
    # ======================================= OK
    def _get_candle_start_base(time: pd.Timestamp, minutes: int) -> pd.Timestamp:
        """محاسبه زمان شروع کندل برای تایم‌فریم‌های دقیقه‌ای و ساعتی."""
        total_minutes = time.hour * 60 + time.minute
        adjusted_minutes = (total_minutes // minutes) * minutes
        return time.floor('D') + pd.Timedelta(minutes=adjusted_minutes)

    # =======================================
    # Internal Func-2
    # ======================================= OK
    def _last_closedcandle_time(
        timeframe: str,
        steps: int,
        base_time: Union[str, pd.Timestamp, datetime]
    ) -> pd.Timestamp:
        """
        محاسبه زمان شروع کندل شماره 'steps' از آخرین کندل کامل قبل از base_time.

        پارامترها:
            timeframe: تایم‌فریم (مثلاً 'D1', 'H4', 'M5', 'W1', 'MN1')
            steps: تعداد کندل‌های کامل به عقب (1 = آخرین کندل کامل)
            base_time: زمان مبنا (پیش‌فرض: زمان حال)

        خروجی:
            زمان شروع کندل (pd.Timestamp با timezone UTC)
        """

        # ================================================================
        # 1) اعتبار سنجی timeframe
        # ================================================================
        tf_key = _TF_MAP.get(timeframe.upper(), timeframe.upper())
        minutes = _TF_MINUTES.get(tf_key)
        if minutes is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        offset = pd.Timedelta(minutes=minutes)

        # ================================================================
        # 2) محاسبه زمان شروع آخرین کندل کامل قبل از base_time
        # ================================================================
        if tf_key == 'W1':
            # start_of_period = base_time - pd.Timedelta(days=base_time.weekday())            # ابتدای هفته (دوشنبه 00:00:00)
            start_of_period = base_time - pd.Timedelta(days=(base_time.weekday() + 1) % 7)  # ابتدای هفته (یکشنبه 00:00:00)
            start_of_period = start_of_period.floor('D')
            period_offset = pd.Timedelta(days=7)
        elif tf_key == 'MN1':
            # ابتدای ماه (روز اول 00:00:00)
            start_of_period = base_time.floor('D') - pd.Timedelta(days=base_time.day - 1)
            period_offset = pd.DateOffset(months=1)  # برای مقایسه باید به Timestamp تبدیل شود
        else:
            # دقیقه‌ای یا ساعتی
            start_of_period = _get_candle_start_base(base_time, minutes)
            period_offset = offset

        # بررسی کامل بودن کندل
        if tf_key == 'MN1':
            # برای ماه، پایان دوره را با DateOffset محاسبه می‌کنیم
            end_of_period = start_of_period + period_offset
        else:
            end_of_period = start_of_period + period_offset

        # اگر base_time بعد از پایان کندل باشد، کندل کامل شده است
        if base_time >= end_of_period:
            last_complete_start = start_of_period
        else:
            last_complete_start = start_of_period - (period_offset if tf_key in ('W1', 'MN1') else offset)

        # برگرداندن کندل شماره 'steps' به عقب
        if tf_key == 'MN1':
            return last_complete_start - (period_offset * (steps - 2))
        else:
            return last_complete_start - offset * (steps - 2)

    # =======================================
    # 1) اعتبارسنجی ها
    # =======================================
    from_none = from_last_n is None
    to_none = to_last_n is None

    if (from_none and not to_none) or (not from_none and to_none):   # 1=None, 1=not None
        if timeframe is None:
            raise ValueError("'timeframe' must be defined.")
    if (not from_none) and (not to_none):
        if from_last_n < to_last_n:
            raise ValueError("'from_last_n' must be greater than 'to_last_n'.")
        if from_last_n < 0 or to_last_n < 0:
            raise ValueError("'from_last_n' and 'to_last_n' must be positive.")
    # =======================================
    # 2) تنظیم base_time به UTC
    # =======================================
    if base_time is None:
        base_time = pd.Timestamp.now(tz='UTC')
        logger.debug(f"new base_time is {base_time}")
    else:
        base_time = pd.to_datetime(base_time, utc=True)
    # =======================================
    # 3) یکسان‌سازی نام تایم‌فریم‌ها
    # =======================================
    if from_last_n is None:
        start = None
    else:
        start = _last_closedcandle_time(timeframe=timeframe, steps=from_last_n, base_time=base_time)
        start = ensure_timezone_aware(start, tz="UTC")
    if to_last_n is None:
        end = None
    else:
        end = _last_closedcandle_time(timeframe=timeframe, steps=to_last_n, base_time=base_time)
        end = ensure_timezone_aware(end, tz="UTC")

    return start, end, base_time

# =======================================================================================
# 3-- ویژگی‌های زمانی و سشن‌ها 
# ======================================================================================= OK=
def ensure_timezone_aware(dt, tz="UTC"):
    """
    بررسی می‌کند که متغیر زمانی از منطقه‌ی زمانی آگاه است یا نه.
    اگر نبود، منطقه‌ی زمانی مشخص شده را به آن اضافه می‌کند.

    پارامترها:
        dt: ورودی می‌تواند pd.Timestamp، datetime.datetime یا str باشد
        tz: منطقه‌ی زمانی مورد نظر (پیش‌فرض: "UTC")

    خروجی:
        pd.Timestamp با منطقه‌ی زمانی مشخص
    """
    # تبدیل به pandas Timestamp
    if not isinstance(dt, pd.Timestamp):
        dt = pd.to_datetime(dt)

    # بررسی آگاه بودن از منطقه‌ی زمانی
    if dt.tz is None:
        # اگر ناآگاه است، منطقه را اضافه کن
        return dt.tz_localize(tz)
    else:
        # اگر آگاه است، به منطقه‌ی مورد نظر تبدیل کن
        return dt.tz_convert(tz)
    
    """ مثالهایی برا استفاده از تابع بالا
        ==========================================
        # مثال 1: زمان ناآگاه (naive)
        naive_dt = datetime(2026, 7, 7, 10, 30)
        aware = ensure_timezone_aware(naive_dt, "UTC")
        print(aware)  # 2026-07-07 10:30:00+00:00

        # مثال 2: زمان آگاه (aware) با منطقه‌ی دیگر
        aware_dt = pd.Timestamp("2026-07-07 10:30:00", tz="Asia/Tehran")
        converted = ensure_timezone_aware(aware_dt, "UTC")
        print(converted)  # 2026-07-07 07:00:00+00:00

        # مثال 3: رشته
        str_dt = "2026-07-07 10:30:00"
        result = ensure_timezone_aware(str_dt, "UTC")
        print(result)  # 2026-07-07 10:30:00+00:00
        =============================================
    """

# =======================================================================================
# کلاس کمکی BuildParams
# =======================================================================================
class BuildParams:
    """ برای یک نماد و در یک تایمفریم مبنای مشخص """
    def __init__(self,
        symbol: str,                       
        base_tf: str,
        timeframes: Optional[List[str]],
        selected_tf: Optional[str] = None,    # برای ذخیره نمودن (نام تایم فریم جاری) که باید بارگیری بشود                       
        #--------------------
        load_format: Literal["csv", "parquet"] = "parquet",   # فرمت خواندن داده های خام از روی هارد
        mode: Literal["number", "time", "periods"] = "number",
        #-------------------- for mode = "number"
        start_lastrows: int = None,
        end_lastrows: int = None,
        #-------------------- for mode = "time"
        start_time: datetime = None,
        end_time: datetime = None,       
        #-------------------- for mode = "periods"
        period_size: str = None,    # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
        from_last_n: int = None,    # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
        to_last_n: int = None,      # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
        base_time: datetime = None  # برای استفاده در تابع _get_range_by_timeframe سه گانه پنجره انتخاب داده های خام
    ) -> None:
        # --- Validations -------------------------------------------
        if symbol is None:
            raise ValueError("Must define 'symbol'")
        
        # -----
        # وقتی که در مصرف کننده این کلاس از config_completer استفاده شده باشد،
        # در بخش زیر، دیگر نیازی به check_tfs نیست.
        _base_tf, _other_tfs, all_tfs = check_tfs(base_tf, timeframes)
        if _base_tf != base_tf:
            logger.info(f"base_tf changed from {base_tf} to {_base_tf}")
        if set(_other_tfs) != set(timeframes):
            logger.info(f"timeframes changed from {timeframes} to {_other_tfs}")
        # -----
        
        if selected_tf is not None:
            selected_tf = _TF_MAP.get(selected_tf.upper().replace(" ", ""))
        # else:    # این بخش حذف شد. بجای آن فالبک شد به تایم فریم مبنا
        #     raise ValueError(f"selected_tf: {selected_tf} is invalid.")
            
        if period_size is not None:
            period_size = _TF_MAP.get(period_size)

        # --- Initial mountings -------------------------------------
        self.symbol = symbol
        self.base_tf = _base_tf
        self.timeframes = _other_tfs         #   <===== توجه شود
        self.selected_tf = selected_tf or _base_tf
        self.load_format = load_format
        self.mode = mode
        self.start_lastrows = start_lastrows
        self.end_lastrows = end_lastrows
        self.start_time = ensure_timezone_aware(start_time, tz="UTC") if start_time is not None else None
        self.end_time = ensure_timezone_aware(end_time, tz="UTC") if end_time is not None else None
        self.from_last_n = from_last_n
        self.to_last_n = to_last_n
        self.period_size = period_size
        self.base_time = ensure_timezone_aware(base_time, tz="UTC") if base_time is not None else None

        # محاسبه بازه زمانی با استفاده از تابع _get_range_by_timeframe
        now_time = None # ==>> برای اطمینان از اینکه تابع _get_range_by_timeframe اجرا شده است
        if mode == "periods":           
            start, end, now_time = _get_range_by_timeframe(
                from_last_n=from_last_n,
                to_last_n=to_last_n,
                timeframe=period_size,
                base_time=self.base_time,
            )
            if now_time is not None:
                self.mode = "time"
                self.start_time = start
                self.end_time = end
                self.base_time = now_time

# =======================================================================================
# کلاس اصلی DataHandler 
# =======================================================================================
class DataHandler:
    """
    سازنده‌ی دیتاست پردازش‌شده‌ی چند-تایم‌فریم برای آموزش/بک‌تست/اجرا.
    """
    # -------------------------------------------------------------------------
    # 1- سازنده 
    # -------------------------------------------------------------------------
    def __init__(self, 
                 cfg: Dict[str, Any],
                 symbol: str,
                 event_bus: Optional[EventBus] = None,
                 ) -> None:
        
        # -1-- Setting config ---------------------------------------
        self.cfg: Dict[str, Any] = cfg
        self.symbol: str = symbol

        # -2-- Setting directories ----------------------------------
        self.raw_dir: Path = resolve_raw_dir(self.cfg)
        self.proc_dir: Path = resolve_process_dir(self.cfg)

        # -3-- Saving format ----------------------------------------
        dl = (self.cfg.get("download_defaults") or {})  #  از همان فرمت دیتا-لودر استفاده میکند
        self.save_format: str = str(dl.get("save_format", "parquet")).lower()
        if self.save_format not in ("csv", "parquet"):
            self.save_format = "parquet"
        
        # -4-- Default timeframes -------------- --------------------
        self.timeframes = cfg["__timeframes_dict"][symbol]

        # -5-- base_tf ----------------------------------------------
        self._base_tf = cfg["__base_tfs_dict"][symbol]

        # -6-- broker_timezone --------------------------------------
        project_cfg = self.cfg.get("project")
        if not project_cfg:
            raise ValueError("'project' key not found in config !")
            
        self.broker_timezone = project_cfg.get("broker_timezone")
        if not self.broker_timezone:
            raise ValueError("'broker_timezone' key not found in 'project' key !")
        
        # -7-- Connection to MarketDataEngine -----------------------
        self.event_bus = event_bus                 # ➕ Version-D
        self._subscriber_id: Optional[str] = None  # ➕ Version-D
        self._running = False                      # ➕ Version-D
    
        self._data_callback = None

        # -8-- Cache for live mode ----------------------------------
        # یک دیکشنری برای یک نماد. شامل دیتافریمهای مختلف متناظر با تایمفریمهای متفاوت
        self._cache_dict: dict[str, pd.DataFrame] = {}

        self._cached_df: Optional[pd.DataFrame] = None             # not used # ➕ Version-E
        self._lookback: int = 100  # تعداد ردیف‌های نگهداری شده   # not used # ➕ Version-E
        self._cached_tfs: List[str] = []                           # not used # ➕ Version-E
        
        # -9-- Warmup dict for every TF -----------------------------
        self._warmup_dict: Dict[str, int] = cfg["__warmups_dicts"][symbol]
        # _warmup_dict = {'M1': 26, 'M5': 14, 'H4': 14, ...}
        
        self._latest_dataset: Optional[MTFDataset] = None

    # -------------------------------------------------------------------------
    # 2- بارگذاری یک تایم‌فریم خام برای حالت batch
    # ------------------------------------------------------------------------- OK ===> FOR BATCH
    def _load_raw(self, params: BuildParams) -> pd.DataFrame:
        symbol: str = params.symbol
        timeframe: str = params.selected_tf
        fmt: Literal["parquet", "csv"] = params.load_format
        mode: Literal["number", "time"] = params.mode
        start_lastrows: int | None = params.start_lastrows
        end_lastrows: int | None = params.end_lastrows
        start_time: datetime | None = params.start_time
        end_time: datetime | None = params.end_time

        logger.debug("==== start _load_raw =================")  # for debug
        logger.debug(f"symbol = {symbol}")                      # for debug
        logger.debug(f"timeframe = {timeframe}")                # for debug
        logger.debug(f"fmt = {fmt}")                            # for debug
        logger.debug(f"mode = {mode}")                          # for debug
        logger.debug(f"start_lastrows = {start_lastrows}")      # for debug
        logger.debug(f"end_lastrows = {end_lastrows}")          # for debug
        logger.debug(f"start_time = {start_time}")              # for debug
        logger.debug(f"end_time = {end_time}")                  # for debug

        """
        *** تمام تایم فریمهای دریافتی از کلاس BuildParams استاندارد پروژه را دارا میباشند
            و نیازی به کنترل مجدد ندارند
        1) مسیر کامل دسترسی به فایل را بوسیله نماد، تایمفریم و فرمت میسازد
        2) فایل داده را با توجه به مد و تعداد کندلها یا زمان ابتدا و انتها، میخواند
        """
        path = full_file_path(self.raw_dir, symbol, timeframe, fmt=fmt)
        # logger.debug(f"path = {path}")                 # for debug
        df = _read_raw_df(path=path, mode=mode,
                          start_lastrows=start_lastrows, end_lastrows=end_lastrows,
                          start_time=start_time, end_time=end_time,
                          broker_timezone=self.broker_timezone)
        
        if df.empty:
            logger.warning("Raw data %s/%s not found or empty: %s", symbol, timeframe, path)
            return df

        logger.debug(f"Len(df) = {len(df)}")                    # for debug   
        logger.debug("====  end  _load_raw =================")  # for debug
        return df
    
    # -------------------------------------------------------------------------
    # 3- ساخت دیتاست برای حالت batch
    # ------------------------------------------------------------------------- ===> FOR BATCH
    def build(self, params: BuildParams) -> MTFDataset:

        dropna_permit = True
        symbol = params.symbol
        _base_tf = params.base_tf
        _tfs: List[str] = list(params.timeframes) if params.timeframes else self.timeframes
    
        # -----1: set timeframes --------------------------
        base_tf, other_tf, all_tfs = check_tfs(_base_tf, _tfs)

        # -----2: loading df of base_tf -------------------
        params.selected_tf = base_tf
        base_raw = self._load_raw(params)

        if base_raw.empty:
            # در اینجا سیاست این است که قبلاً تمام فایلهای مورد نیاز
            # دانلود شده باشند، تا سرعت اجرای این کلاس پایین نیاید
            raise FileNotFoundError(f"Raw data for {symbol}/{base_tf} is not available. Please download first.")
        
        # ----- 3:
        dataset = MTFDataset(symbol=symbol, base_tf=base_tf)
        dataset.add(base_tf, base_raw)
        for tf in other_tf:
            params.selected_tf = tf
            odf = self._load_raw(params)
            if odf.empty:
                continue
            dataset.add(tf, odf)
        
        self._latest_dataset = dataset
        return dataset

    # -------------------------------------------------------------------------
    # 4- ذخیره‌سازی دیتافریم batch
    # ------------------------------------------------------------------------- ===> FOR BATCH
    def save(self, dataset: MTFDataset, symbol: str, base_tf: str, fmt: Optional[str] = None):
        df = dataset.frames[base_tf]

        base_tf = _TF_MAP[base_tf.replace(" ","").upper()] if base_tf is not None else self._base_tf
        fmt = (fmt or self.save_format or "parquet").lower()

        out = full_file_path(self.proc_dir, symbol, base_tf, fmt)  # برای ذخیره فایل پروسس شده 
        
        # ---------- ذخیره فایل
        if fmt == "parquet":
            try:
                df.to_parquet(out)
            except Exception as ex:
                logger.warning("Parquet save failed (%s). Reverting to CSV.", ex)
                out = out.with_suffix(".csv")
                df.to_csv(out)
                fmt = "csv"
        else: 
            df.to_csv(out)
            fmt = "csv"

        # ---------- Manifest برای تکرارپذیری
        manifest = {
            "symbol": symbol,
            "base_timeframe": base_tf.upper(),
            "start_time": df.index[0],
            "end_time": df["closed_time"].iloc[-1],
            "rows": int(len(df)),
            "columns": list(df.columns),
            "format": fmt.lower(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
        # ---------- Manifest انتهای بلوک

        logger.info("Processed data saved: %s (rows=%d, cols=%d)", out, len(df), len(df.columns))
        return out

    # -------------------------------------------------------------------------
    # 5- ➕ اتصال EventBus به کلاس DataHandler
    # ------------------------------------------------------------------------- ////// FOR LIVE (1)
    def subscribe_to_event_bus(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._subscriber_id = event_bus.subscribe(self.symbol)
        logger.info("DataHandler connected to EventBus with id=%s", self._subscriber_id)

    # -------------------------------------------------------------------------
    # 6- ➕ شروع مصرف خودکار
    # ------------------------------------------------------------------------- ////// FOR LIVE (2)
    def start_consuming2(self) -> None:
        if not self.event_bus or not self._subscriber_id:
            raise RuntimeError("EventBus not attached...")
        
        self._running = True
        try:
            while self._running:
                event = self.event_bus.get_event(self._subscriber_id, timeout=1.0)
                if event and event.get("event_type") == "NEW_CANDLE":
                    self.on_new_candle2(event)
        finally:
            if self._subscriber_id:
                self.event_bus.unsubscribe(self._subscriber_id)
                self._subscriber_id = None
    
    # -------------------------------------------------------------------------
    # 7- ➕ دریافت کندل جدید برای حالت live
    # ------------------------------------------------------------------------- ////// FOR LIVE (3)
    def on_new_candle2_old1(self, event: Dict[str, Any]) -> None:
        """
        مصرف رویداد NEW_CANDLE از EventBus برای آپدیت لحظه‌ای
        """
        symbol = event["symbol"]
        timeframe = event["timeframe"]
        all_dfs = event["all_dfs"]

        # دریافت کندل جدید و به‌روزرسانی دیتاست
        # چونکه قرار است در تمام تایمفریمها آپدیت انجام بشود، فقط سطر زیر باید حذف شود
        if timeframe == self._base_tf:
            # آپدیت دیتاست جاری
            self.update_live2(symbol, timeframe, all_dfs)
            logger.debug("Live update for %s/%s", symbol, timeframe)

    def on_new_candle2(self, event: Dict[str, Any]) -> None:
        """
        مصرف رویداد NEW_CANDLE از EventBus برای آپدیت لحظه‌ای
        """
        symbol = event["symbol"]
        timeframe = event["timeframe"]
        all_dfs = event["all_dfs"]
        self.update_live2(
            symbol=symbol,
            timeframe=timeframe,
            all_dfs=all_dfs,
        )
        logger.debug("Live update for %s/%s", symbol, timeframe)

    # -------------------------------------------------------------------------
    # 8- ➕ Live Update 2
    # ------------------------------------------------------------------------- ////// FOR LIVE (4)
    def update_live2(self, symbol: str, timeframe: str, all_dfs: Dict[str, Any]) -> MTFDataset:
        """
        به‌روزرسانی کش برای یک کندل جدید و بازگرداندن cache_dict به لایه‌ی بالاتر.
        
        پارامترها:
            symbol: نماد (مثلاً XAUUSD)
            timeframe: تایم‌فریم کندل جدید (مثلاً M1)
            all_dfs: دیکشنری شامل دیتافریم‌های جدید برای تمام تایم‌فریم‌ها
            
        خروجی:
            cache_dict به‌روز شده (دیکشنری شامل دیتافریم‌های فشرده برای هر تایم‌فریم)
        """
        logger.debug("update_live2 start.")
        timeframe = timeframe.upper()
        
        # اگر لیست تایم‌فریم‌های مورد نیاز هنوز مشخص نشده، ابتدا باید set_warmup_lengths صدا زده شود
        if not self._warmup_dict:
            logger.warning("Required timeframes not set. Call set_warmup_lengths() first.")
            return pd.DataFrame()   #### ایراد دارد. باید دیتاست برگرداند
        
        # اگر تایم‌فریم دریافتی در لیست مورد نیاز نیست، نادیده بگیر
        if timeframe not in self._warmup_dict.keys():
            logger.debug(f"Ignoring {timeframe} (not required)")
            return pd.DataFrame()   #### ایراد دارد. باید دیتاست برگرداند
        
        # به‌روزرسانی کش
        dataset = self._update_cache(symbol, timeframe, all_dfs)
        
        # ----- روش اول برای اجرای تابع کال بک ----------------------
        self._notify_new_data(dataset)

        # ----- روش دوم برای اجرای کال بک ---------------------------
        # if self._data_callback is not None:
        #     self._data_callback(dataset)

        return dataset       

    # -------------------------------------------------------------------------
    # 9- ➕ Helpers for Live Update
    # ------------------------------------------------------------------------- ////// FOR LIVE (5)
    def _update_cache(self, symbol: str, timeframe: str, all_dfs: Dict[str, pd.DataFrame]) -> MTFDataset:
        """
        به‌روزرسانی کش دیکشنری ها برای یک نماد و تایم‌فریم های موجود در warmup_dict.
        all_dfs داده های کندلی نماد مربوطه در تایمفریمهای مختلف است.

        این تابع دیکشنری _cache_dict را بروزرسانی میکند
        """       
        for tf in self._warmup_dict.keys():
            key = f"{symbol}:{tf.upper()}"
            if key not in self._cache_dict:
                self._cache_dict[key] = all_dfs[key].copy()
            else:
                combined = pd.concat([self._cache_dict[key], all_dfs[key]], axis=0)
                combined = combined[~combined.index.duplicated(keep='last')]
                
                warmup = self._warmup_dict.get(tf)  # تعداد کندلهای مورد نیاز برای وارم آپ
                # if warmup and len(combined) > warmup:
                #     combined = combined.iloc[-warmup:]
                if warmup and len(combined) > (warmup + 1):
                    combined = combined.iloc[-(warmup + 1):]
                self._cache_dict[key] = combined
        logger.debug(f"_cache_dict was updated: {self._warmup_dict}")

        # --------- new added
        dataset = MTFDataset(symbol=symbol, base_tf=self._base_tf)
        for tf in self._warmup_dict:
            key = f"{symbol}:{tf}"
            if key in self._cache_dict:
                dataset.add(tf, self._cache_dict[key])
       
        self._latest_dataset = dataset
        return dataset
           
    # -------------------------------------------------------------------------
    def get_latest_dataset(self) -> Optional[MTFDataset]:
        return self._latest_dataset

    # -------------------------------------------------------------------------
    # 10- ➕ ثبت تابع کال بک مصرف کننده در این کلاس
    # ------------------------------------------------------------------------- ////// FOR LIVE (6)
    def set_data_callback(self, callback):
        """ثبت تابع callback برای دریافت دیتافریم جدید"""
        self._data_callback = callback

    # -------------------------------------------------------------------------
    # 11- ➕ فراخوانی تابع کال بک مصرف کننده از این کلاس
    # ------------------------------------------------------------------------- ////// FOR LIVE (7)
    def _notify_new_data(self, dataset: MTFDataset):
        """هر جا که دیتافریم جدید ساخته شد (مثلاً در متد update یا در consumer)، این متد را صدا بزنید"""
        if self._data_callback is not None:
            self._data_callback(dataset)

    # -------------------------------------------------------------------------
    # 12- ➕ متد جدید: توقف مصرف
    # ------------------------------------------------------------------------- ////// FOR LIVE (8)
    def stop_consuming(self) -> None:
        self._running = False
        if self.event_bus and self._subscriber_id:
            self.event_bus.unsubscribe(self._subscriber_id)


# =======================================================================================
# CLI
# =======================================================================================
# ------------------------------------------------------------------- OK
def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        # format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        format="%(asctime)s | %(levelname)-6s | %(filename)-28s | %(lineno)-4d : %(funcName)-24s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ------------------------------------------------------------------- OK
def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Create a processed multi-timeframe dataset from raw MT5 data.")
    
    p.add_argument("-c", "--config", type=str, default=str(project_root() / "f01_config" / "config.yaml"),
                                                                help="Path to the config file (default: f01_config/config.yaml)")
    p.add_argument("--symbol", type=str, required=True,         help="Symbol (example: XAUUSD)")
    p.add_argument("--base-tf", type=str, default=None,         help="Base timeframe")
    p.add_argument("--timeframes", nargs="*", default=None,     help="Timeframes to use. If not provided, config.download_defaults.timeframes will be used.")
    p.add_argument("--load_format", type=str, default=None, choices=["csv", "parquet"], help="Raw Data format (default: from config)")
    p.add_argument("--save_format", type=str, default=None, choices=["csv", "parquet"], help="Output format (default: from config)")
    p.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    return p.parse_args()

# ------------------------------------------------------------------- OK
def main() -> int:
    from f10_utils.config_loader import load_config

    # --- 1 --- استخراج مقادیر از خط فرمان 
    args = _parse_args()

    # --- 2 --- ساخت لاگر و تعیین سطح آن، همراه با تعیین فرمت و فرمت زمان 
    _setup_logging(args.log_level)

    # --- 3 --- بارگذاری کانفیگ با ENV Override
    cfg = load_config(args.config, enable_env_override=True)

    # --- 4 --- ساخت هندلر
    handler = DataHandler(cfg=cfg, symbol=args.symbol)

    # --- 5.1 --- ساخت دیتاست
    base_tf = (args.base_tf or handler._base_tf or "M1").upper()
    base_tf = "4m"
    save_format = (args.save_format or handler.save_format or "parquet").lower()
    
    # --- 5.2 پارامترهای ساخت
    params = BuildParams(
        symbol=args.symbol,                # اجباری است
        base_tf=base_tf,                      # base_tf,
        timeframes=["m20", "1  0  M  "],   #args.timeframes,
        selected_tf=None,
        load_format=(args.load_format or "parquet").lower(),    # فرمت داده های خام که باید خوانده شوند.

        mode = "periods",  # "number", "time", "periods"
        # --- مربوط به مد number:
        start_lastrows=500,
        end_lastrows=200,
        # --- مربوط به مد time:
        start_time=pd.to_datetime("2026-05-31 00:00:00+00:00", utc=True),
        end_time=pd.to_datetime("2026-06-05 00:00:00+00:00", utc=True),
        # --- مربوط به مد periods:
        period_size = "20m",
        from_last_n = 8,
        to_last_n = 4,
        base_time = pd.to_datetime("2026-06-02 20:00:00").tz_localize(handler.broker_timezone),
    )

    # --- 5.4 ساخت دیتاست
    df = handler.build(params)

    # --- 6 --- ذخیره سازی دیتاست
    out = handler.save(df=df, symbol=args.symbol, base_tf=base_tf, fmt=save_format)

    logger.info("Done. Output: %s", out)
    return 0

# ------------------------------------------------------------------- OK
if __name__ == "__main__":
    raise SystemExit(main())

r"""
Run: python -m f02_data.data_handler_F_2  --symbol XAUUSD_i 
"""
# =======================================================================================
# تست پوشش کد (برای توسعه‌دهندگان) 
# =======================================================================================
""" Func Names                                 Used in Functions: ...
                              1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
1  resolve_process_dir      --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --
2  _read_raw_df              --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --
3  _parse_hhmm_in_utc               --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
4  _in_utc_range             --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
5  _add_session_flags        --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
6  prefix_columns           --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  ok  --  --  --  --
7  _merge_on_base            --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
8  _check_ohlc               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --
9  _normalize_ohlc_columns   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Not Used
10  _add_time_features_to_df --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  ok  --  --  --  --
11 _add_qc_flags_to_df       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  ok  --  --  --  --
12 _finalize_dataframe       --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  ok  --  --  --  --

13 Class: BuildParams        --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  ok  --

14 Class: DataHandler        --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --  --  --  --  --  --  ok  --
15    __init__               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --
16    _load_raw              --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  ok  --  --  --  --
17    build                  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
18    save                   --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
19    on_new_candle          --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --  --  --  --  --  --
20    subscribe_to_event_bus --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Not Used
21    start_consuming        --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Not Used
22    stop_consuming         --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  -- Not Used
23    update_live            --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  20  --  --  --  --  --  --  --  --

24 _setup_logging            --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
25 _parse_args               --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok  --
26 main                      --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
27 (Global code)             -/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/-/
"""
