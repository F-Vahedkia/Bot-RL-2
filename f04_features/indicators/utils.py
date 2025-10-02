# -*- coding: utf-8 -*-
# f04_features/indicators/utils.py
# Status in (Bot-RL-2): Completed

"""
ابزارهای کمکی مشترک برای اندیکاتورها (Bot-RL-1)
- کشف تایم‌فریم‌ها از نام ستون‌ها
- نگهبان NaN/Inf و سبک کردن dtype
- zscore، true_range
"""
from __future__ import annotations
from typing import Sequence, Optional, Dict, Tuple, List, Iterable, Any
import re
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd

# وزن‌دهی — نام ستون‌های قابل‌قبول (اولین موجود انتخاب می‌شود)
DEFAULT_MA_SLOPE_CANDIDATES: List[str] = [
    "__ma_slope@M5", "__ma_slope@H1", "__ma_slope@H4"
]
DEFAULT_RSI_SCORE_CANDIDATES: List[str] = [
    "__rsi_zone@H1__rsi_zone_score", "__rsi_zone@H4__rsi_zone_score"
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

""" --------------------------------------------------------------------------- OK C1
کشف تایم‌فریم‌ها از روی نام ستون‌ها
"""
@dataclass
class TFView:
    tf: str
    cols: Dict[str, str]  # mapping: standard_name -> df_column_name
                          #        :        "open" -> "H1_open"
_TF_REGEX = re.compile(r"^(?P<tf>[A-Z0-9]+)_(?P<field>open|high|low|close|tick_volume|spread)$", re.IGNORECASE)


""" --------------------------------------------------------------------------- OK C2
# FiboTestConfig: پیکربندی تستِ مستقل از scripts  (Test-only, not runtime defaults)
# -----------------------------------------------------------------------------
# توضیح آموزشی (فارسی):
# این کلاس فقط برای سناریوهای تست/وایرینگ سبک استفاده می‌شود تا وابستگی به f15_scripts
# حذف شود. اگر فولدر scripts پاک شود، این کلاس هنوز داخل هسته باقی می‌ماند.
# - مقادیر پیش‌فرض مطابق بلوک «پیکربندی تست» قبلی هستند.
# - بعداً اگر کلیدهای متناظر از قبل در config.yaml موجود باشند، می‌توانیم همین کلاس
#   را از روی کانفیگ پُر کنیم (طبق قانون کانفیگ شما: اگر کلید موجود بود، تکراری نسازیم).
# - پیام‌های اجرایی در این کلاس نداریم؛ فقط داده و هِلپرهای سبک.
# -----------------------------------------------------------------------------
"""
@dataclass
class FiboTestConfig:
    # مسیر دیتاست پردازش‌شده (در صورت نیاز برای سناریوی تست)
    DATA_FILE: str = r"f02_data/processed/XAUUSD/H1.parquet"

    # تایم‌فریم‌ها و پنجرهٔ برش دادهٔ اخیر
    TFS: List[str] = field(default_factory=lambda: 
                            ["M1", "M5", "M30", "H1", "H4", "D1", "W1"])
    TAILS: Dict[str, int] = field(default_factory=lambda: 
                            {"M1": 1000, "M5": 1000, "M30": 1000, "H1": 500, "H4": 500, "D1": 500, "W1":200})
    N_LEGS: int = 5                # تعداد لگ‌های اخیر برای هر TF

    # پارامترهای خوشه‌بندی فیبو (درصدها به واحد percent هستند)
    TOL_PCT: float = 0.20          # پنجرهٔ همگرایی خوشه‌ها (٪)
    PREFER_RATIO: float = 0.618    # نسبت مرجح

    # پارامترهای سطوح رُند S/R
    SR_STEP: float = 10.0          # گام سطوح رُند (مثلاً XAUUSD≈10)
    SR_COUNT: int = 25             # تعداد سطوح رُند حول قیمت آخر
    SR_TOL_PCT: float = 0.05       # تلورانس نسبی برای همپوشانی با S/R

    # وزن‌دهی مؤلفه‌های کانفلوئنس
    W_TREND: float = 10.0          # وزن ترند (MA slope)
    W_RSI: float = 10.0            # وزن RSI zone
    W_SR: float = 10.0             # وزن همپوشانی S/R

    # ------------------------ هِلپرهای مصرف ------------------------
    def sr_levels(self, ref_price: float) -> List[float]:
        """
        توضیح آموزشی (فارسی):
          بر اساس قیمت مرجع، سطوح رُند را به‌صورت متقارن می‌سازد تا
          به fib_cluster / fib_cluster_cfg پاس بدهیم.
        """
        return round_levels(ref=ref_price, step=self.SR_STEP, count=self.SR_COUNT)

    def to_cluster_kwargs(self) -> Dict[str, Any]:
        """
        توضیح آموزشی (فارسی):
          پارامترهای مرتبط با خوشه‌بندی را در قالب یک دیکشنری آماده می‌کند
          تا به wrapper یا خود fib_cluster پاس داده شوند.
        """
        return {
            "tol_pct": self.TOL_PCT,
            "prefer_ratio": self.PREFER_RATIO,
            # وزن‌ها و تلورانس SR معمولاً در امضای wrapper مصرف می‌شوند:
            "w_trend": self.W_TREND,
            "w_rsi": self.W_RSI,
            "w_sr": self.W_SR,
            "sr_tol_pct": self.SR_TOL_PCT,
        }


""" --------------------------------------------------------------------------- OK Func2
هِلپر عمومی S/R: تولید سطوح رُندِ متقارن پیرامون ref
"""
def round_levels(anchor: float, step: float, n: int = 10) -> List[float]:
    """
    تولید یک «شبکهٔ سطوح رُند» حول مقدار anchor با فاصلهٔ step.
    مثال: round_levels(1945.3, 10, n=5) → [1895, 1905, ..., 1995]

    پارامترها:
      anchor: لنگر قیمتی (مثلاً آخرین قیمت)
      step: فاصلهٔ شبکه (مثلاً 10.0 برای طلا، یا 0.5 …)
      n: چند سطح به بالا/پایین (دوطرفه)

    خروجی: لیست سطوح رُند (کوچک به بزرگ)
    """
    if step <= 0:
        raise ValueError("step must be positive")

    base = np.floor(anchor / step) * step  # کف رند نزدیک
    levels = [base + k * step for k in range(-n, n + 1)]
    return sorted(levels)


""" --------------------------------------------------------------------------- OK Func3
استخراج نمای استانداردِ OHLC برای TF خواسته‌شده از روی ستون‌های پیشونددار.
خروجی: DataFrame با ستون‌های ['open','high','low','close','tick_volume','spread'] (هر کدام که موجود باشد)
"""
def get_ohlc_view(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    cols = {}
    for k in ["open", "high", "low", "close", "tick_volume", "spread"]:
        c = f"{tf}_{k}"
        if c in df.columns:
            cols[k] = df[c]
    out = pd.DataFrame(cols).dropna(how="all")
    if out.empty:
        raise ValueError(f"OHLC for TF={tf} not found")
    out.index = pd.to_datetime(out.index, utc=True)
    out.sort_index(inplace=True)
    return out


""" --------------------------------------------------------------------------- OK Func4
یک هلپر عمومی برای انتخاب اولین ستون موجود از چند نامِ کاندید.
اولین ستونی که در دیتافریم موجود است را برمی‌گرداند؛ در غیر این‌صورت None.
"""
def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]:

    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s is not None and len(s) > 0:
                return s
    return None


""" --------------------------------------------------------------------------- OK Func5
این تابع از روی نام ستون‌های دیتافریم می فهمد چه «تایم‌فریم»‌هایی داخل داده وجود دارد
 و برای هر تایم‌فریم، نگاشتی از ستون‌های واقعی به اسامی استاندارد OHLC می سازد.
مثال:
 {
  "M30": TFView(tf="M30", cols={"open":"M30_open", "high":"M30_high", "low":"M30_low", "close": "M30_close",
      # اگر ستون‌های دیگری مثل tick_volume/spread هم داشتیم، اینجا اضافه می‌شدند
                                }
                ),
  "H1": TFView(tf="H1", cols={"open":"H1_open", "high":"H1_high", "low":"H1_low", "close": "H1_close",
                                }
                )
}
 """
def detect_timeframes(df: pd.DataFrame) -> Dict[str, TFView]:
    buckets: Dict[str, Dict[str, str]] = {}
    for col in df.columns:
        m = _TF_REGEX.match(col)
        if not m:
            continue
        tf = m.group("tf").upper()
        field = m.group("field").lower()
        buckets.setdefault(tf, {})[field] = col
    return {tf: TFView(tf=tf, cols=mapping) for tf, mapping in buckets.items()}


""" --------------------------------------------------------------------------- OK Func6
برش یک TF با استانداردسازی نام ستون‌ها
"""
def slice_tf(df: pd.DataFrame, view: TFView) -> pd.DataFrame:
    cols = []
    rename_map = {}
    for k_std, c in view.cols.items():
        cols.append(c)
        rename_map[c] = "volume" if k_std == "tick_volume" else k_std
    sdf = df[cols].rename(columns=rename_map).copy()
    return sdf


""" --------------------------------------------------------------------------- OK Func7
نگهبان NaN و dtype سبک
"""
def nan_guard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            # Nullable Int64 را دست‌نخورده می‌گذاریم
            pass
    return df


""" --------------------------------------------------------------------------- OK Func8
z-Score ساده یا همان نرمال سازی
"""
def zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    mp = min_periods or window
    # above line is equal to: 
    # mp = window if min_periods is None else min_periods
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    return ((s - mean) / std.replace(0, np.nan)).astype("float32")


""" --------------------------------------------------------------------------- OK Func9
True Range (برای ATR و ...)
"""
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.astype("float32")


# --- New Added ----------------------------------------------------- 040607
"""
افزودنی‌های کم‌خطر برای Utils اندیکاتورها (Bot-RL-2)
- تشخیص سوئینگ‌ها (بدون SciPy) با فیلترهای prominence و ATR
- محاسبهٔ ATR (کلاسیک با SMA)
- زی‌اسکورِ فاصله (zscore_distance)
- فاصله تا نزدیک‌ترین سطح (nearest_level_distance)

نکته‌ها:
- همهٔ ورودی/خروجی‌ها با ایندکس زمانی UTC مرتب فرض شده‌اند.
- نام‌گذاری مطابق «اولین سند اصلاح اندیکاتورها».
"""


""" --------------------------------------------------------------------------- OK Func10
Additions for swing/metrics
Wilder/EMA/Classic ATR. Returns a pandas Series aligned with df.index.
- این تابع ATR را بر اساس True Range محاسبه می‌کند.
- روش‌ها:
  * "classic": میانگین سادهٔ TR (SMA) با min_periods نیم‌پنجره (مطابق نسخهٔ خودت)
  * "wilder" : هموارسازی وایلدر با α = 1/window
  * "ema"    : هموارسازی نمایی رایج با α = 2/(window+1)
- خروجی: Series هم‌تراز با df.index
"""
def compute_atr(df: pd.DataFrame, window: int = 14, method: str = "classic") -> pd.Series:

    # نگهبان‌های ورودی
    if window < 1:
        raise ValueError("window must be >= 1")
    if not {"high", "low", "close"}.issubset(set(df.columns)):
        raise ValueError("DF must contain columns: high, low, close")

    # نرمال‌سازی روش
    m = (method or "classic").strip().lower()
    if m not in {"classic", "wilder", "ema"}:
        raise ValueError(f"Unknown ATR method: {method!r}. Use 'classic', 'wilder', or 'ema'.")

    # True Range
    tr = true_range(high=df["high"], low=df["low"], close=df["close"])

    # محاسبهٔ ATR بر اساس روش
    if m == "wilder":
        atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    elif m == "ema":
        atr = tr.ewm(alpha=2 / (window + 1), adjust=False, min_periods=window).mean()
    else:  # classic
        #atr = tr.rolling(window=window, min_periods=max(2, window // 2)).mean()
        atr = tr.rolling(window=window, min_periods=window).mean()

    # بهینه‌سازی حافظه و نام‌گذاری یک‌دست
    atr = atr.astype("float32")
    atr.name = f"ATR_{m}_{window}"
    return atr


""" --------------------------------------------------------------------------- OK Func11
Swing Detection (H/L)

English:
    Local swing detection without SciPy:
    - A point is swing-high if it's the maximum in a ±min_distance window.
    - A point is swing-low  if it's the minimum in a ±min_distance window.
    - Optional 'prominence' and 'atr_mult * ATR' filters to drop weak swings.
Persian:
    تشخیص قله/کف محلی بدون SciPy:
    - بیشینه/کمینه در پنجرهٔ ±min_distance
    - فیلتر اختیاری بر اساس prominence و همچنین آستانهٔ ATR (atr_mult * ATR)

# نکته: price باید ایندکس زمانی UTC و مرتب داشته باشد.

تشخیص قله/کف محلی بدون SciPy:
    - نقطه swing-high اگر در پنجرهٔ ±min_distance بیشینه باشد.
    - نقطه swing-low  اگر در پنجرهٔ ±min_distance کمینه باشد.
    - فیلتر اختیاری با prominence و همچنین آستانهٔ ATR (atr_mult * ATR).

ورودی‌ها:
    price: Series اندیس‌گذاری‌شده بر حسب زمان (UTC)
    prominence: حداقل برجستگی نسبت به لبه‌های پنجره (اختیاری)
    min_distance: نصفِ اندازهٔ پنجره به دو طرف
    atr: سری ATR هم‌تراز (اختیاری)
    atr_mult: اگر داده شود، آستانهٔ حذف سوئینگ‌های ضعیف = atr_mult * ATR
    tf: نام تایم‌فریم برای متادیتا (اختیاری)

خروجی:
    DataFrame با ایندکس زمانی، ستون‌ها: ['price','kind','atr','tf']
    kind ∈ {'H','L'}

"""
def detect_swings(
    price: pd.Series,
    prominence: Optional[float] = None,
    min_distance: int = 5,
    atr: Optional[pd.Series] = None,
    atr_mult: Optional[float] = None,
    tf: Optional[str] = None,
    ) -> pd.DataFrame:

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series indexed by time")

    # تضمین ترتیب زمانی سریِ ورودی (پایدار): اگر مرتب نیست، یک‌بار پایدار مرتب می‌کنیم
    if not price.index.is_monotonic_increasing:
        price = price.sort_index(kind="stable")

    idx = price.index
    n = len(price)
    if n < (2 * min_distance + 1):
        logger.debug("detect_swings: insufficient length (n=%d, min_distance=%d)", n, min_distance)
        return pd.DataFrame(columns=["price", "kind", "atr", "tf"])

    highs: list[Tuple[pd.Timestamp, float]] = []
    lows:  list[Tuple[pd.Timestamp, float]] = []

    # پنجره‌ی لغزان برای اکسترمم محلی
    for i in range(min_distance, n - min_distance):
        p = price.iloc[i]
        left = price.iloc[i - min_distance : i]
        right = price.iloc[i + 1 : i + 1 + min_distance]

        is_high = p >= left.max() and p >= right.max()
        is_low  = p <= left.min() and p <= right.min()

        if not (is_high or is_low):
            continue

        prom_left  = abs(p - left.iloc[-1])   # prom_left  = abs(price.iloc[i] - price.iloc[i-1])
        prom_right = abs(p - right.iloc[0])   # prom_right = abs(price.iloc[i] - price.iloc[i+1])        

        # فیلتر prominence ساده: فاصله از نزدیک‌ترین همسایهٔ طرفین
        if (prominence is not None) and (prominence > 0):
            prom_ok = (prom_left >= prominence) and (prom_right >= prominence)
            if not prom_ok:
                continue

        # فیلتر مبتنی بر ATR (اگر دادهٔ ATR و ضریب atr_mult داده شده باشد)
        atr_here = float(atr.iloc[i]) if (atr is not None and pd.notna(atr.iloc[i])) else np.nan
        if (atr is not None) and (atr_mult is not None) and (atr_mult > 0) and not np.isnan(atr_here):
            local_prom = max(prom_left, prom_right)
            if local_prom < atr_mult * atr_here:
                continue

        # اگر p توسط هر دو فیلتر قبول شود:
        ts = idx[i]
        if is_high:
            highs.append((ts, float(p)))
        if is_low:
            lows.append((ts, float(p)))

    # خروجی یکدست بصورت یک لیست از دیکشنری ها
    rows: List[Dict] = []
    
    for ts, val in highs:
        atr_value = float(atr.loc[ts]) if (atr is not None and ts in atr.index) else np.nan
        rows.append({"ts": ts, "price": val, "kind": "H", "atr": atr_value, "tf": tf})
    for ts, val in lows:
        atr_value = float(atr.loc[ts]) if (atr is not None and ts in atr.index) else np.nan
        rows.append({"ts": ts, "price": val, "kind": "L", "atr": atr_value, "tf": tf})

    swings = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

    if not swings.empty:
        swings.set_index("ts", inplace=True)
        swings.index = pd.to_datetime(swings.index, utc=True)
    else:
        logger.debug("detect_swings: no swings detected")
    return swings


""" --------------------------------------------------------------------------- OK Func12
Z-Score distance
English: Return (x - mu) / sigma with small epsilon for stability.
Persian: نرمال‌سازی فاصله با زی‌اسکور.
"""
def zscore_distance(x: float, mu: float, sigma: float, eps: float = 1e-12) -> float:

    s = abs(sigma) if sigma is not None else 0.0
    return float((x - mu) / (s + eps))


""" --------------------------------------------------------------------------- OK Func13
Nearest level distance
English: Find nearest level to 'price' and return distances (signed/abs) and the level.
Persian: نزدیک‌ترین سطح قیمتی به price را برمی‌گرداند.
"""
def nearest_level_distance(price: float, levels: Sequence[float]) -> Dict[str, float]:

    # 1) گارد ایمن‌تر برای توالی خالی
    if levels is None or len(levels) == 0:
        return {"nearest_level": float("nan"), "signed": float("nan"), "abs": float("nan")}    

    # 2) حذف مقادیر غیرمتناهی/NaN از levels
    clean = [lv for lv in levels if np.isfinite(lv)]
    if len(clean) == 0:
        return {"nearest_level": float("nan"), "signed": float("nan"), "abs": float("nan")}

    # 3) محاسبهٔ فاصله‌ها
    diffs = [price - lv for lv in clean]
    j = np.argmin([abs(d) for d in diffs])  # تبدیل به int لازم نیست
    return {"nearest_level": float(clean[j]), "signed": float(diffs[j]), "abs": float(abs(diffs[j]))}


""" --------------------------------------------------------------------------- OK Func14
ساخت سطوح فیبوی رتریسمنت برای «n لگ اخیر» از روی سوئینگ‌های بسته.

ورودی:
    - ohlc_df: DataFrame با ستون‌های open/high/low/close (ایندکس UTC مرتب)
    - n_legs: تعداد لگ‌های اخیر (پیش‌فرض 10)
    - ratios: نسبت‌های فیبو (اگر None → [0.236,0.382,0.5,0.618,0.786])
    - prominence/min_distance/atr_mult: پارامترهای فیلتر سوئینگ (برای حذف نویز)

خروجی:
    DataFrame ستون‌ها: ['ratio','price','leg_up','leg_idx']
    - leg_idx: شمارهٔ لگ از انتها (1 = آخرین لگ، 2 = یکی قبل‌تر، ...)
"""
def levels_from_recent_legs(
    ohlc_df: pd.DataFrame,
    n_legs: int = 10,
    ratios: Optional[Iterable[float]] = None,
    prominence: Optional[float] = None,
    min_distance: int = 5,
    atr_mult: Optional[float] = 1.0,
) -> pd.DataFrame:

    # 1) اگر ratios ندادیم، لیست پیش‌فرض ساخته می‌شود. --------------
    if ratios is None:
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    # 2) محاسبهٔ ATR برای فیلتر سوئینگ (در صورت نیاز) ---------------
    atr = None
    try:
        atr = compute_atr(ohlc_df, window=14, method="wilder")
    except Exception:
        pass

    # 3) محاسبه سوئینگ ها -------------------------------------------
    close = ohlc_df["close"].astype(float)
    try:
        swings = detect_swings(
            close,
            prominence=prominence,
            min_distance=min_distance,
            atr=atr,
            atr_mult=atr_mult,
            tf=None,
        )
    except Exception as ex:
        # اگر detect_swings در دسترس نبود
        return pd.DataFrame(columns=["ratio", "price", "leg_up", "leg_idx"])

    # 4) اگر سوئینگ کمتر از ۲ نقطه بود، خروجی خالی است --------------
    if swings is None or swings.empty or len(swings) < 2:
        return pd.DataFrame(columns=["ratio", "price", "leg_up", "leg_idx"])

    # 5) سورت نمودن با ایندکس و تبدیل لیست قیمت‌های سوئینگ به آرایه -
    s = swings.sort_index()
    prices = s["price"].astype(float).to_numpy()

    rows: List[dict] = []
    # از آخرین نقطه شروع می‌کنیم: (i-1 → i) یک لگ
    # i: اندیس آخرین سوئینگ، i-1: سوئینگ قبلی
    last_i = len(prices) - 1
    max_legs = min(n_legs, last_i)  # به تعداد جفت‌ها می‌تونیم لگ بسازیم

    # 6) از آخرین نقطه شروع می‌کند؛ هر جفت متوالی یک لگ می‌شود (p1→p2).
    for k in range(0, max_legs):
        i = last_i - k
        j = i - 1
        if j < 0:
            break
        p1, p2 = prices[j], prices[i]   # p1: old_price,   p2: new_price
        # 7) جهت لگ (leg_up) تعیین می‌شود، سپس low و high انتخاب می‌شود.
        leg_up = p2 > p1
        low, high = (p1, p2) if leg_up else (p2, p1)

        rng = high - low   # it means "range"
        if rng <= 0:
            continue
        # 8) برای هر نسبت فیبو، قیمت رتریسمنت محاسبه و در لیست rows ذخیره می‌شود.
        for r in ratios:
            # قیمت رتریسمنتِ لگ (استاندارد)
            price = (high - r * rng) if leg_up else (low + r * rng)
            rows.append({
                "ratio": float(r),
                "price": float(price),
                "leg_up": bool(leg_up),
                "leg_idx": int(k+1),   # آخرین لگ یا اخیرترین لگ دارای ایندکس 1 خواهد بود
            })
    # 9)دیتافریم نهایی ساخته و بر اساس price و leg_idx مرتب می‌شود.
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["price", "leg_idx"]).reset_index(drop=True)
    return out
