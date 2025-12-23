# -*- coding: utf-8 -*-
# f03_features/indicators/sr_advanced.py
# Status in (Bot-RL-2): Completed

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import logging

from .core import atr as atr_core

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --- Fair Value Gap FVG (3-bar) ------------------------------------
# تشخیص + lifecycle + انتشار + امتیاز
"""
تشخیص Fair Value Gap (FVG) بدون نگاه به آینده (look-ahead) و قابل‌استفاده در Engine/Registry

قواعد:
- تعریف کلاسیک 3-کندله:
  * FVG صعودی (bull): low[i] > high[i - lookback]  (به‌طور پیش‌فرض lookback=2)
  * FVG نزولی (bear): high[i] < low[i - lookback]
- آستانهٔ حداقل اندازهٔ گپ نسبت به ATR برای حذف نویزهای کوچک
- ضد لوک‌اِهد: فلگ‌ها با shift(+1) اعمال می‌شوند.
- خروجی استاندارد: سری‌های فلگ و محدودهٔ زون‌ها برای مصرف در Engine/Registry

نکات:
- همهٔ خروجی‌ها per-bar و float32/int8 هستند.
- در این گام، «زون‌های زنده/پرشدن زون/عمر زون» به‌صورت ساده ارائه می‌شود (ایجاد اولیهٔ زون).
"""
def detect_fvg(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    lookback: int = 2,
    atr_window: int = 14,
    min_size_pct_of_atr: float = 0.50
) -> Dict[str, pd.Series]:
    """
    تشخیص FVG به‌صورت سری‌های پرچم و بازهٔ قیمتی زون.

    پارامترها:
    - lookback: فاصلهٔ مرجع 3-کندله (به‌طور کلاسیک 2 → مقایسهٔ کندل i با i-2)
    - atr_window: طول ATR برای سنجش حداقل اندازهٔ گپ
    - min_size_pct_of_atr: حداقل نسبت اندازهٔ گپ به ATR (مثلاً 0.5 یعنی 50٪ ATR)

    خروجی:
    - fvg_up:   پرچم FVG صعودی (int8) با shift(+1) برای ضد-لوک‌اِهد
    - fvg_dn:   پرچم FVG نزولی (int8) با shift(+1)
    - fvg_top:  سقف زون FVG (float32) — فقط روی کندلِ ایجاد مقدار دارد، در غیر این‌صورت NaN
    - fvg_bottom: کف زون FVG (float32) — فقط روی کندلِ ایجاد مقدار دارد، در غیر این‌صورت NaN
    """
    # --- checks ----------------------------------------------------
    if not (len(open_) == len(high) == len(low) == len(close)):
        raise ValueError("Input series must have equal length.")

    lookback = int(lookback)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    # --- ATR برای سنجش حداقل اندازهٔ معتبر -------------------------
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")

    # --- شرایط خام FVG (سه-کندلی) ----------------------------------
    # Bullish: low[i] > high[i - lookback]
    bull_raw = (low > high.shift(lookback))

    # Bearish: high[i] < low[i - lookback]
    bear_raw = (high < low.shift(lookback))

    # --- اندازهٔ گپ (فقط مثبت تعریف شود) ---------------------------
    # برای bull: فاصلهٔ low[i] تا high[i-lookback]
    bull_gap = (low - high.shift(lookback)).clip(lower=0.0)

    # برای bear: فاصلهٔ low[i-lookback] تا high[i]
    bear_gap = (low.shift(lookback) - high).clip(lower=0.0)

    # --- آستانهٔ اندازه برحسب ATR ----------------------------------
    thr = (atrv * float(min_size_pct_of_atr)).astype("float32")
    bull_ok = bull_raw & (bull_gap >= thr)
    bear_ok = bear_raw & (bear_gap >= thr)

    # --- محدودهٔ زون -----------------------------------------------
    # bull: کف زون = high[i-lookback]، سقف زون = low[i]
    fvg_top_bull = low.where(bull_ok).astype("float32")
    fvg_bot_bull = high.shift(lookback).where(bull_ok).astype("float32")

    # bear: سقف زون = low[i-lookback]، کف زون = high[i]
    fvg_top_bear = low.shift(lookback).where(bear_ok).astype("float32")
    fvg_bot_bear = high.where(bear_ok).astype("float32")

    # --- پرچم‌های نهایی با ضد-لوک‌اِهد -----------------------------
    fvg_up = bull_ok.astype("int8").shift(1).fillna(0).astype("int8")
    fvg_dn = bear_ok.astype("int8").shift(1).fillna(0).astype("int8")

    # ---- تجمیع زون‌ها (در کندل ایجاد؛ در غیر اینصورت NaN) ---------
    # اگر هر دو رخ دهد (نادر)، اولویت‌بندی ساده: bear بر bull یا برعکس.
    # اینجا bull را مقدم می‌گیریم مگر bear هم‌زمان باشد؛ می‌توان سیاست را در آینده تنظیم کرد.
    fvg_top = fvg_top_bull.where(bull_ok, fvg_top_bear)
    fvg_bottom = fvg_bot_bull.where(bull_ok, fvg_bot_bear)

    # فقط روی کندل ایجاد مقدار می‌خواهیم؛ در بقیهٔ کندل‌ها NaN.
    fvg_top = fvg_top.where(bull_ok | bear_ok)
    fvg_bottom = fvg_bottom.where(bull_ok | bear_ok)

    return {
        "fvg_up": fvg_up.astype("int8"),
        "fvg_dn": fvg_dn.astype("int8"),
        "fvg_top": fvg_top.astype("float32"),
        "fvg_bottom": fvg_bottom.astype("float32"),
    }

def make_fvg(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    Factory رجیستری: دریافت DataFrame با ستون‌های 'open','high','low','close'
    و برگرداندن خروجی استاندارد FVG (برای Engine/Registry).

    cfg شامل پارامترهای detect_fvg است:
      - lookback: int
      - atr_window: int
      - min_size_pct_of_atr: float
    """
    # --- اعتبارسنجی ورودی -----------------------------------------
    needed = ("open", "high", "low", "close")
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")
    
    # --- خروجی پایهٔ FVG -------------------------------------------
    out = detect_fvg(df["open"], df["high"], df["low"], df["close"],
                     lookback=int(cfg.get("lookback", 2)),
                     atr_window=int(cfg.get("atr_window", 14)),
                     min_size_pct_of_atr=float(cfg.get("min_size_pct_of_atr", 0.50)))
    # --- lifecycle یک‌روزه (بدون look-ahead) -----------------------
    prev_top = out["fvg_top"].shift(1)
    prev_bot = out["fvg_bottom"].shift(1)
    born = (out["fvg_top"].notna() & out["fvg_bottom"].notna()).astype("int8")
    filled_next = ((df["low"] <= prev_top) & (df["high"] >= prev_bot) & 
                   prev_top.notna() & prev_bot.notna()
                ).astype("int8")
    expired_next = ((prev_top.notna() & prev_bot.notna()) & (~filled_next.astype(bool))).astype("int8")
    
    out.update({"fvg_born": born, "fvg_filled_next": filled_next, "fvg_expired_next": expired_next})

    # --- lifecycle چند-کندلی (گام ۳) -------------------------------
    import numpy as np
    import pandas as pd
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["fvg_top"].shift(k) for k in range(1, N+1)]
    bots = [out["fvg_bottom"].shift(k) for k in range(1, N+1)]
    # وجود زون در پنجرهٔ 1..N ---------------------------------------
    has_zone_cols = [(t.notna() & b.notna()) for t, b in zip(tops, bots)]
    has_zone = pd.concat(has_zone_cols, axis=1).any(axis=1)
    # تاچ در هر کدام از زون‌های پنجرهٔ 1..N -------------------------
    hit_list = [((df["low"] <= t) & (df["high"] >= b)) for t, b in zip(tops, bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    # منقضی‌شدن: اگر زون N-کندل‌پیش در هیچ‌یک از 1..N تاچ نشود ------
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list, axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")
    out.update({"fvg_alive_n": alive_now, "fvg_filled_window": hit_any, "fvg_expired_now": expired_now})

    # --- انتشار زون روی تایم‌فریم هدف برای مصرف ساده در Engine -----
    # انتخاب «نزدیک‌ترین» زون فعال از 1..N (اولویت با جدیدترین: shift(1), shift(2), ...)
    cand_top = pd.concat(tops, axis=1).astype("float32")   # ستون 0: shift(1)
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    # اولین مقدار غیر NaN از چپ: ------------------------------------
    sel_top = cand_top.bfill(axis=1).iloc[:, 0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:, 0]
    # فقط وقتی زون واقعاً فعال است: ---------------------------------
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # --- اندازهٔ زون و نرمال‌سازی با ATR ---------------------------
    from .core import atr as atr_core
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window", 14))).astype("float32")
    gap_size = (active_top - active_bot).abs().astype("float32")
    size_norm = (gap_size / atrv.replace(0, np.nan)).astype("float32")

    # --- برآورد سن زون (تعداد کندل از جدیدترین زون انتخاب‌شده) -----
    # تعیین شاخص ستونی که از آن مقدار گرفته‌ایم (1..N). ستون‌هایی که NaN بودند، به‌صورت bfill پر شده‌اند.
    # برای تخمین «سن»، با مقایسهٔ cand_top با sel_top، اولین ستونی که برابر sel_top است را پیدا می‌کنیم.
    """
    age_cols = []
    for i in range(N):
        age_cols.append((cand_top.iloc[:, i].notna()) & (cand_top.iloc[:, i].eq(sel_top)))
    age_df = pd.concat(age_cols, axis=1)
    # اندیس اولین True از چپ (0..N-1) → +1 = سن برحسب کندل ----------
    age_bars_est = (age_df.idxmax(axis=1).fillna(0).astype("int64") + 1).where(active_top.notna()).astype("float32")
    """
    # قبل از این پچ، cand_top و sel_top ساخته شده‌اند
    # سن زون = شماره ستونی که sel_top از آن آمده (1..N)
    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])  # ستون‌ها را قطعی عددی کن
    cmp = cand_top.eq(sel_top, axis=0)               # تطابق هر ستون با sel_top
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)  # اگر هیچ True نبود → NaN
    age_bars_est = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())

    # --- لمس فعلی و شمارش لمس‌ها در پنجرهٔ N -----------------------
    touch_now = ((df["low"] <= active_top) & (df["high"] >= active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    # به‌عنوان تقریب عملیاتی: مجموع لمس‌ها در پنجرهٔ ثابت N (برای زون فعال)
    touch_count = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    # --- امتیاز زون: ترکیب اندازه/قدمت/تعداد لمس (بدون look-ahead)
    w_size = float(cfg.get("w_size", 0.5))
    w_age  = float(cfg.get("w_age", 0.3))
    w_touch = float(cfg.get("w_touch", 0.2))
    # نرمال‌سازی سادهٔ age/touch ------------------------------------
    age_score = 1.0 / (1.0 + age_bars_est.replace(0, np.nan))   # جدیدتر → امتیاز بیشتر
    touch_score = (touch_count / float(max(1, N)))               # لمس‌های بیشتر (در N) → امتیاز بالاتر
    # size_norm از قبل محاسبه شده است -------------------------------
    score = (w_size * size_norm.fillna(0.0) +
             w_age  * age_score.fillna(0.0) +
             w_touch * touch_score.fillna(0.0)).astype("float32")

    out.update({
        "fvg_active_top": active_top.astype("float32"),
        "fvg_active_bottom": active_bot.astype("float32"),
        "fvg_gap_size": gap_size.fillna(0).astype("float32"),
        "fvg_size_norm": size_norm.fillna(0).astype("float32"),
        "fvg_age_bars_est": age_bars_est.fillna(0).astype("float32"),
        "fvg_touch_now": touch_now.astype("int8"),
        "fvg_touch_count": touch_count.astype("float32"),
        "fvg_score": score.astype("float32"),
    })
    return out


# --- Supply/Demand (SD) (Base → Impulse → Return) ------------------
def detect_sd(open_, high, low, close, *,
              base_len:int=3, atr_window:int=14,
              base_atr_max:float=0.6, impulse_atr_min:float=1.2) -> Dict[str, pd.Series]:
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")
    rng  = (high - low).astype("float32")
    
    ratio = (rng / atrv.replace(0, np.nan)).astype("float32")
    cond = (ratio <= float(base_atr_max))
    L = int(base_len)
    # بیس: چند کندل کم‌نوسان پشت‌سرهم
    base_flag = (cond.rolling(L, min_periods=L).sum() == L).astype("int8")
    
    # ایمپالس: کندل بعد از بیس با رنج بزرگ
    imp = ((rng / atrv.replace(0, np.nan)) >= float(impulse_atr_min)).astype("int8")
    
    # تولد زون: بیس کامل و سپس ایمپالس (shift برای ضد لوک‌اِهد)
    born_bool = base_flag.shift(1).fillna(0).astype(bool) & imp.astype(bool)
    tmp = born_bool.shift(1)
    tmp = tmp.where(tmp.notna(), False)  # به‌جای fillna(False)
    born = tmp.astype("int8")

    # مرزهای زون: سقف/کف بیس
    base_high = high.rolling(int(base_len), min_periods=int(base_len)).max().astype("float32")
    base_low  = low.rolling(int(base_len),  min_periods=int(base_len)).min().astype("float32")
    sd_top = base_high.where(born.astype(bool))
    sd_bot = base_low.where(born.astype(bool))
    return {"sd_born": born, "sd_top": sd_top.astype("float32"), "sd_bottom": sd_bot.astype("float32")}

def make_sd(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    needed = ("open","high","low","close")
    for c in needed:
        if c not in df.columns: raise KeyError(f"Missing column: {c}")
    out = detect_sd(df["open"], df["high"], df["low"], df["close"],
                    base_len=int(cfg.get("base_len",3)),
                    atr_window=int(cfg.get("atr_window",14)),
                    base_atr_max=float(cfg.get("base_atr_max",0.6)),
                    impulse_atr_min=float(cfg.get("impulse_atr_min",1.2)))
    # lifecycle چند-کندلی
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["sd_top"].shift(k) for k in range(1, N+1)]
    bots = [out["sd_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")
    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))
    # اندازه/سن/لمس و امتیاز
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0,np.nan)).astype("float32")
    
    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())
    
    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")
    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0,np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")
    out.update({
        "sd_alive_n": alive_now.astype("int8"),
        "sd_filled_window": hit_any.astype("int8"),
        "sd_expired_now": expired_now.astype("int8"),
        "sd_active_top": active_top.astype("float32"),
        "sd_active_bottom": active_bot.astype("float32"),
        "sd_size_norm": size_norm.fillna(0).astype("float32"),
        "sd_age_bars_est": age_bars.fillna(0).astype("float32"),
        "sd_touch_now": touch_now.astype("int8"),
        "sd_touch_count": touch_cnt.astype("float32"),
        "sd_score": score,
    })
    return out


# --- Order Block (OB) ----------------------------------------------
# تشخیص + lifecycle + انتشار + امتیاز
def detect_ob(open_, high, low, close, *,
              atr_window:int=14,
              body_atr_min:float=0.6,     # حداقل بدنهٔ کندل کاندید نسبت به ATR
              wick_ratio_max:float=0.6,   # حداکثر نسبت ویک به بدنه برای کاندید
              bos_lookback:int=5          # حداکثر فاصلهٔ شکست ساختار بعد از کاندید
             ) -> Dict[str, pd.Series]:
    """
    تشخیص اولیهٔ Order Block:
    - کندل کاندید با بدنهٔ کافی (نسبت به ATR) و ویک‌های محدود.
    - سپس Break of Structure (BOS): قیمت در bos_lookback کندل بعدی، سقف/کف اخیر را می‌شکند.
    - ضد لوک‌اِهد: پرچم تولد با shift(+1) اعمال می‌شود.
    """
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")
    body = (close - open_).astype("float32")
    up = (body > 0)
    dn = (body < 0)
    body_abs = body.abs()
    rng = (high - low).astype("float32")
    wick_sum = (rng - body_abs).clip(lower=0.0)

    # کاندیدهای معتبر: بدنه کافی و ویک محدود
    valid = (body_abs / atrv.replace(0, np.nan) >= float(body_atr_min)) & \
            ((wick_sum / body_abs.replace(0, np.nan)) <= float(wick_ratio_max))

    # شکست ساختار: اگر کندل‌های بعدی، سقف/کف آخرِ قبل از کاندید را بشکنند
    recent_high = high.shift(1).rolling(int(bos_lookback), min_periods=1).max()
    recent_low  = low.shift(1).rolling(int(bos_lookback),  min_periods=1).min()
    bos_up = (close > recent_high)   # شکست به بالا
    bos_dn = (close < recent_low)    # شکست به پایین

    # تولد OB پس از BOS: 
    # - برای OB صعودی معمولاً آخرین کندل نزولی معتبر قبل از BOS
    # - برای OB نزولی آخرین کندل صعودی معتبر قبل از BOS
    # تقریب عملیاتی: در همان کندل BOS، تولد OB با مرزهای کندلِ مخالفِ معتبر قبل از آن
    # برای ضد لوک‌اِهد، پرچم روی کندل بعدی اعمال می‌شود.
    ob_bull_cand = (dn & valid)  # کندل نزولیِ معتبر (پتانسیل OB صعودی)
    ob_bear_cand = (up & valid)  # کندل صعودیِ معتبر (پتانسیل OB نزولی)

    # اندیس آخرین کاندید معتبر قبل از هر کندل
    last_bull_idx = (
        ob_bull_cand.astype("boolean")
        .where(ob_bull_cand, pd.NA)
        .ffill()
        .notna()
    )
    last_bear_idx = (
        ob_bear_cand.astype("boolean")
        .where(ob_bear_cand, pd.NA)
        .ffill()
        .notna()
    )

    born_bull = bos_up.astype("int8").shift(1).fillna(0).astype("int8") & last_bull_idx.astype("int8")
    born_bear = bos_dn.astype("int8").shift(1).fillna(0).astype("int8") & last_bear_idx.astype("int8")

    # مرزهای زون: بدنهٔ کندل کاندید (close/open)؛ به‌صورت تقریبی
    ob_top_bull = open_.where(ob_bull_cand).ffill().where(born_bull.astype(bool))
    ob_bot_bull = close.where(ob_bull_cand).ffill().where(born_bull.astype(bool))

    ob_top_bear = close.where(ob_bear_cand).ffill().where(born_bear.astype(bool))
    ob_bot_bear = open_.where(ob_bear_cand).ffill().where(born_bear.astype(bool))

    # تجمیع یک زون: اگر هر دو رخ دهد، اولویت با ساختارهای همسو با BOS است
    ob_top = ob_top_bull.where(born_bull.astype(bool), ob_top_bear)
    ob_bot = ob_bot_bull.where(born_bull.astype(bool), ob_bot_bear)

    # پرچم تولد نهایی (بدون همپوشانی): 
    born = ((born_bull.astype(bool) | born_bear.astype(bool))).astype("int8")
    return {
        "ob_born": born,
        "ob_top": ob_top.astype("float32"),
        "ob_bottom": ob_bot.astype("float32"),
    }

def make_ob(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    ساخت خروجی Order Block برای Engine/Registry:
    - lifecycle چند-کندلی (alive/filled/expired)
    - انتشار زون فعال (active_top/bottom)
    - اندازه/سن/تعداد لمس و امتیاز نهایی
    پارامترها:
      atr_window, body_atr_min, wick_ratio_max, bos_lookback,
      max_bars_alive, w_size, w_age, w_touch
    """
    need = ("open","high","low","close")
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    out = detect_ob(
        df["open"], df["high"], df["low"], df["close"],
        atr_window=int(cfg.get("atr_window", 14)),
        body_atr_min=float(cfg.get("body_atr_min", 0.6)),
        wick_ratio_max=float(cfg.get("wick_ratio_max", 0.6)),
        bos_lookback=int(cfg.get("bos_lookback", 5)),
    )

    # lifecycle چند-کندلی (هم‌راستا با FVG/SD)
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["ob_top"].shift(k) for k in range(1, N+1)]
    bots = [out["ob_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")

    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # اندازه/سن/لمس و امتیاز (هم‌وزن با FVG/SD برای یکنواختی)
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0, np.nan)).astype("float32")
    
    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())
    
    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0,np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")

    out.update({
        "ob_alive_n": alive_now.astype("int8"),
        "ob_filled_window": hit_any.astype("int8"),
        "ob_expired_now": expired_now.astype("int8"),
        "ob_active_top": active_top.astype("float32"),
        "ob_active_bottom": active_bot.astype("float32"),
        "ob_size_norm": size_norm.fillna(0).astype("float32"),
        "ob_age_bars_est": age_bars.fillna(0).astype("float32"),
        "ob_touch_now": touch_now.astype("int8"),
        "ob_touch_count": touch_cnt.astype("float32"),
        "ob_score": score,
    })
    return out


# --- Liquidity Sweep (EQH/EQL grab + close back inside) ------------
# تشخیص + lifecycle + انتشار + امتیاز
def detect_liq_sweep(open_, high, low, close, *,
                     lookback:int=5,            # پنجرهٔ مرجع برای سقف/کف اخیر
                     atr_window:int=14,
                     min_tail_atr:float=0.5     # حداقل طول ویک نسبت به ATR
                     ) -> Dict[str, pd.Series]:
    """
    تشخیص اولیهٔ Liquidity Sweep:
    - Bearish sweep: high > recent_high  و  close < recent_high  و  ویک بالایی بلند
    - Bullish sweep: low  < recent_low   و  close > recent_low   و  ویک پایینی بلند
    زونِ پایش:
      - برای bearish: [recent_high , high]
      - برای bullish: [low , recent_low]
    ضدّ لوک‌اِهد: پرچم «تولد» با shift(+1) اعمال می‌شود.
    """
    atrv = atr_core(high, low, close, n=int(atr_window)).astype("float32")

    # سقف/کف اخیر (تا کندل قبل)
    recent_high = high.shift(1).rolling(int(lookback), min_periods=1).max().astype("float32")
    recent_low  = low.shift(1).rolling(int(lookback),  min_periods=1).min().astype("float32")

    # ویک‌ها
    upper_tail = (high - np.maximum(open_, close)).clip(lower=0.0).astype("float32")
    lower_tail = (np.minimum(open_, close) - low).clip(lower=0.0).astype("float32")

    # شرط طول ویک نسبت به ATR
    tail_ok_up = (upper_tail / atrv.replace(0, np.nan) >= float(min_tail_atr))
    tail_ok_dn = (lower_tail / atrv.replace(0, np.nan) >= float(min_tail_atr))

    # الگوی سوئیپ
    bearish_raw = (high > recent_high) & (close < recent_high) & tail_ok_up
    bullish_raw = (low  < recent_low)  & (close > recent_low)  & tail_ok_dn

    # زون‌های سوئیپ (روی همان کندل رخداد)
    ls_top_bear = high.where(bearish_raw).astype("float32")
    ls_bot_bear = recent_high.where(bearish_raw).astype("float32")

    ls_top_bull = recent_low.where(bullish_raw).astype("float32")
    ls_bot_bull = low.where(bullish_raw).astype("float32")

    # پرچم تولد با ضدّ لوک‌اِهد
    born_bear = bearish_raw.astype("int8").shift(1).fillna(0).astype("int8")
    born_bull = bullish_raw.astype("int8").shift(1).fillna(0).astype("int8")

    # تجمیع به یک زون (اگر هر دو نادر اتفاق افتاد، اولویت ساده: bear سپس bull)
    ls_top = ls_top_bear.where(bearish_raw, ls_top_bull)
    ls_bot = ls_bot_bear.where(bearish_raw, ls_bot_bull)
    born   = ((bearish_raw | bullish_raw).astype("int8").shift(1).fillna(0)).astype("int8")

    return {
        "ls_born": born,
        "ls_top":  ls_top.astype("float32"),
        "ls_bottom": ls_bot.astype("float32"),
    }

def make_liq_sweep(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    ساخت خروجی Liquidity Sweep برای Engine/Registry:
    - lifecycle چند-کندلی (alive/filled/expired)
    - انتشار زون فعال (active_top/bottom)
    - اندازه/سن/تعداد لمس و امتیاز نهایی (سازگار با FVG/SD/OB)
    پارامترها: lookback, atr_window, min_tail_atr, max_bars_alive, w_size/w_age/w_touch
    """
    need = ("open","high","low","close")
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    out = detect_liq_sweep(
        df["open"], df["high"], df["low"], df["close"],
        lookback=int(cfg.get("lookback", 5)),
        atr_window=int(cfg.get("atr_window", 14)),
        min_tail_atr=float(cfg.get("min_tail_atr", 0.5)),
    )

    # lifecycle چند-کندلی (هم‌راستا با بقیه)
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["ls_top"].shift(k) for k in range(1, N+1)]
    bots = [out["ls_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")

    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # اندازه/سن/لمس و امتیاز
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0, np.nan)).astype("float32")

    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())

    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0, np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")

    out.update({
        "ls_alive_n": alive_now.astype("int8"),
        "ls_filled_window": hit_any.astype("int8"),
        "ls_expired_now": expired_now.astype("int8"),
        "ls_active_top": active_top.astype("float32"),
        "ls_active_bottom": active_bot.astype("float32"),
        "ls_size_norm": size_norm.fillna(0).astype("float32"),
        "ls_age_bars_est": age_bars.fillna(0).astype("float32"),
        "ls_touch_now": touch_now.astype("int8"),
        "ls_touch_count": touch_cnt.astype("float32"),
        "ls_score": score,
    })
    return out


# --- Breaker / Flip Zone (Invalidated OB → Reverse Zone) -----------
def detect_breaker_flip(open_, high, low, close, *,
                        atr_window:int=14,
                        ob_body_atr_min:float=0.6,
                        ob_wick_ratio_max:float=0.6,
                        ob_bos_lookback:int=5,
                        lookback:int=5) -> Dict[str, pd.Series]:
    """
    تشخیص اولیهٔ Breaker/Flip:
    - ابتدا OBها را استخراج می‌کنیم (همین فایل: detect_ob)؛
    - اگر در 1..lookback کندلِ بعدی، قیمت «کامل» از بالای TOP (یا از زیر BOTTOM) OB عبور کند،
      همان کندلِ عبور، Breaker/Flip می‌سازد.
    - زون Breaker بدنهٔ همان کندلِ عبور است (body-top/body-bottom).
    - ضدّ لوک‌اِهد: پرچم تولد با shift(+1) اعمال می‌شود.
    """
    # OBهای اخیر
    ob = detect_ob(open_, high, low, close,
                   atr_window=atr_window,
                   body_atr_min=ob_body_atr_min,
                   wick_ratio_max=ob_wick_ratio_max,
                   bos_lookback=ob_bos_lookback)
    ob_top = ob["ob_top"]
    ob_bot = ob["ob_bottom"]

    # عبور از زون‌های OB در پنجرهٔ 1..L
    L = int(lookback)
    tops = [ob_top.shift(k) for k in range(1, L+1)]
    bots = [ob_bot.shift(k) for k in range(1, L+1)]

    inv_up_cols = [ (close > t) & t.notna() for t in tops ]   # عبور به بالا
    inv_dn_cols = [ (close < b) & b.notna() for b in bots ]   # عبور به پایین
    inv_up = pd.concat(inv_up_cols, axis=1).any(axis=1)
    inv_dn = pd.concat(inv_dn_cols, axis=1).any(axis=1)

    # زون Breaker = بدنهٔ کندل عبور
    body_top = np.maximum(open_, close).astype("float32")
    body_bot = np.minimum(open_, close).astype("float32")

    bf_top_up = body_top.where(inv_up)
    bf_bot_up = body_bot.where(inv_up)
    bf_top_dn = body_top.where(inv_dn)
    bf_bot_dn = body_bot.where(inv_dn)

    # تجمیع به یک زون (اولویت با inv_up)
    bf_top = bf_top_up.where(inv_up, bf_top_dn)
    bf_bot = bf_bot_up.where(inv_up, bf_bot_dn)

    born = ((inv_up | inv_dn).astype("int8").shift(1).fillna(0)).astype("int8")
    return {
        "bf_born": born,
        "bf_top":  bf_top.astype("float32"),
        "bf_bottom": bf_bot.astype("float32"),
    }

def make_breaker_flip(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    Breaker/Flip برای Engine/Registry:
    - lifecycle چند-کندلی (alive/filled/expired)
    - انتشار زون فعال (active_top/bottom)
    - اندازه/قدمت/تعداد لمس و امتیاز نهایی
    پارامترها:
      atr_window, ob_body_atr_min, ob_wick_ratio_max, ob_bos_lookback, lookback,
      max_bars_alive, w_size, w_age, w_touch
    """
    need = ("open","high","low","close")
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    out = detect_breaker_flip(
        df["open"], df["high"], df["low"], df["close"],
        atr_window=int(cfg.get("atr_window", 14)),
        ob_body_atr_min=float(cfg.get("ob_body_atr_min", 0.6)),
        ob_wick_ratio_max=float(cfg.get("ob_wick_ratio_max", 0.6)),
        ob_bos_lookback=int(cfg.get("ob_bos_lookback", 5)),
        lookback=int(cfg.get("lookback", 5)),
    )

    # lifecycle چند-کندلی (هم‌راستا با FVG/SD/OB)
    N = int(cfg.get("max_bars_alive", 3))
    tops = [out["bf_top"].shift(k) for k in range(1, N+1)]
    bots = [out["bf_bottom"].shift(k) for k in range(1, N+1)]
    has_zone = pd.concat([(t.notna() & b.notna()) for t,b in zip(tops,bots)], axis=1).any(axis=1)
    hit_list = [((df["low"]<=t) & (df["high"]>=b)) for t,b in zip(tops,bots)]
    hit_any = pd.concat(hit_list, axis=1).any(axis=1).astype("int8")
    expired_now = ((tops[-1].notna() & bots[-1].notna()) & (~pd.concat(hit_list,axis=1).any(axis=1))).astype("int8")
    alive_now = (has_zone & (~hit_any.astype(bool))).astype("int8")

    # انتشار زون فعال
    cand_top = pd.concat(tops, axis=1).astype("float32")
    cand_bot = pd.concat(bots, axis=1).astype("float32")
    sel_top = cand_top.bfill(axis=1).iloc[:,0]
    sel_bot = cand_bot.bfill(axis=1).iloc[:,0]
    active_top = sel_top.where(alive_now.astype(bool))
    active_bot = sel_bot.where(alive_now.astype(bool))

    # اندازه/سن/لمس و امتیاز
    atrv = atr_core(df["high"], df["low"], df["close"], n=int(cfg.get("atr_window",14))).astype("float32")
    size = (active_top - active_bot).abs().astype("float32")
    size_norm = (size / atrv.replace(0, np.nan)).astype("float32")

    cand_top = cand_top.copy()
    cand_top.columns = np.arange(cand_top.shape[1])
    cmp = cand_top.eq(sel_top, axis=0)
    any_true = cmp.any(axis=1).to_numpy()
    idx = np.where(any_true, cmp.to_numpy().argmax(axis=1) + 1, np.nan)
    age_bars = pd.Series(idx, index=df.index, dtype="float32").where(active_top.notna())

    touch_now = ((df["low"]<=active_top) & (df["high"]>=active_bot) & active_top.notna() & active_bot.notna()).astype("int8")
    touch_cnt = touch_now.rolling(N, min_periods=1).sum().where(active_top.notna()).fillna(0).astype("float32")

    w_size=float(cfg.get("w_size",0.5))
    w_age=float(cfg.get("w_age",0.3))
    w_touch=float(cfg.get("w_touch",0.2))
    age_score = 1.0/(1.0 + age_bars.replace(0, np.nan))
    touch_score = (touch_cnt/float(max(1,N)))
    score = (w_size*size_norm.fillna(0)+w_age*age_score.fillna(0)+w_touch*touch_score.fillna(0)).astype("float32")

    out.update({
        "bf_alive_n": alive_now.astype("int8"),
        "bf_filled_window": hit_any.astype("int8"),
        "bf_expired_now": expired_now.astype("int8"),
        "bf_active_top": active_top.astype("float32"),
        "bf_active_bottom": active_bot.astype("float32"),
        "bf_size_norm": size_norm.fillna(0).astype("float32"),
        "bf_age_bars_est": age_bars.fillna(0).astype("float32"),
        "bf_touch_now": touch_now.astype("int8"),
        "bf_touch_count": touch_cnt.astype("float32"),
        "bf_score": score,
    })
    return out


# --- SR Fusion (FVG + SD + OB + BF + LS) ---------------------------
# Fair Value Gap, Supply/Demand, Order Block, Breaker/Flip, Liquidity Sweep
def make_sr_fusion(df: pd.DataFrame, **cfg) -> Dict[str, pd.Series]:
    """
    فیوژن 5 گانهٔ SR با کنترل پایداری:
      - EMA smoothing + age decay                 EMA زمانی روی امتیازها + decay برحسب سن زون
      - hysteresis (enter/exit)                   Hysteresis با آستانه‌ی ورود/خروج (enter/exit)
      - optional MTF confirmation (confirm_col)   MTF confirmation اختیاری با ستون/فلگ ورودی (confirm_col)
      - tie-break (تازگی/فعال‌بودن)               Tie-break & conflict (برنده = بالاترین امتیاز؛ در برابری: زون تازه‌تر/Active)
      - cooldown پس از filled/expired             Cooldown پس از filled/expired (خاموش‌کردن سهم مؤلفه طی cooldown_bars)
      - min-confidence                            Min-confidence (زیر حداقل، خروجی صفر)
    پارامترها (نمونهٔ پیش‌فرض):
      ema_span=5, age_norm=5, enter_th=0.55, exit_th=0.35,
      cooldown_bars=3, min_conf=0.30, confirm_col=None
      max_bars_alive=3, atr_window=14, w_size=0.5, w_age=0.3, w_touch=0.2
    چند نکته:
    - برای MTF، اگر ستونی برای تایید داری (مثلاً از TF بالاتر)، 
      نامش را با confirm_col بده تا فقط در همان نقاط، فیوژن فعال شود.
    - هرجا خواستی وزن‌دهی/آستانه‌ها را تغییر بدهی، پارامترها در cfg هستند.
    
    """
    # 1) محاسبهٔ مؤلفه‌ها (با همان سازوکار داخلیِ همین فایل)
    fvg = make_fvg(df, **cfg)
    sd  = make_sd(df,  **cfg)
    ob  = make_ob(df,  **cfg)
    bf  = make_breaker_flip(df, **cfg)
    ls  = make_liq_sweep(df, **cfg)

    # 2) استخراج امتیاز و متادیتا
    comps = [
        ("fvg", fvg, "fvg_score", "fvg_active_top", "fvg_active_bottom", "fvg_age_bars_est", "fvg_alive_n", "fvg_filled_window", "fvg_expired_now"),
        ("sd",  sd,  "sd_score",  "sd_active_top",  "sd_active_bottom",  "sd_age_bars_est",  "sd_alive_n",  "sd_filled_window",  "sd_expired_now"),
        ("ob",  ob,  "ob_score",  "ob_active_top",  "ob_active_bottom",  "ob_age_bars_est",  "ob_alive_n",  "ob_filled_window",  "ob_expired_now"),
        ("bf",  bf,  "bf_score",  "bf_active_top",  "bf_active_bottom",  "bf_age_bars_est",  "bf_alive_n",  "bf_filled_window",  "bf_expired_now"),
        ("ls",  ls,  "ls_score",  "ls_active_top",  "ls_active_bottom",  "ls_age_bars_est",  "ls_alive_n",  "ls_filled_window",  "ls_expired_now"),
    ]

    ema_span   = int(cfg.get("ema_span", 5))
    age_norm   = float(cfg.get("age_norm", 5.0))
    enter_th   = float(cfg.get("enter_th", 0.55))
    exit_th    = float(cfg.get("exit_th", 0.35))
    cooldown_k = int(cfg.get("cooldown_bars", 3))
    min_conf   = float(cfg.get("min_conf", 0.30))
    confirm_col = cfg.get("confirm_col", None)  # نام ستونی در df که True/1 = تایید MTF

    eff_scores = {}
    tops_map, bots_map, ages_map, alive_map = {}, {}, {}, {}
    cool_map = {}  # rolling filled/expired

    for key, dct, s_score, s_top, s_bot, s_age, s_alive, s_filled, s_expired in comps:
        sc  = dct.get(s_score,  pd.Series(0.0, index=df.index, dtype="float32")).astype("float32")
        top = dct.get(s_top,    pd.Series(np.nan, index=df.index, dtype="float32")).astype("float32")
        bot = dct.get(s_bot,    pd.Series(np.nan, index=df.index, dtype="float32")).astype("float32")
        age = dct.get(s_age,    pd.Series(0.0, index=df.index, dtype="float32")).astype("float32")
        liv = dct.get(s_alive,  pd.Series(0, index=df.index, dtype="int8")).astype("int8")
        fil = dct.get(s_filled, pd.Series(0, index=df.index, dtype="int8")).astype("int8")
        exp = dct.get(s_expired,pd.Series(0, index=df.index, dtype="int8")).astype("int8")

        # EMA smoothing + age decay + alive mask
        sc_ema = sc.ewm(span=ema_span, adjust=False, min_periods=1).mean().astype("float32")
        decay  = (1.0 / (1.0 + (age / max(1e-6, age_norm)))).astype("float32")
        alive_mask = liv.astype(bool)
        sc_eff = (sc_ema * decay).where(alive_mask, 0.0)

        # cooldown: اگر اخیرأ filled/expired شده، سهم صفر
        cool = (fil.astype(bool) | exp.astype(bool)).rolling(cooldown_k, min_periods=1).max().fillna(0).astype("int8")
        sc_eff = sc_eff.where(~cool.astype(bool), 0.0)

        eff_scores[key] = sc_eff.astype("float32")
        tops_map[key] = top
        bots_map[key] = bot
        ages_map[key] = age
        alive_map[key] = liv
        cool_map[key] = cool

    # 3) MTF confirmation (اختیاری)
    if confirm_col is not None and confirm_col in df.columns:
        conf = df[confirm_col].astype("int8").astype(bool)
        for k in list(eff_scores.keys()):
            eff_scores[k] = eff_scores[k].where(conf, 0.0)

    # 4) انتخاب برنده و tie-break (تازگی/فعال بودن)
    eff_df = pd.DataFrame(eff_scores).astype("float32")
    sr_score_smooth = eff_df.max(axis=1).astype("float32")
    winner = eff_df.idxmax(axis=1).astype("string")

    # tie-break: اگر چند مؤلفه تقریباً برابرند (در محدودهٔ 0.02)، تازه‌تر/active ترجیح داده شود
    eps = float(cfg.get("tie_eps", 0.02))
    for i in range(len(winner)):
        candidates = [k for k in eff_df.columns if abs(eff_df.iloc[i][k] - sr_score_smooth.iloc[i]) <= eps]
        if len(candidates) > 1:
            # تازه‌تر را انتخاب کن: سن کمتر
            sel = min(candidates, key=lambda k: (ages_map[k].iloc[i] if not np.isnan(ages_map[k].iloc[i]) else 1e9))
            # اگر غیر فعال بود، سعی کن فعال‌ترین را بگیری
            if not bool(alive_map[sel].iloc[i]):
                actives = [k for k in candidates if bool(alive_map[k].iloc[i])]
                if actives:
                    sel = min(actives, key=lambda k: (ages_map[k].iloc[i] if not np.isnan(ages_map[k].iloc[i]) else 1e9))
            winner.iloc[i] = sel

    # 5) فعال‌سازی hysteresis (enter/exit) + min-confidence
    sr_state = pd.Series(0, index=df.index, dtype="int8")
    sr_score = pd.Series(0.0, index=df.index, dtype="float32")
    sr_top   = pd.Series(np.nan, index=df.index, dtype="float32")
    sr_bot   = pd.Series(np.nan, index=df.index, dtype="float32")

    for i in range(len(df)):
        s = sr_score_smooth.iloc[i]
        if sr_state.iloc[i-1] if i>0 else 0:
            # حالت on → بمان مگر زیر exit یا زیر min_conf
            if (s < exit_th) or (s < min_conf):
                sr_state.iloc[i] = 0
            else:
                sr_state.iloc[i] = 1
        else:
            # حالت off → فقط اگر عبور از enter و min_conf
            if (s >= enter_th) and (s >= min_conf):
                sr_state.iloc[i] = 1
            else:
                sr_state.iloc[i] = 0

        if sr_state.iloc[i]:
            w = winner.iloc[i]
            sr_score.iloc[i] = s
            sr_top.iloc[i]   = tops_map[w].iloc[i]
            sr_bot.iloc[i]   = bots_map[w].iloc[i]
        # else: صفر/NaN باقی می‌ماند

    return {
        "sr_score_raw": sr_score_smooth.astype("float32"),
        "sr_score": sr_score.astype("float32"),
        "sr_active_top": sr_top.astype("float32"),
        "sr_active_bottom": sr_bot.astype("float32"),
        "sr_on": sr_state.astype("int8"),
        "sr_source": winner.astype("string"),
    }

