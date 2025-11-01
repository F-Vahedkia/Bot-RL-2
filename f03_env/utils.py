## f03_env/utils.py


# -*- coding: utf-8 -*-
"""
ابزارهای Env:
- بارگذاری دیتای پردازش‌شده (M1 + فیچرها) و فیلتر زمانی train/val/test
- انتخاب ستون‌های مشاهده (Observation)
- نرمال‌سازی (fit روی train، apply روی val/test) + ذخیره/بارگذاری اسکالر
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import pandas as pd

from f10_utils.config_loader import load_config
from f10_utils.config_ops import _deep_get
# --- مسیر پروژه و فایل‌ها ---

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../f03_env → ریشهٔ پروژه

def paths_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """
    خواندن مسیرها از هر دو ساختار:
      - cfg["project"]["paths"][...]
      - cfg["paths"][...]
    و برگرداندن دیکتی با کلیدهای حداقل: processed, cache, models_dir.
    """
    def _p(key: str, default: str) -> Path:
        p_proj = ((cfg.get("project") or {}).get("paths") or {})
        p_flat = (cfg.get("paths") or {})
        raw = p_proj.get(key) or p_flat.get(key) or default
        return _project_root() / raw

    processed = _p("processed_dir", "f02_data/processed")
    cache     = _p("cache_dir",     "f02_cache")
    models    = _p("models_dir",    "f12_models")

    for d in (processed, cache, models):
        d.mkdir(parents=True, exist_ok=True)

    return {"processed": processed, "cache": cache, "models_dir": models}



# --- بارگذاری دیتای پردازش‌شده ---

def read_processed(symbol: str, base_tf: str, cfg: Dict[str, Any], fmt: Optional[str] = None) -> pd.DataFrame:
    paths = paths_from_cfg(cfg)
    sym_dir = (paths["processed"] / symbol.upper())
    pq = sym_dir / f"{base_tf.upper()}.parquet"
    cs = sym_dir / f"{base_tf.upper()}.csv"
    if fmt is None:
        if pq.exists():
            df = pd.read_parquet(pq)
        elif cs.exists():
            df = pd.read_csv(cs, parse_dates=["time"], index_col="time")
        else:
            raise FileNotFoundError(f"Processed file not found: {pq} or {cs}")
    else:
        if fmt.lower() == "parquet":
            df = pd.read_parquet(pq)
        else:
            df = pd.read_csv(cs, parse_dates=["time"], index_col="time")

    # تضمین ایندکس زمانی و مرتب‌سازی
    if df.index.name != "time":
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.set_index("time")
        else:
            raise ValueError("Processed DataFrame must have time index or 'time' column")
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df

# --- برش‌های زمانی (train/val/test) ---

def time_slices(cfg: Dict[str, Any]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    env_cfg = (cfg.get("env") or {})
    split = env_cfg.get("split") or {}
    def _pair(x):
        if not x: return None
        a, b = pd.to_datetime(x[0]), pd.to_datetime(x[1])
        return (a.tz_localize("UTC"), b.tz_localize("UTC"))
    out = {}
    for k in ("train", "val", "test"):
        p = _pair(split.get(k)) if split.get(k) else None
        if p: out[k] = p
    return out


def slice_df_by_range(df: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df.index >= a) & (df.index <= b)].copy()


# --- Split by ratios + session alignment (NY 17:00) ---
def _tf_to_minutes(base_tf: str) -> int:
    s = str(base_tf).upper()
    return int(s[1:]) if s.startswith("M") else (int(s[1:])*60 if s.startswith("H") else (int(s[1:])*1440 if s.startswith("D") else 1))

def _snap_right_to_session(ts: pd.Timestamp, align: Dict[str, Any], tf_mins: int) -> pd.Timestamp:
    tz = (align or {}).get("tz") or "America/New_York"
    hhmm = (align or {}).get("hhmm") or "17:00"
    hh, mm = map(int, hhmm.split(":"))
    loc = ts.tz_convert(tz)
    tgt = loc.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if tgt < loc: tgt = tgt + pd.Timedelta(days=1)
    if tf_mins > 1:
        rem = tgt.minute % tf_mins
        if rem: tgt = tgt + pd.Timedelta(minutes=(tf_mins - rem))
    return tgt.tz_convert("UTC")

def build_slices_from_ratios(idx: pd.DatetimeIndex, cfg: Dict[str, Any]) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    env_cfg = (cfg.get("env") or {})
    split = (env_cfg.get("split") or {})
    ratios = (split.get("ratios") or {})
    try:
        r_tr = float(ratios.get("train", 0)); r_va = float(ratios.get("val", 0)); r_te = float(ratios.get("test", 0))
    except Exception:
        return {}
    if (r_tr <= 0) or (r_va <= 0) or (r_te <= 0) or (len(idx) < 3): return {}
    n = len(idx); i1 = max(1, int(n * r_tr)); i2 = max(i1 + 1, int(n * (r_tr + r_va)))
    # مرزهای اولیه (بین i-1 و i)
    t1 = idx[min(i1, n-1)]; t2 = idx[min(i2, n-1)]
    tf_mins = _tf_to_minutes((env_cfg.get("base_tf") or "M1"))
    align = (split.get("align") or {"tz":"America/New_York","hhmm":"17:00"})
    # اسنپ هر مرز به 17:00 نیویورک (لبهٔ کندل/سشن)
    t1a = _snap_right_to_session(pd.Timestamp(t1).tz_convert("UTC"), align, tf_mins)
    t2a = _snap_right_to_session(pd.Timestamp(t2).tz_convert("UTC"), align, tf_mins)
    # اندیس‌های هم‌تراز روی دیتاست
    j1 = int(idx.searchsorted(t1a, side="left")); j2 = int(idx.searchsorted(t2a, side="left"))
    j1 = max(1, min(j1, n-1)); j2 = max(j1+1, min(j2, n-1))
    
    # clamp to safe bounds
    n = len(idx)
    j1 = max(0, min(j1, n - 2))
    j2 = max(1, min(j2, n - 1))
    if j1 >= j2:
        j1 = max(0, j2 - 1)
    
    return {
        "train": (idx[0],        idx[j1-1]),
        "val":   (idx[j1],       idx[j2-1]),
        "test":  (idx[j2],       idx[-1]),
    }


# --- انتخاب ستون‌های مشاهده ---

@dataclass
class FeatureSelect:
    include_ohlc: bool = True
    include_spread: bool = False
    include_volume: bool = True
    whitelist: Optional[List[str]] = None  # لیست کامل ستون‌ها یا الگوهای ساده (contains)
    blacklist: Optional[List[str]] = None


def infer_observation_columns(df: pd.DataFrame, sel: FeatureSelect) -> List[str]:
    cols: List[str] = []
    if sel.include_ohlc:
        # ستون‌های قیمت خام (همهٔ TFها)
        cols += [c for c in df.columns if c.endswith("_open") or c.endswith("_high") or c.endswith("_low") or c.endswith("_close")]
    if sel.include_volume:
        cols += [c for c in df.columns if c.endswith("_tick_volume") or c.endswith("_volume")]
    if sel.include_spread:
        cols += [c for c in df.columns if c.endswith("_spread")]
    # همهٔ فیچرهای اندیکاتوری (TF__...)
    cols += [c for c in df.columns if "__" in c]
    # یکتا
    cols = list(dict.fromkeys(cols))

    def _apply_list(patterns: Optional[List[str]], base: List[str], keep: bool) -> List[str]:
        if not patterns: return base
        out = []
        for c in base:
            ok = any(pat in c for pat in patterns)
            if (ok and keep) or ((not ok) and (not keep)):
                out.append(c)
        return out

    if sel.whitelist:
        cols = _apply_list(sel.whitelist, cols, keep=True)
    if sel.blacklist:
        cols = _apply_list(sel.blacklist, cols, keep=False)
    return cols

# --- نرمال‌سازی (fit روی train) ---

@dataclass
class Scaler:
    mean_: Dict[str, float]
    std_: Dict[str, float]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            m = self.mean_.get(c, 0.0)
            s = self.std_.get(c, 1.0)
            if s == 0 or np.isclose(s, 0):
                out[c] = 0.0
            else:
                out[c] = (out[c] - m) / s
        return out.astype("float32")


def fit_scaler(df: pd.DataFrame, cols: List[str]) -> Scaler:
    sub = df[cols].astype("float32")
    mean_ = sub.mean(axis=0).to_dict()
    std_ = sub.std(axis=0).replace(0, np.nan).fillna(1.0).to_dict()
    return Scaler(mean_=mean_, std_=std_)


def save_scaler(scaler: Scaler, cache_dir: Path, tag: str) -> Path:
    obj = {"mean_": scaler.mean_, "std_": scaler.std_}
    out = cache_dir / f"scaler_{tag}.json"
    out.write_text(json.dumps(obj, ensure_ascii=False))
    return out


def load_scaler(cache_dir: Path, tag: str) -> Optional[Scaler]:
    p = cache_dir / f"scaler_{tag}.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    return Scaler(mean_=obj.get("mean_", {}), std_=obj.get("std_", {}))

# -------------------------------------
# توسط دو تابع زیر:
# خروجی «گزارش Split + Warmup» (اجرای خودکار هنگام ساخت Env)
# -------------------------------------
def split_summary_path(symbol: str, base_tf: str, cfg: Dict[str, Any]) -> Path:
    """
    مسیر فایل گزارش Split برای یک (symbol, base_tf) را برمی‌گرداند.
    خروجی در کنار processed ذخیره می‌شود تا با نسخهٔ داده هم‌مکان باشد.
    """
    paths = paths_from_cfg(cfg)
    sym_dir = (paths["processed"] / symbol.upper())
    sym_dir.mkdir(parents=True, exist_ok=True)
    return sym_dir / f"{base_tf.upper()}.split.json"


def write_split_summary(idx: pd.DatetimeIndex,
                        slices: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]],
                        warmup_bars: int,
                        symbol: str,
                        base_tf: str,
                        cfg: Dict[str, Any]) -> Path:
    """
    یک گزارش JSON از مرزهای train/val/test (پس از اسنپ)، شمار کندل‌ها،
    و مقدار warmup تولید می‌کند. این گزارش برای QA و تکرارپذیری استفاده می‌شود.
    """
    # -----------------------------
    # محاسبهٔ طول هر بخش روی ایندکس
    # -----------------------------
    a1, b1 = slices.get("train", (idx[0], idx[-1]))
    a2, b2 = slices.get("val",   (idx[0], idx[-1]))
    a3, b3 = slices.get("test",  (idx[0], idx[-1]))

    def _count(a, b):
        return int(idx.searchsorted(b, "right") - idx.searchsorted(a, "left"))

    # نقطهٔ شروع مؤثرِ val/test بعد از warmup (فقط برای QA و اطلاع)
    def _effective_start(a):
        pos = int(idx.searchsorted(a, side="left"))
        pos0 = max(0, pos - int(warmup_bars))
        if pos - pos0 < 1:
            pos0 = max(0, pos - 1)
        return pd.Timestamp(idx[pos0]).tz_convert("UTC").isoformat()

    obj = {
        "symbol": symbol.upper(),
        "base_tf": base_tf.upper(),
        "warmup_bars": int(warmup_bars),
        "train": {"start": pd.Timestamp(a1).tz_convert("UTC").isoformat(),
                  "end":   pd.Timestamp(b1).tz_convert("UTC").isoformat(),
                  "bars":  _count(a1, b1)},
        "val":   {"start": pd.Timestamp(a2).tz_convert("UTC").isoformat(),
                  "end":   pd.Timestamp(b2).tz_convert("UTC").isoformat(),
                  "bars":  _count(a2, b2),
                  "effective_start_after_warmup": _effective_start(a2)},
        "test":  {"start": pd.Timestamp(a3).tz_convert("UTC").isoformat(),
                  "end":   pd.Timestamp(b3).tz_convert("UTC").isoformat(),
                  "bars":  _count(a3, b3),
                  "effective_start_after_warmup": _effective_start(a3)},
    }

    out = split_summary_path(symbol, base_tf, cfg)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# اولویت اسپرد: سری بروکر ← کانفیگ costs.spread_pts ← صفر
def resolve_spread_selection(cfg, df=None, has_broker_series=False):
    bt = _deep_get(cfg, "evaluation.backtest", {})
    trc = _deep_get(cfg, "trading.costs", {})
    col = None
    if (has_broker_series or bool(bt.get("reuse_historical_spread"))) and df is not None:
        cols = [c for c in df.columns if ("spread" in c.lower()) and df[c].notna().any()]
        col = cols[0] if cols else None
    if col:
        return {"use_series": True, "column": col, "value": 0.0, "source": "broker_data"}
        
    v = max(float(trc.get("spread_pts", 0.0)), 0.0)
    src = "config_costs" if "spread_pts" in trc else "zero_default"
    return {"use_series": False, "value": v, "source": src}
