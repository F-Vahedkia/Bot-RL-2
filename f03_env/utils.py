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

# --- مسیر پروژه و فایل‌ها ---

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../f03_env → ریشهٔ پروژه

'''
def paths_from_cfg_old1(cfg: Dict[str, Any]) -> Dict[str, Path]:
    p = cfg.get("paths", {}) or {}
    processed = _project_root() / (p.get("processed_dir") or "f02_data/processed")
    cache = _project_root() / (p.get("cache_dir") or "f02_cache")
    processed.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return {"processed": processed, "cache": cache}
'''
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