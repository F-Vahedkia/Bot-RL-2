# -*- coding: utf-8 -*-
"""
Feature Store (Phase D) — Bot-RL-2
- ساخت فریم فیچر از روی Spec/DSL با استفاده از هستهٔ indicators.engine
- استخراج متادیتای هر ستون فیچر (indicator, tf, key, source, dtype, warmup, coverage)
- ذخیرهٔ داده و متادیتا (Parquet/CSV + sidecar JSON/CSV)

نکات معماری (مطابق الزامات پروژه):
- این فایل در f04_features قرار دارد و فقط از ماژول‌های داخل همین فولدر استفاده می‌کند.
- هیچ import از خارجِ f01_…f14 انجام نشده است.
- پیام‌های runtime انگلیسی هستند؛ توضیحات فارسی در کامنت‌ها آمده است.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# هستهٔ اندیکاتورها و رجیستری (داخل f04_features)
from f04_features.indicators.engine import apply as indicators_apply
from f04_features.indicators.registry import list_all_indicators_v2

logger = logging.getLogger(__name__)


# ============================== ابزار پارس نام ستون‌ها ==============================
# الگوی نام‌گذاری ستون‌ها طبق موتور: __<name>@<TF>__<key>
_FEATURE_COL_RE = re.compile(r"^__(?P<ind>[a-zA-Z0-9_]+)@(?P<tf>[A-Z]\d+)__(?P<key>[a-zA-Z0-9_]+)$")


@dataclass
class FeatureMeta:
    """متادیتای هر ستون فیچر در Feature Store."""
    column: str
    indicator: str                 # نام اندیکاتور (از prefix)
    tf: str                        # تایم‌فریم از نام ستون (مثل M1, M5, D1, …)
    key: str                       # کلید داخلیِ اندیکاتور (suffix)
    source: str                    # 'advanced' یا 'legacy' یا 'base'
    dtype: str                     # dtype پانداس/نام نوع
    first_valid_ts: Optional[str]  # زمان اولین مقدار معتبر (ISO8601)
    warmup: int                    # تعداد NaN/Inf قبل از اولین مقدار معتبر
    coverage_ratio: float          # نسبت نقاط معتبر بعد از warmup به کل (0..1)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    اطمینان از UTC بودن ایندکس زمانی (بدون حدس: اگر tz ندارد، به UTC محلی‌سازی می‌شود).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DatetimeIndex.")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    return df


def _parse_feature_column(col: str) -> Optional[Tuple[str, str, str]]:
    """
    پارس نام ستون فیچر مطابق قرارداد موتور: '__<name>@<TF>__<key>'
    خروجی: (indicator, tf, key) یا None اگر مطابق الگو نبود.
    """
    m = _FEATURE_COL_RE.match(col)
    if not m:
        return None
    return (m.group("ind"), m.group("tf"), m.group("key"))


def _source_lookup_advanced() -> Dict[str, str]:
    """
    از رجیستری v2 برچسب advanced را استخراج می‌کند.
    خروجی: map[indicator] = 'advanced'  (برای دیگر نام‌ها به‌صورت پیش‌فرض 'legacy' استفاده می‌شود)
    """
    # list_all_indicators_v2(False) فقط advanced را برمی‌گرداند.
    return {k: "advanced" for k, tag in list_all_indicators_v2(include_legacy=False).items() if tag == "advanced"}


def _nan_inf_mask(s: pd.Series) -> np.ndarray:
    """برگرداندن ماسک non-finite برای یک سری عددی."""
    arr = s.to_numpy()
    return ~np.isfinite(arr)


def _first_valid_and_warmup(s: pd.Series) -> Tuple[Optional[pd.Timestamp], int]:
    """
    پیدا کردن اولین زمان معتبر و تعداد non-finite قبل از آن (warmup).
    """
    fv = s.first_valid_index()
    if fv is None:
        return None, len(s)
    # تعداد non-finite از ابتدای سری تا قبل از fv
    warmup = int((_nan_inf_mask(s.loc[:fv]).sum()) - (0 if pd.notna(s.loc[fv]) else 1))
    return fv, max(warmup, 0)


def _coverage_after_warmup(s: pd.Series) -> float:
    """
    نسبت نقاط معتبر بعد از warmup به کل طول سری.
    """
    fv = s.first_valid_index()
    if fv is None:
        return 0.0
    tail = s.loc[fv:]
    valid = int(np.isfinite(tail.to_numpy(dtype=float)).sum())
    total = int(len(s))
    return float(valid) / float(total) if total else 0.0


# ============================== هستهٔ ساخت و ذخیرهٔ Feature Store ==============================

def build_feature_frame(df: pd.DataFrame, specs: List[str]) -> pd.DataFrame:
    """
    ساخت فریم فیچر با استفاده از engine.apply و Spec/DSL.
    - df: OHLCV پایه با DatetimeIndex (نرخ دقیقه/…)
    - specs: لیست Specها مطابق DSL (مثلاً: "sma(20)@M1", "rsi(14)@M5", "adr(window=14)@D1", …)
    خروجی: DataFrame شامل ستون‌های ورودی + ستون‌های فیچر استاندارد.
    """
    # تضمین UTC برای سازگاری با موتور
    df = _ensure_utc_index(df)
    out = indicators_apply(df=df, specs=specs)
    if not isinstance(out, pd.DataFrame):
        raise TypeError("Engine apply did not return a DataFrame.")
    logger.info("Built feature frame with %d rows and %d columns.", len(out), len(out.columns))
    return out


def extract_metadata(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    استخراج متادیتا برای هر ستون فیچر.
    - فقط ستون‌هایی که با '__' شروع می‌شوند به عنوان فیچر شمارش می‌گردند.
    """
    adv_map = _source_lookup_advanced()  # نام اندیکاتورهای advanced
    rows: List[FeatureMeta] = []

    for col in df_feat.columns:
        if not col.startswith("__"):
            # ستون‌های base (OHLCV/…)؛ در صورت نیاز می‌توان ثبت کرد. فعلاً صرفاً عبور.
            continue

        parsed = _parse_feature_column(col)
        if not parsed:
            # اگر طبق قرارداد نبود، به‌عنوان legacy در نظر گرفته می‌شود.
            ind, tf, key = ("unknown", "M1", col.strip("_"))
        else:
            ind, tf, key = parsed

        src = adv_map.get(ind, "legacy")
        s = df_feat[col]

        fv, warm = _first_valid_and_warmup(s)
        cov = _coverage_after_warmup(s)

        meta = FeatureMeta(
            column=col,
            indicator=ind,
            tf=tf,
            key=key,
            source=src,
            dtype=str(s.dtype),
            first_valid_ts=(None if fv is None else fv.isoformat()),
            warmup=warm,
            coverage_ratio=float(round(cov, 6)),
        )
        rows.append(meta)

    meta_df = pd.DataFrame([asdict(r) for r in rows]).sort_values(["indicator", "tf", "key"]).reset_index(drop=True)
    logger.info("Extracted metadata for %d feature columns.", len(meta_df))
    return meta_df


def save_feature_store(
    df_feat: pd.DataFrame,
    meta_df: pd.DataFrame,
    out_dir: str | Path,
    base_name: str,
    data_format: str = "parquet",
    compression: Optional[str] = "snappy",
    write_index: bool = True,
) -> Dict[str, str]:
    """
    ذخیرهٔ دیتافریم فیچر و متادیتا در دیسک.
    - out_dir: مسیر پوشهٔ خروجی (ایجاد می‌شود اگر وجود نداشت)
    - base_name: پیشوند نام فایل‌ها (مثلاً 'XAUUSD_2022Q1')
    - data_format: 'parquet' یا 'csv'
    - compression: الگوریتم فشرده‌سازی پارکت (snappy/zstd/…)، برای CSV نادیده گرفته می‌شود
    - write_index: ذخیرهٔ ایندکس زمانی
    خروجی: مسیر فایل‌های تولیدشده
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    # ذخیرهٔ داده
    if data_format.lower() == "parquet":
        data_path = out / f"{base_name}.parquet"
        df_feat.to_parquet(data_path, compression=compression, index=write_index)
        paths["data"] = str(data_path)
    elif data_format.lower() == "csv":
        data_path = out / f"{base_name}.csv"
        df_feat.to_csv(data_path, index=write_index)
        paths["data"] = str(data_path)
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")

    # ذخیرهٔ متادیتا (CSV + JSON sidecar)
    meta_csv = out / f"{base_name}.meta.csv"
    meta_json = out / f"{base_name}.meta.json"
    meta_df.to_csv(meta_csv, index=False)
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

    paths["meta_csv"] = str(meta_csv)
    paths["meta_json"] = str(meta_json)

    logger.info("Feature store saved: data=%s, meta_csv=%s, meta_json=%s", paths["data"], paths["meta_csv"], paths["meta_json"])
    return paths


def build_and_save_feature_store(
    df: pd.DataFrame,
    specs: List[str],
    out_dir: str | Path,
    base_name: str,
    data_format: str = "parquet",
    compression: Optional[str] = "snappy",
) -> Dict[str, str]:
    """
    تابع ترکیبی: ساخت فریم فیچر ← استخراج متادیتا ← ذخیرهٔ هر دو.
    مناسب برای استفاده در اسکریپت‌های Batch تولید Feature Store.
    """
    feat = build_feature_frame(df=df, specs=specs)
    meta = extract_metadata(feat)
    return save_feature_store(
        df_feat=feat,
        meta_df=meta,
        out_dir=out_dir,
        base_name=base_name,
        data_format=data_format,
        compression=compression,
    )
