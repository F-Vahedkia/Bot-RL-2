# -*- coding: utf-8 -*-
# f03_features/feature_store.py
# Status in (Bot-RL-2): Completed

"""
Run: python -m f03_features.feature_store --symbol XAUUSD --base-tf M1

Feature Store (Phase D) — Bot-RL-2
- ساخت فریم فیچر از روی Spec/DSL با استفاده از هستهٔ indicators.engine
- استخراج متادیتای هر ستون فیچر (indicator, tf, key, source, dtype, warmup, coverage)
- ذخیرهٔ داده و متادیتا (Parquet/CSV + sidecar JSON/CSV)

نکات معماری (مطابق الزامات پروژه):
- این فایل در f03_features قرار دارد و فقط از ماژول‌های داخل همین فولدر استفاده می‌کند.
- هیچ import از خارجِ f01_…f14 انجام نشده است.
- پیام‌های runtime انگلیسی هستند؛ توضیحات فارسی در کامنت‌ها آمده است.
"""
# =============================================================================
# Imports & Logger
# =============================================================================
from __future__ import annotations

import json
import logging
import argparse
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from f04_env.utils import read_processed, paths_from_cfg

# هستهٔ اندیکاتورها و رجیستری (داخل f03_features)
from f03_features.feature_engine import apply as indicators_apply
from f03_features.feature_registry import list_all_indicators

# ثبت بیلدرهای Price Action در رجیستری واحد اندیکاتورها (بدون لوپ ایمپورت)
import f03_features.feature_bootstrap  # noqa: F401

from f10_utils.config_loader import load_config


logger = logging.getLogger(__name__)


# ============================== ابزار پارس نام ستون‌ها ==============================
# الگوی نام‌گذاری ستون‌ها طبق موتور: __<name>@<TF>__<key>
_FEATURE_COL_RE = re.compile(r"^__(?P<ind>[a-zA-Z0-9_]+)@(?P<tf>[A-Z]\d+)__(?P<key>[a-zA-Z0-9_]+)$")

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
def _filter_specs_by_registry(specs: List[str]) -> List[str]:
    """
    فیلتر کردن Spec هایی که نام اندیکاتورشان در رجیستری v2 وجود ندارد.
    این تابع هیچ حدسی نمی‌زند؛ فقط بر اساس خروجی list_all_indicators عمل می‌کند.
    """
    reg = list_all_indicators(include_legacy=True)
    known = set(reg.keys())
    kept: List[str] = []
    dropped: List[str] = []

    for raw in specs:
        name = re.split(r"[(@]", str(raw), 1)[0].strip()
        if not name:
            continue
        if name in known:
            kept.append(raw)
        else:
            dropped.append(str(raw))

    if dropped:
        logger.warning(
            "Dropping %d specs not present in unified REGISTRY: %s",
            len(dropped),
            ", ".join(dropped),
        )
    return kept

# -------------------------------------------------------------------
def _write_debug_column_lists(
    df_processed: pd.DataFrame,
    df_feat: pd.DataFrame,
    symbol: str,
    base_tf: str,
    ind_specs: List[str],
    pa_specs: List[str],
    out_root: Path = Path("f15_testcheck") / "_MyFiles",
) -> None:
    """
    نوشتن سه فایل CSV برای مشاهدهٔ نام ستون‌ها:
      - ستون‌های دیتافریم processed بعد از publish
      - ستون‌های فیچر مربوط به اندیکاتورها
      - ستون‌های فیچر مربوط به price_action
    نام ستون‌ها مستقیماً از خود دیتافریم‌ها استخراج می‌شود.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    sym_u = symbol.upper()
    tf_u = base_tf.upper()

    ### --- Added_1
    logger.info(
        "Debug columns: starting for symbol=%s tf=%s (processed_cols=%d, feat_cols=%d, out_root=%s)",
        sym_u, tf_u, len(df_processed.columns), len(df_feat.columns), out_root)


    # 1) همهٔ ستون‌های processed (بعد از publish)
    pd.DataFrame({"column": list(df_processed.columns)}).to_csv(
        out_root / f"columns_processed_{sym_u}_{tf_u}.csv",
        index=False,
    )

    # 2) ستون‌های فیچر (هر چیزی که با '__' شروع شود)
    feat_cols = [c for c in df_feat.columns if c.startswith("__")]

    def _base_name(spec: str) -> str:
        return re.split(r"[(@]", str(spec), 1)[0].strip()

    ind_names = {_base_name(s) for s in (ind_specs or [])}
    pa_names = {_base_name(s) for s in (pa_specs or [])}

    ind_cols: List[str] = []
    pa_cols: List[str] = []

    ### --- Added_1
    logger.info("Debug columns: indicator specs=%s | price_action specs=%s",
        list(ind_specs or []), list(pa_specs or []) )

    for c in feat_cols:
        parsed = _parse_feature_column(c)
        if not parsed:
            ### --- Added_1
            logger.debug("Debug columns: skipped feature column (pattern mismatch): %s", c)
            continue
        ind, tf, key = parsed
        name = ind

        ### --- Added_1
        logger.debug("Debug columns: parsed feature col=%s -> (ind=%s, tf=%s, key=%s), in_ind=%s, in_pa=%s",
            c, ind, tf, key, name in ind_names, name in pa_names)
        
        if name in ind_names:
            ind_cols.append(c)
        if name in pa_names:
            pa_cols.append(c)

    ### --- Added_1
    logger.info("Debug columns: final counts -> indicators=%d, price_action=%d",
        len(ind_cols), len(pa_cols))
    
    pd.DataFrame({"column": ind_cols}).to_csv(
        out_root / f"columns_indicators_{sym_u}_{tf_u}.csv",
        index=False,
    )
    pd.DataFrame({"column": pa_cols}).to_csv(
        out_root / f"columns_price_action_{sym_u}_{tf_u}.csv",
        index=False,
    )

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
def _parse_feature_column(col: str) -> Optional[Tuple[str, str, str]]:
    """
    پارس نام ستون فیچر مطابق قرارداد موتور: '__<name>@<TF>__<key>'
    خروجی: (indicator, tf, key) یا None اگر مطابق الگو نبود.
    """
    m = _FEATURE_COL_RE.match(col)
    if not m:
        return None
    return (m.group("ind"), m.group("tf"), m.group("key"))

# -------------------------------------------------------------------
def _source_lookup_advanced() -> Dict[str, str]:
    """
    از رجیستری v2 برچسب advanced را استخراج می‌کند.
    خروجی: map[indicator] = 'advanced'  (برای دیگر نام‌ها به‌صورت پیش‌فرض 'legacy' استفاده می‌شود)
    """
    # list_all_indicators(False) فقط advanced را برمی‌گرداند.
    return {k: "advanced" for k, tag in list_all_indicators(include_legacy=False).items() if tag == "advanced"}

# -------------------------------------------------------------------
def _nan_inf_mask(s: pd.Series) -> np.ndarray:
    """برگرداندن ماسک non-finite برای یک سری عددی."""
    arr = s.to_numpy()
    return ~np.isfinite(arr)

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
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
# -------------------------------------------------------------------
def build_feature_frame(df: pd.DataFrame, specs: List[str]) -> pd.DataFrame:
    """
    ساخت فریم فیچر با استفاده از engine.apply و Spec/DSL.
    - df: OHLCV پایه با DatetimeIndex (نرخ دقیقه/…)
    - specs: لیست Specها مطابق DSL (مثلاً: "sma(20)@M1", "rsi(14)@M5", "adr(window=14)@D1", …)
    خروجی: DataFrame شامل ستون‌های ورودی + ستون‌های فیچر استاندارد.
    """
    # تضمین UTC برای سازگاری با موتور
    df = _ensure_utc_index(df)

    # اگر هیچ ستون حجمی نداریم، اندیکاتورهای حجمی را از specs حذف می‌کنیم
    has_vol = any(c.endswith(("volume", "tick_volume")) or c in ("volume", "tick_volume") for c in df.columns)
    if not has_vol:
        vol_inds = {"vwap", "vwap_roll", "adl", "cmf"}
        dropped = [s for s in specs if re.split(r"[(@]", str(s), 1)[0].strip() in vol_inds]
        specs = [s for s in specs if s not in dropped]
        if dropped:
            logger.warning("No volume/tick_volume columns; dropping volume-based indicators: %s",
                           ", ".join(map(str, dropped)))

    """
    # Persian: اگر ستون volume نداریم، از tick_volume به عنوان alias استفاده می‌کنیم
    if "volume" not in df.columns:
        # ستون‌های *_tick_volume (مثل M1_tick_volume) و در صورت وجود tick_volume ساده
        candidates = [c for c in df.columns if c.endswith("tick_volume")] + [
            c for c in ("tick_volume",) if c in df.columns
        ]
        if candidates:
            df = df.copy()
            df["volume"] = df[candidates[0]]
    """
    out = indicators_apply(df=df, specs=specs)
    if not isinstance(out, pd.DataFrame):
        raise TypeError("Engine apply did not return a DataFrame.")
    logger.info("Built feature frame with %d rows and %d columns.", len(out), len(out.columns))
    return out

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
def build_and_save_feature_store(
    df: pd.DataFrame,
    specs: List[str],
    out_dir: str | Path,
    base_name: str,
    data_format: str = "parquet",
    compression: Optional[str] = "snappy",
    *, cfg=None, symbol=None, base_tf=None, publish=False
) -> Dict[str, str]:
    """
    تابع ترکیبی: ساخت فریم فیچر ← استخراج متادیتا ← ذخیرهٔ هر دو.
    مناسب برای استفاده در اسکریپت‌های Batch تولید Feature Store.
    """
    feat = build_feature_frame(df=df, specs=specs)
    meta = extract_metadata(feat)
    paths = save_feature_store(df_feat=feat,
                               meta_df=meta,
                               out_dir=out_dir,
                               base_name=base_name,
                               data_format=data_format,
                               compression=compression)
    
    if publish and (cfg is not None):
        d,e = (cfg.get("data") or {}), (cfg.get("env") or {})
        sym = d.get("symbol") or (d.get("symbols") or [None])[0]
        if sym and (e.get("base_tf")):
            proc = paths_from_cfg(cfg)["processed"] / str(sym).upper() / f"{str(e['base_tf']).upper()}.parquet"
            publish_to_processed(cfg, paths["data"], proc)
    return paths

# -------------------------------------------------------------------
def publish_to_processed(cfg, store_parquet, processed_parquet):
    df_p = pd.read_parquet(processed_parquet)
    df_f = pd.read_parquet(store_parquet)
    df_f = df_f.loc[df_p.index]  # هم‌ترازی روی ایندکسِ مشترک
    cols_new = [c for c in df_f.columns if c not in df_p.columns]
    if not cols_new:
        logging.getLogger(__name__).info("No new feature columns to publish"); return 0
    df_p.join(df_f[cols_new]).to_parquet(processed_parquet, index=True)
    return len(cols_new)

# -------------------------------------------------------------------
def ensure_published_from_cfg(cfg, symbol: str, base_tf: str) -> int:
    """
    ساخت Feature Store از روی config و تزریق به processed.
    خروجی: تعداد ستون‌های جدیدی که به دیتافریم processed اضافه شده است.
    """

    lg = logging.getLogger(__name__)

    # --- استخراج Spec ها از config (بدون هیچ حدس اضافی) ---
    feat_cfg = (cfg.get("features") or {})
    ind_specs: List[str] = list(feat_cfg.get("indicators") or [])
    pa_specs: List[str] = list(feat_cfg.get("price_action") or [])
    specs_raw: List[str] = ind_specs + pa_specs

    if not specs_raw:
        lg.info("Auto-publish skipped: no feature specs (indicators/price_action) in config.")
        return 0

    # --- فیلتر کردن Specها بر اساس رجیستری واقعی ---
    specs = _filter_specs_by_registry(specs_raw)
    if not specs:
        lg.info("Auto-publish skipped: no valid specs after registry filter.")
        return 0

    sym_u = symbol.upper()
    tf_u = base_tf.upper()

    # --- خواندن دیتافریم processed فعلی ---
    df_proc = read_processed(sym_u, tf_u, cfg)

    # --- ساخت و ذخیرهٔ Feature Store از روی df_proc و specs معتبر ---
    out_paths = build_and_save_feature_store(
        df=df_proc,
        specs=specs,
        out_dir=Path("f15_testcheck/_reports/feature_store"),
        base_name=f"{sym_u}_{tf_u}",
    )

    # --- publish به فایل processed اصلی ---
    processed_pq = (paths_from_cfg(cfg)["processed"] / sym_u / f"{tf_u}.parquet")
    n = int(publish_to_processed(cfg, out_paths["data"], processed_pq))
    lg.info("Auto-publish merged %d columns into processed (symbol=%s, tf=%s).",
            n, sym_u, tf_u)

    # --- نوشتن ۳ فایل CSV دیباگ بر اساس خود دیتافریم‌ها ---
    try:
        ### --- Added_1
        lg.info("Debug columns: loading feature and processed frames for CSV dump (data=%s, processed=%s)",
            out_paths["data"], processed_pq)
        

        df_feat = pd.read_parquet(out_paths["data"])
        df_proc_after = pd.read_parquet(processed_pq)
        _write_debug_column_lists(
            df_processed=df_proc_after,
            df_feat=df_feat,
            symbol=sym_u,
            base_tf=tf_u,
            ind_specs=ind_specs,
            pa_specs=pa_specs,
        )
        ### --- Added_1
        lg.info("Debug columns: CSV column dump finished successfully.")
    except Exception:
        lg.exception("Failed to write debug column CSVs.")

    return n


# -------------------------------------------------------------------
def main() -> int:
    """اجرای مستقل publish فیچرها بر اساس config."""
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--base-tf", required=True, dest="base_tf")
    args = p.parse_args()
    cfg = load_config()
    n = ensure_published_from_cfg(cfg, args.symbol, args.base_tf)
    logging.getLogger(__name__).info(
        "Feature publish finished: %d new columns (symbol=%s, tf=%s).",
        n, args.symbol.upper(), args.base_tf.upper(),
    )
    return 0

# -------------------------------------------------------------------
if __name__ == "__main__":  # اجرای مستقل Stage-B
    
    ### --- Added_1
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--base-tf", required=True, dest="base_tf")
    
    a = ap.parse_args()
    cfg = load_config()
    ensure_published_from_cfg(cfg, a.symbol, a.base_tf)
    print(f"[FeatureStore] done for symbol={a.symbol} base_tf={a.base_tf}")

# -------------------------------------------------------------------
