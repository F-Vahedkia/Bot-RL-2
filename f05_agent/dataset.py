# -*- coding: utf-8 -*-
"""
FeatureDataset — Bot-RL-2 (Phase E)
- بارگذاری Feature Store (Parquet/CSV) و متادیتا
- ساخت پنجرهٔ مشاهدات (state) بدون لوک‌اِهد
- انتخاب ستون‌های فیچر (prefix='__') و ستون‌های پایهٔ قیمت (OHLCV)
- API ساده برای پیمایش گام‌به‌گام/مینی‌بچ

یادداشت معماری:
- فقط به f04_features.* تکیه می‌کند؛ بیرون از f01..f14 چیزی import نمی‌شود.
- پیام‌های runtime انگلیسی هستند؛ توضیحات فارسی در کامنت‌ها.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    """پیکربندی پایه برای Dataset."""
    window: int = 64                 # طول پنجرهٔ مشاهدات
    feature_prefix: str = "__"       # ستون‌های فیچر با این پیشوند شروع می‌شوند
    price_cols: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    adr_indicator_name: str = "adr"  # برای کشف ستون ADR از متادیتا
    adr_key_contains: str = "adr_"   # کلید داخلی ستون ADR (مثلاً adr_14)
    min_coverage: float = 0.5        # حداقل پوشش پس از warmup برای پذیرش ستون


class FeatureDataset:
    """
    کلاسی برای مدیریت دادهٔ فیچرها جهت RL، با تضمین عدم لوک‌اِهد.
    - state_t: از ردیف‌های [t-window+1 .. t] ساخته می‌شود.
    - reward_t: از حرکت قیمتِ [t -> t+1] ساخته می‌شود (محاسبه بیرون از این کلاس هم ممکن است).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        meta_path: Optional[Union[str, Path]] = None,
        cfg: DatasetConfig = DatasetConfig(),
    ):
        self.cfg = cfg
        self.data_path = Path(data_path)
        self.meta_path = Path(meta_path) if meta_path else None

        # --- بارگذاری داده ---
        if self.data_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix.lower() == ".csv":
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported data file format: {self.data_path}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Feature data must have a DatetimeIndex.")

        self.df: pd.DataFrame = df

        # --- بارگذاری متادیتا (اختیاری اما توصیه‌شده) ---
        self.meta: Optional[pd.DataFrame] = None
        if self.meta_path:
            if self.meta_path.suffix.lower().endswith(".csv"):
                self.meta = pd.read_csv(self.meta_path)
            elif self.meta_path.suffix.lower().endswith(".json"):
                self.meta = pd.read_json(self.meta_path)
            else:
                raise ValueError(f"Unsupported meta file format: {self.meta_path}")

        # --- کشف ستون‌های فیچر و قیمت ---
        self.price_cols: List[str] = [c for c in self.cfg.price_cols if c in self.df.columns]
        self.feature_cols: List[str] = [c for c in self.df.columns if c.startswith(self.cfg.feature_prefix)]

        # اگر متادیتا داریم، ستون‌هایی با پوشش کم را حذف می‌کنیم (برای پایداری RL)
        if self.meta is not None and {"column", "coverage_ratio"}.issubset(self.meta.columns):
            ok_set = set(
                self.meta.loc[self.meta["coverage_ratio"] >= float(self.cfg.min_coverage), "column"].tolist()
            )
            self.feature_cols = [c for c in self.feature_cols if c in ok_set]

        # --- ایندکس‌های معتبر برای شروع اپیزود/گام ---
        self.window: int = int(self.cfg.window)
        self.t_min: int = self.window - 1
        self.t_max: int = len(self.df) - 2  # t+1 باید وجود داشته باشد (برای reward آینده)

        if self.t_max <= self.t_min:
            raise RuntimeError("Not enough rows for the requested window; increase data or reduce window.")

        # --- کشف ستون ADR برای نرمال‌سازی (اختیاری) ---
        self.adr_cols: List[str] = []
        if self.meta is not None and {"column", "indicator", "key"}.issubset(self.meta.columns):
            _m = self.meta
            mask = (_m["indicator"] == self.cfg.adr_indicator_name) & (_m["column"].astype(str).str.contains(self.cfg.adr_key_contains))
            self.adr_cols = _m.loc[mask, "column"].tolist()

    # ====================== API ======================

    def get_state(self, t: int, use_prices: bool = True) -> np.ndarray:
        """
        استخراج state در زمان t به‌صورت پنجرهٔ [t-window+1 .. t] بدون لوک‌اِهد.
        خروجی: ndarray با شکل [L, F] که L=window و F=تعداد ویژگی‌ها (+ قیمت‌ها در صورت نیاز)
        """
        if not (self.t_min <= t <= self.t_max):
            raise IndexError(f"t out of range: {t} not in [{self.t_min}, {self.t_max}]")

        cols = (self.price_cols if use_prices else []) + self.feature_cols
        # --- PATCH: از ایندکس‌های «برچسبی» استفاده کن؛ عدد صحیح روی DatetimeIndex خطا می‌دهد ---
        left_label = self.df.index[t - self.window + 1]
        right_label = self.df.index[t]
        sl = self.df.loc[left_label:right_label, cols]
        # -----------------------------------------------------------------------
        return sl.to_numpy(dtype=float)

    def get_prices(self, t: int) -> Dict[str, float]:
        """بازگرداندن OHLCV در زمان t (برای محاسبهٔ reward/PNL خارج از کلاس)."""
        row = self.df.iloc[t]
        return {k: float(row[k]) for k in self.price_cols if k in row}

    def get_adr_value(self, t: int, prefer: Optional[str] = None) -> Optional[float]:
        """برگرداندن مقدار ADR در زمان t (اگر موجود باشد)."""
        if not self.adr_cols:
            return None
        col = prefer if (prefer in self.adr_cols) else self.adr_cols[0]
        v = self.df.iloc[t][col]
        return float(v) if np.isfinite(v) else None

    def iter_range(self, t_start: Optional[int] = None, t_end: Optional[int] = None):
        """تکرارگر سادهٔ t برای آموزش/ارزیابی بدون لوک‌اِهد."""
        i0 = self.t_min if t_start is None else max(self.t_min, int(t_start))
        i1 = self.t_max if t_end is None else min(self.t_max, int(t_end))
        for t in range(i0, i1 + 1):
            yield t
