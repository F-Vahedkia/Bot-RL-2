# f04_features/indicators/engine.py
# # -*- coding: utf-8 -*-
"""IndicatorEngine: اجرای Specها روی دیتای MTF و تولید DataFrame ویژگی‌ها"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from .utils import detect_timeframes, slice_tf, nan_guard
from .parser import parse_spec

@dataclass
class EngineConfig:
    specs: List[str]                 # مثل: ["rsi(14)@M1", "ema(close,20)@M5", ...]
    timeframes: Optional[List[str]] = None  # اگر None، همهٔ TFهای موجود
    shift_all: int = 1               # شیفت عمومی جهت جلوگیری از look-ahead
    drop_na_head: bool = True        # حذف NaNهای ابتدای سری‌ها

class IndicatorEngine:
    def __init__(self, cfg: EngineConfig, registry: Dict[str, Any]):
        self.cfg = cfg
        self.registry = registry      # نگاشت نام → تابع سازندهٔ Series map

    def _available_tfs(self, df: pd.DataFrame) -> List[str]:
        views = detect_timeframes(df)
        if self.cfg.timeframes:
            return [tf for tf in self.cfg.timeframes if tf in views]
        return list(views.keys())

    def _apply_one(self, base_df: pd.DataFrame, tf: str, spec_str: str) -> Dict[str, pd.Series]:
        """
        اجرای یک Spec روی یک تایم‌فریم مشخص.
        - اگر آرگومان اول Spec نامِ ستون باشد (open/high/low/close/volume)،
        فقط وقتی پاس می‌دهیم که تابع رجیستری پارامتر «col» داشته باشد؛
        در این صورت «col» را به شکلِ positional قبل از بقیهٔ آرگومان‌ها می‌فرستیم
        تا با عدد بعدی (مثلاً period) تداخل «multiple values for 'col'» پیش نیاید.
        - اگر تابع «col» نداشت، آن توکن ستونی را نادیده می‌گیریم (تابع خودش از df[...] می‌خواند).
        - خروجی‌ها با پیشوند TF برگردانده می‌شوند.
        """
        import inspect  # فقط برای تشخیص پارامترهای تابع (سبک و بی‌هزینه)

        spec = parse_spec(spec_str)
        if spec.name not in self.registry:
            raise ValueError(f"Unknown indicator: {spec.name} (spec={spec_str})")

        views = detect_timeframes(base_df)
        if tf not in views:
            raise KeyError(f"Timeframe {tf} not found in DataFrame. (spec={spec_str})")

        # برشِ دیتای مربوط به TF و استانداردسازی نام ستون‌ها به open/high/low/close/volume
        sdf = slice_tf(base_df, views[tf])

        fn = self.registry[spec.name]
        fn_params = list(inspect.signature(fn).parameters.keys())

        # آماده‌سازی آرگومان‌ها از روی Spec
        args = list(spec.args)  # کپی ایمن
        col_token = None
        if args and isinstance(args[0], str) and args[0].lower() in {"open", "high", "low", "close", "volume"}:
            col_token = args.pop(0).lower()

        # استراتژی پاس‌دادن آرگومان‌ها:
        # 1) اگر تابع پارامتر «col» دارد و کاربر ستون داده، col را positional قبل از بقیه args می‌فرستیم.
        # 2) اگر «col» در امضای تابع نیست، col_token را نادیده می‌گیریم (تابع خودش از df[...] استفاده می‌کند).
        if col_token is not None and "col" in fn_params:
            out_map: Dict[str, pd.Series] = fn(sdf, col_token, *args)
        else:
            out_map: Dict[str, pd.Series] = fn(sdf, *args)

        # پیشونددهی نام ستون‌های خروجی با TF مقصد
        prefixed = {f"{tf}__{k}": v for k, v in out_map.items()}
        return prefixed


    def apply_all(self, df: pd.DataFrame, base_tf_fallback: Optional[str] = None) -> pd.DataFrame:
        tfs_all = self._available_tfs(df)
        if not tfs_all:
            raise ValueError("No timeframe columns detected.")
        base_tf = base_tf_fallback or tfs_all[0]
        features: Dict[str, pd.Series] = {}
        for spec_str in self.cfg.specs:
            tf = parse_spec(spec_str).tf or base_tf
            vals = self._apply_one(df, tf, spec_str)
            features.update(vals)
        feat_df = pd.DataFrame(features, index=df.index).sort_index()
        if self.cfg.shift_all and self.cfg.shift_all > 0:
            feat_df = feat_df.shift(self.cfg.shift_all)
        feat_df = nan_guard(feat_df)
        if self.cfg.drop_na_head:
            feat_df = feat_df.dropna(how="all")
        return feat_df