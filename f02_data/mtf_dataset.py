# f02_data/mtf_dataset.py
# Reviewed at 1405/04/28

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List
import pandas as pd

# =============================================================================
# MAIN CLASS
# =============================================================================
@dataclass
class MTFDataset:
    """
    نگهدارنده‌ی دیتافریم‌های چند تایم‌فریم.
    هیچ همترازسازی انجام نمی‌دهد.
    هر تایم‌فریم دیتافریم مستقل خودش را دارد.
    """
    symbol: str
    base_tf: str

    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # --------------------------------------------------------------- 1=
    def add(self, timeframe: str, df: pd.DataFrame) -> None:
        self.frames[timeframe.upper()] = df

    # --------------------------------------------------------------- 2=
    def get(self, timeframe: str) -> pd.DataFrame:
        return self.frames[timeframe.upper()]

    # --------------------------------------------------------------- 3=
    def __getitem__(self, timeframe: str) -> pd.DataFrame:
        """
        توسط این متد عبارت df=dataset["M5"] همانند عبارت df=dataset.get("M5") عمل میکند
        """
        return self.get(timeframe)

    # --------------------------------------------------------------- 4=
    def replace(self, timeframe: str, df: pd.DataFrame) -> None:
        """
        جایگزین کردن دیتافریم یک تایم‌فریم.

        اگر تایم‌فریم قبلاً وجود نداشته باشد، خطا می‌دهد تا
        جایگزینی اشتباه با add اشتباه گرفته نشود.
        """
        timeframe = timeframe.upper()

        if timeframe not in self.frames:
            raise KeyError(f"Timeframe '{timeframe}' does not exist.")

        self.frames[timeframe] = df

    # --------------------------------------------------------------- 5=
    def copy(self) -> "MTFDataset":     # Forward Reference
        out = MTFDataset(
            symbol=self.symbol,
            base_tf=self.base_tf,
        )

        for tf, df in self.frames.items():
            out.frames[tf] = df.copy()

        return out

    # --------------------------------------------------------------- 6=
    @property
    def timeframes(self) -> List[str]:
        return list(self.frames.keys())
    
    # --------------------------------------------------------------- 7=
    def apply(
        self,
        func: Callable[..., pd.DataFrame],
        *args: Any,
        **kwargs: Any
    ) -> None:
        """
        روشهای  استفاده:
        dataset.apply(_add_time_features_to_df, self.cfg)
        dataset.apply(compute_indicators)
        dataset.apply(build_price_action)
        dataset.apply(build_patterns)
        """
        for tf in self.frames:
            self.frames[tf] = func(
                self.frames[tf],
                *args,
                **kwargs
            )

    # --------------------------------------------------------------- 8=
    def apply_each(
        self,
        func: Callable[..., pd.DataFrame],
        *args: Any,
        **kwargs: Any
    ) -> "MTFDataset":
        """
        اعمال یک تابع روی تمام تایم‌فریم‌ها.

        تابع باید دیتافریم و نام تایم‌فریم را دریافت کرده 
        و یک دیتافریم جدید برگرداند.

        امضا:
            func(df, timeframe, *args, **kwargs) -> pd.DataFrame

        مثال:
            dataset.apply_each(
                _finalize_dataframe,
                self.cfg,
                symbol,
            )
        """
        for tf in self.frames:
            self.frames[tf] = func(
                self.frames[tf],
                tf,
                *args,
                **kwargs
            )
        return self

# ============================================================================= END






""" آموزشی:
-----------
     @property
    عبارت @property یک Decorator در پایتون است که باعث می‌شود یک متد،
    از بیرون کلاس مثل یک متغیر (Attribute) رفتار کند، نه مثل یک تابع.
    بنابراین میتوانیم بجای 
        .timeframe()
    به سادگی از 
        .timeframe
    استفاده کنیم و دیگر نیازی به پرانتزهای نشان دهنده تابع یا متد نداریم
    
"""