from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import inspect
import logging

from f03_features.feature_B_registry_1 import get_indicator, IndicatorSpec
from f10_utils.parser import parse_spec, ParsedSpec
from f02_data.data_handler_F_2 import _TF_MINUTES
from f02_data.mtf_dataset import MTFDataset

logger = logging.getLogger(__name__)

class FeatureEngine:

    # -------------------------------------------------------- 0
    def __init__(self, config: Dict[str, Any]):

        self._live_cache: Dict[str, Any] = {}   # نگهداری نمونه‌های Stateful (SMA, RSI, Supertrend, ...)
        self.mode: Optional[str] = None  # در execute مقداردهی می‌شود (train/live) و در متدهای دیگر برای تشخیص حالت استفاده می‌شود.
        self._run_id: int = 0
        if config is None:
            raise ValueError("config is required for FeatureEngine")
        self.config = config  # 👈 برای خواندن لیست مشخصات از کانفیگ

    # -------------------------------------------------------- 1
    def execute_old1(self, df: pd.DataFrame, specs: List[str], mode: str = "train") -> pd.DataFrame:
        """
        نقطه ورود اصلی برای محاسبه فیچرها.
        
        پارامترها:
            df: دیتافریم ورودی حاوی داده‌های خام (OHLCV و سایر ستون‌ها)
            specs: لیست رشته‌های مشخصات فیچرها (مثلاً ['sma(20)@H1', 'rsi(14)@M1'])
            mode: حالت اجرا ('train' یا 'live')
        
        خروجی:
            دیتافریم با ستون‌های فیچر اضافه‌شده
        
        عملکرد:
            1. پارس کردن همه مشخصات با استفاده از parse_spec
            2. ساختن گراف وابستگی بین فیچرها
            3. تعیین ترتیب اجرا بر اساس وابستگی‌ها
            4. اجرای فیچرها به ترتیب (هر فیچر روی خروجی قبلی اعمال می‌شود)
            5. در حالت train، کش live ریست می‌شود تا اجراها مستقل باشند
        """
        if df is None or df.empty:
            logger.info("df is empty")
            return df
        
        self.mode = mode
        self._run_id += 1

        # HARD RESET: isolate runs (critical for reproducibility)
        if mode == "train":
            self._live_cache = {}

        result_df = df.copy()
        parsed_specs: List[ParsedSpec] = []

        # --- 1. PARSE ALL SPECS
        for spec in specs:
            try:
                parsed_specs.append(parse_spec(spec))
            except Exception as e:
                logger.warning("Invalid spec skipped: %s | %s", spec, e)
        logger.info("✅ Parsed %d feature specs: %s", len(parsed_specs), [ps.raw for ps in parsed_specs])

        # --- 2. BUILD DEPENDENCY GRAPH
        graph = self._build_graph(parsed_specs)
        ordered = self._resolve_order(graph)
        logger.info("✅ Execution order: %s", ordered)

        # SAFE EXECUTION ORDER MAP (raw -> ParsedSpec)
        ordered_map = {ps.raw: ps for ps in parsed_specs}

        # 3. EXECUTE STRICT GRAPH ORDER (dependency-respecting)
        for node in ordered:
            ps = ordered_map.get(node)
            if ps is None:
                logger.warning("Graph node without spec ignored: %s", node)
                continue
            result_df = self._apply_spec(result_df, ps, mode)

        logger.info("✅ Feature calculation complete. Result shape: %s, Columns: %s", 
                   result_df.shape, list(result_df.columns))
        return result_df

    # -------------------------------------------------------- 2
    def _apply_spec_old1(self, df: pd.DataFrame, ps: ParsedSpec, mode: str) -> pd.DataFrame:
        import re
        
        # استخراج tf از ps.raw
        if not hasattr(ps, 'tf') or ps.tf is None:
            match = re.search(r'@(\w+)', ps.raw)
            ps.tf = match.group(1) if match else None

        spec = get_indicator(ps.name, self.mode)
        if spec is None:
            logger.warning("Indicator not found: %s", ps.name)
            return df

        # اعتبارسنجی با در نظر گرفتن tf
        if ps.tf:
            missing = [c for c in spec.required_cols if f"{ps.tf}_{c}" not in df.columns]
            if missing:
                for c in missing:
                    logger.warning("Contract violation [%s]: missing column %s", ps.name, f"{ps.tf}_{c}")
                return df
        else:
            if not self._validate_contract(spec, df):
                return df

        fn = spec.fn

        logger.info("🔄 Applying %s (mode=%s) with args=%s, kwargs=%s", 
                   ps.name, self.mode, ps.args, ps.kwargs)
        
        # حالت batch یا live
        if spec.is_batch_mode(self.mode):
            out = self._call_batch(fn, df, ps)
            if isinstance(out, pd.DataFrame):
                df = self._merge(df, out)
            return df

        if spec.is_incremental_mode(self.mode):
            df = self._apply_live(spec, df, ps)
            return df

        return df
    
    # -------------------------------------------------------- 3
    def _call_batch_old1(self, fn, df: pd.DataFrame, ps: ParsedSpec):
        # فقط کلیدهایی که در امضای تابع Batch وجود دارند را نگه دار
        sig = inspect.signature(fn)
        valid_keys = set(sig.parameters.keys()) - {'df'}  # df را حذف کن
        filtered_kwargs = {k: v for k, v in ps.kwargs.items() if k in valid_keys}
        return fn(df, **filtered_kwargs)


    def _call_batch_old2(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        # حذف 'df' از لیست پارامترها
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        # 1. نگاشت آرگومان‌های موقعیتی به نام پارامترها
        bound_args = {}
        for i, arg in enumerate(ps.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg
        
        # 2. اعمال kwargs (اولویت با kwargs است)
        bound_args.update(ps.kwargs)
        
        # 3. فیلتر نهایی: فقط کلیدهای معتبر امضای تابع
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        return fn(df, **filtered_kwargs)
    

    def _call_batch_old3(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        # حذف 'df' از لیست پارامترها (اولین پارامتر معمولاً df است)
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        # 1. نگاشت آرگومان‌های موقعیتی به نام پارامترها
        bound_args = {}
        for i, arg in enumerate(ps.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg
        
        # 2. اعمال kwargs (اولویت با kwargs است)
        bound_args.update(ps.kwargs)
        
        # 3. ✅ افزودن پیشوند تایم‌فریم به نام ستون‌ها (اگر تایم‌فریم وجود داشته باشد)
        tf = getattr(ps, 'timeframe', None)
        column_like = {'open', 'high', 'low', 'close', 'volume', 'spread'}
        if tf:
            for key, value in bound_args.items():
                # اگر مقدار یک رشته و در مجموعه‌ی ستون‌ها باشد، پیشوند بزن
                if isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
                # همچنین اگر کلید خودش نام ستون باشد (مثل close_col) مقدار آن را پیشوند بزن
                elif key.endswith('_col') and isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
        
        # 4. فیلتر نهایی: فقط کلیدهای معتبر امضای تابع را نگه دار
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        return fn(df, **filtered_kwargs)
    

    def _call_batch_old4(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        # حذف 'df' از لیست پارامترها (اولین پارامتر معمولاً df است)
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        # 1. نگاشت آرگومان‌های موقعیتی به نام پارامترها (با اولویت پارامترهای عددی)
        bound_args = {}
        
        # اگر ps.args شامل یک مقدار عددی باشد و ps.kwargs خالی باشد،
        # آن را به عنوان 'period' (یا 'length' در صورت وجود) در نظر بگیر
        if ps.args and not ps.kwargs:
            # استخراج مقدار اول (معمولاً period)
            first_arg = ps.args[0]
            if isinstance(first_arg, (int, float)):
                # اگر پارامتری به نام 'period' در تابع وجود دارد، آن را به آن نسبت بده
                if 'period' in param_names:
                    bound_args['period'] = first_arg
                elif 'length' in param_names:
                    bound_args['length'] = first_arg
                else:
                    # در غیر این صورت، به ترتیب param_names نسبت بده
                    for i, arg in enumerate(ps.args):
                        if i < len(param_names):
                            bound_args[param_names[i]] = arg
            else:
                # اگر آرگومان اول عددی نبود، به ترتیب param_names نسبت بده
                for i, arg in enumerate(ps.args):
                    if i < len(param_names):
                        bound_args[param_names[i]] = arg
        else:
            # در صورت وجود kwargs یا آرگومان‌های متعدد، به ترتیب param_names نسبت بده
            for i, arg in enumerate(ps.args):
                if i < len(param_names):
                    bound_args[param_names[i]] = arg
        
        # 2. اعمال kwargs (اولویت با kwargs است)
        bound_args.update(ps.kwargs)
        
        # 3. افزودن پیشوند تایم‌فریم به نام ستون‌ها (اگر تایم‌فریم وجود داشته باشد)
        tf = getattr(ps, 'timeframe', None)
        column_like = {'open', 'high', 'low', 'close', 'volume', 'spread'}
        if tf:
            for key, value in bound_args.items():
                # اگر مقدار یک رشته و در مجموعه‌ی ستون‌ها باشد، پیشوند بزن
                if isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
                # همچنین اگر کلید خودش نام ستون باشد (مثل close_col) مقدار آن را پیشوند بزن
                elif key.endswith('_col') and isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
        
        # 4. فیلتر نهایی: فقط کلیدهای معتبر امضای تابع را نگه دار
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        return fn(df, **filtered_kwargs)
    

    def _call_batch_old5(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        # حذف 'df' از لیست پارامترها
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        # 1. نگاشت آرگومان‌های موقعیتی به نام پارامترها
        bound_args = {}
        for i, arg in enumerate(ps.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg
        
        # 2. اعمال kwargs (اولویت با kwargs است)
        bound_args.update(ps.kwargs)
        
        # 3. ✅ تکمیل پارامترهای ستون‌ها با مقادیر پیش‌فرض از امضای تابع
        # این کار باعث می‌شود که حتی اگر کاربر فقط 'period' را پاس دهد،
        # پارامترهای ستون مانند high_col, low_col, close_col مقدار پیش‌فرض خود را بگیرند.
        for param_name, param in sig.parameters.items():
            if param_name == 'df':
                continue
            # اگر پارامتر در bound_args نبود و دارای مقدار پیش‌فرض بود
            if param_name not in bound_args and param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default
        
        # 4. ✅ افزودن پیشوند تایم‌فریم به نام ستون‌ها
        tf = getattr(ps, 'timeframe', None)
        column_like = {'open', 'high', 'low', 'close', 'volume', 'spread'}
        if tf:
            for key, value in bound_args.items():
                # اگر مقدار یک رشته و در مجموعه‌ی ستون‌ها باشد، پیشوند بزن
                if isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
                # همچنین اگر کلید به '_col' ختم شود و مقدارش یک نام ستون باشد، پیشوند بزن
                elif key.endswith('_col') and isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"

        # 5. فیلتر نهایی: فقط کلیدهای معتبر امضای تابع را نگه دار
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        logger.info("📊 Calling %s with filtered kwargs: %s", fn.__name__, filtered_kwargs)
        out = fn(df, **filtered_kwargs)
        
        # افزودن پیشوند تایم‌فریم به نام ستون‌های خروجی
        tf = getattr(ps, 'timeframe', None)
        if tf and isinstance(out, pd.DataFrame):
            out.columns = [f"{tf}_{col}" for col in out.columns]
        
        return out


    def _call_batch_old6(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        # حذف 'df' از لیست پارامترها
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        # 1. نگاشت آرگومان‌های موقعیتی به نام پارامترها
        bound_args = {}
        for i, arg in enumerate(ps.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg
        
        # 2. اعمال kwargs (اولویت با kwargs است)
        bound_args.update(ps.kwargs)
        
        # 3. ✅ تکمیل پارامترهای ستون‌ها با مقادیر پیش‌فرض از امضای تابع
        # این کار باعث می‌شود که حتی اگر کاربر فقط 'period' را پاس دهد،
        # پارامترهای ستون مانند high_col, low_col, close_col مقدار پیش‌فرض خود را بگیرند.
        for param_name, param in sig.parameters.items():
            if param_name == 'df':
                continue
            if param_name not in bound_args and param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default
        
        # 4. افزودن پیشوند تایم‌فریم به نام ستون‌ها و مقیاس‌دهی period
        tf = getattr(ps, 'timeframe', None)
        column_like = {'open', 'high', 'low', 'close', 'volume', 'spread'}
        if tf:
            # 4a. پیشوند به نام ستون‌های ورودی
            for key, value in bound_args.items():
                if isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
                elif key.endswith('_col') and isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
            
            # 4b. مقیاس‌دهی period بر اساس نسبت تایم‌فریم (برای جلوگیری از خطای look-ahead)
            import re
            base_tf = None
            for col in df.columns:
                if col.endswith('_open'):
                    match = re.match(r'^([A-Za-z0-9]+)_', col)
                    if match:
                        base_tf = match.group(1)
                        break
            if base_tf:
                from f02_data.data_handler_F_2 import _TF_MINUTES
                base_min = _TF_MINUTES.get(base_tf.upper(), 60)
                tf_min = _TF_MINUTES.get(tf.upper(), 60)
                if base_min and tf_min:
                    ratio = tf_min / base_min
                    # پارامترهای عددی که باید مقیاس‌دهی شوند
                    numeric_params = {'period', 'n', 'length', 'window', 'fast', 'slow', 'signal', 'k_period', 'd_period'}
                    for param in numeric_params:
                        if param in bound_args and isinstance(bound_args[param], (int, float)):
                            original = bound_args[param]
                            scaled = int(round(original * ratio))
                            if scaled != original:
                                bound_args[param] = max(1, scaled)
                                logger.debug(f"Scaled {param}: {original}@{tf} → {scaled} on {base_tf}")
        
        # 5. فیلتر نهایی: فقط کلیدهای معتبر امضای تابع را نگه دار
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        logger.info("📊 Calling %s with filtered kwargs: %s", fn.__name__, filtered_kwargs)
        out = fn(df, **filtered_kwargs)
        
        # افزودن پیشوند تایم‌فریم به نام ستون‌های خروجی
        if tf and isinstance(out, pd.DataFrame):
            out.columns = [f"{tf}_{col}" for col in out.columns]
        
        return out


    def _call_batch_old7(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        # حذف 'df' از لیست پارامترها
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        # 1. نگاشت آرگومان‌های موقعیتی به نام پارامترها
        bound_args = {}
        for i, arg in enumerate(ps.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg
        
        # 2. اعمال kwargs (اولویت با kwargs است)
        bound_args.update(ps.kwargs)
        
        # 3. تکمیل پارامترهای ستون‌ها با مقادیر پیش‌فرض از امضای تابع
        for param_name, param in sig.parameters.items():
            if param_name == 'df':
                continue
            if param_name not in bound_args and param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default
        
        # 4. افزودن پیشوند تایم‌فریم به نام ستون‌ها
        tf = getattr(ps, 'timeframe', None)
        column_like = {'open', 'high', 'low', 'close', 'volume', 'spread'}
        if tf:
            for key, value in bound_args.items():
                if isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
                elif key.endswith('_col') and isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
        
        # 5. فیلتر نهایی: فقط کلیدهای معتبر امضای تابع را نگه دار
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        logger.info("📊 Calling %s with filtered kwargs: %s", fn.__name__, filtered_kwargs)
        out = fn(df, **filtered_kwargs)
        
        # افزودن پیشوند تایم‌فریم به نام ستون‌های خروجی
        if tf and isinstance(out, pd.DataFrame):
            out.columns = [f"{tf}_{col}" for col in out.columns]
        
        return out


    def _call_batch_old8(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        
        if param_names and param_names[0] == 'df':
            param_names = param_names[1:]
        
        bound_args = {}
        for i, arg in enumerate(ps.args):
            if i < len(param_names):
                bound_args[param_names[i]] = arg
        
        bound_args.update(ps.kwargs)
        
        for param_name, param in sig.parameters.items():
            if param_name == 'df':
                continue
            if param_name not in bound_args and param.default != inspect.Parameter.empty:
                bound_args[param_name] = param.default
        
        tf = getattr(ps, 'timeframe', None)
        column_like = {'open', 'high', 'low', 'close', 'volume', 'spread'}
        if tf:
            for key, value in bound_args.items():
                if isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
                elif key.endswith('_col') and isinstance(value, str) and value.lower() in column_like:
                    bound_args[key] = f"{tf}_{value}"
        
        # ===== ✅ راه‌حل نهایی برای تایم‌فریم‌های بالاتر =====
        # اگر تایم‌فریم اندیکاتور از base_tf بزرگ‌تر است،
        # باید اندیکاتور را روی داده‌های اصلی آن تایم‌فریم محاسبه کرده و سپس پخش کنیم.
        base_tf = None
        for col in df.columns:
            if col.endswith('_open'):
                import re
                match = re.match(r'^([A-Za-z0-9]+)_', col)
                if match:
                    base_tf = match.group(1)
                    break
        
        if tf and base_tf and tf != base_tf:
            if _TF_MINUTES.get(tf.upper(), 0) > _TF_MINUTES.get(base_tf.upper(), 0):
                # 1. شناسایی ستون ورودی (مثلاً H4_close)
                col_name = None
                for key, value in bound_args.items():
                    if key.endswith('_col') and isinstance(value, str) and value.startswith(tf):
                        col_name = value
                        break
                    if key in ['column', 'price_col'] and isinstance(value, str) and value.startswith(tf):
                        col_name = value
                        break
                
                if col_name:
                    # 2. ریسمپل به تایم‌فریم اصلی
                    tf_minutes = _TF_MINUTES.get(tf.upper(), 0)
                    if tf_minutes == 0:
                        raise ValueError(f"Unknown timeframe: {tf}")
                    freq = f"{tf_minutes}T"
                    resampled = df[col_name].resample(freq).last().dropna()

                    # 3. ساخت kwargs جدید برای تابع (بدون ستون اصلی)
                    temp_kwargs = {k: v for k, v in bound_args.items() if k not in ['column', 'price_col', 'high_col', 'low_col', 'close_col']}
                    temp_kwargs['column'] = col_name  # ستون برای تابع (اما روی resampled اعمال می‌شود)
                    
                    # 4. فراخوانی تابع روی resampled
                    temp_df = pd.DataFrame({col_name: resampled})
                    temp_out = fn(temp_df, **temp_kwargs)
                    
                    # 5. پخش نتیجه به شبکه‌ی اصلی با ffill
                    if isinstance(temp_out, pd.DataFrame):
                        out_col = temp_out.columns[0]
                        out = pd.DataFrame({
                            out_col: temp_out[out_col].reindex(df.index, method='ffill')
                        }, index=df.index)
                        # اضافه کردن پیشوند (که قبلاً در خروجی نهایی انجام می‌شود)
                        out.columns = [f"{tf}_{col}" for col in out.columns]
                        return out
        
        # ===== اگر تایم‌فریم پایه بود یا شرط بالا اجرا نشد =====
        valid_keys = set(sig.parameters.keys()) - {'df'}
        filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        
        logger.info("📊 Calling %s with filtered kwargs: %s", fn.__name__, filtered_kwargs)
        out = fn(df, **filtered_kwargs)
        
        if tf and isinstance(out, pd.DataFrame):
            out.columns = [f"{tf}_{col}" for col in out.columns]
        
        return out

    
    def _call_batch_old11(self, fn, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        Batch execution روی دیتافریم واقعی همان تایم‌فریم.
        """

        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        if params and params[0] == "df":
            params = params[1:]

        kwargs = {}

        # positional args
        for i, arg in enumerate(ps.args):
            if i < len(params):
                kwargs[params[i]] = arg

        # keyword args
        kwargs.update(ps.kwargs)

        # default values
        for name, p in sig.parameters.items():
            if name == "df":
                continue
            if name not in kwargs and p.default is not inspect.Parameter.empty:
                kwargs[name] = p.default

        tf = ps.tf
        df = dataset.get(tf)

        if df is None:
            raise KeyError(f"Timeframe '{tf}' not found in dataset.")

        # تبدیل close -> close
        column_like = {"open", "high", "low", "close", "volume", "spread"}

        for k, v in list(kwargs.items()):
            if isinstance(v, str):
                if v in column_like:
                    kwargs[k] = v

        out = fn(df, **kwargs)

        if isinstance(out, pd.DataFrame):
            out.columns = [f"{tf}_{c}" for c in out.columns]

        dataset.frames[tf] = df.join(out)

        return dataset


    # -------------------------------------------------------- 5
    def _update_live_old1(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        cols = getattr(spec, "required_cols", None)
        if not cols:
            return obj.update()

        tf = getattr(ps, "tf", None)
        values = []
        for c in cols:
            target_col = f"{tf}_{c}" if tf else c
            if target_col not in row:
                logger.warning("Missing column %s for %s", target_col, spec)
                return None
            values.append(row[target_col])

        raw_out = obj.update(*values)
        return self._normalize_output(spec, raw_out)    
    

    # -------------------------------------------------------- 13 new
    def process_live_data_old11(self, payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        پردازش داده‌های لحظه‌ای دریافتی از DataHandler.
        
        پارامترها:
            payload: دیکشنری شامل 'base_df' و 'cache_dict'
            
        خروجی:
            دیتافریم حاوی فیچرهای محاسبه‌شده یا None در صورت خطا
        """
        base_df = payload.get("base_df")
        cache_dict = payload.get("cache_dict")
        
        if base_df is None or base_df.empty:
            logger.warning("Empty base_df received in FeatureEngine.process_live_data")
            return None
        
        # 1. خواندن لیست مشخصات از کانفیگ
        feature_specs = self.config.get("features", {}).get("live_specs", [])
        if not feature_specs:
            logger.warning("No live feature specs found in config. Skipping feature calculation.")
            return base_df  # برگرداندن دیتا بدون فیچر
        
        # 2. اجرای FeatureEngine روی base_df با حالت live
        try:
            result_df = self.execute(
                df=base_df,
                specs=feature_specs,
                mode="live"
            )
            logger.info(f"FeatureEngine processed live data: {len(result_df)} rows, {len(result_df.columns)} cols")
            return result_df
        except Exception as e:
            logger.exception(f"FeatureEngine failed to process live data: {e}")
            return None
    




