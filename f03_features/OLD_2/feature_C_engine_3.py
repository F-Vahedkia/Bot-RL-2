# f03_features/feature_C_engine_3.py

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import inspect
import logging

from f03_features.feature_B_registry_1 import get_indicator, IndicatorSpec
from f10_utils.parser import parse_spec, ParsedSpec
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================================
# ENGINE CORE (PURE FEATURE TRANSFORMER)
# ============================================================
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
    def execute(self, df: pd.DataFrame, specs: List[str], mode: str = "train") -> pd.DataFrame:
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
            return df
        
        self.mode = mode
        self._run_id += 1

        # HARD RESET: isolate runs (critical for reproducibility)
        if mode == "train":
            self._live_cache = {}

        result_df = df.copy()
        parsed_specs: List[ParsedSpec] = []

        # 1. PARSE ALL SPECS
        for spec in specs:
            try:
                parsed_specs.append(parse_spec(spec))
            except Exception as e:
                logger.warning("Invalid spec skipped: %s | %s", spec, e)

        # 2. BUILD DEPENDENCY GRAPH
        graph = self._build_graph(parsed_specs)
        ordered = self._resolve_order(graph)

        # SAFE EXECUTION ORDER MAP (raw -> ParsedSpec)
        ordered_map = {ps.raw: ps for ps in parsed_specs}

        # 3. EXECUTE STRICT GRAPH ORDER (dependency-respecting)
        for node in ordered:
            ps = ordered_map.get(node)
            if ps is None:
                logger.warning("Graph node without spec ignored: %s", node)
                continue
            result_df = self._apply_spec(result_df, ps, mode)

        return result_df

    # -------------------------------------------------------- 2
    def _apply_spec_old1(self, df: pd.DataFrame, ps: ParsedSpec, mode: str) -> pd.DataFrame:
        # استخراج tf از ps.raw اگر هنوز تعیین نشده
        if not hasattr(ps, 'tf') or ps.tf is None:
            import re
            match = re.search(r'@(\w+)', ps.raw)
            ps.tf = match.group(1) if match else None

        import re
        if not hasattr(ps, 'tf') or ps.tf is None:
            match = re.search(r'@(\w+)', ps.raw)
            ps.tf = match.group(1) if match else None

        # ----------------------------------------------------
        # 1. get registry entry
        # ----------------------------------------------------
        spec: Optional[IndicatorSpec] = get_indicator(ps.name, self.mode)
        if spec is None:
            logger.warning("Indicator not found in registry: %s", ps.name)
            return df

        # استخراج tf از ps.raw
        if not hasattr(ps, 'tf') or ps.tf is None:
            import re
            match = re.search(r'@(\w+)', ps.raw)
            ps.tf = match.group(1) if match else None

        # اعتبارسنجی ستون‌ها با در نظر گرفتن tf
        if ps.tf:
            missing = [c for c in spec.required_cols if f"{ps.tf}_{c}" not in df.columns]
            if missing:
                for c in missing:
                    logger.warning("Contract violation [%s]: missing column %s", ps.name, f"{ps.tf}_{c}")
                return df
        else:
            if not self._validate_contract(spec, df):
                return df
            
        # ----------------------------------------------------
        # 2. extract function
        # ----------------------------------------------------
        fn = spec.fn

        try:
            # ------------------------------------------------
            # 3. BATCH MODE (TRAIN)
            # ------------------------------------------------
            if spec.is_batch_mode(self.mode):
                out = self._call_batch(fn, df, ps)
                if isinstance(out, pd.DataFrame):
                    df = self._merge(df, out)
                return df

            # ------------------------------------------------
            # 4. LIVE MODE (STATEFUL)
            # ------------------------------------------------
            # print(ps.name, mode, spec,                            # for debug
            #     spec.is_batch_mode(mode) if spec else None,       # for debug
            #     spec.is_incremental_mode(mode) if spec else None) # for debug
            
            if spec.is_incremental_mode(self.mode):
                # print("=====================")   # for debug
                df = self._apply_live(spec, df, ps)
                return df

        except Exception as e:
            logger.exception("Execution failed for %s: %s", ps.name, e)

        return df

    def _apply_spec(self, df: pd.DataFrame, ps: ParsedSpec, mode: str) -> pd.DataFrame:
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
    def _call_batch_old1(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:

        # resolve args
        kwargs = dict(ps.kwargs)

        # IMPORTANT: inject positional args dynamically
        # (registry-driven, no hardcoding)

        try:
            return fn(df, *ps.args, **kwargs)
        except TypeError as e1:
            try:
                return fn(df, **kwargs)
            except TypeError:
                raise e1
            
    def _call_batch_old2(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        kwargs = dict(ps.kwargs)
        tf = getattr(ps, "tf", None)
        
        if tf:
            new_args = []
            for a in ps.args:
                if isinstance(a, str) and a.lower() in fn.__defaults__:
                    new_args.append(f"{tf}_{a}")
                else:
                    new_args.append(a)
            args_to_use = new_args
        else:
            args_to_use = ps.args

        try:
            return fn(df, *args_to_use, **kwargs)
        except TypeError:
            return fn(df, **kwargs)           

    def _call_batch_old3(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        kwargs = dict(ps.kwargs)
        tf = getattr(ps, "tf", None)
        
        # استخراج tf از ps.raw در صورت نیاز
        if tf is None:
            import re
            match = re.search(r'@(\w+)', ps.raw)
            tf = match.group(1) if match else None
            ps.tf = tf

        # ساخت آرگومان‌های جدید با پیشوند tf
        if tf:
            new_args = []
            for a in ps.args:
                if isinstance(a, str) and a.lower() in ['open', 'high', 'low', 'close', 'volume', 'spread']:
                    new_args.append(f"{tf}_{a}")
                else:
                    new_args.append(a)
            args_to_use = new_args
        else:
            args_to_use = ps.args

        try:
            return fn(df, *args_to_use, **kwargs)
        except TypeError as e:
            # اگر با آرگومان‌ها خطا داد، بدون آرگومان‌های اضافی امتحان کن
            try:
                return fn(df, **kwargs)
            except TypeError:
                raise e     

    def _call_batch_old4(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        # ===== DEBUG =====
        logger.info(f"DEBUG: ps.raw = {ps.raw}")
        logger.info(f"DEBUG: ps.args = {ps.args}")
        logger.info(f"DEBUG: ps.kwargs = {ps.kwargs}")
        logger.info(f"DEBUG: ps.tf = {getattr(ps, 'tf', None)}")
        logger.info(f"DEBUG: df columns = {df.columns.tolist()}")
        # =================
    
        kwargs = dict(ps.kwargs)
        
        # اطمینان از وجود tf در ps
        if not hasattr(ps, 'tf') or ps.tf is None:
            import re
            match = re.search(r'@(\w+)', ps.raw)
            ps.tf = match.group(1) if match else None
        
        tf = ps.tf
        args_to_use = []
        
        for a in ps.args:
            if isinstance(a, str) and a.lower() in ['open', 'high', 'low', 'close', 'volume', 'spread']:
                if tf:
                    args_to_use.append(f"{tf}_{a}")
                else:
                    args_to_use.append(a)
            else:
                args_to_use.append(a)
        
        try:
            return fn(df, *args_to_use, **kwargs)
        except TypeError:
            return fn(df, **kwargs)
    
    def _call_batch_old5(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        # ===== DEBUG =====
        logger.info(f"DEBUG: ps.raw = {ps.raw}")
        logger.info(f"DEBUG: ps.args = {ps.args}")
        logger.info(f"DEBUG: ps.kwargs = {ps.kwargs}")
        logger.info(f"DEBUG: ps.tf = {getattr(ps, 'tf', None)}")
        logger.info(f"DEBUG: df columns = {df.columns.tolist()}")
        # =================
        if not hasattr(ps, 'tf') or ps.tf is None:
            import re
            match = re.search(r'@(\w+)', ps.raw)
            ps.tf = match.group(1) if match else None

        tf = ps.tf
        kwargs = dict(ps.kwargs)
        
        # اصلاح kwargs برای ستون‌ها
        if tf:
            for key, value in kwargs.items():
                if isinstance(value, str) and value.lower() in ['open','high','low','close','volume','spread']:
                    kwargs[key] = f"{tf}_{value}"

        args_to_use = []
        for a in ps.args:
            if isinstance(a, str) and a.lower() in ['open','high','low','close','volume','spread']:
                args_to_use.append(f"{tf}_{a}" if tf else a)
            else:
                args_to_use.append(a)

        try:
            return fn(df, *args_to_use, **kwargs)
        except TypeError:
            return fn(df, **kwargs)
        
    def _call_batch_old6(self, fn, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        kwargs = dict(ps.kwargs)
        tf = ps.timeframe  # از خود ParsedSpec (نه regex)
        
        # اصلاح نام ستون‌ها در kwargs
        if tf:
            for key, value in kwargs.items():
                if isinstance(value, str) and value.lower() in ['open','high','low','close','volume','spread']:
                    kwargs[key] = f"{tf}_{value}"
        
        # اصلاح نام ستون‌ها در args
        args_to_use = []
        for a in ps.args:
            if isinstance(a, str) and a.lower() in ['open','high','low','close','volume','spread']:
                args_to_use.append(f"{tf}_{a}" if tf else a)
            else:
                args_to_use.append(a)
        
        try:
            return fn(df, *args_to_use, **kwargs)
        except TypeError:
            return fn(df, **kwargs)
    
    def _call_batch(self, fn, df: pd.DataFrame, ps: ParsedSpec):
        # فقط کلیدهایی که در امضای تابع Batch وجود دارند را نگه دار
        sig = inspect.signature(fn)
        valid_keys = set(sig.parameters.keys()) - {'df'}  # df را حذف کن
        filtered_kwargs = {k: v for k, v in ps.kwargs.items() if k in valid_keys}
        return fn(df, **filtered_kwargs)

    # -------------------------------------------------------- 4
    def _apply_live(self, spec: IndicatorSpec, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:

        # SAFE LIVE CACHE KEY (prevents cross-context state pollution)     pollution: آلودگی
        context = getattr(df, "attrs", {}).get("context", {})
        ctx_id = (
            context.get("symbol", "NA"),
            context.get("tf", "NA"),
            context.get("session", "NA"),
        )
        key = f"{self.mode}::{ps.raw}::{ctx_id}"

        if key not in self._live_cache:
            ctor_args, ctor_kwargs = self._build_live_ctor_args(spec, ps)
            try:
                self._live_cache[key] = spec.fn(*ctor_args, **ctor_kwargs)
            # except Exception:  # old
            except TypeError:    # new
                self._live_cache[key] = spec.fn(**ctor_kwargs)

        obj = self._live_cache[key]
        rows = []
        for _, row in df.iterrows():
            try:
                # out = self._update_live(obj, row, spec)
                out = self._update_live(obj, row, spec, ps)
                rows.append(out)
            except Exception:
                rows.append(None)

        return self._attach_live_output(df=df, ps=ps, spec=spec, outputs=rows)    
    
    # -------------------------------------------------------- 5
    def _update_live_old1(self, obj: Any, row: pd.Series, spec: IndicatorSpec):

        cols = getattr(spec, "required_cols", None)
        if not cols:
            return obj.update()

        values = []
        for c in cols:
            if c not in row:
                logger.warning("Missing column %s for %s", c, spec)
                return None
            values.append(row[c])

        raw_out = obj.update(*values)
        normalized = self._normalize_output(spec, raw_out)
        return normalized
    
    def _update_live(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
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
    
    # -------------------------------------------------------- 6
    def _merge(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """
        Merge strategy:
        - avoid column collision
        - preserve index alignment
        """
        if out is None or out.empty:
            return df

        # prevent overwrite
        for col in out.columns:
            if col not in df.columns:
                df[col] = out[col]
                continue
            if df[col].equals(out[col]):
                continue
            df[f"{col}__dup"] = out[col]

        return df
    
    # -------------------------------------------------------- 7
    def _build_live_ctor_args(self, spec: IndicatorSpec, ps: ParsedSpec):

        column_like = {"open", "high", "low", "close", "volume"}
        args = [
            a for a in ps.args
            if not (isinstance(a, str) and a.lower() in column_like)
        ]
        return args, ps.kwargs

    # -------------------------------------------------------- 8
    def _attach_live_output(self, df: pd.DataFrame, ps: ParsedSpec, spec: IndicatorSpec, outputs):

        if not outputs:
            return df

        # SAFE ALIGNMENT GUARANTEE
        n = len(df)

        columns_map = {}

        for out in outputs:
            if out is None:
                continue

            if isinstance(out, dict):
                for k in out.keys():
                    col = self._build_live_column_name(k, ps)
                    if col not in columns_map:
                        columns_map[col] = [None] * n

        for idx, out in enumerate(outputs):
            if not isinstance(out, dict):
                continue
            for k, v in out.items():
                col = self._build_live_column_name(k, ps)
                columns_map[col][idx] = v

        for col, values in columns_map.items():
            if len(values) < len(df):
                values.extend([None] * (len(df) - len(values)))
            df[col] = values
        
        return df

    # -------------------------------------------------------- 9
    def _build_live_column_name(self, output_name, ps):

        raw = ps.raw
        if output_name == ps.name:
            return f"{raw}_live"

        params_part = raw[len(ps.name):]
        return f"{output_name}{params_part}_live"

    # -------------------------------------------------------- 10
    def _normalize_output(self, spec, output):
        names = getattr(spec, "output_names", None)

        # scalar
        if names is None:
            return {spec.name: output}

        # tuple / list
        if isinstance(output, (tuple, list)):
            out = {}
            for i, name in enumerate(names):
                if i < len(output):
                    out[name] = output[i]
                else:
                    out[name] = None
            return out

        # single value fallback
        return {names[0]: output}

    # -------------------------------------------------------- 11
    def _validate_contract(self, spec, df):

        # 1. required cols
        for c in spec.required_cols:
            if c not in df.columns:
                logger.warning(
                    "Contract violation [%s]: missing column %s",
                    spec.name, c
                )
                return False

        # 2. mode check
        if hasattr(spec, "modes") and getattr(self, "mode", None) not in spec.modes:
            logger.warning(
                "Contract violation [%s]: mode not supported %s",
                spec.name, self.mode
            )
            return False

        return True
    
    # -------------------------------------------------------- 12
    def _build_graph(self, parsed_specs):
        graph = {}
        
        valid_nodes = {ps.raw for ps in parsed_specs}
        
        for ps in parsed_specs:
            spec = get_indicator(ps.name, self.mode)
            if spec is None:
                continue

            node = ps.raw
            graph[node] = []

            deps = getattr(spec, "depends_on", None)

            # STRICT DEPENDENCY VALIDATION
            if isinstance(deps, (list, tuple, set)):
                for d in deps:
                    if d in valid_nodes:
                        graph[node].append(d)
                    else:
                        logger.warning("Unknown dependency ignored [%s -> %s]", node, d)
        return graph
    
    # -------------------------------------------------------- 13
    def _resolve_order(self, graph):

        resolved = []
        visited = set()
        visiting = set()

        def visit(node: str):
            if node in visited:
                return
            if node in visiting:
                raise RuntimeError(f"Circular feature dependency detected: {node}")

            visiting.add(node)

            for dep in graph.get(node, []):
                if dep in graph:
                    visit(dep)
            
            visiting.remove(node)
            visited.add(node)
            resolved.append(node)

        # DETERMINISTIC ORDERING (critical for reproducibility)
        for n in sorted(graph.keys()):
            visit(n)

        return resolved

    # -------------------------------------------------------- 14 new
    def process_live_data(self, payload: Dict[str, Any]) -> Optional[pd.DataFrame]:
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
            logger.debug(f"FeatureEngine processed live data: {len(result_df)} rows, {len(result_df.columns)} cols")
            return result_df
        except Exception as e:
            logger.exception(f"FeatureEngine failed to process live data: {e}")
            return None
    
    # -------------------------------------------------------- END

# ===================================================================
# وابستگی متدهای کلاس DataHandler
# ===================================================================
""" Method Name             1   2   3      4   5   6   7   8   9  10
1  _load_raw               --  --  --     --  --  --  --  --  --  --
2  build                   --  --  --     --  --  --  --  --  --  -- For External Use (build)
3  save                    --  --  --     --  --  --  --  --  --  -- For External Use (save)


"""