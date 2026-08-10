# f03_features/feature_C_engine_4.py

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import inspect
import logging

from f03_features.feature_C_registry_1 import get_indicator, IndicatorSpec
from f10_utils.parser import parse_spec, ParsedSpec
from f10_utils.constants import _TF_MINUTES
from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_B_cache_6 import (
    GLOBAL_FEATURE_CACHE,
    ExecutionContract,
    cached_compute,
)

logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

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

        self._contract = ExecutionContract(
            engine_version="1",
            resolver_version="1",
            config_version="1",
            registry_version="1",
        )
    
    # -------------------------------------------------------- 1=new
    def execute(
        self,
        dataset: MTFDataset,
        specs: List[str],
        mode: str = "train",
    ) -> MTFDataset:
        """
        محاسبه فیچرها روی MTFDataset.

        Parameters
        ----------
        dataset : MTFDataset
            مجموعه دیتافریم‌های تایم‌فریم‌های مختلف.
        specs : list[str]
            لیست feature specification ها.
        mode : {"train", "live"}

        Returns
        -------
        MTFDataset
        """

        if dataset is None or len(dataset.frames) == 0:
            logger.info("MTFDataset is empty.")
            return dataset

        self.mode = mode
        self._run_id += 1

        if mode == "train":
            self._live_cache.clear()

        parsed_specs: List[ParsedSpec] = []

        # ---------------------------------------------------
        # Parse
        # ---------------------------------------------------
        for spec in specs:
            try:
                parsed_specs.append(parse_spec(spec))
            except Exception as e:
                logger.warning("Invalid spec skipped: %s | %s", spec, e)

        logger.info(
            "Parsed %d feature specs.",
            len(parsed_specs),
        )

        # ---------------------------------------------------
        # Dependency Graph
        # ---------------------------------------------------
        graph = self._build_graph(parsed_specs)
        ordered = self._resolve_order(graph)

        ordered_map = {ps.raw.replace("'", '"'): ps for ps in parsed_specs}   #KKK

        # ---------------------------------------------------
        # Execute
        # ---------------------------------------------------
        for node in ordered:
            ps = ordered_map.get(node)
            if ps is None:
                continue

            dataset = cached_compute(
                cache=GLOBAL_FEATURE_CACHE,
                dataset=dataset,
                ps=ps,
                mode=mode,
                contract=self._contract,
                compute_fn=lambda ds=dataset, p=ps: self._apply_spec(dataset=ds, ps=p, mode=mode),
            )

        logger.info(
            "Feature calculation completed on %d timeframes.",
            len(dataset.frames),
        )

        return dataset

    # -------------------------------------------------------- 2=new
    def _apply_spec(
        self,
        dataset: MTFDataset,
        ps: ParsedSpec,
        mode: str,
    ) -> MTFDataset:
        """ اجرای یک Feature روی تایم‌فریم مربوطه. """

        spec = get_indicator(ps.name, mode)
        if spec is None:
            logger.warning("Indicator not found: %s", ps.name)
            return dataset

        tf = getattr(ps, "timeframe", None)
        if tf is None:
            logger.warning("Spec has no timeframe: %s", ps.raw)
            return dataset

        # ---------- قسمت حذف شده قدیمی
        # df = dataset.get(tf)
        # if df is None or df.empty:
        #     logger.warning("Dataset has no dataframe for timeframe %s", tf)
        #     return dataset
        # ---------- قسمت اضافه شده جدید
        try:
            df = dataset.get(tf)
        except KeyError:
            logger.warning(
                "Dataset has no dataframe for timeframe %s",
                tf,
            )
            return dataset
        if df.empty:
            logger.warning(
                "Dataset dataframe is empty for timeframe %s",
                tf,
            )
            return dataset
        # ---------- پایان قسمت اضافه شده جدید

        # ---------- Contract ----------
        missing = [c for c in spec.required_cols if c not in df.columns]
        if missing:
            logger.warning(
                "Contract violation [%s]: missing columns %s",
                ps.name,
                missing,
            )
            return dataset

        # ---------- Batch ----------
        if spec.is_batch_mode(mode):
            dataset = self._call_batch(
                fn=spec.fn,
                dataset=dataset,
                ps=ps,
            )
            return dataset

        # ---------- Live ----------
        if spec.is_incremental_mode(mode):
            out_df = self._apply_live(
                spec=spec,
                df=df,
                ps=ps,
            )
            dataset.replace(tf, out_df)
            return dataset

        return dataset
    
    # -------------------------------------------------------- 3=new
    def _call_batch_old1(self, fn, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        اجرای Indicator در حالت Batch روی تایم‌فریم مربوطه.
        مسئولیت:
            - استخراج DataFrame از MTFDataset
            - ساخت kwargs
            - اجرای Indicator
            - الحاق خروجی به همان DataFrame
            - جایگزینی DataFrame داخل Dataset
        """
        tf = ps.timeframe
        df = dataset.get(tf)
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())

        if params and params[0] == "df":
            params = params[1:]

        kwargs = {}

        # positional
        for i, value in enumerate(ps.args):
            if i < len(params):
                kwargs[params[i]] = value

        # keyword
        kwargs.update(ps.kwargs)

        # defaults
        for name, p in sig.parameters.items():
            if name == "df":
                continue
            if (
                name not in kwargs
                and p.default is not inspect.Parameter.empty
            ):
                kwargs[name] = p.default

        column_like = {"open", "high", "low", "close", "volume", "spread"}

        for k, v in list(kwargs.items()):
            if isinstance(v, str):
                if v in column_like:
                    kwargs[k] = v
        
        out = fn(df, **kwargs)

        print("=" * 60)                                                # for debug
        print(ps.raw)                                                  # for debug
        print(type(out))                                               # for debug
        print(out.columns if isinstance(out, pd.DataFrame) else out)   # for debug
        print("=" * 60)                                                # for debug

        if out is None:
            return dataset
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{ps.name} must return DataFrame in batch mode.")
        
        # نام ستون دقیقاً برابر Spec خام باشد
        if len(out.columns) == 1:
            out.columns = [ps.raw.replace("'", '"')]   #KKK
        else:
            out.columns = [
                f"{ps.raw.replace("'", '"')}::{c}"     #KKK
                for c in out.columns
            ]

        df = self._merge(df, out)
        dataset.replace(tf, df)

        return dataset


    def _call_batch_old2(self, fn, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        اجرای Indicator در حالت Batch روی تایم‌فریم مربوطه.
        مسئولیت:
            - استخراج DataFrame از MTFDataset
            - ساخت kwargs
            - اجرای Indicator
            - الحاق خروجی به همان DataFrame
            - جایگزینی DataFrame داخل Dataset
        """
        tf = ps.timeframe
        df = dataset.get(tf)

        sig = inspect.signature(fn)
        valid = {
            k: v
            for k, v in ps.kwargs.items()
            if k in sig.parameters
        }
        out = fn(df, **valid)

        print("=" * 60)                                                # for debug
        print(ps.raw)                                                  # for debug
        print(type(out))                                               # for debug
        print(out.columns if isinstance(out, pd.DataFrame) else out)   # for debug
        print("=" * 60)                                                # for debug

        if out is None:
            return dataset
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{ps.name} must return DataFrame in batch mode.")
        
        # نام ستون دقیقاً برابر Spec خام باشد
        if len(out.columns) == 1:
            out.columns = [ps.raw.replace("'", '"')]   #KKK
        else:
            out.columns = [
                f"{ps.raw.replace("'", '"')}::{c}"     #KKK
                for c in out.columns
            ]

        df = self._merge(df, out)
        dataset.replace(tf, df)

        return dataset


    def _call_batch(self, fn, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        Execute one Batch Indicator.
        این متد:
            • DataFrame مناسب را استخراج می‌کند.
            • args و kwargs را با امضای تابع تطبیق می‌دهد.
            • فقط پارامترهای معتبر را عبور می‌دهد.
            • خروجی را به Dataset الحاق می‌کند.
        """

        tf = ps.timeframe
        df = dataset.get(tf)

        sig = inspect.signature(fn)

        parameters = list(sig.parameters.values())

        # حذف پارامتر df
        if parameters and parameters[0].name == "df":
            parameters = parameters[1:]

        kwargs = {}

        # --------------------------------------------------
        # positional arguments
        # --------------------------------------------------
        positional_params = [
            p
            for p in parameters
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        for value, param in zip(ps.args, positional_params):
            kwargs[param.name] = value

        # --------------------------------------------------
        # keyword arguments
        # --------------------------------------------------
        kwargs.update(ps.kwargs)

        # --------------------------------------------------
        # فقط پارامترهای موجود در Signature
        # --------------------------------------------------
        filtered = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
        }
        out = fn(
            df,
            **filtered,
        )

        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"{ps.name} must return DataFrame."
            )

        raw = ps.raw.replace("'", '"')
        if len(out.columns) == 1:
            out.columns = [raw]
        else:
            out.columns = [
                f"{raw}::{c}"
                for c in out.columns
            ]

        dataset.replace(tf, self._merge(df, out))
        return dataset

    # -------------------------------------------------------- 4===
    def _apply_live(self, spec: IndicatorSpec, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:

        # SAFE LIVE CACHE KEY (prevents cross-context state pollution)     pollution: آلودگی
        context = getattr(df, "attrs", {}).get("context", {})
        ctx_id = (
            context.get("symbol", "NA"),
            context.get("tf", "NA"),
            context.get("session", "NA"),
        )
        key = f"{self.mode}::{ps.raw.replace("'", '"')}::{ctx_id}"    #KKK

        logger.info("🔄 Applying LIVE %s (mode=%s)", ps.name, self.mode)
        # if key not in self._live_cache:
        #     sig = inspect.signature(spec.fn)
        #     param_names = list(sig.parameters.keys())
            
        #     # نگاشت args به نام پارامترها
        #     bound_args = {}
        #     for i, arg in enumerate(ps.args):
        #         if i < len(param_names):
        #             bound_args[param_names[i]] = arg
            
        #     # اعمال kwargs (اولویت با kwargs)
        #     bound_args.update(ps.kwargs)
            
        #     # فیلتر نهایی
        #     valid_keys = set(sig.parameters.keys())
        #     filtered_kwargs = {k: v for k, v in bound_args.items() if k in valid_keys}
        #     self._live_cache[key] = spec.fn(**filtered_kwargs)

        if key not in self._live_cache:
            self._live_cache[key] = self._build_live_instance(spec, ps)

        obj = self._live_cache[key]

        rows = []
        for _, row in df.iterrows():
            try:
                out = self._update_live(obj, row, spec, ps)
                rows.append(out)
            except Exception:
                logger.exception("LIVE update failed for %s", ps.raw)
                rows.append(None)

        return self._attach_live_output(df=df, ps=ps, spec=spec, outputs=rows)    
    
    # -------------------------------------------------------- 5.1===
    def _update_live_old1(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        column_name = ps.kwargs.get('column', ps.kwargs.get('close_col', 'close'))
        tf = getattr(ps, 'timeframe', None)
        target_col = f"{tf}_{column_name}" if tf else column_name

        if target_col not in row:
            logger.warning("Missing column %s for %s", target_col, spec.name)
            return None

        raw_out = obj.update(row[target_col])
        return self._normalize_output(spec, raw_out)  

    # -------------------------------------------------------- 5.1 ===New
    def _update_live_old2(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        args = self._build_update_arguments(obj, row)
        raw_out = obj.update(*args)
        return self._normalize_output(spec, raw_out)


    def _update_live(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        """
        اجرای یک update روی Indicator Live.
        ورودی‌های update() به صورت خودکار از DataFrame استخراج
        می‌شوند و هیچ وابستگی به نوع Indicator وجود ندارد.
        """
        sig = inspect.signature(obj.update)
        values = []
        tf = ps.timeframe
        for name in sig.parameters:
            # kwargs همیشه اولویت دارد
            source = ps.kwargs.get(name, name)

            # aliases
            alias = {
                "column": ps.kwargs.get("column"),
                "close_col": ps.kwargs.get("close_col"),
                "high_col": ps.kwargs.get("high_col"),
                "low_col": ps.kwargs.get("low_col"),
                "open_col": ps.kwargs.get("open_col"),
                "volume_col": ps.kwargs.get("volume_col"),
            }

            if source in alias and alias[source] is not None:
                source = alias[source]

            canonical = {
                "open_": "open",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            source = canonical.get(source, source)
            col = f"{tf}_{source}" if tf else source

            if col not in row.index:
                logger.warning(
                    "Missing column %s for %s",
                    col,
                    spec.name,
                )
                return None

            values.append(row[col])

        raw = obj.update(*values)

        return self._normalize_output(spec, raw)

    # -------------------------------------------------------- 5.2===
    def _build_live_instance_old1(self, spec, ps):
        # "TODO": candidate for removal
        # فقط کلیدهایی که در امضای سازنده وجود دارند را نگه دار
        sig = inspect.signature(spec.fn)
        valid_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in ps.kwargs.items() if k in valid_keys}
        return spec.fn(**filtered_kwargs)

    # -------------------------------------------------------- 5.2 ===New
    def _build_live_instance(self, spec: IndicatorSpec, ps: ParsedSpec):
        """
        Create one stateful indicator instance.
        Only constructor parameters are forwarded.
        """
        sig = inspect.signature(spec.fn)
        kwargs = {}
        for name in sig.parameters:
            if name in ps.kwargs:
                kwargs[name] = ps.kwargs[name]
        return spec.fn(**kwargs)

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
    
    # -------------------------------------------------------- 7===
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
                    tf = getattr(ps, 'timeframe', None)
                    if tf:
                        col = f"{tf}_{col}"

                    if col not in columns_map:
                        columns_map[col] = [None] * n

        for idx, out in enumerate(outputs):
            if not isinstance(out, dict):
                continue
            for k, v in out.items():
                col = self._build_live_column_name(k, ps)
                tf = getattr(ps, 'timeframe', None)
                if tf:
                    col = f"{tf}_{col}"

                columns_map[col][idx] = v

        for col, values in columns_map.items():
            if len(values) < len(df):
                values.extend([None] * (len(df) - len(values)))
            df[col] = values
        
        return df

    # -------------------------------------------------------- 8===
    def _build_live_column_name(self, output_name, ps):

        raw = ps.raw.replace("'", '"')   #KKK
        if output_name == ps.name:
            return f"{raw}_live"

        params_part = raw[len(ps.name):]
        return f"{output_name}{params_part}_live"

    # -------------------------------------------------------- 9===
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

    # -------------------------------------------------------- 10
    def _validate_contract(self, spec, df):
        # "TODO": remove or integrate
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
    
    # -------------------------------------------------------- 11
    def _build_graph(self, parsed_specs):
        graph = {}
        
        valid_nodes = {ps.raw.replace("'", '"') for ps in parsed_specs}   #KKK
        
        for ps in parsed_specs:
            spec = get_indicator(ps.name, self.mode)
            if spec is None:
                continue

            node = ps.raw.replace("'", '"')   #KKK
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
    
    # -------------------------------------------------------- 12
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

    # -------------------------------------------------------- 13 new===
    def process_live_data(self, dataset: MTFDataset) -> Optional[MTFDataset]:
        """
        اجرای FeatureEngine روی MTFDataset زنده.
        """

        if dataset is None:
            return None

        if len(dataset.frames) == 0:
            return dataset

        feature_specs = (self.config.get("features", {}).get("live_specs", []))

        if not feature_specs:
            return dataset

        try:
            return self.execute(dataset=dataset, specs=feature_specs, mode="live")
        except Exception:
            logger.exception("FeatureEngine live execution failed.")
            return None

    # -------------------------------------------------------- NEW ===
    def _build_update_arguments(self, obj: Any, row: pd.Series) -> list[Any]:
        """
        Build positional arguments for obj.update(...) by inspecting
        the update() signature and mapping parameter names to DataFrame columns.
        """

        sig = inspect.signature(obj.update)

        COLUMN_ALIASES = {
            "open": "open",
            "open_": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        args = []

        for name, param in sig.parameters.items():

            if name == "self":
                continue

            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            column = COLUMN_ALIASES.get(name, name)

            if column not in row.index:
                raise KeyError(
                    f"Column '{column}' required by "
                    f"{obj.__class__.__name__}.update() was not found."
                )

            args.append(row[column])

        return args
    
    # -------------------------------------------------------- END

# ===================================================================
# وابستگی متدهای کلاس DataHandler
# ===================================================================
""" class FeatureEngine       1   2   3   4  51  52   6   7   8   9  10  11  12  13
1   execute                  --  --  --  --  --  --  --  --  --  --  --  --  --  ok
2   _apply_spec              ok  --  --  --  --  --  --  --  --  --  --  --  --  --
3   _call_batch              --  ok  --  --  --  --  --  --  --  --  --  --  --  --
4   _apply_live              --  ok  --  --  --  --  --  --  --  --  --  --  --  --
5.1 _update_live             --  --  --  ok  --  --  --  --  --  --  --  --  --  --
5.2 _build_live_instance     --  --  --  --  --  --  --  --  --  --  --  --  --  -- Not Used
6   _merge                   --  ok  --  --  --  --  --  --  --  --  --  --  --  --
7   _attach_live_output      --  --  --  ok  --  --  --  --  --  --  --  --  --  --
8   _build_live_column_name  --  --  --  --  --  --  --  ok  --  --  --  --  --  --
9   _normalize_output        --  --  --  --  ok  --  --  --  --  --  --  --  --  --
10  _validate_contract       --  ok  --  --  --  --  --  --  --  --  --  --  --  --
11  _build_graph             ok  --  --  --  --  --  --  --  --  --  --  --  --  --
12  _resolve_order           ok  --  --  --  --  --  --  --  --  --  --  --  --  --
13  process_live_data        --  --  --  --  --  --  --  --  --  --  --  --  --  -- Not Used
"""


"""
نام مندها:
def execute(self, dataset: MTFDataset, specs: List[str], mode: str = "train") -> MTFDataset:
def _apply_spec(self, dataset: MTFDataset, ps: ParsedSpec, mode: str) -> MTFDataset:
def _call_batch(self, fn, dataset, ps):
def _apply_live(self, spec: IndicatorSpec, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
def _update_live(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
def _build_live_instance(self, spec, ps):
def _merge(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
def _attach_live_outputt(self, df: pd.DataFrame, ps: ParsedSpec, spec: IndicatorSpec, outputs):
def _build_live_column_name(self, output_name, ps):
def _normalize_output(self, spec, output):
def _validate_contract(self, spec, df):
def _build_graph(self, parsed_specs):
def _resolve_order(self, graph):
def process_live_data(self, dataset: MTFDataset) -> Optional[MTFDataset]:
"""