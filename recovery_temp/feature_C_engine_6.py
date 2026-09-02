# f03_features/feature_C_engine_6.py

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import inspect
import logging

from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_C_registry_1 import get_indicator, IndicatorSpec
from f10_utils.functions.parser import parse_spec, ParsedSpec

# این بخش برای train و optimize است.
# هدف: اگر یک feature قبلاً محاسبه شده، دوباره محاسبه نشود.
from f03_features.feature_B_cache_6 import (
    GLOBAL_FEATURE_CACHE_MANAGER,
    ExecutionContract,
    cached_compute,
)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

# =============================================================================
# ENGINE CORE (PURE FEATURE TRANSFORMER)
# =============================================================================
class FeatureEngine:

    # ===================================================== 0 خوانده شد و فهمیده شد.
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        این متد 5 کار انجام میدهد:
            1) بررسی می‌کند config معتبر است.
            2) config را ذخیره می‌کند.
            3) mode را خالی می‌گذارد.
            4) cache مربوط به اندیکاتورهای live را آماده می‌کند.
            5) قرارداد نسخه‌بندی اجرای Engine را می‌سازد.
        """
        if config is None:
            raise ValueError("config is required for FeatureEngine")

        if not isinstance(config, dict):
            raise TypeError(f"config must be dict, got {type(config).__name__}")

        self.config: Dict[str, Any] = config
        self.mode: Optional[str] = None     # تعیین واقعی این پارامتر در تابع execute() انجام میشود.
        self._run_id: int = 0               # این پارامتر در ابتدای هر execute() یک واحد زیاد میشود.
        self._live_cache: Dict[str, Any] = {}
        self._contract = ExecutionContract(
            engine_version="1",
            resolver_version="1",
            config_version="1",
            registry_version="1",
        )
    
    # ===================================================== 1 خوانده شد و تقریباً فهمیده شد.
    def execute(self, dataset: MTFDataset, specs: List[str], mode: str = "train") -> MTFDataset:
        """
        این متد موتور اصلی اجرای FeatureEngine است.
        کار اصلی که این متد انجام میدهد:
            1) یک MTFDataset می‌گیرد.
            2) لیست feature specification ها را می‌گیرد.
            3) هر specification را توسط Parser تبدیل می‌کند.
            4) سپس هر feature را اجرا می‌کند (batch یا live).
            5) در نهایت همان MTFDataset را با ستون‌های feature جدید برمی‌گرداند.
        """
        # -- (1) -- کنترل داده های ورودی
        if dataset is None:
            raise ValueError("dataset is required")

        if len(dataset.frames) == 0:
            logger.info("MTFDataset is empty.")
            return dataset

        # ------------------- Daleted=1
        # previous_mode = self.mode
        # self.mode = mode
        # self._run_id += 1
        # if mode in {"train", "optimize"}:
        #     self._live_cache.clear()
        # elif previous_mode is not None and previous_mode != mode:
        #     self._live_cache.clear()
        # ------------------- Added=1
        # -- (2) -- تنظیم مود و ران-آی دی
        self.mode = mode
        self._run_id += 1
        # ------------------- End=1

        # -- (3) -- تبدیل اسپک های متنی به ساختار داخلی از نوع ParsedSpec
        parsed_specs: List[ParsedSpec] = []
        for raw_spec in specs:
            try:
                parsed_specs.append(parse_spec(raw_spec))
            except Exception as exc:
                logger.warning("Invalid feature spec skipped: %s | %s", raw_spec, exc)

        # -- (4) -- بررسی وجود و اعنبار اسپکهای. در واقع بررسی اینکه اصلاً چیزی برای محاسبه وجود دارد یا نه.
        if not specs:
            return dataset

        if not parsed_specs:
            raise ValueError("No valid feature specifications were parsed.")

        # -- (5) -- فرصتی برای مرتب سازی ترتیب اجرای فیچرها
        """
        در اینجا میشود که مرتب‌سازی بر اساس dependency
        یا اجرای featureهای پایه قبل از feature های وابسته را گنجانید
        """
        ordered_specs = parsed_specs

        # -- (6) -- حلقه اصلی
        for ps in ordered_specs:
            if mode in {"train", "optimize"}:
                dataset = cached_compute(
                    # cache=GLOBAL_FEATURE_CACHE,
                    cache=GLOBAL_FEATURE_CACHE_MANAGER.get(dataset.symbol), # type: FeatureCache
                    dataset=dataset,                                        # type: 
                    ps=ps,                                                  # type: ParsedSpec
                    mode=mode,                                              # type: str
                    contract=self._contract,                                # type: ExecutionContract
                    # در سطر زیر محاسبات بچ انجام میشود
                    compute_fn=lambda ds=dataset, p=ps:                     # type: Callable[[], MTFDataset]
                        self._apply_spec(dataset=ds, ps=p, mode=mode)
                )
            else:
                dataset = self._apply_spec(dataset=dataset, ps=ps, mode=mode)

        logger.info("Feature calculation completed on %d timeframes.", len(dataset.frames))
        return dataset

    # ===================================================== 2
    def _apply_spec(self, dataset: MTFDataset, ps: ParsedSpec, mode: str) -> MTFDataset:
        """
        Execute one feature according to the Registry contract.
        """

        spec = get_indicator(ps.name, mode)
        if spec is None:
            logger.warning("Indicator not found in registry: %s (mode=%s)", ps.name, mode)
            return dataset

        tf = ps.timeframe
        if tf is None:
            logger.warning("Spec has no timeframe: %s", ps.raw)
            return dataset

        tf = tf.upper()
        try:
            df = dataset.get(tf)
        except KeyError:
            logger.warning("Dataset has no dataframe for timeframe %s (indicator=%s)", tf, ps.name)
            return dataset

        if df.empty:
            logger.warning("Dataset dataframe is empty for timeframe %s (indicator=%s)", tf, ps.name)
            return dataset

        if not self._validate_contract(
            spec=spec,
            df=df,
            mode=mode,
            timeframe=tf,
        ):
            return dataset

        if spec.is_batch_mode(mode):
            return self._call_batch(
                spec=spec,
                dataset=dataset,
                ps=ps,
            )

        if spec.is_incremental_mode(mode):
            out_df = self._apply_live(
                spec=spec,
                dataset=dataset,
                df=df,
                ps=ps,
            )
            dataset.replace(tf, out_df)
            return dataset

        logger.warning("Unsupported execution mode '%s' for indicator '%s'", mode, ps.name)
        return dataset

    # ===================================================== 3
    def _call_batch_old(self, spec: IndicatorSpec, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        Execute one Batch indicator using the Registry contract.
        """
        tf = ps.timeframe

        if tf is None:
            raise ValueError(f"Batch execution requires timeframe: {ps.raw}")

        tf = tf.upper()
        df = dataset.get(tf)

        fn = spec.fn
        sig = inspect.signature(fn)

        parameters_spec = getattr(spec, "parameters", None)
        alias_map: Dict[str, str] = {}

        if parameters_spec:
            for parameter in parameters_spec:
                for alias in getattr(parameter, "aliases", ()):
                    alias_map[alias] = parameter.name

        kwargs: Dict[str, Any] = {}

        for key, value in ps.kwargs.items():
            canonical_key = alias_map.get(key, key)

            if canonical_key in kwargs:
                raise ValueError(
                    f"Duplicate parameter for indicator "
                    f"'{ps.name}': '{key}' conflicts with "
                    f"'{canonical_key}'."
                )

            kwargs[canonical_key] = value

        try:
            bound = sig.bind(df, *ps.args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Invalid arguments for Batch indicator "
                f"'{ps.name}': {exc}"
            ) from exc

        out = fn(*bound.args, **bound.kwargs)

        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"Batch indicator '{ps.name}' must return DataFrame, "
                f"got {type(out).__name__}."
            )

        out = out.copy()

        output_names = getattr(spec, "output_names", None)
        canonical = ps.canonical

        if output_names:
            if len(output_names) != len(out.columns):
                raise ValueError(
                    f"Output count mismatch for '{ps.name}': "
                    f"registry defines {len(output_names)} outputs, "
                    f"function returned {len(out.columns)}."
                )

            if len(output_names) == 1:
                out.columns = [canonical]
            else:
                out.columns = [
                    f"{name}{canonical[len(ps.name):]}"
                    for name in output_names
                ]

        elif len(out.columns) == 1:
            out.columns = [canonical]

        else:
            out.columns = [
                f"{canonical}::{column}"
                for column in out.columns
            ]

        dataset.replace(
            tf,
            self._merge(df, out),
        )

        return dataset

    # ------------------/////
    def _call_batch(self, spec: IndicatorSpec, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
        """
        Execute one Batch indicator using the validated Registry contract.
        """

        tf = ps.timeframe

        if tf is None:
            raise ValueError(f"Batch execution requires timeframe: {ps.raw}")

        tf = tf.upper()
        df = dataset.get(tf)

        kwargs = dict(ps.kwargs)
        out = spec.fn(df, **kwargs)

        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"Batch indicator '{ps.name}' must return DataFrame, "
                f"got {type(out).__name__}."
            )

        out = out.copy()

        output_names = spec.output_names
        canonical = ps.canonical

        if output_names:
            if len(output_names) != len(out.columns):
                raise ValueError(
                    f"Output count mismatch for '{ps.name}': "
                    f"registry defines {len(output_names)} outputs, "
                    f"function returned {len(out.columns)}."
                )

            if len(output_names) == 1:
                out.columns = [canonical]
            else:
                params_part = canonical[len(ps.name):]
                out.columns = [
                    f"{name}{params_part}"
                    for name in output_names
                ]

        elif len(out.columns) == 1:
            out.columns = [canonical]

        else:
            out.columns = [
                f"{canonical}::{column}"
                for column in out.columns
            ]

        dataset.replace(
            tf,
            self._merge(df, out),
        )

        return dataset

    # ===================================================== 4
    def _apply_live(self, spec: IndicatorSpec, dataset: MTFDataset, df: pd.DataFrame, ps: ParsedSpec) -> pd.DataFrame:
        """
        Execute one stateful indicator over the current MTF frame.
        """

        tf = ps.timeframe.upper()

        key = (
            self.mode,
            dataset.symbol,
            dataset.base_tf,
            tf,
            ps.canonical,
        )

        logger.info(
            "Applying incremental feature %s (mode=%s)",
            ps.name,
            self.mode,
        )

        if key not in self._live_cache:
            self._live_cache[key] = self._build_live_instance(
                spec,
                ps,
            )

        obj = self._live_cache[key]

        outputs = []

        for _, row in df.iterrows():
            try:
                outputs.append(
                    self._update_live(
                        obj=obj,
                        row=row,
                        spec=spec,
                        ps=ps,
                    )
                )
            except Exception:
                logger.exception(
                    "Incremental update failed for %s",
                    ps.raw,
                )
                outputs.append(None)

        return self._attach_live_output(
            df=df,
            ps=ps,
            spec=spec,
            outputs=outputs,
        )

    # ===================================================== 5
    def _update_live_old(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        """
        Feed one row into a stateful indicator.
        """

        sig = inspect.signature(obj.update)
        values = []

        column_aliases = {
            "open_": "open",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }

        selected_column = ps.kwargs.get("column")

        for parameter in sig.parameters.values():

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            name = parameter.name

            if name in {"value", "column"}:
                source = selected_column or "close"
            else:
                source = name

            source = column_aliases.get(source, source)

            if source in row.index:
                values.append(row[source])
                continue

            tf = ps.timeframe.upper()

            prefixed = f"{tf}_{source}"

            if prefixed in row.index:
                values.append(row[prefixed])
                continue

            if parameter.default is not inspect.Parameter.empty:
                values.append(parameter.default)
                continue

            logger.warning(
                "Missing live input column '%s' for %s",
                source,
                spec.name,
            )
            return None

        raw = obj.update(*values)

        return self._normalize_output(
            spec=spec,
            output=raw,
        )

    # ------------------/////
    def _update_live(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        """
        Feed one row into a stateful indicator.
        Input order is defined by the Registry's required_cols contract.
        """
        values = []
        required_cols = list(spec.required_cols)
        selected_column = ps.kwargs.get("column")

        if selected_column:
            required_cols = [selected_column]

        tf = ps.timeframe.upper() if ps.timeframe else None

        for source in required_cols:

            if source in row.index:
                values.append(row[source])
                continue

            if tf is not None:
                prefixed = f"{tf}_{source}"

                if prefixed in row.index:
                    values.append(row[prefixed])
                    continue

            logger.warning(
                "Missing live input column '%s' for %s",
                source,
                spec.name,
            )
            return None

        raw = obj.update(*values)

        return self._normalize_output(
            spec=spec,
            output=raw,
        )

    # ===================================================== 6
    def _build_live_instance_old(self, spec: IndicatorSpec, ps: ParsedSpec) -> Any:
        """
        Build one stateful Live indicator instance.
        """

        sig = inspect.signature(spec.fn)

        parameters_spec = getattr(spec, "parameters", None)
        alias_map: Dict[str, str] = {}

        if parameters_spec:
            for parameter in parameters_spec:
                for alias in getattr(parameter, "aliases", ()):
                    alias_map[alias] = parameter.name

        kwargs: Dict[str, Any] = {}

        for key, value in ps.kwargs.items():
            canonical_key = alias_map.get(key, key)

            if canonical_key in kwargs:
                raise ValueError(
                    f"Duplicate parameter for indicator "
                    f"'{ps.name}': '{key}' conflicts with "
                    f"'{canonical_key}'."
                )

            kwargs[canonical_key] = value

        try:
            bound = sig.bind(*ps.args, **kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Invalid constructor arguments for "
                f"Live indicator '{ps.name}': {exc}"
            ) from exc

        return spec.fn(*bound.args, **bound.kwargs)

    # ------------------/////
    def _build_live_instance(self, spec: IndicatorSpec, ps: ParsedSpec) -> Any:
        """
        Build one stateful Live indicator instance.

        Parser has already normalized, validated and defaulted
        all Registry parameters.
        """

        kwargs = dict(ps.kwargs)

        return spec.fn(**kwargs)

    # ===================================================== 7
    def _merge(self, df: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        """
        Merge feature columns without modifying the input frame.
        """

        if out is None or out.empty:
            return df

        result = df.copy()

        for col in out.columns:
            if col not in result.columns:
                result[col] = out[col]
                continue

            if result[col].equals(out[col]):
                continue

            raise ValueError(
                f"Feature column collision: '{col}'"
            )

        return result

    # ===================================================== 8
    def _attach_live_output(self, df: pd.DataFrame, ps: ParsedSpec, spec: IndicatorSpec, outputs) -> pd.DataFrame:

        if not outputs:
            return df

        result = df.copy()
        n = len(result)

        columns_map: Dict[str, List[Any]] = {}

        for idx in range(n):
            out = outputs[idx]

            if not isinstance(out, dict):
                continue

            for name, value in out.items():
                col = self._build_live_column_name(
                    output_name=name,
                    ps=ps,
                )

                if col not in columns_map:
                    columns_map[col] = [None] * n

                columns_map[col][idx] = value

        for col, values in columns_map.items():
            result[col] = values

        return result

    # ===================================================== 9
    def _build_live_column_name(self, output_name: str, ps: ParsedSpec) -> str:
        canonical = ps.canonical

        if output_name == ps.name:
            return canonical

        params_part = canonical[len(ps.name):]

        return f"{output_name}{params_part}"

    # ===================================================== 10
    def _normalize_output(self, spec: IndicatorSpec, output: Any) -> Dict[str, Any]:

        if isinstance(output, dict):
            return output

        names = getattr(spec, "output_names", None)

        if not names:
            return {
                spec.name: output,
            }

        if isinstance(output, (tuple, list)):
            return {
                name: output[i] if i < len(output) else None
                for i, name in enumerate(names)
            }

        if len(names) == 1:
            return {
                names[0]: output,
            }

        raise TypeError(
            f"Indicator '{spec.name}' returned scalar output "
            f"but defines multiple output_names."
        )

    # ===================================================== 11
    def _validate_contract(self, spec: IndicatorSpec, df: pd.DataFrame, mode: str, timeframe: str) -> bool:
        if not spec.supports(mode):
            logger.warning(
                "Contract violation [%s]: mode '%s' is not supported",
                spec.name,
                mode,
            )
            return False

        missing = []

        for column in spec.required_cols:
            if column in df.columns:
                continue

            prefixed = f"{timeframe}_{column}"

            if prefixed in df.columns:
                continue

            missing.append(column)

        if missing:
            logger.warning(
                "Contract violation [%s]: missing columns %s",
                spec.name,
                missing,
            )
            return False

        return True

    # ===================================================== 12
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

    # ===================================================== END




















"""
    # ===================================================== 12
    def _build_graph(self, parsed_specs):
        graph = {}
        valid_nodes = {ps.canonical for ps in parsed_specs}
        for ps in parsed_specs:
            spec = get_indicator(ps.name, self.mode)
            if spec is None:
                continue

            node = ps.canonical
            graph[node] = []

            deps = getattr(spec, "depends_on", None)

            # STRICT DEPENDENCY VALIDATION
            if isinstance(deps, (list, tuple, set)):
                for dep in deps:
                    # Dependencyها نیز باید Canonical باشند.
                    dep = dep.replace("'", '"')
                    if dep in valid_nodes:
                        graph[node].append(dep)
                    else:
                        logger.warning("Unknown dependency ignored [%s -> %s]", node, dep)

        return graph
    
    # ===================================================== 13
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


"""