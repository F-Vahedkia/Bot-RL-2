# f03_features/feature_C_engine_5.py

from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import inspect
import logging

from f03_features.feature_C_registry_1 import get_indicator, IndicatorSpec
from f10_utils.parser import parse_spec, ParsedSpec
from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_B_cache_6 import (
    GLOBAL_FEATURE_CACHE,
    ExecutionContract,
    cached_compute,
)
logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

# =============================================================================
# ENGINE CORE (PURE FEATURE TRANSFORMER)
# =============================================================================
class FeatureEngine:

    # ===================================================== 0
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
    
    # ===================================================== 1
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

        # ---------------------------------------
        # Parse
        # ---------------------------------------
        for spec in specs:
            try:
                parsed_specs.append(parse_spec(spec))
            except Exception as e:
                logger.warning("Invalid spec skipped: %s | %s", spec, e)

        logger.info(
            "Parsed %d feature specs.",
            len(parsed_specs),
        )

        # ---------------------------------------
        # Dependency Graph
        # ---------------------------------------
        graph = self._build_graph(parsed_specs)
        ordered = self._resolve_order(graph)

        ordered_map = {ps.canonical: ps for ps in parsed_specs}

        # ---------------------------------------
        # Execute
        # ---------------------------------------
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

    # ===================================================== 2
    def _apply_spec_F(
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
                dataset=dataset,
                df=df,
                ps=ps,
            )
            dataset.replace(tf, out_df)
            return dataset

        return dataset
    

    def _apply_spec_B(
        self,
        dataset: MTFDataset,
        ps: ParsedSpec,
        mode: str,
    ) -> MTFDataset:
        """
        اجرای یک Feature روی DataFrame مربوط به timeframe مشخص.

        Contract:
        - Registry منبع حقیقت IndicatorSpec است.
        - انتخاب indicator بر اساس (name, mode) انجام می‌شود.
        - اعتبار DataFrame و mode توسط _validate_contract بررسی می‌شود.
        - در Batch، نتیجه توسط _call_batch به Dataset برگردانده می‌شود.
        - در Incremental، stateful computation توسط _apply_live انجام می‌شود.
        """

        # ---------------------------------------
        # 1. Resolve indicator from Registry
        # ---------------------------------------
        spec = get_indicator(ps.name, mode)

        if spec is None:
            logger.warning(
                "Indicator not found: %s (mode=%s)",
                ps.name,
                mode,
            )
            return dataset

        # ---------------------------------------
        # 2. Timeframe contract
        # ---------------------------------------
        tf = ps.timeframe

        if tf is None:
            logger.warning(
                "Spec has no timeframe: %s",
                ps.raw,
            )
            return dataset

        tf = tf.upper()

        # ---------------------------------------
        # 3. Dataset must contain requested timeframe
        # ---------------------------------------
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

        # ---------------------------------------
        # 4. Indicator Contract
        #
        # _validate_contract is the single validation point
        # inside FeatureEngine for required columns + mode.
        # ---------------------------------------
        if not self._validate_contract(spec, df):
            return dataset

        # ---------------------------------------
        # 5. Batch execution
        # ---------------------------------------
        if spec.is_batch_mode(mode):
            return self._call_batch(
                fn=spec.fn,
                dataset=dataset,
                ps=ps,
            )

        # ---------------------------------------
        # 6. Incremental / Live execution
        # ---------------------------------------
        if spec.is_incremental_mode(mode):
            out_df = self._apply_live(
                spec=spec,
                dataset=dataset,
                df=df,
                ps=ps,
            )

            dataset.replace(tf, out_df)
            return dataset

        # ---------------------------------------
        # 7. Unsupported mode
        # ---------------------------------------
        logger.warning(
            "Unsupported execution mode '%s' for indicator '%s'",
            mode,
            ps.name,
        )

        return dataset


    def _apply_spec_D(self, dataset: MTFDataset, ps: ParsedSpec, mode: str) -> MTFDataset:
        """
        اجرای یک Feature روی DataFrame مربوط به timeframe مشخص.

        Contract
        --------
        - Parser مسئول parse و canonicalize کردن Spec است.
        - Registry منبع حقیقت IndicatorSpec است.
        - Engine فقط orchestration اجرای Feature را انجام می‌دهد.
        - اعتبار mode و required columns توسط _validate_contract انجام می‌شود.
        - در Batch، اجرای محاسبه به _call_batch واگذار می‌شود.
        - در Incremental/Live، اجرای محاسبه به _apply_live واگذار می‌شود.
        - هر Symbol مسیر مستقل خود را در MTFDataset طی می‌کند.
        """

        # -------------------------------------------------
        # 1. Resolve Indicator from Registry
        # -------------------------------------------------
        spec = get_indicator(ps.name, mode)

        if spec is None:
            logger.warning(
                "Indicator not found in registry: %s (mode=%s)",
                ps.name,
                mode,
            )
            return dataset

        # -------------------------------------------------
        # 2. Timeframe Contract
        # -------------------------------------------------
        tf = ps.timeframe

        if tf is None:
            logger.warning(
                "Spec has no timeframe: %s",
                ps.raw,
            )
            return dataset

        tf = tf.upper()

        # -------------------------------------------------
        # 3. Dataset Contract
        #
        # FeatureEngine فقط روی همان timeframeای کار می‌کند
        # که در ParsedSpec مشخص شده است.
        # -------------------------------------------------
        try:
            df = dataset.get(tf)
        except KeyError:
            logger.warning(
                "Dataset has no dataframe for timeframe %s "
                "(indicator=%s)",
                tf,
                ps.name,
            )
            return dataset

        if df.empty:
            logger.warning(
                "Dataset dataframe is empty for timeframe %s "
                "(indicator=%s)",
                tf,
                ps.name,
            )
            return dataset

        # -------------------------------------------------
        # 4. Indicator Contract
        #
        # تمام کنترل‌های required_cols و supported mode
        # در یک نقطه متمرکز شده‌اند.
        # -------------------------------------------------
        if not self._validate_contract(spec, df):
            return dataset

        # -------------------------------------------------
        # 5. Batch Execution
        # -------------------------------------------------
        if spec.is_batch_mode(mode):
            return self._call_batch(
                fn=spec.fn,
                dataset=dataset,
                ps=ps,
            )

        # -------------------------------------------------
        # 6. Incremental / Live Execution
        # -------------------------------------------------
        if spec.is_incremental_mode(mode):
            out_df = self._apply_live(
                spec=spec,
                dataset=dataset,
                df=df,
                ps=ps,
            )

            dataset.replace(tf, out_df)
            return dataset

        # -------------------------------------------------
        # 7. Unsupported Mode
        # -------------------------------------------------
        logger.warning(
            "Unsupported execution mode '%s' for indicator '%s'",
            mode,
            ps.name,
        )

        return dataset

    # ===================================================== 3
    def _call_batch_C(self, fn, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
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

        # ---------------------------------------
        # positional arguments
        # ---------------------------------------
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

        # ---------------------------------------
        # keyword arguments
        # ---------------------------------------
        kwargs.update(ps.kwargs)

        # ---------------------------------------
        # فقط پارامترهای موجود در Signature
        # ---------------------------------------
        filtered = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters
        }

        out = fn(df, **filtered)
        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{ps.name} must return DataFrame.")

        raw = ps.canonical
        if len(out.columns) == 1:
            out.columns = [raw]
        else:
            out.columns = [f"{raw}::{c}" for c in out.columns]
        dataset.replace(tf, self._merge(df, out))
        return dataset


    def _call_batch_B(self, fn, dataset: MTFDataset, ps: ParsedSpec) -> MTFDataset:
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

        # ---------------------------------------
        # positional arguments
        # ---------------------------------------
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

        # ------------------------------------------
        # Canonical keyword arguments
        #
        # Parser/Registry قبلاً aliasها را
        # به نام canonical تبدیل کرده‌اند.
        # ------------------------------------------
        kwargs.update(ps.kwargs)

        # ---------------------------------------
        # Canonicalize parameter names through Registry
        # ---------------------------------------
        parameters_spec = getattr(
            get_indicator(ps.name, self.mode),
            "parameters",
            None,
        )
        if parameters_spec:
            alias_map = {}

            for param in parameters_spec:
                for alias in getattr(param, "aliases", ()):
                    alias_map[alias] = param.name

            canonical_kwargs = {}

            for key, value in kwargs.items():
                canonical_key = alias_map.get(key, key)

                if canonical_key in canonical_kwargs:
                    raise ValueError(
                        f"Duplicate parameter for indicator "
                        f"'{ps.name}': '{key}' conflicts with "
                        f"'{canonical_key}'."
                    )

                canonical_kwargs[canonical_key] = value

            kwargs = canonical_kwargs
        # ---------------------------------------
        # فقط پارامترهای موجود در Signature          old
        # فقط پارامترهایی که تابع واقعاً می‌شناسد     new
        # ---------------------------------------
        accepted = set(sig.parameters)
        filtered = {
            k: v
            for k, v in kwargs.items()
            if k in accepted
        }

        out = fn(df, **filtered)
        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"{ps.name} must return DataFrame.")

        raw = ps.canonical
        if len(out.columns) == 1:
            out.columns = [raw]
        else:
            out.columns = [f"{raw}::{c}" for c in out.columns]
        dataset.replace(tf, self._merge(df, out))
        return dataset


    def _call_batch_A(
        self,
        fn,
        dataset: MTFDataset,
        ps: ParsedSpec,
    ) -> MTFDataset:
        """
        اجرای یک Indicator در Batch mode.

        Contract:
        - Parser/Registry منبع canonical parameter names هستند.
        - positional arguments از ParsedSpec به ترتیب Signature تابع
        متصل می‌شوند.
        - keyword arguments از ParsedSpec دریافت می‌شوند.
        - aliasها در صورت وجود، با تعریف Registry به canonical name
        تبدیل می‌شوند.
        - فقط پارامترهایی که تابع Batch واقعاً در Signature خود دارد
        به آن ارسال می‌شوند.
        - خروجی باید DataFrame باشد.
        - نام Feature خروجی بر اساس ps.canonical ساخته می‌شود.
        """

        tf = ps.timeframe
        if tf is None:
            logger.warning(
                "Batch execution requires timeframe: %s",
                ps.raw,
            )
            return dataset

        tf = tf.upper()
        df = dataset.get(tf)

        # --------------------------------------------------
        # 1. Resolve function signature
        # --------------------------------------------------
        sig = inspect.signature(fn)
        parameters = list(sig.parameters.values())

        # اولین پارامتر Batch Indicator باید DataFrame باشد.
        if parameters and parameters[0].name == "df":
            parameters = parameters[1:]

        kwargs: Dict[str, Any] = {}

        # --------------------------------------------------
        # 2. Positional arguments
        # --------------------------------------------------
        positional_params = [
            p
            for p in parameters
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if len(ps.args) > len(positional_params):
            raise TypeError(
                f"Too many positional arguments for indicator "
                f"'{ps.name}': expected at most "
                f"{len(positional_params)}, got {len(ps.args)}"
            )

        for value, param in zip(ps.args, positional_params):
            kwargs[param.name] = value

        # --------------------------------------------------
        # 3. Keyword arguments from ParsedSpec
        # --------------------------------------------------
        for key, value in ps.kwargs.items():

            # جلوگیری از تداخل positional + keyword
            if key in kwargs:
                raise ValueError(
                    f"Duplicate parameter for indicator "
                    f"'{ps.name}': '{key}' is supplied both "
                    f"positionally and by keyword."
                )

            kwargs[key] = value

        # --------------------------------------------------
        # 4. Canonicalize parameter names through Registry
        # --------------------------------------------------
        registry_spec = get_indicator(ps.name, self.mode)

        parameters_spec = getattr(
            registry_spec,
            "parameters",
            None,
        )

        if parameters_spec:
            alias_map: Dict[str, str] = {}

            for param in parameters_spec:
                for alias in getattr(param, "aliases", ()):
                    alias_map[alias] = param.name

            canonical_kwargs: Dict[str, Any] = {}

            for key, value in kwargs.items():
                canonical_key = alias_map.get(key, key)

                if canonical_key in canonical_kwargs:
                    raise ValueError(
                        f"Duplicate parameter for indicator "
                        f"'{ps.name}': '{key}' conflicts with "
                        f"'{canonical_key}'."
                    )

                canonical_kwargs[canonical_key] = value

            kwargs = canonical_kwargs

        # --------------------------------------------------
        # 5. Filter against actual Batch function Signature
        # --------------------------------------------------
        accepted = {
            name
            for name, parameter in sig.parameters.items()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted
        }

        # --------------------------------------------------
        # 6. Execute Batch Indicator
        # --------------------------------------------------
        out = fn(df, **filtered)

        if out is None:
            return dataset

        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"Batch indicator '{ps.name}' must return DataFrame, "
                f"got {type(out).__name__}."
            )

        # --------------------------------------------------
        # 7. Canonical Feature naming
        #
        # raw نباید برای نام‌گذاری استفاده شود؛
        # canonical قرارداد پایدار Feature Engine است.
        # --------------------------------------------------
        canonical = ps.canonical

        if len(out.columns) == 1:
            out = out.copy()
            out.columns = [canonical]
        else:
            out = out.copy()
            out.columns = [
                f"{canonical}::{column}"
                for column in out.columns
            ]

        # --------------------------------------------------
        # 8. Merge result into the same Symbol/timeframe dataset
        # --------------------------------------------------
        dataset.replace(
            tf,
            self._merge(df, out),
        )

        return dataset

    # ===================================================== 4
    def _apply_live(self,
                    spec: IndicatorSpec,
                    dataset: MTFDataset,
                    df: pd.DataFrame,
                    ps: ParsedSpec
    ) -> pd.DataFrame:

        # بلوک حذف شده قدیمی:
        # SAFE LIVE CACHE KEY (prevents cross-context state pollution)     pollution: آلودگی
        # context = getattr(df, "attrs", {}).get("context", {})
        # ctx_id = (
        #     context.get("symbol", "NA"),
        #     context.get("tf", "NA"),
        #     context.get("session", "NA"),
        # )
        # بلوک اضافه شده جدید:
        ctx_id = (
            dataset.symbol,
            ps.timeframe,
            dataset.base_tf,
        )
        key = f"{self.mode}::{ps.canonical}::{ctx_id}"                              # حذف شده ی قدیمی
        key = f"{self.mode}::{dataset.symbol}::{dataset.base_tf}::{ps.canonical}"   # اضافه شده جدید

        logger.info("🔄 Applying LIVE %s (mode=%s)", ps.name, self.mode)

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
    
    # ===================================================== 5
    def _update_live_G(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
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
            canon_price = {       # canonical prices
                "open_": "open",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            source = canon_price.get(source, source)
            # ---------- بخش قدیمی حذف شده
            # col = f"{tf}_{source}" if tf else source
            # ---------- بخش جدید اضافه شده
            col = source
            if col not in row.index:
                prefixed = f"{tf}_{source}"
                if prefixed in row.index:
                    col = prefixed
                else:
                    logger.warning("Missing column %s for %s", source, spec.name)
                    return None
            # ---------- پایان بخش جدید اضافه شده
            if col not in row.index:
                logger.warning("Missing column %s for %s", col, spec.name)
                return None
            values.append(row[col])
        raw = obj.update(*values)
        return self._normalize_output(spec, raw)


    def _update_live_F(self, obj: Any, row: pd.Series, spec: IndicatorSpec, ps: ParsedSpec):
        """
        اجرای یک update روی Indicator Live.

        ورودی‌های update() از روی Signature واقعی متد update
        استخراج می‌شوند و به ستون متناظر DataFrame متصل می‌گردند.

        نکته معماری:
        - نام Indicator در این متد hard-code نشده است.
        - برای ورودی‌های عمومی مانند `value` و `column`،
        ستون منبع از پارامتر canonical `column` در ParsedSpec گرفته می‌شود.
        - ورودی‌های OHLCV مستقیماً به ستون canonical متناظر متصل می‌شوند.
        - ابتدا نام ساده ستون بررسی می‌شود و در صورت نبودن آن،
        نام دارای prefix تایم‌فریم بررسی می‌گردد.
        """

        sig = inspect.signature(obj.update)
        values = []
        tf = ps.timeframe

        for name in sig.parameters:

            # ---------------------------------------
            # 1. تعیین منبع داده برای ورودی update()
            # ---------------------------------------
            source = name

            # `value` و `column` ورودی‌های عمومی هستند و باید
            # از column موجود در ParsedSpec تغذیه شوند.
            if name in {"value", "column"}:
                source = ps.kwargs.get("column", "close")

            # ---------------------------------------
            # 2. تبدیل نام‌های ورودی به نام canonical ستون
            # ---------------------------------------
            canon_price = {
                "open_": "open",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
            source = canon_price.get(source, source)

            # ---------------------------------------
            # 3. پیدا کردن ستون در DataFrame
            #
            # ابتدا نام canonical بررسی می‌شود.
            # اگر وجود نداشت، نام TF-prefixed بررسی می‌شود.
            # ---------------------------------------
            col = source
            if col not in row.index:
                prefixed = f"{tf}_{source}"

                if prefixed in row.index:
                    col = prefixed
                else:
                    logger.warning(
                        "Missing column %s for %s",
                        source,
                        spec.name,
                    )
                    return None

            values.append(row[col])

        # ------------------------------------------
        # 4. اجرای یک update با ورودی‌های استخراج‌شده
        # ------------------------------------------
        raw = obj.update(*values)

        # ------------------------------------------
        # 5. تبدیل خروجی scalar/tuple/list به ساختار استاندارد
        # ------------------------------------------
        return self._normalize_output(spec, raw)

    # ===================================================== 6
    def _build_live_instance_H(self, spec: IndicatorSpec, ps: ParsedSpec):
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


    def _build_live_instance_D(self, spec: IndicatorSpec, ps: ParsedSpec):
        """
        ساخت یک instance از اندیکاتور Live.

        قرارداد:
        - Parser مسئول parse و canonicalize کردن پارامترهای Spec است.
        - Registry مسئول تعریف پارامترهای معتبر و aliasهای آنهاست.
        - Engine فقط پارامترهای canonical موجود در ParsedSpec را
        به constructor اندیکاتور Live منتقل می‌کند.
        - پارامترهای مربوط به انتخاب ستون (column) به constructor
        فرستاده نمی‌شوند، مگر اینکه constructor واقعاً آن را بپذیرد.
        """
        sig = inspect.signature(spec.fn)

        # پارامترهای canonical که Parser/Registry در اختیار Engine
        # قرار داده‌اند.
        parsed_kwargs = dict(ps.kwargs)
        kwargs = {}
        for name, parameter in sig.parameters.items():

            # **kwargs در constructor، پارامتر واقعی Indicator نیست.
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if name in parsed_kwargs:
                kwargs[name] = parsed_kwargs[name]

        return spec.fn(**kwargs)


    def _build_live_instance_E(self, spec: IndicatorSpec, ps: ParsedSpec):
        """
        ساخت یک instance از اندیکاتور Live.

        Contract:
        - Parser مسئول parse کردن Spec است.
        - Registry مسئول تعریف نام canonical پارامترها و aliasهاست.
        - Engine مسئول تطبیق ParsedSpec با constructor واقعی Indicator است.
        - آرگومان‌های positional و keyword هر دو پشتیبانی می‌شوند.
        - aliasها در اینجا به canonical name تبدیل می‌شوند.
        - فقط پارامترهایی که constructor واقعاً می‌پذیرد ارسال می‌شوند.
        - پارامترهای مربوط به DataFrame column مانند `column`
        فقط در صورتی ارسال می‌شوند که constructor واقعاً آنها را
        در Signature خود داشته باشد.
        """

        sig = inspect.signature(spec.fn)

        # ---------------------------------------
        # 1. دریافت پارامترهای تعریف‌شده در Registry
        # ---------------------------------------
        parameters_spec = getattr(spec, "parameters", None)

        alias_map = {}

        if parameters_spec:
            for param in parameters_spec:
                for alias in getattr(param, "aliases", ()):
                    alias_map[alias] = param.name

        # ---------------------------------------
        # 2. تبدیل positional arguments به canonical kwargs
        #
        # ترتیب positionalها بر اساس پارامترهای واقعی
        # constructor تعیین می‌شود.
        # ---------------------------------------
        constructor_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        kwargs = {}

        for value, param in zip(ps.args, constructor_params):
            kwargs[param.name] = value

        # ---------------------------------------
        # 3. افزودن keyword arguments
        # ---------------------------------------
        for key, value in ps.kwargs.items():

            canonical_key = alias_map.get(key, key)

            if canonical_key in kwargs:
                raise ValueError(
                    f"Duplicate parameter for indicator "
                    f"'{ps.name}': '{key}' conflicts with "
                    f"'{canonical_key}'."
                )

            kwargs[canonical_key] = value

        # ---------------------------------------
        # 4. فقط پارامترهایی که constructor واقعاً می‌شناسد
        # ---------------------------------------
        accepted = {
            name
            for name, parameter in sig.parameters.items()
            if parameter.kind != inspect.Parameter.VAR_KEYWORD
        }

        filtered = {
            key: value
            for key, value in kwargs.items()
            if key in accepted
        }

        # ---------------------------------------
        # 5. ساخت instance
        # ---------------------------------------
        return spec.fn(**filtered)

    # ===================================================== 7
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
    
    # ===================================================== 8
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

    # ===================================================== 9
    def _build_live_column_name(self, output_name, ps):
        raw = ps.canonical
        if output_name == ps.name:
            return f"{raw}_live"

        params_part = raw[len(ps.name):]
        return f"{output_name}{params_part}_live"

    # ===================================================== 10
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

    # ===================================================== 11
    def _validate_contract(self, spec, df):
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

    # ===================================================== 14
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
