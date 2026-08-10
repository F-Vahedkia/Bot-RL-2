# f03_features/data_pipeline.py

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from f02_data.data_handler_F_2_3_n import BuildParams
from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_B_graph import FeatureGraph
from f03_features.feature_C_engine_4 import FeatureEngine
from f03_features.feature_B_store import FeatureStoreV2
from f03_features.observation_B_builder import ObservationBuilder

logger = logging.getLogger(__name__)

class DataPipeline:
    """ اتصال رسمی بین لایه Data و Features.
    جریان داده:
        MTFDataset ->  FeatureEngine -> FeatureStore -> ObservationBuilder -> Observation
    """
    # -------------------------------------------------------------------------
    def __init__(
        self,
        feature_engine: FeatureEngine,
        feature_store: FeatureStoreV2,
        observation_builder: ObservationBuilder,
        feature_graph: FeatureGraph,
        feature_specs: List[str],
        symbol: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.feature_engine = feature_engine
        self.feature_store = feature_store
        self.observation_builder = observation_builder
        self.feature_graph = feature_graph
        self.feature_specs = list(feature_specs)

        self.config = config or {}

        # --- Symbol awareness ----
        self.symbol = symbol
        logger.info("DataPipeline initialized for symbol=%s", self.symbol)

        # Runtime state
        self._dataset: Optional[MTFDataset] = None
        self._features: Optional[MTFDataset] = None
        self._observation = None

        # Cache
        self.feature_cache: Dict[str, Any] = {}

        #####################
        self._dataset = None
        self._features = None
        self._observation = None

        self.feature_cache = {}

        # ASSUMPTION
        self.specs = list(feature_specs)

        # ASSUMPTION
        self.data_handler = None 
        #####################
    
    # -------------------------------------------------------------------------
    @property
    def base_tf(self) -> str:
        """ دریافت base timeframe مربوط به symbol جاری. """
        base_tfs = self.config.get("__base_tfs_dict", {})
        if self.symbol not in base_tfs:
            raise RuntimeError(f"No base_tf found for symbol={self.symbol}")

        return base_tfs[self.symbol]

    # -------------------------------------------------------------------------
    def run(self, dataset: MTFDataset,
        *,
        mode: str = "train",
        save_features: bool = False,
        save_dir: Optional[str] = None,
        save_name: str = "features",
    ):
        """
        اجرای کامل Pipeline:
        DataHandler -> MTFDataset -> FeatureEngine -> FeatureStore -> ObservationBuilder
        """
        if dataset is None:
            raise ValueError("dataset is None")

        if not isinstance(dataset, MTFDataset):
            raise TypeError(
                f"Expected MTFDataset, got {type(dataset).__name__}"
            )
        logger.debug("Running DataPipeline (mode=%s)", mode)

        # ==============================================================
        # Stage 1: محاسبه Featureها
        # ==============================================================
        features = self.feature_engine.execute(
            dataset=dataset,
            specs=self.feature_specs,
            mode=mode,
        )
        if features is None:
            raise RuntimeError("FeatureEngine returned None.")

        # ==============================================================
        # Stage 2: FeatureStore
        # ==============================================================
        feature_dataset = self.feature_store.build(
            dataset=dataset,
            features=features,
        )
        metadata = self.feature_store.extract_metadata(
            feature_dataset
        )

        # ==============================================================
        # Stage 3: ذخیره Featureها (اختیاری)
        # ==============================================================
        if save_features:
            if save_dir is None:
                raise ValueError("save_dir must be provided when save_features=True")
            
            self.feature_store.save(
                dataset=feature_dataset,
                metadata=metadata,
                out_dir=save_dir,
                name=save_name,
            )
            logger.info("FeatureStore saved successfully.")

        # ==============================================================
        # Stage 4: ساخت Observation
        # ==============================================================
        base_tf = self.base_tf
        observation_df = self.observation_builder.build(
            feature_dataset,    #.get(self.base_tf), باید کل دیتاست ارسال شود. نه اینکه فقط دیتافریم تایم فریم مبنا ارسال شود.
            self.feature_graph,
        )

        if observation_df is None:
            raise RuntimeError("ObservationBuilder returned None.")

        logger.debug("Observation built successfully.")

        # ==============================================================
        # Stage 5: خروجی Pipeline
        # ==============================================================
        result = {
            "dataset": dataset,
            "features": feature_dataset,
            "metadata": metadata,
            "observation": observation_df,
        }
        logger.info("DataPipeline finished successfully.")

        self._dataset = dataset
        self._features = feature_dataset
        self._observation = observation_df

        return result
    
    # =========================================================================
    # DataPipeline: Part 1/5 - Section 5/5
    #
    # این بخش انتهای اسکلت کلاس را کامل می‌کند.
    # فقط از APIهایی که شما ارائه کرده‌اید استفاده شده است.
    # هر قسمت حدسی با # ASSUMPTION مشخص شده است.
    # =========================================================================

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_dataset(self) -> MTFDataset:
        """ آخرین Dataset موجود در Pipeline را برمی‌گرداند. """
        return self._dataset

    def get_features(self) -> Optional[MTFDataset]:
        """ آخرین Dataset فیچرها. """
        return self._features

    def get_observation(self):
        """ آخرین Observation.
        ASSUMPTION: خروجی ObservationBuilder ممکن است بعداً pd.DataFrame یا np.ndarray باشد.
        """
        return self._observation

    def clear_cache(self) -> None:
        """ پاک کردن Feature Cache """
        self.cache.clear()

    def reset(self) -> None:
        """ ریست کامل وضعیت Pipeline """
        self._dataset = None
        self._features = None
        self._observation = None
        self.clear_cache()

    # -------------------------------------------------------------------------
    # Live API
    # -------------------------------------------------------------------------
    def process_live(
        self,
        dataset: MTFDataset,
    ):
        """
        پردازش یک آپدیت Live.

        فقط Engine مسئول محاسبه Incremental است.
        """

        self._dataset = dataset

        features = self.engine.process_live_data(dataset)

        if features is None:
            return None

        self._features = features

        # ASSUMPTION
        # ObservationBuilder در نسخه Live
        # بعداً به Dataset مجهز خواهد شد.
        return features
    
    # =========================================================================
    # Part 2
    # اجرای Pipeline (Batch)
    # =========================================================================
    # -------------------------------------------------------------------------
    # Batch Entry
    # -------------------------------------------------------------------------
    def run_from_build_params(
        self,
        params: BuildParams,
        specs: List[str],
        mode: str = "train",
    ):
        """
        اجرای کامل Pipeline
        DataHandler -> MTFDataset -> FeatureEngine -> ObservationBuilder -> FeatureStore
        """
        # -------------------------------------------------------------
        # Stage 1: دریافت Dataset
        # -------------------------------------------------------------
        dataset = self.data_handler.build(params)
        self._dataset = dataset
        # -------------------------------------------------------------
        # Stage 2: محاسبه Featureها
        # -------------------------------------------------------------
        features = self.feature_engine.execute(
            dataset=dataset,
            specs=specs,
            mode=mode,
        )
        self._features = features
        # -------------------------------------------------------------
        # Stage 3: ساخت Observation
        # -------------------------------------------------------------
        observation = self._build_observation(features)
        self._observation = observation
        return observation
    
    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------
    def _build_observation(self, dataset: MTFDataset):
        """ تبدیل Feature Dataset به Observation """

        # ASSUMPTION
        # ObservationBuilder فعلاً فقط DataFrame دریافت می‌کند.
        # بنابراین از تایم‌فریم پایه استفاده می‌شود.
        # اگر نسخه جدید Builder مستقیماً MTFDataset را قبول کند
        # این قسمت حذف خواهد شد.

        return self.observation_builder.build(
            dataset,
            self.feature_graph,
        )

    # =========================================================================
    # Part 3
    # FeatureStore + Persistence
    # =========================================================================
    # -------------------------------------------------------------------------
    # Save Feature Dataset
    # -------------------------------------------------------------------------
    def save_features(
        self,
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ) -> Dict[str, Dict[str, str]]:
        """ ذخیره Feature Dataset """
        if self._features is None:
            raise RuntimeError("Feature dataset has not been created.")

        metadata = self.store.extract_metadata(
            self._features
        )
        return self.store.save(
            dataset=self._features,
            metadata=metadata,
            out_dir=out_dir,
            name=name,
            fmt=fmt,
        )
    
    # -------------------------------------------------------------------------
    # Build FeatureStore
    # -------------------------------------------------------------------------
    def build_store(self) -> MTFDataset:
        """ ساخت FeatureStoreV2 """
        if self._dataset is None:
            raise RuntimeError("Dataset is empty.")

        if self._features is None:
            raise RuntimeError("Feature dataset is empty.")

        result = self.store.build(
            dataset=self._dataset,
            features=self._features,
        )
        return result
    
    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def export(
        self,
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ):
        """ Build + Save """
        dataset = self.build_store()
        metadata = self.store.extract_metadata(dataset)
        return self.store.save(
            dataset=dataset,
            metadata=metadata,
            out_dir=out_dir,
            name=name,
            fmt=fmt,
        )
    
    # -------------------------------------------------------------------------
    # Optional Utility
    # -------------------------------------------------------------------------
    def has_features(self) -> bool:
        return self._features is not None
    
    def has_dataset(self) -> bool:
        return self._dataset is not None
    
    def has_observation(self) -> bool:
        return self._observation is not None
    
    # -------------------------------------------------------------------------
    # ASSUMPTION:
    # FeatureStoreV2.build(...)
    # خروجی را به صورت MTFDataset برمی‌گرداند
    # (مطابق امضایی که ارائه کرده‌اید.)
    #
    # همچنین extract_metadata و save دقیقاً همان APIهای
    # اعلام‌شده در پروژه استفاده شده‌اند و API جدیدی اختراع
    # نشده است.
    # -------------------------------------------------------------------------
    # =====================================================================
    # Observation
    # =====================================================================
    def build_observation(self, feature_dataset: MTFDataset):
        """ ساخت Observation برای لایه Agent. """
        observation = self.observation_builder.build(
            feature_dataset,
            self.feature_graph,
        )
        return observation

    # ===========================================
    def build_numpy_observation(self, feature_dataset: MTFDataset):
        """ خروجی numpy برای مدل RL. """
        return self.observation_builder.build_numpy(
            feature_dataset,
            self.feature_graph,
        )

    # =====================================================================
    # Feature Store
    # =====================================================================
    def build_feature_store(
        self,
        raw_dataset: MTFDataset,
        feature_dataset: MTFDataset,
    ) -> MTFDataset:
        """ ساخت نسخه قابل ذخیره FeatureStore. """
        return self.feature_store.build(
            raw_dataset,
            feature_dataset,
        )
    
    # ===========================================
    def save_feature_store(
        self,
        dataset: MTFDataset,
        *,
        out_dir: str | Path,
        name: str,
        fmt: str = "parquet",
    ):
        """ ذخیره Dataset و Metadata. """
        metadata = self.feature_store.extract_metadata(dataset)
        return self.feature_store.save(
            dataset=dataset,
            metadata=metadata,
            out_dir=out_dir,
            name=name,
            fmt=fmt,
        )

    # =====================================================================
    # Full Pipeline
    # =====================================================================
    def run_dataset(self, dataset: MTFDataset, mode: str = "train"):
        """
        اجرای کامل Pipeline.
        MTFDataset -> FeatureEngine.execute() -> features -> build_feature_store() -> 
        -> feature_dataset -> build_observation() -> observation
        خروجی:
            observation
            feature_dataset
        """
        features = self.feature_engine.execute(
            dataset=dataset,
            specs=self.feature_specs,
            mode=mode
        )
        feature_dataset = self.build_feature_store(
            dataset,
            features
        )

        print("=" * 60)                                             # for debug
        print(feature_dataset.get(self.base_tf).columns.tolist())   # for debug
        print("=" * 60)                                             # for debug

        observation = self.build_observation(feature_dataset)
        ##################### ADDED TEMPORARY start
        self._dataset = dataset
        self._features = feature_dataset
        self._observation = observation
        ##################### ADDED TEMPORARY end
        return observation, feature_dataset

    # =====================================================================
    # Live Processing
    # =====================================================================
    def process_live(self, dataset: MTFDataset) -> Optional[MTFDataset]:
        """ اجرای Pipeline در حالت Incremental. """
        return self.feature_engine.process_live_data(dataset)

    # =====================================================================
    # Cache
    # =====================================================================
    def clear_cache(self) -> None:
        """ پاک کردن Feature Cache. """
        self.feature_cache.clear()

    # =====================================================================
    # Reload
    # =====================================================================
    def rebuild_graph(self, specs: Optional[List[str]] = None) -> None:
        """ بازسازی FeatureGraph. """
        if specs is not None:
            self.specs = specs
        self.feature_graph = FeatureGraph(self.specs)

    def reload_config(self, config: Dict[str, Any]) -> None:
        """ بارگذاری مجدد Config. """
        self.config = config
        self.feature_engine = FeatureEngine(config)
        self.observation_builder = ObservationBuilder(config)
        self.rebuild_graph()

    # =====================================================================
    # Accessors
    # =====================================================================
    @property
    def dataset(self) -> Optional[MTFDataset]: 
        return self._dataset

    @property
    def features(self) -> Optional[MTFDataset]:
        return self._features

    @property
    def observation(self):
        return self._observation
    
    @property
    def graph(self) -> FeatureGraph:
        return self.feature_graph

    @property
    def engine(self) -> FeatureEngine:
        return self.feature_engine

    @property
    def store(self) -> FeatureStoreV2:
        return self.feature_store

    @property
    def cache(self):
        return self.feature_cache

    # =====================================================================
    # Information
    # =====================================================================
    def info(self) -> Dict[str, Any]:
        """ اطلاعات Pipeline. """
        return {
            "base_tf": self.base_tf,
            "spec_count": len(self.specs),
            "specs": list(self.specs),
            "timeframe_count": len(self.feature_graph.all_nodes()),
        }
    
    # ===========================================
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(base_tf={self.base_tf!r}, "
            f"specs={len(self.specs)})"
        )


""" متدهای زیر فعلاً باقی بمانند
run_from_build_params
_build_observation
build_store
export
save_features
info
reload_config
rebuild_graph

run
run_dataset
"""