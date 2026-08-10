# f02_data/feature_orchestrator.py
"""
Production Data Pipeline

Responsibilities
----------------
        DataHandler
                ↓
        MTFDataset(raw)
                ↓
        FeatureEngine
                ↓
        MTFDataset(features)
                ↓
        FeatureStore
                ↓
        Merged MTFDataset
                ↓
        ObservationBuilder
                ↓
        Observation (RL Ready)

This module is intentionally independent of MT5.
It only orchestrates the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any

import logging
import pandas as pd

from f02_data.mtf_dataset import MTFDataset

from f03_features.feature_C_engine_4 import FeatureEngine
from f03_features.feature_B_store import FeatureStoreV2
from f03_features.feature_B_graph import FeatureGraph
from f03_features.observation_B_builder import ObservationBuilder

logger = logging.getLogger(__name__)

# ============================================================
# Pipeline State
# ============================================================
class PipelineStage(Enum):
    EMPTY = auto()
    RAW_READY = auto()
    FEATURES_READY = auto()
    MERGED_READY = auto()
    OBSERVATION_READY = auto()

# ============================================================
# Pipeline Result
# ============================================================
@dataclass(slots=True)
class PipelineResult:
    raw: Optional[MTFDataset] = None
    features: Optional[MTFDataset] = None
    merged: Optional[MTFDataset] = None
    observation: Optional[pd.DataFrame] = None
    graph: Optional[FeatureGraph] = None

# ============================================================
# Pipeline Context
# ============================================================
@dataclass(slots=True)
class PipelineContext:
    config: Dict[str, Any]
    engine: FeatureEngine
    store: FeatureStoreV2
    observation_builder: ObservationBuilder

# ============================================================
# Data Pipeline
# ============================================================
class DataPipeline:
    """
    Orchestrates the complete data flow.
    This class owns no market data.
    It only coordinates the stages.
    """
    # ---------------------------------------------------------
    def __init__(
        self,
        context: PipelineContext,
    ):
        self.ctx = context
        self.stage = PipelineStage.EMPTY
        self.result = PipelineResult()
    # ---------------------------------------------------------
    @property
    def config(self):
        return self.ctx.config

    @property
    def engine(self):
        return self.ctx.engine

    @property
    def store(self):
        return self.ctx.store

    @property
    def observation_builder(self):
        return self.ctx.observation_builder

    # ---------------------------------------------------------
    def reset(self):
        self.stage = PipelineStage.EMPTY
        self.result = PipelineResult()
        self.engine._live_cache.clear()

    # ---------------------------------------------------------
    def run(
        self,
        dataset: MTFDataset,
        specs: list[str],
        mode: str = "train",
    ) -> PipelineResult:
        """
        Complete pipeline.
            raw -> features -> merged -> observation
        """
        self.reset()
        self.result.raw = dataset
        self.stage = PipelineStage.RAW_READY
        self._build_features(
            specs=specs,
            mode=mode,
        )
        self._merge()
        self._build_observation()
        return self.result

    # ---------------------------------------------------------
    def _build_features(
        self,
        datasets: dict[str, MTFDataset],
    ) -> dict[str, MTFDataset]:

        feature_datasets: dict[str, MTFDataset] = {}
        feature_specs = (
            self.cfg
            .get("features", {})
            .get("train_specs", [])
        )
        for symbol, dataset in datasets.items():
            feature_datasets[symbol] = self.engine.execute(
                dataset=dataset.copy(),
                specs=feature_specs,
                mode=self.mode,
            )
        return feature_datasets
    
    # ---------------------------------------------------------
    def _merge(
        self,
        raw: dict[str, MTFDataset],
        features: dict[str, MTFDataset],
    ) -> dict[str, MTFDataset]:

        merged: dict[str, MTFDataset] = {}
        for symbol in raw:
            merged[symbol] = self.store.build(
                dataset=raw[symbol],
                features=features[symbol],
            )
        return merged

    # ---------------------------------------------------------
    def _build_observation(
        self,
        merged: dict[str, MTFDataset],
    ) -> pd.DataFrame:

        dfs = []
        specs = (
            self.cfg
            .get("features", {})
            .get("train_specs", [])
        )
        graph = FeatureGraph(specs)

        for symbol in sorted(merged.keys()):
            obs = self.observation_builder.build(
                dataset=merged[symbol],
                graph=graph,
            )
            dfs.append(obs)

        if not dfs:
            return pd.DataFrame()

        observation = pd.concat(
            dfs,
            axis=1,
            join="inner",
        )
        return observation

    # ---------------------------------------------------------
    def observation(self) -> pd.DataFrame:
        if self.result.observation is None:
            raise RuntimeError("Observation has not been built.")
        return self.result.observation

    # ---------------------------------------------------------
    def merged_dataset(self) -> MTFDataset:
        if self.result.merged is None:
            raise RuntimeError("Merged dataset has not been built.")
        return self.result.merged









