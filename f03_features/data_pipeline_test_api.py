# f03_features/data_pipeline_test_api.py
# Run: pytest -v -s f03_features/data_pipeline_test_api.py

from unittest.mock import MagicMock

import pytest

from f02_data.mtf_dataset import MTFDataset

from f03_features.data_pipeline import DataPipeline
from f03_features.feature_B_graph import FeatureGraph
from f10_utils.config_completer import config_completer

cfg = config_completer()
SYMBOL = "XAUUSD_i"
# =============================================================================
# Helpers
# =============================================================================

def make_pipeline():
    feature_engine = MagicMock()
    feature_store = MagicMock()
    observation_builder = MagicMock()

    graph = FeatureGraph([])

    pipeline = DataPipeline(
        feature_engine=feature_engine,
        feature_store=feature_store,
        observation_builder=observation_builder,
        feature_graph=graph,
        feature_specs=["ema(20)", "rsi(14)"],
        symbol=SYMBOL,
        config=cfg,
    )

    return (
        pipeline,
        feature_engine,
        feature_store,
        observation_builder,
    )


# =============================================================================
# API
# =============================================================================

def test_pipeline_has_required_methods():

    pipeline, *_ = make_pipeline()

    assert hasattr(pipeline, "run_dataset")
    assert hasattr(pipeline, "process_live")

    assert hasattr(pipeline, "build_feature_store")
    assert hasattr(pipeline, "build_observation")
    assert hasattr(pipeline, "build_numpy_observation")

    assert hasattr(pipeline, "clear_cache")
    assert hasattr(pipeline, "reload_config")
    assert hasattr(pipeline, "rebuild_graph")


# =============================================================================
# run_dataset
# =============================================================================

def test_run_dataset_calls_all_layers():

    pipeline, engine, store, builder = make_pipeline()

    raw_dataset = MagicMock(spec=MTFDataset)

    features_dataset = MagicMock(spec=MTFDataset)

    stored_dataset = MagicMock(spec=MTFDataset)

    observation = MagicMock()

    # FeatureEngine.execute(...)
    engine.execute.return_value = features_dataset

    # FeatureStore.build(...)
    store.build.return_value = stored_dataset

    # ObservationBuilder.build(...)
    builder.build.return_value = observation

    # ASSUMPTION
    # build_observation از feature_dataset.get(base_tf) استفاده می‌کند.
    stored_dataset.get.return_value = MagicMock()

    obs, feature_ds = pipeline.run_dataset(raw_dataset)

    assert obs is observation
    assert feature_ds is stored_dataset

    engine.execute.assert_called_once()

    store.build.assert_called_once()

    builder.build.assert_called_once()


# =============================================================================
# build_feature_store
# =============================================================================

def test_build_feature_store():

    pipeline, _, store, _ = make_pipeline()

    raw_dataset = MagicMock(spec=MTFDataset)

    feature_dataset = MagicMock(spec=MTFDataset)

    result_dataset = MagicMock(spec=MTFDataset)

    store.build.return_value = result_dataset

    result = pipeline.build_feature_store(
        raw_dataset,
        feature_dataset,
    )

    assert result is result_dataset

    store.build.assert_called_once_with(
        raw_dataset,
        feature_dataset,
    )


# =============================================================================
# Observation
# =============================================================================

def test_build_observation():

    pipeline, _, _, builder = make_pipeline()

    feature_dataset = MagicMock(spec=MTFDataset)

    df = MagicMock()

    feature_dataset.get.return_value = df

    observation = MagicMock()

    builder.build.return_value = observation

    result = pipeline.build_observation(feature_dataset)

    assert result is observation

    builder.build.assert_called_once()


# =============================================================================
# numpy observation
# =============================================================================

def test_build_numpy_observation():

    pipeline, _, _, builder = make_pipeline()

    feature_dataset = MagicMock(spec=MTFDataset)

    df = MagicMock()

    feature_dataset.get.return_value = df

    arr = MagicMock()

    builder.build_numpy.return_value = arr

    result = pipeline.build_numpy_observation(feature_dataset)

    assert result is arr

    builder.build_numpy.assert_called_once()


# =============================================================================
# process_live
# =============================================================================

def test_process_live():

    pipeline, engine, _, _ = make_pipeline()

    dataset = MagicMock(spec=MTFDataset)

    live_dataset = MagicMock(spec=MTFDataset)

    engine.process_live_data.return_value = live_dataset

    result = pipeline.process_live(dataset)

    assert result is live_dataset

    engine.process_live_data.assert_called_once_with(dataset)