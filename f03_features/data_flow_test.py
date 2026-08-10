# f03_features/data_flow_test.py
# Run: python -m f03_features.data_flow_test

import logging
# from f10_utils.logging_utils import setup_logging

# from f10_utils.config_loader import load_config
from f10_utils.config_completer import config_completer

from f02_data.data_handler_F_2_3_n import DataHandler, BuildParams

from f03_features.feature_B_bootstrap import build_feature_system
from f03_features.feature_B_graph import FeatureGraph
from f03_features.observation_B_builder import ObservationBuilder
from f03_features.data_pipeline import DataPipeline


def main():

    logging.basicConfig(level=logging.INFO)

    cfg = config_completer()
    # setup_logging(cfg)

    symbols = list(cfg["features"]["symbols"].keys())
    print(symbols)
    symbol = symbols[1]
    print(symbol)    

    # feature_specs = cfg["features"]["indicators"]
    feature_specs = cfg["features"]["symbols"][str(symbol)]["indicators"]

    feature_system = build_feature_system()

    feature_engine = feature_system.get_engine()
    feature_store = feature_system.get_store()

    graph = FeatureGraph(feature_specs)

    obs_builder = ObservationBuilder(cfg)

    pipeline = DataPipeline(
        feature_engine=feature_engine,
        feature_store=feature_store,
        observation_builder=obs_builder,
        feature_graph=graph,
        feature_specs=feature_specs,
        symbol=symbol,
        config=cfg,
    )

    handler = DataHandler(
        cfg=cfg,
        symbol=symbol,
    )

    params = BuildParams(
        symbol=symbol,
        base_tf=cfg["__base_tfs_dict"][symbol],
        timeframes=cfg["__timeframes_dict"][symbol],
    )

    dataset = handler.build(params)

    observation = pipeline.run_dataset(
        dataset,
        mode="train",
    )[0]

    print("\n========== SUCCESS ==========")
    print(type(dataset))
    print(type(observation))

    if hasattr(observation, "shape"):
        print(observation.shape)

    print("=============================\n")


if __name__ == "__main__":
    main()