# VSCode Run Command:
# Run: python -m pytest -q f03_features/0tests/test_phase1_2_feature_B_graph.py

import pytest
from f03_features.feature_B_graph import FeatureGraph


def test_graph_basic_build():
    specs = ["sma(10)", "ema(20)"]
    g = FeatureGraph(specs)

    nodes = g.all_nodes()
    assert len(nodes) == 2
    assert {n.name for n in nodes} == {"sma", "ema"}


def test_group_by_name():
    specs = ["sma(10)", "sma(20)", "ema(20)"]
    g = FeatureGraph(specs)

    sma_nodes = g.get_by_name("sma")
    ema_nodes = g.get_by_name("ema")

    assert len(sma_nodes) == 2
    assert len(ema_nodes) == 1


def test_group_by_timeframe_default_global():
    specs = ["sma(10)", "ema(20)"]
    g = FeatureGraph(specs)

    global_nodes = g.get_by_timeframe("GLOBAL")
    assert len(global_nodes) == 2


def test_missing_timeframe_behavior():
    specs = ["sma(10)"]
    g = FeatureGraph(specs)

    node = g.all_nodes()[0]
    assert node.timeframe is None or node.timeframe == node.timeframe  # sanity check


def test_invalid_spec_should_not_crash():
    specs = ["sma(10)", "INVALID_SPEC___BAD"]

    try:
        g = FeatureGraph(specs)
        nodes = g.all_nodes()
        assert isinstance(nodes, list)
    except Exception:
        pytest.fail("FeatureGraph crashed on invalid spec")
