# f03_features/feature_C_engine_5_tester_B.py
# Run: pytest -v -s f03_features/feature_C_engine_5_tester_B.py

from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from f03_features.feature_C_engine_5 import FeatureEngine
from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_C_registry_1 import IndicatorSpec


# =============================================================================
# Fake ParsedSpec
# =============================================================================
@dataclass
class FakeParsedSpec:
    raw: str
    name: str
    timeframe: str
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

# =============================================================================
# Fake Indicators
# =============================================================================
# --------------------------------------------------------- 1
class FakeSMA:
    def __init__(self, period=3):
        self.period = period

    def update(self, close):
        return close + 100

# --------------------------------------------------------- 2
class FakeATR:
    def __init__(self, period=14):
        self.period = period

    def update(self, high, low, close):
        return high + low + close

# --------------------------------------------------------- 3
class FakeMACD:
    def __init__(self):
        pass

    def update(self, close):
        return (close, close + 1, close + 2)

# --------------------------------------------------------- 4
class FakeHeikin:
    def update(self, open_, high, low, close):
        return (open_, high, low, close)

# =============================================================================
# Batch indicator
# ============================================================================= 5
def fake_batch(df, period=5):
    out = pd.DataFrame(index=df.index)
    out["value"] = np.arange(len(df))
    return out

# =============================================================================
# Helpers
# =============================================================================
# --------------------------------------------------------- 6
def make_dataframe():
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    df = pd.DataFrame(
        {
            "M1_open": [1,2,3,4,5],
            "M1_high": [2,3,4,5,6],
            "M1_low": [0,1,2,3,4],
            "M1_close": [1.5,2.5,3.5,4.5,5.5],
            "M1_volume": [10,11,12,13,14],
        },
        index=idx,
    )
    df.attrs["context"] = {
        "symbol":"EURUSD",
        "tf":"M1",
        "session":"TEST",
    }
    return df

# --------------------------------------------------------- 7
def make_dataset():
    ds = MTFDataset(symbol="EURUSD", base_tf="M1")
    ds.add("M1", make_dataframe())
    return ds

# =============================================================================
# Fake Config
# =============================================================================
# --------------------------------------------------------- 8
@pytest.fixture
def engine():
    cfg = {
        "features": {
            "live_specs": [],
        }
    }
    return FeatureEngine(cfg)

# --------------------------------------------------------- 9
@pytest.fixture
def dataframe():
    return make_dataframe()

# --------------------------------------------------------- 10
@pytest.fixture
def dataset():
    return make_dataset()

# =============================================================================
# _normalize_output
# =============================================================================
# --------------------------------------------------------- 11
def test_normalize_output_scalar(engine):
    spec = IndicatorSpec(
        name="sma",
        fn=None,
        is_stateful=True,
        output_names=None,
    )
    out = engine._normalize_output(spec, 12.5)
    assert isinstance(out, dict)
    assert out == {"sma": 12.5,}

# --------------------------------------------------------- 12
def test_normalize_output_tuple(engine):
    spec = IndicatorSpec(
        name="macd",
        fn=None,
        is_stateful=True,
        output_names=["macd", "signal", "histogram",],
    )
    out = engine._normalize_output(spec,(10, 20, 30),)
    assert out == {
        "macd": 10,
        "signal": 20,
        "histogram": 30,
    }

# --------------------------------------------------------- 13
def test_normalize_output_short_tuple(engine):

    spec = IndicatorSpec(
        name="macd",
        fn=None,
        is_stateful=True,
        output_names=["macd", "signal", "histogram",],
    )
    out = engine._normalize_output(spec, (5,6,),)
    assert out["macd"] == 5
    assert out["signal"] == 6
    assert out["histogram"] is None

# --------------------------------------------------------- 14
def test_normalize_output_single_fallback(engine):
    spec = IndicatorSpec(
        name="bb",
        fn=None,
        is_stateful=True,
        output_names=["upper", "middle", "lower",],
    )
    out = engine._normalize_output(spec, 99,)
    assert out == {"upper": 99,}

# =============================================================================
# _merge
# =============================================================================
# --------------------------------------------------------- 15
def test_merge_add_new_column(engine, dataframe):
    out = pd.DataFrame(
        {
            "feature": [1,2,3,4,5],
        },
        index=dataframe.index,
    )
    merged = engine._merge(
        dataframe.copy(),
        out,
    )
    assert "feature" in merged.columns
    assert len(merged) == len(dataframe)

# --------------------------------------------------------- 16
def test_merge_same_column_same_values(engine, dataframe):
    out = pd.DataFrame(
        {
            "M1_close": dataframe["M1_close"],
        },
        index=dataframe.index,
    )
    merged = engine._merge(
        dataframe.copy(),
        out,
    )
    assert "M1_close__dup" not in merged.columns

# --------------------------------------------------------- 17
def test_merge_duplicate_column(engine, dataframe):
    out = pd.DataFrame(
        {
            "M1_close": dataframe["M1_close"] + 100,
        },
        index=dataframe.index,
    )
    merged = engine._merge(
        dataframe.copy(),
        out,
    )
    assert "M1_close__dup" in merged.columns

# --------------------------------------------------------- 18
def test_merge_empty_output(engine, dataframe):
    out = pd.DataFrame()
    merged = engine._merge(
        dataframe.copy(),
        out,
    )
    assert merged.equals(dataframe)

# =============================================================================
# _build_live_column_name
# =============================================================================
# --------------------------------------------------------- 19
def test_build_live_column_name_scalar(engine):
    ps = FakeParsedSpec(
        raw="sma(period=20)@M1",
        name="sma",
        timeframe="M1",
    )
    col = engine._build_live_column_name("sma", ps,)
    assert col == "sma(period=20)@M1_live"

# --------------------------------------------------------- 20
def test_build_live_column_name_multi(engine):
    ps = FakeParsedSpec(
        raw="macd()@M1",
        name="macd",
        timeframe="M1",
    )
    col = engine._build_live_column_name("signal", ps,)
    assert col == "signal()@M1_live"

# --------------------------------------------------------- 21
def test_build_live_column_name_histogram(engine):
    ps = FakeParsedSpec(
        raw="macd()@M1",
        name="macd",
        timeframe="M1",
    )
    col = engine._build_live_column_name("histogram", ps,)
    assert col.endswith("_live")

# =============================================================================
# _build_live_instance
# =============================================================================
# --------------------------------------------------------- 22
def test_build_live_instance_sma(engine):
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma(period=15)@M1",
        name="sma",
        timeframe="M1",
        kwargs={
            "period":15,
            "unknown":999,
        },
    )
    obj = engine._build_live_instance(spec, ps,)
    assert isinstance(obj, FakeSMA)
    assert obj.period == 15

# --------------------------------------------------------- 23
def test_build_live_instance_default(engine):
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    obj = engine._build_live_instance(spec, ps,)
    assert isinstance(obj, FakeSMA)
    assert obj.period == 3

# =============================================================================
# _update_live
# =============================================================================
# --------------------------------------------------------- 24
def test_update_live_single_value(engine, dataframe):
    row = dataframe.iloc[0]
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    obj = FakeSMA()
    out = engine._update_live(obj, row, spec, ps,)
    assert isinstance(out, dict)
    assert "sma" in out

# --------------------------------------------------------- 25
def test_update_live_macd(engine, dataframe):
    row = dataframe.iloc[0]
    spec = IndicatorSpec(
        name="macd",
        fn=FakeMACD,
        is_stateful=True,
        output_names=["macd", "signal", "histogram",],
    )
    ps = FakeParsedSpec(
        raw="macd()@M1",
        name="macd",
        timeframe="M1",
    )
    obj = FakeMACD()
    out = engine._update_live(obj, row, spec, ps,)
    assert out["macd"] == row["M1_close"]
    assert out["signal"] == row["M1_close"] + 1
    assert out["histogram"] == row["M1_close"] + 2

# --------------------------------------------------------- 26
def test_update_live_atr(engine, dataframe):
    row = dataframe.iloc[0]
    spec = IndicatorSpec(
        name="atr",
        fn=FakeATR,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="atr()@M1",
        name="atr",
        timeframe="M1",
    )
    obj = FakeATR()
    out = engine._update_live(obj, row, spec, ps,)
    expected = (row["M1_high"] + row["M1_low"] + row["M1_close"])
    assert out["atr"] == expected

# --------------------------------------------------------- 27
def test_update_live_heikin(engine, dataframe):
    row = dataframe.iloc[0]
    spec = IndicatorSpec(
        name="heikin",
        fn=FakeHeikin,
        is_stateful=True,
        output_names=["ha_open", "ha_high", "ha_low", "ha_close",],
    )
    ps = FakeParsedSpec(
        raw="heikin()@M1",
        name="heikin",
        timeframe="M1",
    )
    obj = FakeHeikin()
    out = engine._update_live(obj, row, spec, ps,)
    assert out["ha_open"] == row["M1_open"]
    assert out["ha_high"] == row["M1_high"]
    assert out["ha_low"] == row["M1_low"]
    assert out["ha_close"] == row["M1_close"]

# --------------------------------------------------------- 28
def test_update_live_missing_column(engine, dataframe):
    row = dataframe.iloc[0].drop("M1_close")
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    obj = FakeSMA()
    out = engine._update_live(obj, row, spec, ps,)
    assert out is None

# =============================================================================
# _apply_live
# =============================================================================
# --------------------------------------------------------- 29
def test_apply_live(engine, dataframe):
    engine.mode = "live"
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    out = engine._apply_live(
        spec,
        dataframe.copy(),
        ps,
    )
    assert len(out) == len(dataframe)
    cols = [c for c in out.columns if "live" in c]
    assert len(cols) == 1

# --------------------------------------------------------- 30
def test_apply_live_cache(engine, dataframe):
    engine.mode = "live"
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma(period=5)@M1",
        name="sma",
        timeframe="M1",
        kwargs={"period":5},
    )
    engine._apply_live(
        spec,
        dataframe.copy(),
        ps,
    )
    size1 = len(engine._live_cache)
    engine._apply_live(
        spec,
        dataframe.copy(),
        ps,
    )
    size2 = len(engine._live_cache)
    assert size1 == size2 == 1

# =============================================================================
# _attach_live_output
# =============================================================================
# --------------------------------------------------------- 31
def test_attach_live_output_scalar(engine, dataframe):
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma(period=5)@M1",
        name="sma",
        timeframe="M1",
    )
    outputs = [
        {"sma": 10},
        {"sma": 20},
        {"sma": 30},
        {"sma": 40},
        {"sma": 50},
    ]
    out = engine._attach_live_output(
        dataframe.copy(),
        ps,
        spec,
        outputs,
    )
    live_cols = [c for c in out.columns if c.endswith("_live")]
    assert len(live_cols) == 1
    assert list(out[live_cols[0]]) == [10,20,30,40,50]

# --------------------------------------------------------- 32
def test_attach_live_output_macd(engine, dataframe):
    spec = IndicatorSpec(
        name="macd",
        fn=FakeMACD,
        is_stateful=True,
        output_names=[
            "macd",
            "signal",
            "histogram",
        ],
    )
    ps = FakeParsedSpec(
        raw="macd()@M1",
        name="macd",
        timeframe="M1",
    )
    outputs = []
    for i in range(len(dataframe)):
        outputs.append(
            {
                "macd": i,
                "signal": i+1,
                "histogram": i+2,
            }
        )
    out = engine._attach_live_output(
        dataframe.copy(),
        ps,
        spec,
        outputs,
    )
    live_cols = [
        c
        for c in out.columns
        if c.endswith("_live")
    ]
    assert len(live_cols) == 3

# --------------------------------------------------------- 33
def test_attach_live_output_with_none(engine, dataframe):
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    outputs = [
        {"sma":1},
        None,
        {"sma":3},
        None,
        {"sma":5},
    ]
    out = engine._attach_live_output(
        dataframe.copy(),
        ps,
        spec,
        outputs,
    )
    live = [c for c in out.columns if c.endswith("_live")][0]
    assert pd.isna(out.iloc[1][live])
    assert pd.isna(out.iloc[3][live])

# --------------------------------------------------------- 34
def test_attach_live_output_empty(engine, dataframe):
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    out = engine._attach_live_output(
        dataframe.copy(),
        ps,
        spec,
        [],
    )
    assert out.equals(dataframe)

# =============================================================================
# _build_graph
# =============================================================================
# --------------------------------------------------------- 35
def test_build_graph_without_dependency(engine, monkeypatch):
    spec = IndicatorSpec(
        name="sma",
        fn=None,
        is_stateful=False,
    )
    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: spec,
    )
    ps = FakeParsedSpec(
        raw="sma()@M1",
        name="sma",
        timeframe="M1",
    )
    graph = engine._build_graph([ps])
    assert len(graph) == 1

# --------------------------------------------------------- 36
def test_build_graph_with_dependency(engine, monkeypatch):
    spec = IndicatorSpec(
        name="supertrend",
        fn=None,
        is_stateful=False,
    )
    spec.depends_on = ["atr()@M1",]
    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: spec,
    )
    ps1 = FakeParsedSpec(
        raw="atr()@M1",
        name="atr",
        timeframe="M1",
    )
    ps2 = FakeParsedSpec(
        raw="supertrend()@M1",
        name="supertrend",
        timeframe="M1",
    )
    graph = engine._build_graph([ps1, ps2,])
    assert "supertrend()@M1" in graph
    assert graph["supertrend()@M1"] == ["atr()@M1"]

# =============================================================================
# _resolve_order
# =============================================================================
# --------------------------------------------------------- 37
def test_resolve_order_single_node(engine):
    graph = {
        "sma()@M1": [],
    }
    order = engine._resolve_order(graph)
    assert order == ["sma()@M1"]

# --------------------------------------------------------- 38
def test_resolve_order_chain(engine):
    graph = {
        "atr()@M1": [],
        "supertrend()@M1": ["atr()@M1"],
    }
    order = engine._resolve_order(graph)
    assert order.index("atr()@M1") < order.index("supertrend()@M1")

# --------------------------------------------------------- 39
def test_resolve_order_multiple_dependencies(engine):
    graph = {
        "ema()@M1": [],
        "atr()@M1": [],
        "supertrend()@M1": [
            "ema()@M1",
            "atr()@M1",
        ],
    }
    order = engine._resolve_order(graph)
    assert order.index("ema()@M1") < order.index("supertrend()@M1")
    assert order.index("atr()@M1") < order.index("supertrend()@M1")

# --------------------------------------------------------- 40
def test_resolve_order_cycle(engine):
    graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A"],
    }
    with pytest.raises(RuntimeError):
        engine._resolve_order(graph)

# ==========================================================
# _apply_spec (batch)
# ==========================================================
# def test_apply_spec_batch(monkeypatch, engine, dataset):

#     spec = IndicatorSpec(
#         name="sma",
#         fn=fake_batch_indicator,
#         is_stateful=False,
#         modes={"train"},
#     )

#     monkeypatch.setattr(
#         "f03_features.feature_C_engine_5.get_indicator",
#         lambda *args, **kwargs: spec,
#     )

#     called = {"value": False}

#     def fake_call_batch(fn, dataset, ps):
#         called["value"] = True
#         return dataset

#     monkeypatch.setattr(
#         engine,
#         "_call_batch",
#         fake_call_batch,
#     )

#     ps = FakeParsedSpec(
#         raw="sma(period=5)@M1",
#         name="sma",
#         timeframe="M1",
#     )

#     engine._apply_spec(
#         dataset,
#         ps,
#         "train",
#     )

#     assert called["value"] is True

# =============================================================================
# _apply_spec (live)
# ============================================================================= 41
def test_apply_spec_live(monkeypatch, engine, dataset):
    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
        modes={"live"},
        required_cols=[
            "M1_open",
            "M1_high",
            "M1_low",
            "M1_close",
            "M1_volume",
        ],
    )
    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: spec,
    )
    called = {"value": False}
    # ---------------------------------
    def fake_apply_live(spec, df, ps):
        called["value"] = True
        return df
    # ---------------------------------
    monkeypatch.setattr(
        engine,
        "_apply_live",
        fake_apply_live,
    )
    ps = FakeParsedSpec(
        raw="sma(period=5)@M1",
        name="sma",
        timeframe="M1",
    )
    engine._apply_spec(
        dataset,
        ps,
        "live",
    )
    assert called["value"] is True

# =============================================================================
# _apply_spec (missing dataframe)
# ============================================================================= 42
def test_apply_spec_missing_dataframe(monkeypatch, engine):

    ds = MTFDataset(
        symbol="EURUSD",
        base_tf="M1",
    )

    spec = IndicatorSpec(
        name="sma",
        fn=FakeSMA,
        is_stateful=True,
    )

    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: spec,
    )

    ps = FakeParsedSpec(
        raw="sma()@M5",
        name="sma",
        timeframe="M5",
    )

    out = engine._apply_spec(
        ds,
        ps,
        "live",
    )
    assert out is ds

# =============================================================================
# _apply_spec (missing indicator)
# ============================================================================= 43
def test_apply_spec_unknown_indicator(monkeypatch, engine, dataset):
    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: None,
    )
    ps = FakeParsedSpec(
        raw="unknown()@M1",
        name="unknown",
        timeframe="M1",
    )
    out = engine._apply_spec(
        dataset,
        ps,
        "live",
    )
    assert out is dataset

# ============================================================================= END