# f03_features/feature_C_engine_5_tester_A.py
# Run: pytest -v -s f03_features/feature_C_engine_5_tester_A.py

from __future__ import annotations

import pandas as pd
import pytest

from f02_data.mtf_dataset import MTFDataset
from f03_features.feature_C_engine_5 import FeatureEngine
from f10_utils.parser import parse_spec

# ---------------------------------------------------------
# Dummy Config
# ---------------------------------------------------------
@pytest.fixture
def cfg():
    return {
        "features": {
            "live_specs": [
                "sma('close',5)@M1",
            ]
        }
    }

# ---------------------------------------------------------
# Sample Dataset
# ---------------------------------------------------------
@pytest.fixture
def dataset():
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    n = 200
    m1 = pd.DataFrame({
        "open": range(n),
        "high": [x + 1 for x in range(n)],
        "low": range(n),
        "close": range(n),
        "volume": [100] * n,
    })
    m5 = pd.DataFrame({
        "open": range(n),
        "high": [x + 1 for x in range(n)],
        "low": range(n),
        "close": range(n),
        "volume": [100] * n,
    })
    ds.add("M1", m1)
    ds.add("M5", m5)

    return ds

# ---------------------------------------------------------
# Engine
# ---------------------------------------------------------
@pytest.fixture
def engine(cfg):
    return FeatureEngine(cfg)

# ---------------------------------------------------------
# Execute (Train)
# ---------------------------------------------------------
def test_execute_train(engine, dataset):
    specs = ["sma('close',10)@M1"]
    out = engine.execute(
        dataset=dataset,
        specs=specs,
        mode="train",
    )
    assert isinstance(out, MTFDataset)
    assert "M1" in out.frames

# ---------------------------------------------------------
# Execute (Live)
# ---------------------------------------------------------
def test_execute_live(engine, dataset):
    specs = ["sma('close',10)@M1"]
    out = engine.execute(
        dataset=dataset,
        specs=specs,
        mode="live",
    )
    assert isinstance(out, MTFDataset)

# ---------------------------------------------------------
# Empty Dataset
# ---------------------------------------------------------
def test_empty_dataset(engine):
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    out = engine.execute(
        dataset=ds,
        specs=[],
        mode="train",
    )
    assert out is ds

# ---------------------------------------------------------
# Invalid Indicator
# ---------------------------------------------------------
def test_unknown_indicator(engine, dataset):
    out = engine.execute(dataset, ["abcdefg()@M1"], mode="train")
    assert out is dataset

# ---------------------------------------------------------
# Missing Timeframe
# ---------------------------------------------------------
def test_missing_tf(engine, dataset):
    out = engine.execute(dataset, ["sma(10)@H4"], mode="train")
    assert out is dataset

# ---------------------------------------------------------
# Dataset Replace
# ---------------------------------------------------------
def test_dataset_replace(engine, dataset):
    old = dataset.get("M1")
    out = engine.execute(dataset, ["sma('close',5)@M1"], mode="train")
    new = out.get("M1")
    assert isinstance(new, pd.DataFrame)
    assert len(new) == len(old)

# ---------------------------------------------------------
# Multiple Specs
# ---------------------------------------------------------
def test_multiple_specs(engine, dataset):
    specs = ["sma('close',10)@M1", "wma(10)@M1"]
    out = engine.execute(dataset, specs, mode="train")
    assert isinstance(out, MTFDataset)

# ---------------------------------------------------------
# Execute Twice
# ---------------------------------------------------------
def test_execute_twice(engine, dataset):
    engine.execute(dataset, ["sma('close',5)@M1"], mode="train")
    engine.execute(dataset, ["sma('close',10)@M1"], mode="train")
    assert engine._run_id == 2

# ---------------------------------------------------------
# _merge()
# ---------------------------------------------------------
def test_merge_new_columns(engine):
    df = pd.DataFrame({"close": [1, 2, 3]})
    out = pd.DataFrame({"M1_sma": [1, 2, 3]})
    merged = engine._merge(df.copy(), out)
    assert "M1_sma" in merged.columns

def test_merge_duplicate_equal(engine):
    df = pd.DataFrame({"M1_sma": [1, 2, 3]})
    out = pd.DataFrame({"M1_sma": [1, 2, 3]})
    merged = engine._merge(df.copy(), out)
    assert list(merged.columns) == ["M1_sma"]

def test_merge_duplicate_different(engine):
    df = pd.DataFrame({"M1_sma": [1, 2, 3]})
    out = pd.DataFrame({"M1_sma": [9, 9, 9]})
    merged = engine._merge(df.copy(), out)
    assert "M1_sma__dup" in merged.columns

# ---------------------------------------------------------
# _normalize_output()
# ---------------------------------------------------------
class DummySpec:
    name = "sma"
    output_names = ["value"]

def test_normalize_scalar(engine):
    spec = DummySpec()
    out = engine._normalize_output(spec, 12)
    assert out == {"value": 12}

def test_normalize_tuple(engine):
    spec = DummySpec()
    spec.output_names = ["a", "b", "c"]
    out = engine._normalize_output(spec, (1, 2, 3))
    assert out["a"] == 1
    assert out["b"] == 2
    assert out["c"] == 3

def test_normalize_short_tuple(engine):
    spec = DummySpec()
    spec.output_names = ["a", "b", "c"]
    out = engine._normalize_output(spec,(5,))
    assert out["a"] == 5
    assert out["b"] is None
    assert out["c"] is None

# ---------------------------------------------------------
# _build_live_column_name()
# ---------------------------------------------------------
class DummyPS:
    raw = "sma('close',10)@M1"
    name = "sma"

def test_live_column_name_same(engine):
    ps = DummyPS()
    col = engine._build_live_column_name("sma", ps)
    assert col.endswith("_live")

def test_live_column_name_other(engine):
    ps = DummyPS()
    col = engine._build_live_column_name("signal", ps)
    assert "signal" in col

# ---------------------------------------------------------
# _attach_live_output()
# ---------------------------------------------------------
def test_attach_live_output(engine):
    df = pd.DataFrame({"close": [1, 2, 3]})
    ps = DummyPS()
    spec = DummySpec()
    outputs = [{"sma": 1}, {"sma": 2}, {"sma": 3}]
    out = engine._attach_live_output(df.copy(), ps, spec, outputs)
    assert len(out.columns) == 2

# ---------------------------------------------------------
# _build_graph()
# ---------------------------------------------------------
def test_build_graph(engine, monkeypatch):
    class DummySpec:
        depends_on = ['ema("close", 20)@M1']
    # dep = parse_spec("ema('close',20)@M1").raw.replace("'", '"')
    # class DummySpec:
    #     depends_on = [dep]

    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: DummySpec()
    )

    ps1 = parse_spec("ema('close', 20)@M1")
    ps2 = parse_spec("sma('close', 10)@M1")

    graph = engine._build_graph([ps1, ps2])

    key = ps2.raw.replace("'", '"')
    dep = ps1.raw.replace("'", '"')

    assert key in graph
    assert graph[key] == [dep]

# ---------------------------------------------------------
# _resolve_order()
# ---------------------------------------------------------
def test_resolve_order(engine):
    graph = {
        "c": ["b"],
        "b": ["a"],
        "a": [],
    }
    order = engine._resolve_order(graph)
    assert order == ["a", "b", "c"]

def test_resolve_order_cycle(engine):
    graph = {
        "a": ["b"],
        "b": ["a"],
    }
    with pytest.raises(RuntimeError):
        engine._resolve_order(graph)

# ---------------------------------------------------------
# _validate_contract()
# ---------------------------------------------------------
class ContractSpec:
    name = "dummy"
    required_cols = ["close"]
    modes = {"train"}

def test_validate_contract_ok(engine):
    engine.mode = "train"
    df = pd.DataFrame({"close": [1, 2]})
    assert engine._validate_contract(ContractSpec(), df)

def test_validate_contract_missing(engine):
    engine.mode = "train"
    df = pd.DataFrame({"open": [1, 2]})
    assert not engine._validate_contract(ContractSpec(), df,)

def test_validate_contract_wrong_mode(engine):
    engine.mode = "live"
    df = pd.DataFrame({"close": [1]})
    assert not engine._validate_contract(ContractSpec(), df)

# ---------------------------------------------------------
# _call_batch()
# ---------------------------------------------------------
def test_call_batch(engine, dataset):
    def fake_indicator(df, period=5):
        return pd.DataFrame({"sma": df["close"] + period})
    ps = parse_spec("sma('close',10)@M1")
    engine._call_batch(fake_indicator, dataset, ps)
    df = dataset.get("M1")
    assert ps.raw.replace("'", '"') in df.columns

def test_call_batch_none(engine, dataset):
    def fake(df):
        return None
    ps = parse_spec("sma(column='close',period=5)@M1")
    out = engine._call_batch(fake, dataset, ps)
    assert out is dataset

def test_call_batch_invalid(engine, dataset):
    def fake(df):
        return 123
    ps = parse_spec("sma(column='close',period=5)@M1")
    with pytest.raises(TypeError):
        engine._call_batch(fake, dataset, ps)

# ---------------------------------------------------------
# _apply_spec()
# ---------------------------------------------------------
def test_apply_spec_indicator_not_found(engine, dataset, monkeypatch):
    monkeypatch.setattr(
        "f03_features.feature_C_engine_5.get_indicator",
        lambda *args, **kwargs: None,
    )
    ps = parse_spec("sma('close',10)@M1")
    out = engine._apply_spec(dataset, ps, "train")
    assert out is dataset

# ---------------------------------------------------------
# process_live_data()
# ---------------------------------------------------------
def test_process_live_empty(engine):
    ds = MTFDataset(symbol="XAUUSD", base_tf="M1")
    out = engine.process_live_data(ds)
    assert out is ds

def test_process_live_no_specs(engine, dataset):
    engine.config = {}
    out = engine.process_live_data(dataset)
    assert out is dataset

def test_process_live_exception(engine, dataset, monkeypatch):
    monkeypatch.setattr(
        engine,
        "execute",
        lambda **k: (_ for _ in ()).throw(RuntimeError())
    )
    engine.config = {
        "features": {
            "live_specs": [
                "sma('close',10)@M1"
            ]
        }
    }
    out = engine.process_live_data(dataset)
    assert out is None

# --------------------------------------------------------- END