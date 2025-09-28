# test_parser.py
import pprint
from f04_features.indicators.parser import (
    _split_top_level_commas,
    _parse_value,
    _parse_args_kwargs,
    parse_spec_v2,
)

def test_split_top_level_commas():
    print("=== test_split_top_level_commas ===")
    s = "0.382, (1,2), [a,b,c], foo=1"
    result = _split_top_level_commas(s)
    pprint.pprint(result)

def test_parse_value():
    print("=== test_parse_value ===")
    tokens = ["True", "false", "None", "123", "-4.56", "[M1,H1]", "'hello'", "xyz"]
    for t in tokens:
        val = _parse_value(t)
        print(f"{t!r} -> {val!r} ({type(val).__name__})")

def test_parse_args_kwargs():
    print("=== test_parse_args_kwargs ===")
    samples = [
        "0.382,0.618",
        "tf=[H1,H4,D1], tol_pct=0.1, prefer_ratio=0.618",
        "atr_mult=1.5",
        "window=20, method='ema'",
    ]
    for s in samples:
        args, kwargs = _parse_args_kwargs(s)
        print(f"{s!r} -> args={args}, kwargs={kwargs}")

def test_parse_spec_v2():
    print("=== test_parse_spec_v2 ===")
    specs = [
        "golden_zone(0.382,0.618)@H1",
        "fib_cluster(tf=[H1,H4,D1], tol_pct=0.1, prefer_ratio=0.618)",
        "fib_ext_targets(atr_mult=1.5)@H1",
        "ma_slope(window=20, method='ema')@M5",
        "rsi_zone(period=14)@H1",
    ]
    for s in specs:
        parsed = parse_spec_v2(s)
        pprint.pprint(parsed)

if __name__ == "__main__":
    test_split_top_level_commas()
    test_parse_value()
    test_parse_args_kwargs()
    test_parse_spec_v2()


