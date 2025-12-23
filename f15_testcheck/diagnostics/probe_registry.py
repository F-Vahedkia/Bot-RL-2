# Run: python f15_testcheck/diagnostics/probe_registry.py
# در مورخ 1404/08/17 به درستی اجرا شد.

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from f03_features.feature_registry import get_indicator, list_all_indicators

fn_adr = get_indicator("adr")
fn_ado = get_indicator("adr_distance_to_open")

print("adr:", fn_adr is not None, getattr(fn_adr, "__name__", None))
print("adr_distance_to_open:", fn_ado is not None, getattr(fn_ado, "__name__", None))
print("is_adr_advanced:", list_all_indicators(include_legacy=False).get("adr", "missing"))
