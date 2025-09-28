from f04_features.indicators.registry import get_indicator_v2, list_all_indicators_v2
fn_adr = get_indicator_v2("adr")
fn_ado = get_indicator_v2("adr_distance_to_open")
print("adr:", fn_adr is not None, getattr(fn_adr, "__name__", None))
print("adr_distance_to_open:", fn_ado is not None, getattr(fn_ado, "__name__", None))
print("is_adr_advanced:", list_all_indicators_v2(include_legacy=False).get("adr", "missing"))
