# Python Functions List

## File_1: `f03_features/price_action/breakouts_numba_3.py`

- `_rolling_max(arr, window, min_periods)`
- `_rolling_min(arr, window, min_periods)`
- `_confirm_consecutive(cond, n)`
- `_retest_fail_kernel(h, l, c, breakout_up, breakout_dn, upper_prev, lower_prev, retest_lookahead, fail_lookahead)`
- `_build_breakouts_core(h, l, c, range_window, min_periods, confirm_closes, retest_lookahead, fail_break_lookahead, anti_lookahead)`
- `build_breakouts(df: pd.DataFrame) -> pd.DataFrame`

## File_2: `f03_features/price_action/config_wiring.py`

- `_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any`
- `_maybe_put(d: Dict[str, Any], key: str, val: Any) -> None`
- `extract_pa_kwargs_from_config(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]`
- `build_pa_features_from_config(df_base: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame`

## File_3: `f03_features/price_action/confluence_numba_6.py`

- `_clip01(x)`
- `_norm_distance_nb(dist)`
- `structure_score_nb(a, b, c)`
- `zones_score_nb(z1, z2)`
- `imbalance_score_nb(x)`
- `mtf_score_nb(x)`
- `extras_score_nb(x)`
- `combine_nb(s, z, im, m, ex, ws, wz, wi, wm, wex)`
- `build_confluence(df: pd.DataFrame, anti_lookahead: bool = True, weights: dict | None = None)`

## File_4: `f03_features/price_action/imbalance.py`

- `_call_detect_fvg(df: pd.DataFrame)`
- `_call_make_fvg(df: pd.DataFrame, proto)`
- `_call_detect_lp(df: pd.DataFrame)`
- `_call_detect_sweep(df: pd.DataFrame)`
- `_call_make_sweep(df: pd.DataFrame, proto)`
- `_coerce_fvg_df(fvg_df: pd.DataFrame) -> pd.DataFrame`
- `_coerce_lp_df(lp_df: pd.DataFrame) -> pd.DataFrame`
- `_coerce_sweep_df(sw_df: pd.DataFrame) -> pd.DataFrame`
- `build_imbalance_liquidity(df: pd.DataFrame) -> pd.DataFrame`

## File_5: `f03_features/price_action/market_structure.py`

- `detect_swings(df: pd.DataFrame, depth: int = 12, deviation: float = 5.0, backstep: int = 10, point: float = 0.01) -> pd.DataFrame`
- `build_market_structure(df: pd.DataFrame, depth: int = 12, deviation: float = 5.0, backstep: int = 10, point: float = 0.01) -> pd.DataFrame`
- `detect_bos_choch(structure: pd.DataFrame, price_df: pd.DataFrame, eps: float = 1e-06) -> pd.DataFrame`
- `market_structure_pipeline(df: pd.DataFrame, depth: int = 12, deviation: float = 5.0, backstep: int = 10, point: float = 0.01) -> pd.DataFrame`
- `build_regime_state(df: pd.DataFrame) -> pd.DataFrame`
- `build_regime_state_pro(df: pd.DataFrame, atr_window: int = 14, adr_window: int = 14, tf_higher: str = '5min', smoothing: Literal['atr', 'adr', None] = 'atr', atr_method: Literal['classic', 'wilder', 'ema'] = 'wilder') -> pd.DataFrame`

## File_6: `f03_features/price_action/microchannels.py`

- `build_microchannels(df: pd.DataFrame) -> pd.DataFrame`

## File_7: `f03_features/price_action/mtf_context.py`

- `_safe_series(x) -> pd.Series`
- `_robust_scale(s: pd.Series, window: int, min_periods: int) -> pd.Series`
- `_align_to_base(higher_series: pd.Series, base_index: pd.Index) -> pd.Series`
- `compute_soft_bias(close: pd.Series) -> pd.Series`
- `compute_confluence(bias_local: pd.Series, bias_higher: pd.Series) -> tuple[pd.Series, pd.Series]`
- `build_mtf_context(df_base: pd.DataFrame) -> pd.DataFrame`

## File_8: `f03_features/price_action/regime.py`

- `_ensure_series(s, name = None) -> pd.Series`
- `_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series`
- `_rolling_iqr_width(x: pd.Series, window: int, min_periods: int) -> pd.Series`
- `_norm01(s: pd.Series) -> pd.Series`
- `_clip01(s: pd.Series) -> pd.Series`
- `build_regime(df: pd.DataFrame) -> pd.DataFrame`

## File_9: `f03_features/price_action/registry_adapter.py`

- `list_pa_features() -> list[str]`
- `get_pa_builder(name: str) -> Callable`
- `register_price_action_to_indicators_registry(registry_dict: Dict[str, Callable]) -> Dict[str, Callable]`
- `get_price_action_registry() -> Dict[str, Callable]`
- `build_all_price_action_features(df_base: pd.DataFrame) -> pd.DataFrame`

## File_10: `f03_features/price_action/zones.py`

- `_call_detect_sd(df: pd.DataFrame) -> pd.DataFrame | tuple | None`
- `_call_make_sd(df: pd.DataFrame, proto) -> pd.DataFrame | None`
- `_call_detect_ob(df: pd.DataFrame) -> pd.DataFrame | tuple | None`
- `_call_make_ob(df: pd.DataFrame, proto) -> pd.DataFrame | None`
- `_coerce_sd_df(sd_df: pd.DataFrame) -> pd.DataFrame`
- `_coerce_ob_df(ob_df: pd.DataFrame) -> pd.DataFrame`
- `build_zones(df: pd.DataFrame) -> pd.DataFrame`

