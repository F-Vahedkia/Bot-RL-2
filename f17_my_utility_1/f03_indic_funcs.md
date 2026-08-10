# Python Functions List

## File_1: `f03_features/indicators/__init__.py`

- `run_specs_v2()`
- `_setup_logging(level: str = 'INFO') -> None`
- `main() -> int`

## File_2: `f03_features/indicators/__main__.py`

_No top-level functions found_

## File_3: `f03_features/indicators/core.py`

- `sma(s: pd.Series, n: int) -> pd.Series`
- `ema(s: pd.Series, n: int) -> pd.Series`
- `ema_new(s: pd.Series, n: int) -> pd.Series`
- `wma(s: pd.Series, n: int) -> pd.Series`
- `rsi(close: pd.Series, length: int = 14, method: Literal['ema', 'wilders'] = 'ema') -> pd.Series`
- `roc(close: pd.Series, n: int = 10) -> pd.Series`
- `atr_old(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series`
- `atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series`
- `macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]`
- `bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]`
- `keltner(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, m: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]`
- `stochastic(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, d: int = 3) -> Tuple[pd.Series, pd.Series]`
- `cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series`
- `mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series`
- `obv(close: pd.Series, volume: pd.Series) -> pd.Series`
- `williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series`
- `parabolic_sar_orig(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series`
- `_parabolic_sar_njit_core(h, l, af_start, af_step, af_max)`
- `parabolic_sar_njit(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series`
- `parabolic_sar(high: pd.Series, low: pd.Series, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> pd.Series`
- `heikin_ashi_numpy(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]`
- `_heikin_ashi_njit_core(o, h, l, c)`
- `heikin_ashi_njit(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]`
- `heikin_ashi(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]`
- `registry() -> IndicatorMap`

## File_4: `f03_features/indicators/divergences.py`

- `pivots_numpy(series: pd.Series, k: int = 2)`
- `_pivots_njit_core(x, k)`
- `pivots_njit(series: pd.Series, k: int = 2)`
- `pivots(series: pd.Series, k: int = 2)`
- `divergence_flags_old(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = 'classic')`
- `divergence_flags_numpy(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = 'classic')`
- `_divergence_flags_njit_core(p, o, ph, pl, oh, ol, mode_flag)`
- `divergence_flags_njit(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = 'classic')`
- `divergence_flags(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = 'classic')`
- `registry_flag() -> Dict[str, callable]`
- `divergence_values_old(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = 'classic')`
- `divergence_values(price: pd.Series, osc: pd.Series, k: int = 2, mode: str = 'classic')`
- `registry() -> Dict[str, callable]`

## File_5: `f03_features/indicators/extras_channel.py`

- `_safe_div(a: pd.Series, b: pd.Series) -> pd.Series`
- `donchian(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20)`
- `chaikin_volatility(high: pd.Series, low: pd.Series, n: int = 10, roc: int = 10)`
- `bollinger_position(close: pd.Series, n: int = 20, k: float = 2.0)`
- `keltner_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20, atr_mult: float = 2.0)`
- `registry() -> Dict[str, callable]`

## File_6: `f03_features/indicators/extras_trend.py`

- `_ensure_series(x: pd.Series | np.ndarray, index: pd.Index, name: str | None = None, dtype: str | None = 'float64') -> pd.Series`
- `_safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-12) -> pd.Series`
- `supertrend_numpy(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.Series`
- `_supertrend_njit_core(high, low, close, atr, multiplier)`
- `supertrend_njit(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.Series`
- `supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0) -> pd.Series`
- `_compute_dm_di_adx_adxr_njit(h, l, c, window)`
- `adx_di(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]`
- `aroon_numpy(high: pd.Series, low: pd.Series, n: int = 25) -> tuple[pd.Series, pd.Series, pd.Series]`
- `_aroon_njit_core(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- `aroon_njit(high: pd.Series, low: pd.Series, n: int = 25) -> tuple[pd.Series, pd.Series, pd.Series]`
- `aroon(high: pd.Series, low: pd.Series, n: int = 25) -> tuple[pd.Series, pd.Series, pd.Series]`
- `kama_orig(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series`
- `kama_numpy(arr: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series`
- `kama_njit_core(arr, n = 10, fast = 2, slow = 30) -> np.ndarray`
- `kama_njit(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series`
- `kama(s: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series`
- `dema(s: pd.Series, n: int = 20) -> pd.Series`
- `tema(s: pd.Series, n: int = 20) -> pd.Series`
- `hma(s: pd.Series, n: int = 20) -> pd.Series`
- `ichimoku(high: pd.Series, low: pd.Series, close: pd.Series, tenkan: int = 9, kijun: int = 26, span_b: int = 52) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]`
- `ma_slope(time_series: pd.Series, window: int = 20, method: Literal['sma', 'ema'] = 'ema', norm: Literal['none', 'price', 'stdev'] = 'stdev', eps: float = 1e-12) -> pd.Series`
- `ma_slope_multistep(time_series: pd.Series, window: int = 20, step: int = 5, method: Literal['sma', 'ema'] = 'ema', norm: Literal['none', 'price', 'stdev'] = 'stdev', eps: float = 1e-12) -> pd.Series`
- `_ema(series: pd.Series, window: int, min_periods: int = None)`
- `_sma(series: pd.Series, window: int, min_periods: int = None)`
- `_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int, min_periods: int = None)`
- `_linreg_slope_rolling(series: pd.Series, window: int, min_periods: int = None)`
- `_linreg_slope_rolling_fast(series: pd.Series, window: int, min_periods: int = None)`
- `ma_slope_multistep_selective_1(time_series: pd.Series, window: int, step: int = 3, method: Literal['sma', 'ema'] = 'sma', norm: Literal['none', 'price', 'stdev', 'atr', 'selective'] = 'selective', norm_window: int = None, vol_mode: Literal['atr', 'stdev'] = 'atr', vol_threshold: float = 0.8, eps: float = 1e-09, min_periods: int = None, high: pd.Series = None, low: pd.Series = None, close: pd.Series = None)`
- `ma_slope_multistep_selective_2(time_series: pd.Series, window: int, step: int = 3, method: Literal['sma', 'ema'] = 'sma', norm: Literal['none', 'price', 'stdev', 'atr', 'selective'] = 'selective', norm_window: int = None, vol_mode: Literal['atr', 'stdev'] = 'atr', vol_threshold: float = 0.8, eps: float = 1e-09, min_periods: int = None, high: pd.Series = None, low: pd.Series = None, close: pd.Series = None)`
- `ma_slope_regression_selective(time_series: pd.Series, window: int, reg_window: int = None, method: Literal['sma', 'ema'] = 'sma', norm: Literal['none', 'price', 'stdev', 'atr', 'selective'] = 'selective', norm_window: int = None, vol_mode: Literal['atr', 'stdev'] = 'atr', vol_threshold: float = 0.8, eps: float = 1e-09, min_periods: int = None, high: pd.Series = None, low: pd.Series = None, close: pd.Series = None)`
- `rsi_zone(df: pd.DataFrame, price_col: str = 'close', period: int = 14, overbought: float = 70.0, oversold: float = 30.0, mid: float = 50.0, band: float = 5.0) -> pd.DataFrame`
- `registry() -> Mapping[str, Callable[..., Dict[str, pd.Series]]]`

## File_7: `f03_features/indicators/fibo_pipeline.py`

- `_ratios_map_from_cfg(cfg_all: Dict[str, Any]) -> List[Tuple[str, float]]`
- `build_fibo_levels_per_tf_multi(df: pd.DataFrame, k: int, max_legs_per_tf: int = 3, cfg_all: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]`
- `compute_adr_value(df: pd.DataFrame, window: int, tz: str) -> Optional[float]`
- `extract_sr_levels_from_fractals(df: pd.DataFrame, k: int = 2, lookback: int = 1500, max_levels: int = 30) -> Optional[List[float]]`
- `enhance_cluster_scores(cluster_df: pd.DataFrame, ma_slope: Optional[pd.Series], sr_levels: Optional[List[float]], cfg_all: dict, ref_time: Optional[pd.Timestamp], adr_value: Optional[float]) -> pd.DataFrame`
- `run_fibo_cluster(symbol: str, tf_dfs: Dict[str, pd.DataFrame], base_tf: str, k_fractal: int = 2, max_legs_per_tf: int = 3, tz: str = 'UTC', swings_for_abc: Optional[pd.DataFrame] = None) -> FiboRunResult`
- `get_abc_projections_from_swings(swings_df, ratios = None, cfg_all = None)`
- `attach_abc_projections_to_result(result: 'FiboRunResult', swings_df, cfg_all = None, ratios = None) -> 'FiboRunResult'`
- `compose_fibo_with_abc_full(result, swings_df, ratios = None, cfg_all = None)`

## File_8: `f03_features/indicators/fibonacci.py`

- `_fib_levels_for_leg(old_price: float, new_price: float, ratios: Sequence[float] = DEFAULT_RETR_RATIOS) -> pd.DataFrame`
- `_last_valid_leg(swings: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp, float, float]`
- `last_leg_levels(ohlc_df: pd.DataFrame, prominence: float | None = None, min_distance: int = 5, atr_mult: float | None = 1.0, ratios: Iterable[float] = DEFAULT_RETR_RATIOS) -> pd.DataFrame`
- `select_legs_from_swings(swings: pd.DataFrame) -> List[dict]`
- `levels_from_legs(legs: List[dict], ratios: Sequence[float] = DEFAULT_RETR_RATIOS) -> pd.DataFrame`
- `golden_zone(swings: pd.DataFrame, ratios: Tuple[float, float] = (0.382, 0.618), extra_ratios: Sequence[float] = DEFAULT_RETR_RATIOS) -> pd.DataFrame`
- `fib_cluster(tf_levels: Dict[str, pd.DataFrame], tol_pct: float = 0.08, prefer_ratio: float = 0.618, tf_weights: Optional[Dict[str, float]] = None, ma_slope: Optional[pd.Series] = None, rsi_zone_score: Optional[pd.Series] = None, sr_levels: Optional[Sequence[float]] = None, ref_time: Optional[pd.Timestamp] = None, w_trend: float = 10.0, w_rsi: float = 10.0, w_sr: float = 10.0, sr_tol_pct: float = 0.05) -> pd.DataFrame`
- `fib_ext_targets(entry_price: float, leg_low: float, leg_high: float, side: str, ext_ratios: Sequence[float] = DEFAULT_EXT_RATIOS, sl_atr: Optional[float] = None, sl_atr_mult: float = 1.5) -> pd.DataFrame`
- `_load_fibo_cfg() -> Dict[str, Any]`
- `golden_zone_cfg(swings, ratios: Optional[Tuple[float, float]] = None, extra_ratios: Optional[Sequence[float]] = None)`
- `_adaptive_tol_pct_from_df(df: pd.DataFrame) -> float`
- `fib_cluster_cfg(tf_levels: Dict[str, pd.DataFrame]) -> pd.DataFrame`
- `_infer_ref_price_from_tf_levels(tf_levels: Dict[str, 'pd.DataFrame'], ref_time: Optional['pd.Timestamp'] = None) -> Optional[float]`
- `_compute_adaptive_tol_pct(ref_price: Optional[float], vol_value: Optional[float], k: float, min_pct: float, max_pct: float) -> Optional[float]`
- `_merge_overrides(base: Dict[str, Any], symbol: str, tf: str) -> Dict[str, Any]`
- `_load_fibo_params(global_cfg: Dict[str, Any], symbol: str, tf: str) -> FiboParams`
- `_adaptive_tol_pct(df: pd.DataFrame, params: FiboParams, adr_col: str = 'ADR') -> float`
- `_rsi_zone_score(df: pd.DataFrame, params: FiboParams) -> float`
- `fib_ext_targets_cfg(last_leg: Tuple[float, float], global_cfg: Dict[str, Any], symbol: str, tf: str) -> pd.DataFrame`

## File_9: `f03_features/indicators/levels.py`

- `pivots_classic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, ...]`
- `sr_from_zigzag_legs_orig_1(df: pd.DataFrame) -> pd.DataFrame`
- `sr_from_zigzag_legs_njit_1(df: pd.DataFrame) -> pd.DataFrame`
- `sr_from_zigzag_legs_orig(df: pd.DataFrame) -> pd.DataFrame`
- `sr_from_zigzag_legs_njit(df: pd.DataFrame) -> pd.DataFrame`
- `sr_from_zigzag_legs(df: pd.DataFrame) -> pd.DataFrame`
- `sr_distance_from_levels(df: pd.DataFrame, sr: pd.DataFrame) -> pd.DataFrame`
- `_zigzag_leg_mask_orig(zz: pd.Series) -> pd.Series`
- `_zigzag_leg_mask_njit(zz: pd.Series) -> pd.Series`
- `_zigzag_leg_mask(zz: pd.Series, _njit_threshold: int = 1400000) -> pd.Series`
- `fibo_levels_from_legs_orig(df: pd.DataFrame, zz: pd.Series, ratios: Optional[Sequence[float]] = None, extend_last_leg: bool = False) -> pd.DataFrame`
- `fibo_levels_from_legs_njit(df: pd.DataFrame, zz: pd.Series, ratios: Optional[Sequence[float]] = None, extend_last_leg: bool = False) -> pd.DataFrame`
- `fibo_levels_from_legs(df: pd.DataFrame, zz: pd.Series, ratios: Optional[Sequence[float]] = None, threshold_bytes: int = 12800000, extend_last_leg: bool = False) -> pd.DataFrame`
- `registry() -> Dict[str, callable]`
- `compute_adr(df: pd.DataFrame, window: int = 14, tz: str = 'UTC') -> pd.Series`
- `adr_distance_to_open(df: pd.DataFrame, adr: pd.Series, tz: str = 'UTC') -> pd.DataFrame`
- `sr_overlap_score_simple(price: float, sr_levels: Sequence[float], tol_pct: float = 0.05) -> float`
- `sr_overlap_score(price: float, sr_levels: Sequence[float], tol_pct: float = 0.05, sr_weights: Optional[Sequence[float]] = None) -> float`

## File_10: `f03_features/indicators/parser.py`

- `_split_top_level_commas(s: str) -> List[str]`
- `_parse_value(token: str) -> Any`
- `_parse_args_kwargs(argstr: Optional[str]) -> Tuple[List[Any], Dict[str, Any]]`
- `_align_args_with_signature(ind_name: str, args_in: List[Any], kwargs_in: Dict[str, Any]) -> tuple[list, dict]`
- `parse_spec(spec: str) -> ParsedSpec`

## File_11: `f03_features/indicators/patterns.py`

- `_apply_scale(kwargs: dict, rules: list[tuple[str, float, float, float]]) -> dict`
- `_fmtf(x: float, nd: int = 2) -> str`
- `_body(open_: pd.Series, close: pd.Series) -> pd.Series`
- `_abs_body(open_: pd.Series, close: pd.Series) -> pd.Series`
- `_range(high: pd.Series, low: pd.Series) -> pd.Series`
- `_body_wicks(open_, high, low, close) -> Tuple[pd.Series, pd.Series, pd.Series]`
- `engulfing_flags(open_, high, low, close) -> Tuple[pd.Series, pd.Series]`
- `doji_flag(open_, close, atr: pd.Series | None = None, atr_ratio_thresh: float = 0.1, range_ratio_thresh: float = 0.2) -> pd.Series`
- `pinbar_flags(open_, high, low, close, ratio: float = 2.0) -> Tuple[pd.Series, pd.Series]`
- `hammer_shooting_flags(open_, high, low, close, min_body_frac: float = 0.0, wick_ratio: float = 2.0, opp_wick_k: float = 1.25) -> Tuple[pd.Series, pd.Series]`
- `harami_flags(open_, high, low, close) -> Tuple[pd.Series, pd.Series]`
- `inside_outside_flags(open_, high, low, close, min_range_k_atr: float = 0.0, atr_win: int = 14) -> Tuple[pd.Series, pd.Series]`
- `marubozu_flags(open_, high, low, close, wick_frac: float = 0.1) -> Tuple[pd.Series, pd.Series]`
- `tweezer_flags(high, low, tol_frac: float | None = 0.001, tol_k: float | None = None, tol_mode: str = 'atr_price', atr_win: int = 14, close: pd.Series | None = None) -> Tuple[pd.Series, pd.Series]`
- `three_soldiers_crows_flags(open_, close, atr_ref: pd.Series | None = None, min_body_atr: float = 0.2) -> Tuple[pd.Series, pd.Series]`
- `morning_evening_star_flags(open_, high, low, close, small_body_atr: float = 0.3, atr_win: int = 14) -> Tuple[pd.Series, pd.Series]`
- `piercing_darkcloud_flags(open_, close, min_body_ratio: float = 0.2) -> Tuple[pd.Series, pd.Series]`
- `belt_hold_flags(open_, high, low, close, wick_frac: float = 0.1) -> Tuple[pd.Series, pd.Series]`
- `registry() -> Dict[str, callable]`
- `detect_ab_equal_cd(swings: pd.DataFrame, ratio_tol: float = 0.05) -> Optional[Dict[str, Any]]`
- `abc_projection_adapter_from_abcd(abcd: dict)`

## File_12: `f03_features/indicators/sr_advanced.py`

- `detect_fvg_legacy(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]`
- `detect_fvg_optimized(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]`
- `make_fvg(df: pd.DataFrame) -> Dict[str, pd.Series]`
- `detect_sd(open_, high, low, close)`
- `_detect_sd_numba_core(open_, high, low, close, atr, base_len, base_atr_max, impulse_atr_min)`
- `detect_sd_numba(open_, high, low, close)`
- `make_sd(df: pd.DataFrame) -> Dict[str, pd.Series]`
- `detect_ob(open_, high, low, close) -> Dict[str, pd.Series]`
- `make_ob(df: pd.DataFrame) -> Dict[str, pd.Series]`
- `detect_liq_sweep(open_, high, low, close) -> Dict[str, pd.Series]`
- `make_liq_sweep(df: pd.DataFrame) -> Dict[str, pd.Series]`
- `detect_breaker_flip(open_, high, low, close) -> Dict[str, pd.Series]`
- `make_breaker_flip(df: pd.DataFrame) -> Dict[str, pd.Series]`
- `make_sr_fusion(df: pd.DataFrame) -> Dict[str, pd.Series]`

## File_13: `f03_features/indicators/utils.py`

- `round_levels(anchor: float, step: float, n: int = 10) -> List[float]`
- `get_ohlc_view(df: pd.DataFrame, tf: str) -> pd.DataFrame`
- `pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[pd.Series]`
- `detect_timeframes(df: pd.DataFrame) -> Dict[str, TFView]`
- `slice_tf(df: pd.DataFrame, view: TFView) -> pd.DataFrame`
- `nan_guard(df: pd.DataFrame) -> pd.DataFrame`
- `zscore(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series`
- `true_range_old(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series`
- `_true_range_numba(high, low, close)`
- `true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series`
- `_rma_wilder(tr, window)`
- `_sma_classic(tr, window)`
- `_ema(tr, window)`
- `compute_atr(df: pd.DataFrame, window: int = 14, method: str = 'wilder') -> pd.Series`
- `detect_swings(price: pd.Series, prominence: Optional[float] = None, min_distance: int = 5, atr: Optional[pd.Series] = None, atr_mult: Optional[float] = None, tf: Optional[str] = None) -> pd.DataFrame`
- `zscore_distance(x: float, mu: float, sigma: float, eps: float = 1e-12) -> float`
- `nearest_level_distance(price: float, levels: Sequence[float]) -> Dict[str, float]`
- `levels_from_recent_legs(ohlc_df: pd.DataFrame, n_legs: int = 10, ratios: Optional[Iterable[float]] = None, prominence: Optional[float] = None, min_distance: int = 5, atr_mult: Optional[float] = 1.0) -> pd.DataFrame`

## File_14: `f03_features/indicators/volume.py`

- `vwap_daily(high, low, close, volume)`
- `vwap_rolling(high, low, close, volume, n: int = 100)`
- `adl(high, low, close, volume)`
- `cmf(high, low, close, volume, n: int = 20)`
- `registry() -> Dict[str, callable]`

## File_15: `f03_features/indicators/zigzag.py`

- `_zigzag_mql_numpy_complete(high: np.ndarray, low: np.ndarray, depth: int = 12, deviation: float = 5.0, backstep: int = 3, point: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
- `_zigzag_mql_njit_loopwise_complete(high: np.ndarray, low: np.ndarray, depth: int = 12, deviation: float = 5.0, backstep: int = 3, point: float = 1e-05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
- `zigzag(high: pd.Series, low: pd.Series, depth: int = 12, deviation: float = 5.0, backstep: int = 10, point: float = 0.01, addmeta: bool = True, final_check: bool = True) -> pd.DataFrame`
- `zigzag_mtf_adapter(high: pd.Series, low: pd.Series, tf_higher: str, depth: int = 12, deviation: float = 5.0, backstep: int = 10, point: float = 0.01, mode: Literal['last', 'forward_fill'] = 'forward_fill', extend_last_leg: bool = False, use_timeshift: bool = False) -> pd.Series`
- `zigzag_legs(high: pd.Series, low: pd.Series, tf: str, depth: int = 12, deviation: float = 5.0, backstep: int = 10, point: float = 0.01, extend_last_leg: bool = False, use_timeshift: bool = False) -> pd.DataFrame`

