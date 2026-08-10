class Supertrend:
    def __init__(self, 
                 period: int = 10, 
                 multiplier: float = 3.0,
                 atr_method: Literal["classic", "wilder", "ema"] = "wilder",
                 min_periods: Optional[int] = None):

        self.period = period
        self.multiplier = multiplier
        self.atr_method = atr_method
        self.min_periods = min_periods if min_periods is not None else period
        
        # Map string method to integer for jitclass
        method_map = {"classic": 0, "wilder": 1, "ema": 2}
        atr_method_int = method_map.get(atr_method, 1)  # Default to wilder
        
        # Initialize state with integer method
        self.state = SupertrendState(
            period=period,
            multiplier=multiplier,
            atr_method=atr_method_int,
            min_periods=self.min_periods
        )
    
    def update(self, high: float, low: float, close: float, **kwargs) -> Tuple[float, int]:

        return self.state.update(high, low, close)
    
    def reset(self):

        self.state.reset()



class Stochastic:
    def __init__(
        self, 
        k_period: int = 14, 
        d_period: int = 3,
        smooth_k: int = 3,
        method: Literal["sma", "ema"] = "sma",
        min_periods: Optional[int] = None,
        **kwargs,
    ):
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.method = method
        self.min_periods = min_periods
        
        # تبدیل method از string به int برای jitclass
        method_map = {"sma": 0, "ema": 1}
        method_int = method_map[method]
        
        self.state = StochasticState(
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
            method=method_int,
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, float]:
        return self.state.update(high, low, close)

    def reset(self):
        self.state.reset()


class BollingerBands:
    def __init__(self, period: int = 20, std_dev: float = 2.0, min_periods: Optional[int] = None, **kwargs):
        self.period = period
        self.std_dev = std_dev
        self.min_periods = min_periods
        
        # تبدیل min_periods به فرمت jitclass
        mp = -1 if min_periods is None else min_periods
        self.state = BollingerState(n=period, k=std_dev, min_periods=mp)
    
    def update(self, close: float) -> Tuple[float, float, float, float, float]:
        """Returns (upper, middle, lower, width, percent)"""
        return self.state.update(close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


class KeltnerChannel:
    """ Keltner Channel wrapper """
    
    def __init__(
        self, 
        period: int = 20, 
        multiplier: float = 2.0,
        min_periods: Optional[int] = None,
        **kwargs,
    ):
        self.period = period
        self.multiplier = multiplier
        self.min_periods = min_periods
        self.state = KeltnerState(
            n=period, 
            m=multiplier, 
            min_periods=min_periods if min_periods is not None else -1
        )
    
    def update(self, high: float, low: float, close: float) -> Tuple[float, float, float, float, float]:
        """Returns (upper, middle, lower, width, percentile)"""
        return self.state.update(high, low, close)

    def reset(self):
        """Reset indicator state"""
        self.state.reset()


class HeikinAshi:
    """ Heikin-Ashi wrapper """
    
    def __init__(self):
        self.state = HeikinAshiState()
    
    def update(self, open_: float, high: float, low: float, close: float, **kwargs) -> Tuple[float, float, float, float]:
        """
        Returns: (ha_open, ha_high, ha_low, ha_close)
        On first call, HA Open = (Open + Close) / 2
        """
        return self.state.update(open_, high, low, close)

    def reset(self):
        """Reset state for new sequence"""
        self.state.reset()


"""
_apply_live()
_update_live()           # به نظر خودم، سبک تر میشود
_build_live_instance()     # هیچ تغییر خاصی لازم ندارد.
_attach_live_output()
_build_live_column_name()
_normalize_output()
"""