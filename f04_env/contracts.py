# f04_env/contracts.py

# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Class-1
# =============================================================================
@dataclass(frozen=True, slots=True)
class ObservationContract:
    """
    Contract for one-symbol market observation.

    The observation itself is kept outside this contract.
    The contract only describes its structural identity.
    """

    symbol: str
    base_tf: str
    columns: Tuple[str, ...]
    row_count: int

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()
        base_tf = str(self.base_tf).upper().strip()

        if not symbol:
            raise ValueError("symbol is required")

        if not base_tf:
            raise ValueError("base_tf is required")

        if int(self.row_count) < 0:
            raise ValueError("row_count must be >= 0")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "base_tf", base_tf)
        object.__setattr__(self, "columns", tuple(str(c) for c in self.columns))
        object.__setattr__(self, "row_count", int(self.row_count))

    @property
    def feature_count(self) -> int:
        return len(self.columns)


# =============================================================================
# Class-2
# =============================================================================
@dataclass(frozen=True, slots=True)
class PositionIntent_old:
    """
    Requested position intent for one symbol.

    This is NOT a broker order and NOT a final execution result.
    """

    symbol: str
    target_side: int = 0
    target_lots: float = 0.0

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()

        if not symbol:
            raise ValueError("symbol is required")

        side = int(self.target_side)
        if side not in (-1, 0, 1):
            raise ValueError("target_side must be -1, 0, or 1")

        lots = float(self.target_lots)
        if not np.isfinite(lots):
            raise ValueError("target_lots must be finite")

        if lots < 0.0:
            raise ValueError("target_lots must be >= 0")

        if side == 0 and lots != 0.0:
            raise ValueError("flat intent must have target_lots=0")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "target_side", side)
        object.__setattr__(self, "target_lots", lots)


@dataclass(frozen=True, slots=True)
class PositionIntent:
    """
    Requested position intent for one symbol.

    This is NOT a broker order and NOT a final execution result.

    target_lots:
        Final position quantity produced after risk validation /
        position sizing.

    stop_price:
        Protective stop-loss price associated with this position intent.
        It is optional only for flat intents.
    """

    symbol: str
    target_side: int = 0
    target_lots: float = 0.0
    stop_price: Optional[float] = None

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()
        if not symbol:
            raise ValueError("symbol is required")

        side = int(self.target_side)
        if side not in (-1, 0, 1):
            raise ValueError("target_side must be -1, 0, or 1")

        lots = float(self.target_lots)
        if not np.isfinite(lots):
            raise ValueError("target_lots must be finite")

        if lots < 0.0:
            raise ValueError("target_lots must be >= 0")

        stop_price = self.stop_price
        if stop_price is not None:
            stop_price = float(stop_price)

            if not np.isfinite(stop_price) or stop_price <= 0.0:
                raise ValueError("stop_price must be a positive finite value")

        if side == 0:
            if lots != 0.0:
                raise ValueError("flat intent must have target_lots=0")

            if stop_price is not None:
                raise ValueError("flat intent must have stop_price=None")

        else:
            # stop_price is optional for a non-flat position.
            # Risk / portfolio layers may provide, override, or omit it.
            pass
        
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "target_side", side)
        object.__setattr__(self, "target_lots", lots)
        object.__setattr__(self, "stop_price", stop_price)
        
# =============================================================================
# Class-3
# =============================================================================
@dataclass(frozen=True, slots=True)
class PortfolioAction:
    """
    Portfolio-level action.

    The Meta-Agent will eventually produce this semantic object.
    Numerical RL encoding is intentionally excluded here.
    """

    intents: Tuple[PositionIntent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        normalized = tuple(self.intents)

        seen = set()
        for intent in normalized:
            if not isinstance(intent, PositionIntent):
                raise TypeError(
                    "PortfolioAction.intents must contain PositionIntent objects"
                )

            if intent.symbol in seen:
                raise ValueError(
                    f"Duplicate symbol in PortfolioAction: {intent.symbol}"
                )

            seen.add(intent.symbol)

        object.__setattr__(self, "intents", normalized)


# =============================================================================
# Class-4
# =============================================================================
@dataclass(frozen=True, slots=True)
class StepResult:
    """
    Environment step result.

    observation:
        Next observation supplied to the agent.
    reward:
        Scalar training reward.
    terminated:
        Natural/environment termination.
    truncated:
        Artificial/time-limit truncation.
    info:
        Diagnostic information; never used as hidden state.
    """

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        obs = np.asarray(self.observation)

        if not np.issubdtype(obs.dtype, np.number):
            raise TypeError("StepResult.observation must be numeric")

        reward = float(self.reward)

        if not np.isfinite(reward):
            raise ValueError("StepResult.reward must be finite")

        object.__setattr__(self, "observation", obs)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(self, "info", dict(self.info))


# =============================================================================
# Class-5
# =============================================================================
@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """
    Minimal v8 environment contract.

    This deliberately contains only semantics required by the core.
    """

    base_tf: str = "M1"
    window_size: int = 128
    initial_balance: float = 10_000.0
    max_episode_steps: Optional[int] = None
    max_drawdown_pct: Optional[float] = None
    stop_out_level: Optional[float] = None
    leverage: float = 1.0

    def __post_init__(self) -> None:
        base_tf = str(self.base_tf).upper().strip()

        if not base_tf:
            raise ValueError("base_tf is required")

        if int(self.window_size) <= 0:
            raise ValueError("window_size must be > 0")

        balance = float(self.initial_balance)
        if not np.isfinite(balance) or balance <= 0.0:
            raise ValueError("initial_balance must be > 0")

        if self.max_episode_steps is not None:
            steps = int(self.max_episode_steps)
            if steps <= 0:
                raise ValueError("max_episode_steps must be > 0")
        else:
            steps = None

        # =========== added_1 start
        if self.max_drawdown_pct is not None:
            dd = float(self.max_drawdown_pct)
            if not (0.0 < dd < 1.0):
                raise ValueError("max_drawdown_pct must be between 0 and 1")

        if self.stop_out_level is not None:
            sol = float(self.stop_out_level)
            if sol <= 0.0:
                raise ValueError("stop_out_level must be > 0")
        # =========== added_1 end

        # =========== added_2 start
        leverage = float(self.leverage)
        if not np.isfinite(leverage) or leverage <= 0.0:
            raise ValueError(
                "leverage must be > 0"
            )
        # =========== added_2 end
            
        object.__setattr__(self, "base_tf", base_tf)
        object.__setattr__(self, "window_size", int(self.window_size))
        object.__setattr__(self, "initial_balance", balance)
        object.__setattr__(self, "max_episode_steps", steps)
        object.__setattr__(self, "leverage", leverage)

# ============================================================================= END
