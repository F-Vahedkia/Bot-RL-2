# f04_env/execution_simulator.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from enum import Enum
from f04_env.contracts import PositionIntent
from f04_env.portfolio_state import PositionState


class ExecutionType(str, Enum):
    OPEN = "open"
    INCREASE = "increase"
    REDUCE = "reduce"
    REVERSE = "reverse"
    CLOSE = "close"


@dataclass(frozen=True, slots=True)
class ExecutionCost:
    """
    Cost model for simulated execution.

    Values are expressed in price units except commission,
    which is expressed directly in account currency.
    """

    spread: float = 0.0
    slippage: float = 0.0
    commission: float = 0.0

    def total_price_cost(self) -> float:
        return abs(float(self.spread)) + abs(float(self.slippage))


class ExecutionSimulator:
    """
    Deterministic execution simulator.

    Responsibilities:
        - apply target position intents
        - calculate synthetic fill price
        - calculate realized PnL
        - calculate execution costs

    Non-responsibilities:
        - risk decisions
        - position sizing policy
        - reward calculation
        - broker communication
    """

    def __init__(
        self,
        *,
        contract_size: float = 1.0,
        point_value: float = 1.0,
    ) -> None:
        if contract_size <= 0.0:
            raise ValueError("contract_size must be > 0")

        if point_value <= 0.0:
            raise ValueError("point_value must be > 0")

        self.contract_size = float(contract_size)
        self.point_value = float(point_value)

    @staticmethod
    def _side_sign(side: int) -> int:
        if side > 0:
            return 1
        if side < 0:
            return -1
        return 0

    @staticmethod
    def _classify_transition(
        old_side: int,
        old_lots: float,
        new_side: int,
        new_lots: float,
    ) -> ExecutionType:

        if old_side == 0 and new_side != 0:
            return ExecutionType.OPEN

        if old_side != 0 and new_side == 0:
            return ExecutionType.CLOSE

        if old_side == new_side:
            if new_lots > old_lots:
                return ExecutionType.INCREASE
            if new_lots < old_lots:
                return ExecutionType.REDUCE
            return ExecutionType.REDUCE

        if old_side != new_side:
            return ExecutionType.REVERSE

        raise RuntimeError("Invalid execution transition")

    def execute(
        self,
        *,
        position: PositionState,
        intent: PositionIntent,
        market_price: float,
        cost: ExecutionCost | None = None,
    ) -> Dict[str, float]:
        """
        Apply a target position to one symbol.

        Current implementation uses a deterministic close-price fill.
        More advanced bid/ask/slippage models can be added later.
        """
        if position.symbol != intent.symbol:
            raise ValueError(
                f"Position symbol mismatch: "
                f"{position.symbol!r} != {intent.symbol!r}"
            )

        price = float(market_price)

        if not (price > 0.0):
            raise ValueError("market_price must be > 0")

        cost = cost or ExecutionCost()

        old_side = int(position.side)
        old_lots = float(position.lots)
        old_entry = position.entry_price

        new_side = int(intent.target_side)
        new_lots = float(intent.target_lots)

        execution_type = self._classify_transition(
            old_side,
            old_lots,
            new_side,
            new_lots,
        )

        realized_pnl = 0.0

        # Close existing exposure first.
        if old_side != 0 and old_lots > 0.0:
            if old_entry is None:
                raise RuntimeError(
                    f"Open position {position.symbol} has no entry_price"
                )

            price_diff = (price - float(old_entry)) * old_side

            realized_pnl = (
                price_diff
                * old_lots
                * self.contract_size
                * self.point_value
            )

        # Execution cost is applied once per changed exposure.
        changed = (
            old_side != new_side
            or abs(old_lots - new_lots) > 1e-12
        )

        if changed:
            commission = float(cost.commission)
        else:
            commission = 0.0

        # Replace position with target.
        if new_side == 0 or new_lots <= 0.0:
            position.side = 0
            position.lots = 0.0
            position.entry_price = None
            position.current_price = price
        else:
            position.side = new_side
            position.lots = new_lots
            position.entry_price = price
            position.current_price = price

        position.realized_pnl += realized_pnl - commission

        return {
            "realized_pnl": float(realized_pnl - commission),
            "commission": commission,
            "spread": float(cost.spread),
            "slippage": float(cost.slippage),
            "changed": float(changed),
            "execution_type": execution_type.value,
        }

    def mark_to_market(
        self,
        *,
        positions: Dict[str, PositionState],
        prices: Dict[str, float],
    ) -> float:
        """
        Calculate total unrealized PnL at current prices.
        """
        total = 0.0

        for symbol, position in positions.items():
            if position.side == 0 or position.lots <= 0.0:
                position.unrealized_pnl = 0.0
                continue

            if position.entry_price is None:
                raise RuntimeError(
                    f"Open position {symbol} has no entry_price"
                )

            if symbol not in prices:
                raise KeyError(
                    f"Missing market price for open symbol {symbol}"
                )

            current = float(prices[symbol])
            position.current_price = current

            pnl = (
                (current - float(position.entry_price))
                * self._side_sign(position.side)
                * position.lots
                * self.contract_size
                * self.point_value
            )

            position.unrealized_pnl = float(pnl)
            total += pnl

        return float(total)

