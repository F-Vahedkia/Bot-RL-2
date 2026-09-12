# f04_env/trading_env_tester_A.py
# Run: pytest -v -s f04_env/trading_env_tester_A.py

import numpy as np
import pytest

from f04_env.contracts import (
    EnvironmentConfig,
    PortfolioAction,
    PositionIntent,
)
from f04_env.execution_simulator import (
    ExecutionCost,
    ExecutionSimulator,
)
from f04_env.trading_env import TradingEnvironment
from f04_env.portfolio_state import PositionState


SYMBOLS = ("XAUUSD", "EURUSD")


def make_env(max_episode_steps=None):
    n = 5

    observations = {
        "XAUUSD": np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        "EURUSD": np.arange(100, 100 + n * 2, dtype=np.float32).reshape(n, 2),
    }

    prices = {
        "XAUUSD": np.array([100, 101, 102, 103, 104], dtype=np.float64),
        "EURUSD": np.array([10, 11, 12, 13, 14], dtype=np.float64),
    }

    cfg = EnvironmentConfig(
        base_tf="M1",
        window_size=1,
        initial_balance=10_000.0,
        max_episode_steps=max_episode_steps,
    )

    return TradingEnvironment(
        observations=observations,
        prices=prices,
        config=cfg,
        execution=ExecutionSimulator(
            contract_size=1.0,
            point_value=1.0,
        ),
        execution_costs={
            "XAUUSD": ExecutionCost(commission=2.0),
            "EURUSD": ExecutionCost(commission=1.0),
        },
    )


def test_reset_is_multi_symbol():
    env = make_env()

    obs, info = env.reset(seed=42)

    assert obs.dtype == np.float32
    assert obs.shape == (4,)
    assert info["symbols"] == SYMBOLS
    assert env.portfolio.equity == 10_000.0


def test_open_position_and_mark_to_market():
    env = make_env()

    env.reset()

    action = PortfolioAction(
        (
            PositionIntent(
                symbol="XAUUSD",
                target_side=1,
                target_lots=1.0,
            ),
        )
    )

    result = env.step(action)

    position = env.portfolio.get_position("XAUUSD")

    assert position.side == 1
    assert position.lots == 1.0
    assert position.entry_price == 100.0

    # Current step has moved to price 101.
    assert position.unrealized_pnl == pytest.approx(1.0)


def test_close_position_realizes_pnl():
    env = make_env()

    env.reset()

    env.step(
        PortfolioAction(
            (
                PositionIntent(
                    "XAUUSD",
                    target_side=1,
                    target_lots=1.0,
                ),
            )
        )
    )

    env.step(
        PortfolioAction(
            (
                PositionIntent(
                    "XAUUSD",
                    target_side=0,
                    target_lots=0.0,
                ),
            )
        )
    )

    position = env.portfolio.get_position("XAUUSD")

    assert position.side == 0
    assert position.lots == 0.0
    assert position.entry_price is None


def test_symbols_are_independent():
    env = make_env()

    env.reset()

    env.step(
        PortfolioAction(
            (
                PositionIntent("XAUUSD", 1, 1.0),
                PositionIntent("EURUSD", -1, 2.0),
            )
        )
    )

    assert env.portfolio.get_position("XAUUSD").side == 1
    assert env.portfolio.get_position("EURUSD").side == -1


def test_unknown_symbol_rejected():
    env = make_env()
    env.reset()

    with pytest.raises(ValueError, match="unknown symbols"):
        env.step(
            PortfolioAction(
                (
                    PositionIntent("GBPUSD", 1, 1.0),
                )
            )
        )


def test_done_environment_rejects_further_steps():
    env = make_env()
    env.reset()

    while not env._terminated and not env._truncated:
        env.step(PortfolioAction())

    with pytest.raises(RuntimeError, match="Call reset"):
        env.step(PortfolioAction())


def test_max_episode_steps_produces_truncation():
    env = make_env(max_episode_steps=2)
    env.reset()

    env.step(PortfolioAction())
    result = env.step(PortfolioAction())

    assert result.truncated is True
    assert result.terminated is False


def test_same_inputs_are_deterministic():
    def run():
        env = make_env()
        env.reset(seed=123)

        actions = [
            PortfolioAction(
                (
                    PositionIntent("XAUUSD", 1, 1.0),
                )
            ),
            PortfolioAction(),
            PortfolioAction(
                (
                    PositionIntent("XAUUSD", 0, 0.0),
                )
            ),
        ]

        out = []

        for action in actions:
            result = env.step(action)
            out.append(
                (
                    result.reward,
                    result.observation.copy(),
                    result.terminated,
                    result.truncated,
                    env.portfolio.equity,
                )
            )

        return out

    a = run()
    b = run()

    for x, y in zip(a, b):
        assert x[0] == pytest.approx(y[0])
        np.testing.assert_array_equal(x[1], y[1])
        assert x[2:] == y[2:]


def test_execution_transition_types():

    simulator = ExecutionSimulator()
    position = PositionState(symbol="XAUUSD")

    # ------------- flat -> long 1
    result = simulator.execute(
        position=position,
        intent=PositionIntent("XAUUSD", 1, 1.0),
        market_price=100.0,
    )
    assert result["execution_type"] == "open"


    # ------------- long 1 -> long 2
    result = simulator.execute(
        position=position,
        intent=PositionIntent("XAUUSD", 1, 2.0),
        market_price=101.0,
    )
    assert result["execution_type"] == "increase"


    # ------------- long 2 -> long 1
    result = simulator.execute(
        position=position,
        intent=PositionIntent("XAUUSD", 1, 1.0),
        market_price=102.0,
    )
    assert result["execution_type"] == "reduce"


    # ------------- long -> short
    result = simulator.execute(
        position=position,
        intent=PositionIntent("XAUUSD", -1, 1.0),
        market_price=103.0,
    )
    assert result["execution_type"] == "reverse"


    # ------------- short -> flat
    result = simulator.execute(
        position=position,
        intent=PositionIntent("XAUUSD", 0, 0.0),
        market_price=104.0,
    )
    assert result["execution_type"] == "close"

