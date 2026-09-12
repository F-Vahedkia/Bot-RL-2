# f05_agents/f05_agents_tester_A.py (5)
#
# Created:
#     1405/06/19
#
# Run:
#     pytest -v -s f05_agents/f05_agents_tester_A.py


from __future__ import annotations
from datetime import datetime, timezone
import pytest


from f05_agents.agent_state import (
    MetaAgentState,
    SymbolAgentState,
)
from f05_agents.contracts import (
    AgentObservation,
    DecisionMode,
    MetaPolicyOutput,
    ModelIdentity,
    PortfolioContext,
    PolicyOutput,
    SymbolAgentOutput,
    SymbolContext,
)
from f05_agents.meta_agent import (
    MetaAgent,
    MetaAgentConfig,
)
from f05_agents.symbol_agent import (
    SymbolAgent,
    SymbolAgentConfig,
)


# =====================================================================
# Test Constants
# =====================================================================

TIMESTAMP = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


MODEL_XAU = ModelIdentity(
    model_name="symbol_xauusd",
    model_version="1.0.0",
    policy_version="policy-1",
    config_version="cfg-1",
    experiment_id="exp-1",
)


MODEL_META = ModelIdentity(
    model_name="meta_portfolio",
    model_version="1.0.0",
    policy_version="policy-1",
    config_version="cfg-1",
    experiment_id="exp-1",
)


# =====================================================================
# Test Policy - Symbol
# =====================================================================

class FixedSymbolPolicy:
    """
    Policy قطعی برای تست.

    هیچ منطق واقعی بازار ندارد.
    فقط برای deterministic unit test است.
    """

    def __init__(
        self,
        *,
        signal: int,
        confidence: float,
        expected_return: float,
        risk_score: float,
        desired_exposure: float,
    ) -> None:

        self.output = PolicyOutput(
            signal=signal,
            confidence=confidence,
            expected_return=expected_return,
            risk_score=risk_score,
            desired_exposure=desired_exposure,
        )


    def predict(
        self,
        *,
        observation: AgentObservation,
        context: SymbolContext,
    ) -> PolicyOutput:

        return self.output


# =====================================================================
# Test Policy - Meta
# =====================================================================

class FixedMetaPolicy:
    """
    Policy قطعی برای تست Meta-Agent.
    """

    def __init__(
        self,
        output: MetaPolicyOutput,
    ) -> None:

        self.output = output


    def predict(
        self,
        *,
        signals,
        portfolio,
    ) -> MetaPolicyOutput:

        return self.output


# =====================================================================
# Helpers
# =====================================================================

def make_observation(
    symbol: str,
) -> AgentObservation:

    return AgentObservation(
        symbol=symbol,
        base_tf="M1",
        timestamp=TIMESTAMP,
        values=(0.1, 0.2, 0.3),
        feature_names=(
            "f1",
            "f2",
            "f3",
        ),
    )


def make_context(
    symbol: str,
) -> SymbolContext:

    return SymbolContext(
        symbol=symbol,
        current_side=0,
        current_lots=0.0,
        exposure=0.0,
        drawdown=0.0,
        volatility=0.10,
        local_risk_score=0.20,
    )


def make_portfolio(
    *,
    drawdown: float = 0.0,
    risk_blocked: bool = False,
    correlation=None,
) -> PortfolioContext:

    return PortfolioContext(
        timestamp=TIMESTAMP,
        equity=10_000.0,
        balance=10_000.0,
        used_margin=1_000.0,
        free_margin=9_000.0,
        margin_level=10.0,
        drawdown=drawdown,
        daily_drawdown=drawdown,
        exposure={},
        concentration={},
        correlation=correlation or {},
        risk_blocked=risk_blocked,
        mode=DecisionMode.BACKTEST,
    )


# =====================================================================
# Contract Tests
# =====================================================================

def test_model_identity():

    model = ModelIdentity(
        model_name="test",
        model_version="1",
        policy_version="p1",
    )

    assert model.model_name == "test"
    assert model.model_version == "1"
    assert model.policy_version == "p1"


def test_observation_normalizes_symbol():

    observation = AgentObservation(
        symbol="xauusd",
        base_tf="m1",
        timestamp=TIMESTAMP,
        values=(1.0, 2.0),
    )

    assert observation.symbol == "XAUUSD"
    assert observation.base_tf == "M1"


def test_observation_rejects_naive_timestamp():

    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):

        AgentObservation(
            symbol="XAUUSD",
            base_tf="M1",
            timestamp=datetime(
                2026,
                1,
                1,
            ),
            values=(1.0,),
        )


# =====================================================================
# Symbol-Agent Tests
# =====================================================================

def test_symbol_agent_decision():

    policy = FixedSymbolPolicy(
        signal=1,
        confidence=0.9,
        expected_return=0.02,
        risk_score=0.2,
        desired_exposure=0.4,
    )

    agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="XAUUSD",
            mode=DecisionMode.BACKTEST,
            model=MODEL_XAU,
            min_confidence=0.5,
        ),
        policy=policy,
    )

    output = agent.decide(
        observation=make_observation(
            "XAUUSD"
        ),
        context=make_context(
            "XAUUSD"
        ),
        decision_id="decision-001",
    )

    assert output.symbol == "XAUUSD"
    assert output.signal == 1
    assert output.confidence == pytest.approx(
        0.9
    )
    assert output.desired_exposure == pytest.approx(
        0.4
    )

    assert agent.state.decision_count == 1


def test_symbol_agent_isolated_per_symbol():

    xau_policy = FixedSymbolPolicy(
        signal=1,
        confidence=0.9,
        expected_return=0.01,
        risk_score=0.1,
        desired_exposure=0.3,
    )

    eur_policy = FixedSymbolPolicy(
        signal=-1,
        confidence=0.8,
        expected_return=-0.01,
        risk_score=0.2,
        desired_exposure=0.2,
    )

    xau_agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="XAUUSD",
            mode=DecisionMode.BACKTEST,
            model=MODEL_XAU,
        ),
        policy=xau_policy,
    )

    eur_agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="EURUSD",
            mode=DecisionMode.BACKTEST,
            model=MODEL_XAU,
        ),
        policy=eur_policy,
    )

    xau = xau_agent.decide(
        observation=make_observation("XAUUSD"),
        context=make_context("XAUUSD"),
        decision_id="xau-1",
    )

    eur = eur_agent.decide(
        observation=make_observation("EURUSD"),
        context=make_context("EURUSD"),
        decision_id="eur-1",
    )

    assert xau.signal == 1
    assert eur.signal == -1

    assert xau_agent.state.last_signal == 1
    assert eur_agent.state.last_signal == -1


def test_symbol_agent_rejects_wrong_symbol():

    agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="XAUUSD",
            mode=DecisionMode.BACKTEST,
            model=MODEL_XAU,
        ),
        policy=FixedSymbolPolicy(
            signal=1,
            confidence=0.9,
            expected_return=0.01,
            risk_score=0.1,
            desired_exposure=0.3,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Observation symbol mismatch",
    ):

        agent.decide(
            observation=make_observation(
                "EURUSD"
            ),
            context=make_context(
                "XAUUSD"
            ),
            decision_id="wrong-1",
        )


def test_symbol_agent_confidence_guardrail():

    agent = SymbolAgent(
        config=SymbolAgentConfig(
            symbol="XAUUSD",
            mode=DecisionMode.BACKTEST,
            model=MODEL_XAU,
            min_confidence=0.8,
        ),
        policy=FixedSymbolPolicy(
            signal=1,
            confidence=0.5,
            expected_return=0.01,
            risk_score=0.1,
            desired_exposure=0.3,
        ),
    )

    output = agent.decide(
        observation=make_observation(
            "XAUUSD"
        ),
        context=make_context(
            "XAUUSD"
        ),
        decision_id="confidence-1",
    )

    assert output.signal == 0
    assert output.confidence == pytest.approx(
        0.5
    )


# =====================================================================
# Agent State Tests
# =====================================================================

def test_symbol_agent_state_reset():

    state = SymbolAgentState(
        symbol="XAUUSD"
    )

    state.record_decision(
        signal=1,
        confidence=0.9,
        expected_return=0.01,
        risk_score=0.1,
        desired_exposure=0.2,
        timestamp=TIMESTAMP,
        model=MODEL_XAU,
        mode=DecisionMode.BACKTEST,
    )

    state.add_reward(5.0)

    assert state.decision_count == 1
    assert state.last_signal == 1
    assert state.cumulative_reward == pytest.approx(
        5.0
    )

    state.reset()

    assert state.decision_count == 0
    assert state.last_signal == 0
    assert state.cumulative_reward == 0.0
    assert state.model is None


def test_meta_agent_state_reset():

    state = MetaAgentState()

    state.record_decision(
        approved=True,
        equity=10_000.0,
        drawdown=0.1,
        total_exposure=0.4,
        capital_allocation={
            "XAUUSD": 0.2
        },
        margin_allocation={
            "XAUUSD": 2_000.0
        },
        timestamp=TIMESTAMP,
        model=MODEL_META,
        mode=DecisionMode.BACKTEST,
    )

    assert state.approved_count == 1

    state.reset()

    assert state.decision_count == 0
    assert state.approved_count == 0
    assert state.capital_allocation == {}


# =====================================================================
# Meta-Agent Tests
# =====================================================================

def make_signal(
    symbol: str,
    signal: int,
    confidence: float,
) -> SymbolAgentOutput:

    return SymbolAgentOutput(
        symbol=symbol,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        signal=signal,
        confidence=confidence,
        expected_return=0.01,
        risk_score=0.2,
        desired_exposure=0.3,
        model=MODEL_XAU,
        decision_id=f"{symbol}-signal",
    )


def test_meta_agent_approval():

    policy = FixedMetaPolicy(
        MetaPolicyOutput(
            capital_allocation={
                "XAUUSD": 0.30,
                "EURUSD": 0.20,
            },
            target_signals={
                "XAUUSD": 1,
                "EURUSD": -1,
            },
            target_exposure={
                "XAUUSD": 0.30,
                "EURUSD": -0.20,
            },
            margin_allocation={
                "XAUUSD": 3_000.0,
                "EURUSD": 2_000.0,
            },
            portfolio_risk=0.40,
            reason_codes=(
                "policy_output",
            ),
        )
    )

    agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=MODEL_META,
            max_symbol_allocation=0.5,
            max_total_allocation=1.0,
            max_total_exposure=1.0,
        ),
        policy=policy,
    )

    decision = agent.decide(
        signals={
            "XAUUSD": make_signal(
                "XAUUSD",
                1,
                0.9,
            ),
            "EURUSD": make_signal(
                "EURUSD",
                -1,
                0.8,
            ),
        },
        portfolio=make_portfolio(),
        decision_id="meta-001",
    )

    assert decision.approved is True

    assert decision.target_signals["XAUUSD"] == 1
    assert decision.target_signals["EURUSD"] == -1

    assert decision.capital_allocation["XAUUSD"] == pytest.approx(
        0.30
    )

    assert agent.state.approved_count == 1


def test_meta_agent_symbol_concentration_cap():

    policy = FixedMetaPolicy(
        MetaPolicyOutput(
            capital_allocation={
                "XAUUSD": 0.80,
                "EURUSD": 0.10,
            },
            target_signals={
                "XAUUSD": 1,
                "EURUSD": 1,
            },
            target_exposure={
                "XAUUSD": 0.80,
                "EURUSD": 0.10,
            },
            margin_allocation={
                "XAUUSD": 8_000.0,
                "EURUSD": 1_000.0,
            },
            portfolio_risk=0.5,
        )
    )

    agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=MODEL_META,
            max_symbol_allocation=0.25,
        ),
        policy=policy,
    )

    decision = agent.decide(
        signals={
            "XAUUSD": make_signal(
                "XAUUSD",
                1,
                0.9,
            ),
            "EURUSD": make_signal(
                "EURUSD",
                1,
                0.8,
            ),
        },
        portfolio=make_portfolio(),
        decision_id="meta-002",
    )

    assert decision.approved is True

    assert decision.capital_allocation["XAUUSD"] == pytest.approx(
        0.25
    )


def test_meta_agent_correlation_cap():

    policy = FixedMetaPolicy(
        MetaPolicyOutput(
            capital_allocation={
                "XAUUSD": 0.40,
                "EURUSD": 0.40,
            },
            target_signals={
                "XAUUSD": 1,
                "EURUSD": 1,
            },
            target_exposure={
                "XAUUSD": 0.40,
                "EURUSD": 0.40,
            },
            margin_allocation={
                "XAUUSD": 4_000.0,
                "EURUSD": 4_000.0,
            },
            portfolio_risk=0.5,
        )
    )

    agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=MODEL_META,
            max_correlated_allocation=0.50,
            high_correlation_threshold=0.85,
        ),
        policy=policy,
    )

    decision = agent.decide(
        signals={
            "XAUUSD": make_signal(
                "XAUUSD",
                1,
                0.9,
            ),
            "EURUSD": make_signal(
                "EURUSD",
                1,
                0.9,
            ),
        },
        portfolio=make_portfolio(
            correlation={
                "XAUUSD": {
                    "EURUSD": 0.95,
                },
                "EURUSD": {
                    "XAUUSD": 0.95,
                },
            }
        ),
        decision_id="meta-003",
    )

    combined = (
        decision.capital_allocation["XAUUSD"]
        +
        decision.capital_allocation["EURUSD"]
    )

    assert combined == pytest.approx(
        0.50
    )


def test_meta_agent_risk_block():

    policy = FixedMetaPolicy(
        MetaPolicyOutput(
            capital_allocation={
                "XAUUSD": 0.30,
            },
            target_signals={
                "XAUUSD": 1,
            },
            target_exposure={
                "XAUUSD": 0.30,
            },
            margin_allocation={
                "XAUUSD": 3_000.0,
            },
            portfolio_risk=0.3,
        )
    )

    agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=MODEL_META,
        ),
        policy=policy,
    )

    decision = agent.decide(
        signals={
            "XAUUSD": make_signal(
                "XAUUSD",
                1,
                0.9,
            ),
        },
        portfolio=make_portfolio(
            risk_blocked=True
        ),
        decision_id="meta-risk-001",
    )

    assert decision.approved is False

    assert decision.capital_allocation == {}

    assert (
        "portfolio_risk_blocked"
        in decision.reason_codes
    )

    assert agent.state.rejected_count == 1


def test_meta_agent_deterministic():

    policy = FixedMetaPolicy(
        MetaPolicyOutput(
            capital_allocation={
                "XAUUSD": 0.20,
                "EURUSD": 0.20,
            },
            target_signals={
                "XAUUSD": 1,
                "EURUSD": -1,
            },
            target_exposure={
                "XAUUSD": 0.20,
                "EURUSD": -0.20,
            },
            margin_allocation={
                "XAUUSD": 2_000.0,
                "EURUSD": 2_000.0,
            },
            portfolio_risk=0.3,
        )
    )

    agent = MetaAgent(
        config=MetaAgentConfig(
            mode=DecisionMode.BACKTEST,
            model=MODEL_META,
        ),
        policy=policy,
    )

    signals = {
        "XAUUSD": make_signal(
            "XAUUSD",
            1,
            0.9,
        ),
        "EURUSD": make_signal(
            "EURUSD",
            -1,
            0.9,
        ),
    }

    portfolio = make_portfolio()

    first = agent.decide(
        signals=signals,
        portfolio=portfolio,
        decision_id="det-001",
    )

    agent.reset()

    second = agent.decide(
        signals=signals,
        portfolio=portfolio,
        decision_id="det-001",
    )

    assert first == second