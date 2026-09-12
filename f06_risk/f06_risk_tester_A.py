# f06_risk/f06_risk_tester_A.py (6)
#
# Run: pytest -v -s f06_risk/f06_risk_tester_A.py


from __future__ import annotations
from datetime import datetime, timezone
import pytest

from f04_env.contracts import PortfolioAction
from f05_agents.contracts import (
    DecisionMode,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
)
from f06_risk.limits import RiskLimits
from f06_risk.risk_engine import RiskEngine
from f06_risk.action_builder import RiskActionBuilder
from f06_risk.contracts import (
    RiskDecisionStatus,
    RiskRequest,
)


TIMESTAMP = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

MODEL = ModelIdentity(
    model_name="meta-policy",
    model_version="1.0.0",
    policy_version="meta-policy-1",
    config_version="cfg-1",
    experiment_id="risk-test",
)


def make_portfolio(
    *,
    drawdown: float = 0.0,
    daily_drawdown: float = 0.0,
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
        daily_drawdown=daily_drawdown,
        exposure={},
        concentration={},
        correlation=correlation or {},
        risk_blocked=risk_blocked,
        mode=DecisionMode.BACKTEST,
    )


def make_decision(
    *,
    allocation=0.20,
    exposure=0.20,
    margin=2_000.0,
    risk=0.20,
) -> PortfolioDecision:

    return PortfolioDecision(
        approved=True,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        decision_id="decision-001",
        capital_allocation={"XAUUSD": allocation},
        margin_allocation={"XAUUSD": margin},
        target_exposure={"XAUUSD": exposure},
        target_signals={"XAUUSD": 1},
        portfolio_risk=risk,
        reason_codes=("test",),
        model=MODEL,
    )


# =====================================================================
# 1
# =====================================================================

def test_safe_decision_is_approved():

    engine = RiskEngine(
        limits=RiskLimits()
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(),
            portfolio=make_portfolio(),
        )
    )

    assert result.status == RiskDecisionStatus.APPROVED
    assert result.approved is True
    assert result.rejected is False


# =====================================================================
# 2
# =====================================================================

def test_symbol_exposure_is_reduced():

    engine = RiskEngine(
        limits=RiskLimits(
            max_symbol_exposure=0.25,
        )
    )
    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(
                exposure=0.60
            ),
            portfolio=make_portfolio(),
        )
    )
    assert result.status == RiskDecisionStatus.MODIFIED
    assert result.target_exposure["XAUUSD"] == pytest.approx(0.25)


# =====================================================================
# 3
# =====================================================================

def test_max_drawdown_rejects():

    engine = RiskEngine(
        limits=RiskLimits(
            max_drawdown=0.20,
        )
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(),
            portfolio=make_portfolio(
                drawdown=0.20
            ),
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED
    assert ("MAX_DRAWDOWN" in {x.code for x in result.violations})


# =====================================================================
# 4
# =====================================================================

def test_daily_drawdown_rejects():

    engine = RiskEngine(
        limits=RiskLimits(
            max_daily_drawdown=0.05,
        )
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(),
            portfolio=make_portfolio(
                daily_drawdown=0.05
            ),
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED


# =====================================================================
# 5
# =====================================================================

def test_portfolio_risk_limit_rejects():

    engine = RiskEngine(
        limits=RiskLimits(
            max_portfolio_risk=0.30,
        )
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(
                risk=0.40
            ),
            portfolio=make_portfolio(),
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED


# =====================================================================
# 6
# =====================================================================

def test_correlation_exposure_is_reduced():

    engine = RiskEngine(
        limits=RiskLimits(
            high_correlation_threshold=0.85,
            max_correlated_exposure=0.50,
        )
    )

    decision = PortfolioDecision(
        approved=True,
        timestamp=TIMESTAMP,
        mode=DecisionMode.BACKTEST,
        decision_id="corr-001",
        capital_allocation={
            "XAUUSD": 0.30,
            "EURUSD": 0.30,
        },
        margin_allocation={
            "XAUUSD": 2_000.0,
            "EURUSD": 2_000.0,
        },
        target_exposure={
            "XAUUSD": 0.30,
            "EURUSD": 0.30,
        },
        target_signals={
            "XAUUSD": 1,
            "EURUSD": 1,
        },
        portfolio_risk=0.20,
        reason_codes=("test",),
        model=MODEL,
    )

    portfolio = make_portfolio(
        correlation={
            "XAUUSD": {
                "EURUSD": 0.95,
            },
            "EURUSD": {
                "XAUUSD": 0.95,
            },
        }
    )

    result = engine.evaluate(
        RiskRequest(
            decision=decision,
            portfolio=portfolio,
        )
    )

    assert result.status == (
        RiskDecisionStatus.MODIFIED
    )

    combined = sum(
        abs(value)
        for value
        in result.target_exposure.values()
    )

    assert combined == pytest.approx(
        0.50
    )


# =====================================================================
# 7
# =====================================================================

def test_margin_utilization_modifies_decision():

    engine = RiskEngine(
        limits=RiskLimits(
            max_margin_utilization=0.50,
        )
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(
                allocation=0.80,
                exposure=0.80,
                margin=8_000.0,
            ),
            portfolio=make_portfolio(),
        )
    )

    assert result.status == RiskDecisionStatus.MODIFIED
    assert sum(
        result.margin_allocation.values()
    ) == pytest.approx(
        5_000.0
    )


# =====================================================================
# 8
# =====================================================================

def test_risk_block_rejects():

    engine = RiskEngine(
        limits=RiskLimits()
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(),
            portfolio=make_portfolio(
                risk_blocked=True
            ),
        )
    )

    assert result.status == RiskDecisionStatus.REJECTED


# =====================================================================
# 9
# =====================================================================

def test_rejected_decision_creates_empty_action():

    engine = RiskEngine(
        limits=RiskLimits()
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(),
            portfolio=make_portfolio(
                risk_blocked=True
            ),
        )
    )

    class DummySizer:

        def size(
            self,
            *,
            symbol: str,
            target_exposure: float,
        ) -> float:

            return 1.0

    builder = RiskActionBuilder(
        position_sizer=DummySizer()
    )

    action = builder.build(decision=result)
    assert isinstance(action, PortfolioAction)
    assert action.intents == ()


# =====================================================================
# 10
# =====================================================================

def test_approved_decision_creates_action():

    engine = RiskEngine(
        limits=RiskLimits()
    )

    result = engine.evaluate(
        RiskRequest(
            decision=make_decision(
                exposure=0.20
            ),
            portfolio=make_portfolio(),
        )
    )

    class DummySizer:

        def size(
            self,
            *,
            symbol: str,
            target_exposure: float,
        ) -> float:

            return {
                "XAUUSD": 1.5,
            }[symbol]

    builder = RiskActionBuilder(
        position_sizer=DummySizer()
    )
    action = builder.build(
        decision=result,
        allowed_symbols={
            "XAUUSD",
        },
    )
    assert len(action.intents) == 1
    intent = action.intents[0]

    assert intent.symbol == "XAUUSD"
    assert intent.target_side == 1
    assert intent.target_lots == pytest.approx(1.5)


# =====================================================================
# 11
# =====================================================================

def test_engine_state_is_recorded():

    engine = RiskEngine(
        limits=RiskLimits()
    )
    engine.evaluate(
        RiskRequest(
            decision=make_decision(),
            portfolio=make_portfolio(),
        )
    )
    assert engine.state.evaluation_count == 1
    assert engine.state.approved_count == 1
    assert engine.state.last_decision_id == ("decision-001")


# =====================================================================
# 12
# =====================================================================

def test_deterministic_risk_evaluation():

    request = RiskRequest(
        decision=make_decision(
            allocation=0.40,
            exposure=0.40,
            margin=6_000.0,
        ),
        portfolio=make_portfolio(),
    )

    first = RiskEngine(
        limits=RiskLimits()
    ).evaluate(request)

    second = RiskEngine(
        limits=RiskLimits()
    ).evaluate(request)

    assert first == second

