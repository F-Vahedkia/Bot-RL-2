# f05_agents/meta_agent.py (4)
#
# Created:
#     1405/06/19
#
# Production-grade Meta-Agent foundation.
#
# مسئولیت:
#     - دریافت خروجی Symbol-Agentها
#     - مشاهده وضعیت کل Portfolio
#     - اعمال Portfolio guardrails
#     - کنترل:
#           capital allocation
#           margin allocation
#           exposure
#           concentration
#           correlation
#           portfolio risk
#           drawdown
#
# عدم مسئولیت:
#     - broker
#     - order execution
#     - MT5
#     - market data
#     - feature calculation
#
# Meta-Agent خروجی Decision تولید می‌کند.
# اتصال این Decision به f04_env در یک Adapter مستقل
# انجام خواهد شد.


from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Protocol

from f05_agents.agent_state import (
    MetaAgentState,
)

from f05_agents.contracts import (
    DecisionMode,
    MetaPolicyOutput,
    ModelIdentity,
    PortfolioContext,
    PortfolioDecision,
    SymbolAgentOutput,
)


# =====================================================================
# Meta Policy Interface
# =====================================================================

class MetaPolicy(Protocol):
    """
    قرارداد Policy سطح Portfolio.

    این Policy می‌تواند در آینده:
        - RL
        - Transformer
        - Attention
        - Ensemble
    باشد.
    """

    def predict(
        self,
        *,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
    ) -> MetaPolicyOutput:
        ...


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True, slots=True)
class MetaAgentConfig:
    """
    محدودیت‌های Portfolio-level.

    اینها hard guardrail هستند، نه Policy.
    """

    mode: DecisionMode

    model: ModelIdentity

    # مجموع سرمایه تخصیص داده‌شده
    max_total_allocation: float = 1.0

    # حداکثر تخصیص به یک symbol
    max_symbol_allocation: float = 0.25

    # حداکثر exposure کل
    max_total_exposure: float = 1.0

    # حداکثر risk کل Portfolio
    max_portfolio_risk: float = 1.0

    # اگر correlation مطلق دو نماد از این حد بیشتر شود،
    # آنها به عنوان exposure همبسته شدید در نظر گرفته می‌شوند.
    high_correlation_threshold: float = 0.85

    # سقف مجموع allocation دو نماد heavily correlated
    max_correlated_allocation: float = 0.50

    # در صورت drawdown بالاتر از این مقدار،
    # تصمیم جدید block می‌شود.
    max_drawdown: float = 1.0


    def __post_init__(self) -> None:

        if not (
            0.0 < self.max_total_allocation <= 1.0
        ):
            raise ValueError(
                "max_total_allocation must be in (0,1]"
            )

        if not (
            0.0 < self.max_symbol_allocation <= 1.0
        ):
            raise ValueError(
                "max_symbol_allocation must be in (0,1]"
            )

        if not (
            0.0 < self.max_total_exposure <= 1.0
        ):
            raise ValueError(
                "max_total_exposure must be in (0,1]"
            )

        if not (
            0.0 <= self.max_portfolio_risk <= 1.0
        ):
            raise ValueError(
                "max_portfolio_risk must be in [0,1]"
            )

        if not (
            0.0 <= self.high_correlation_threshold <= 1.0
        ):
            raise ValueError(
                "high_correlation_threshold must be in [0,1]"
            )

        if not (
            0.0 < self.max_correlated_allocation <= 1.0
        ):
            raise ValueError(
                "max_correlated_allocation must be in (0,1]"
            )

        if not (
            0.0 <= self.max_drawdown <= 1.0
        ):
            raise ValueError(
                "max_drawdown must be in [0,1]"
            )


# =====================================================================
# Meta-Agent
# =====================================================================

class MetaAgent:
    """
    Meta-Agent سطح Portfolio.

    Symbol-Agentها فقط پیشنهاد می‌دهند.
    تصمیم نهایی Portfolio در این کلاس شکل می‌گیرد.

    """

    def __init__(
        self,
        *,
        config: MetaAgentConfig,
        policy: MetaPolicy,
    ) -> None:

        self.config = config

        self.policy = policy

        self.state = MetaAgentState()


    # =================================================================
    # Decision
    # =================================================================

    def decide(
        self,
        *,
        signals: Mapping[str, SymbolAgentOutput],
        portfolio: PortfolioContext,
        decision_id: str,
    ) -> PortfolioDecision:

        try:

            if not str(decision_id).strip():
                raise ValueError(
                    "decision_id is required"
                )


            self._validate_signal_set(
                signals
            )


            # ---------------------------------------------------------
            # Hard risk block
            # ---------------------------------------------------------

            if portfolio.risk_blocked:

                return self._rejected_decision(
                    portfolio=portfolio,
                    decision_id=decision_id,
                    reason="portfolio_risk_blocked",
                )


            if portfolio.drawdown >= self.config.max_drawdown:

                return self._rejected_decision(
                    portfolio=portfolio,
                    decision_id=decision_id,
                    reason="max_drawdown_reached",
                )


            # ---------------------------------------------------------
            # Policy inference
            # ---------------------------------------------------------

            proposal = self.policy.predict(
                signals=signals,
                portfolio=portfolio,
            )


            allocations = {
                symbol: float(value)
                for symbol, value
                in proposal.capital_allocation.items()
            }


            margin_allocation = {
                symbol: float(value)
                for symbol, value
                in proposal.margin_allocation.items()
            }


            target_signals = {
                symbol: int(signal)
                for symbol, signal
                in proposal.target_signals.items()
            }


            target_exposure = {
                symbol: float(value)
                for symbol, value
                in proposal.target_exposure.items()
            }


            # ---------------------------------------------------------
            # Guardrail 1:
            # Symbol concentration
            # ---------------------------------------------------------

            allocations = (
                self._apply_symbol_cap(
                    allocations
                )
            )


            # ---------------------------------------------------------
            # Guardrail 2:
            # Correlation concentration
            # ---------------------------------------------------------

            allocations = (
                self._apply_correlation_cap(
                    allocations=allocations,
                    correlation=portfolio.correlation,
                )
            )


            # ---------------------------------------------------------
            # Guardrail 3:
            # Total allocation
            # ---------------------------------------------------------

            allocations = (
                self._scale_to_limit(
                    values=allocations,
                    limit=self.config.max_total_allocation,
                )
            )


            # ---------------------------------------------------------
            # Guardrail 4:
            # Total exposure
            # ---------------------------------------------------------

            total_exposure = sum(
                abs(value)
                for value in target_exposure.values()
            )


            if total_exposure > self.config.max_total_exposure:

                factor = (
                    self.config.max_total_exposure
                    /
                    total_exposure
                )

                target_exposure = {
                    symbol: value * factor
                    for symbol, value
                    in target_exposure.items()
                }


            # ---------------------------------------------------------
            # Guardrail 5:
            # Portfolio risk
            # ---------------------------------------------------------

            portfolio_risk = min(
                float(proposal.portfolio_risk),
                self.config.max_portfolio_risk,
            )


            # ---------------------------------------------------------
            # Margin allocation must follow final capital allocation.
            #
            # مقدار خروجی Policy می‌تواند توسط guardrailها کاهش یابد.
            # بنابراین margin allocation نیز proportional اصلاح می‌شود.
            # ---------------------------------------------------------

            margin_allocation = (
                self._resize_mapping(
                    values=margin_allocation,
                    reference=allocations,
                )
            )


            total_final_exposure = sum(
                abs(value)
                for value in target_exposure.values()
            )


            self.state.record_decision(
                approved=True,
                equity=portfolio.equity,
                drawdown=portfolio.drawdown,
                total_exposure=total_final_exposure,
                capital_allocation=dict(allocations),
                margin_allocation=dict(margin_allocation),
                timestamp=portfolio.timestamp,
                model=self.config.model,
                mode=self.config.mode,
            )


            return PortfolioDecision(
                approved=True,
                timestamp=portfolio.timestamp,
                mode=self.config.mode,
                decision_id=decision_id,
                capital_allocation=allocations,
                margin_allocation=margin_allocation,
                target_exposure=target_exposure,
                target_signals=target_signals,
                portfolio_risk=portfolio_risk,
                reason_codes=tuple(
                    proposal.reason_codes
                ) + ("meta_agent_approved",),
                model=self.config.model,
            )


        except Exception as exc:

            self.state.record_error(exc)

            raise


    # =================================================================
    # Validation
    # =================================================================

    def _validate_signal_set(
        self,
        signals: Mapping[str, SymbolAgentOutput],
    ) -> None:

        for symbol, signal in signals.items():

            normalized = str(symbol).upper()

            if normalized != signal.symbol:
                raise ValueError(
                    "Signal symbol key mismatch: "
                    f"{symbol} != {signal.symbol}"
                )


    # =================================================================
    # Symbol concentration cap
    # =================================================================

    def _apply_symbol_cap(
        self,
        values: Mapping[str, float],
    ) -> dict[str, float]:

        return {
            symbol: min(
                max(0.0, float(value)),
                self.config.max_symbol_allocation,
            )
            for symbol, value in values.items()
        }


    # =================================================================
    # Total limit scaling
    # =================================================================

    @staticmethod
    def _scale_to_limit(
        *,
        values: Mapping[str, float],
        limit: float,
    ) -> dict[str, float]:

        total = sum(
            max(0.0, float(value))
            for value in values.values()
        )

        if total <= limit:
            return dict(values)

        factor = limit / total

        return {
            symbol: max(0.0, float(value)) * factor
            for symbol, value in values.items()
        }


    # =================================================================
    # Correlation cap
    # =================================================================

    def _apply_correlation_cap(
        self,
        *,
        allocations: Mapping[str, float],
        correlation: Mapping[
            str,
            Mapping[str, float],
        ],
    ) -> dict[str, float]:

        result = {
            symbol: max(0.0, float(value))
            for symbol, value in allocations.items()
        }


        for symbol_a, symbol_b in combinations(
            result.keys(),
            2,
        ):

            corr_ab = float(
                correlation
                .get(symbol_a, {})
                .get(symbol_b, 0.0)
            )

            corr_ba = float(
                correlation
                .get(symbol_b, {})
                .get(symbol_a, corr_ab)
            )


            correlation_value = max(
                abs(corr_ab),
                abs(corr_ba),
            )


            if (
                correlation_value
                <
                self.config.high_correlation_threshold
            ):
                continue


            combined = (
                result[symbol_a]
                +
                result[symbol_b]
            )


            if (
                combined
                <= self.config.max_correlated_allocation
            ):
                continue


            factor = (
                self.config.max_correlated_allocation
                /
                combined
            )


            result[symbol_a] *= factor

            result[symbol_b] *= factor


        return result


    # =================================================================
    # Margin Allocation
    # =================================================================

    @staticmethod
    def _resize_mapping(
        *,
        values: Mapping[str, float],
        reference: Mapping[str, float],
    ) -> dict[str, float]:

        reference_total = sum(
            max(0.0, value)
            for value in reference.values()
        )


        source_total = sum(
            max(0.0, float(value))
            for value in values.values()
        )


        if reference_total <= 0.0 or source_total <= 0.0:

            return {
                symbol: 0.0
                for symbol in reference
            }


        result: dict[str, float] = {}


        for symbol, ref_value in reference.items():

            original = max(
                0.0,
                float(values.get(symbol, 0.0)),
            )


            result[symbol] = (
                original
                *
                reference_total
                /
                source_total
            )


        return result


    # =================================================================
    # Rejection
    # =================================================================

    def _rejected_decision(
        self,
        *,
        portfolio: PortfolioContext,
        decision_id: str,
        reason: str,
    ) -> PortfolioDecision:

        self.state.record_decision(
            approved=False,
            equity=portfolio.equity,
            drawdown=portfolio.drawdown,
            total_exposure=0.0,
            capital_allocation={},
            margin_allocation={},
            timestamp=portfolio.timestamp,
            model=self.config.model,
            mode=self.config.mode,
            rejection_reason=reason,
        )


        return PortfolioDecision(
            approved=False,
            timestamp=portfolio.timestamp,
            mode=self.config.mode,
            decision_id=decision_id,
            capital_allocation={},
            margin_allocation={},
            target_exposure={},
            target_signals={},
            portfolio_risk=0.0,
            reason_codes=(
                reason,
            ),
            model=self.config.model,
        )


    # =================================================================
    # Lifecycle
    # =================================================================

    def update_reward(
        self,
        reward: float,
    ) -> None:

        self.state.add_reward(
            reward
        )


    def reset(self) -> None:

        self.state.reset()