# f05_agents/symbol_agent.py (3)
#
# Created:
#     1405/06/19
#
# Production-grade Symbol-Agent foundation.
#
# مسئولیت:
#     - دریافت Observation یک symbol
#     - اجرای Policy مربوط به همان symbol
#     - تولید SymbolAgentOutput
#     - نگهداری runtime state
#
# عدم مسئولیت:
#     - Portfolio allocation
#     - Global risk
#     - Correlation
#     - Margin allocation
#     - Execution
#     - Broker
#
# Policy به‌صورت dependency injection وارد می‌شود.
# بنابراین در آینده می‌توان:
#     RL / Transformer / Ensemble / دیگر مدل‌ها
# را بدون تغییر معماری Symbol-Agent جایگزین کرد.


from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from f05_agents.agent_state import (
    SymbolAgentState,
)

from f05_agents.contracts import (
    AgentObservation,
    ModelIdentity,
    PolicyOutput,
    SymbolAgentOutput,
    SymbolContext,
    DecisionMode,
)


# =====================================================================
# Policy Interface
# =====================================================================

class SymbolPolicy(Protocol):
    """
    قرارداد Policy مربوط به یک Symbol-Agent.

    Policy فقط inference انجام می‌دهد.
    """

    def predict(
        self,
        *,
        observation: AgentObservation,
        context: SymbolContext,
    ) -> PolicyOutput:
        ...


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True, slots=True)
class SymbolAgentConfig:
    """
    Configuration یک Symbol-Agent.
    """

    symbol: str

    mode: DecisionMode

    model: ModelIdentity

    min_confidence: float = 0.0

    enabled: bool = True


    def __post_init__(self) -> None:

        symbol = str(
            self.symbol
        ).upper().strip()

        if not symbol:
            raise ValueError(
                "symbol is required"
            )

        if not (
            0.0 <= self.min_confidence <= 1.0
        ):
            raise ValueError(
                "min_confidence must be between 0 and 1"
            )


# =====================================================================
# Symbol-Agent
# =====================================================================

class SymbolAgent:
    """
    Symbol-Agent مستقل برای یک symbol.

    هیچ اطلاعات Portfolio-wide دریافت نمی‌کند.
    """

    def __init__(
        self,
        *,
        config: SymbolAgentConfig,
        policy: SymbolPolicy,
    ) -> None:

        self.config = config

        self.symbol = config.symbol

        self.policy = policy

        self.state = SymbolAgentState(
            symbol=self.symbol
        )


    # =================================================================
    # Decision
    # =================================================================

    def decide(
        self,
        *,
        observation: AgentObservation,
        context: SymbolContext,
        decision_id: str,
    ) -> SymbolAgentOutput:
        """
        تولید تصمیم Symbol-Agent.
        """

        try:

            if observation.symbol != self.symbol:
                raise ValueError(
                    "Observation symbol mismatch: "
                    f"{observation.symbol} != {self.symbol}"
                )

            if context.symbol != self.symbol:
                raise ValueError(
                    "Context symbol mismatch: "
                    f"{context.symbol} != {self.symbol}"
                )

            if not str(decision_id).strip():
                raise ValueError(
                    "decision_id is required"
                )

            if not self.config.enabled:

                output = SymbolAgentOutput(
                    symbol=self.symbol,
                    timestamp=observation.timestamp,
                    mode=self.config.mode,
                    signal=0,
                    confidence=0.0,
                    expected_return=0.0,
                    risk_score=1.0,
                    desired_exposure=0.0,
                    model=self.config.model,
                    decision_id=decision_id,
                    metadata={
                        "disabled": 1.0,
                    },
                )

                self.state.record_decision(
                    signal=output.signal,
                    confidence=output.confidence,
                    expected_return=output.expected_return,
                    risk_score=output.risk_score,
                    desired_exposure=output.desired_exposure,
                    timestamp=output.timestamp,
                    model=output.model,
                    mode=output.mode,
                )

                return output


            policy_output = self.policy.predict(
                observation=observation,
                context=context,
            )


            signal = int(
                policy_output.signal
            )

            confidence = float(
                policy_output.confidence
            )


            # confidence filter:
            # Policy می‌تواند signal تولید کند،
            # ولی اگر confidence کافی نباشد،
            # تصمیم به neutral تبدیل می‌شود.

            if confidence < self.config.min_confidence:
                signal = 0


            output = SymbolAgentOutput(
                symbol=self.symbol,
                timestamp=observation.timestamp,
                mode=self.config.mode,
                signal=signal,
                confidence=confidence,
                expected_return=float(
                    policy_output.expected_return
                ),
                risk_score=float(
                    policy_output.risk_score
                ),
                desired_exposure=float(
                    policy_output.desired_exposure
                ),
                model=self.config.model,
                decision_id=decision_id,
                metadata=dict(
                    policy_output.metadata
                ),
            )


            self.state.record_decision(
                signal=output.signal,
                confidence=output.confidence,
                expected_return=output.expected_return,
                risk_score=output.risk_score,
                desired_exposure=output.desired_exposure,
                timestamp=output.timestamp,
                model=output.model,
                mode=output.mode,
            )


            return output


        except Exception as exc:

            self.state.record_error(exc)

            raise


    # =================================================================
    # Reward feedback
    # =================================================================

    def update_reward(
        self,
        reward: float,
    ) -> None:
        """
        دریافت reward پس از transition.

        محاسبه reward در Environment انجام می‌شود.
        """

        self.state.add_reward(reward)


    # =================================================================
    # Lifecycle
    # =================================================================

    def reset(self) -> None:
        """
        Reset فقط runtime state را انجام می‌دهد.
        Configuration و Policy دست‌نخورده می‌مانند.
        """

        self.state.reset()