# f05_agents/decision_engine_old1.py (6)
#
# Created:
#     1405/06/19
#
# Chapter 3 - Symbol-Agent + Meta-Agent
#
# این فایل orchestration لایه تصمیم‌گیری چندنمادی را انجام می‌دهد.
#
# مسئولیت:
#     - نگهداری Symbol-Agent های مستقل
#     - ارسال Observation صحیح به Agent صحیح
#     - جمع‌آوری خروجی Symbol-Agent ها
#     - ارسال خروجی‌ها به Meta-Agent
#     - تولید PortfolioDecision
#
# عدم مسئولیت:
#     - اجرای معامله
#     - Broker / MT5
#     - Portfolio accounting
#     - Reward calculation
#
# Architecture:
#
#     Symbol Observations
#            |
#            v
#     +------+------+------+
#     |             |      |
#   XAU Agent    EUR Agent  ...
#     |             |      |
#     +------+------+------+
#            |
#            v
#        Meta-Agent
#            |
#            v
#    PortfolioDecision
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from f05_agents.contracts import (
    AgentObservation,
    PortfolioContext,
    PortfolioDecision,
    SymbolAgentOutput,
)

from f05_agents.meta_agent import MetaAgent

from f05_agents.symbol_agent import SymbolAgent


# =====================================================================
# Decision Engine
# =====================================================================

@dataclass(slots=True)
class MultiSymbolDecisionEngine:
    """
    Orchestrator لایه تصمیم‌گیری چندنمادی.

    هر Symbol-Agent دقیقاً یک symbol را کنترل می‌کند.

    ترتیب Symbol ها deterministic است تا:
        - replay
        - backtest
        - evaluation
        - debugging

    reproducible باقی بماند.
    """

    symbol_agents: Mapping[str, SymbolAgent]
    meta_agent: MetaAgent

    def __post_init__(self) -> None:

        normalized: dict[str, SymbolAgent] = {}

        for symbol, agent in self.symbol_agents.items():

            key = str(symbol).upper().strip()

            if not key:
                raise ValueError(
                    "symbol key must not be empty"
                )

            if not isinstance(agent, SymbolAgent):
                raise TypeError(
                    f"Invalid SymbolAgent for {key}"
                )

            if agent.symbol != key:
                raise ValueError(
                    f"Agent symbol mismatch: "
                    f"{agent.symbol} != {key}"
                )

            if key in normalized:
                raise ValueError(
                    f"Duplicate symbol: {key}"
                )

            normalized[key] = agent

        if not normalized:
            raise ValueError(
                "symbol_agents must not be empty"
            )

        self.symbol_agents = dict(
            sorted(
                normalized.items()
            )
        )


    # =================================================================
    # Decision
    # =================================================================

    def decide(
        self,
        *,
        observations: Mapping[str, AgentObservation],
        contexts: Mapping[str, object],
        portfolio: PortfolioContext,
        decision_id: str,
    ) -> PortfolioDecision:
        """
        اجرای یک چرخه کامل تصمیم‌گیری چندنمادی.

        observations:
            Observation هر symbol.

        contexts:
            SymbolContext متناظر با هر symbol.

        portfolio:
            PortfolioContext برای Meta-Agent.

        decision_id:
            شناسه یکتای چرخه تصمیم‌گیری.
        """

        if not str(decision_id).strip():
            raise ValueError(
                "decision_id is required"
            )

        if portfolio.mode != self.meta_agent.config.mode:
            raise ValueError(
                "Portfolio mode does not match Meta-Agent mode"
            )

        expected_symbols = set(
            self.symbol_agents
        )

        observation_symbols = {
            str(symbol).upper()
            for symbol in observations
        }

        context_symbols = {
            str(symbol).upper()
            for symbol in contexts
        }

        if observation_symbols != expected_symbols:
            raise ValueError(
                "Observation symbols do not match "
                "registered Symbol-Agents"
            )

        if context_symbols != expected_symbols:
            raise ValueError(
                "Context symbols do not match "
                "registered Symbol-Agents"
            )

        outputs: dict[str, SymbolAgentOutput] = {}


        # -------------------------------------------------------------
        # Symbol-Agent inference
        # -------------------------------------------------------------

        for symbol in sorted(expected_symbols):

            observation = observations[symbol]

            context = contexts[symbol]

            if observation.symbol != symbol:
                raise ValueError(
                    f"Observation symbol mismatch: "
                    f"{observation.symbol} != {symbol}"
                )

            if getattr(context, "symbol", None) != symbol:
                raise ValueError(
                    f"Context symbol mismatch: "
                    f"{getattr(context, 'symbol', None)} != {symbol}"
                )

            output = self.symbol_agents[symbol].decide(
                observation=observation,
                context=context,
                decision_id=decision_id,
            )

            outputs[symbol] = output


        # -------------------------------------------------------------
        # Meta-Agent inference
        # -------------------------------------------------------------

        return self.meta_agent.decide(
            signals=outputs,
            portfolio=portfolio,
            decision_id=decision_id,
        )


    # =================================================================
    # Symbol-Agent access
    # =================================================================

    def get_symbol_agent(
        self,
        symbol: str,
    ) -> SymbolAgent:

        key = str(symbol).upper().strip()

        if key not in self.symbol_agents:
            raise KeyError(
                f"Unknown Symbol-Agent: {key}"
            )

        return self.symbol_agents[key]


    # =================================================================
    # Runtime reset
    # =================================================================

    def reset(self) -> None:
        """
        Reset runtime state همه Agent ها.
        """

        for agent in self.symbol_agents.values():
            agent.reset()

        self.meta_agent.reset()