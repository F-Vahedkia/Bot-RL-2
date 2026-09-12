# f06_risk/risk_engine.py (4)
# =============================================================================
# BOT-RL-2 v8
# Chapter 4 - Risk Layer
#
# Production-oriented deterministic pre-trade risk engine.
#
# Flow:
#
#   RiskRequest
#       |
#       +--> RiskContext
#       |
#       +--> RiskProjection
#       |
#       +--> ProjectedRiskState
#       |
#       +--> hard constraints / modifications
#       |
#       +--> RiskDecision
#
# Backward compatibility:
#   When RiskRequest.risk_context is None, the legacy decision-only risk path
#   remains available so existing Chapter-4 tests continue to work.
# =============================================================================
"""
نکاتی که در مرحله دوم chatGPT اعلام کرده:
------------------------------------------
    - RiskEngine هستهٔ deterministic pre-trade risk است و
        مسئولیت‌هایش شامل exposure، concentration، correlation،
        margin utilization، portfolio-risk و projected-state evaluation 
        ست؛ execution و broker و position accounting خارج از مسئولیت آن هستند.
    - مسیر production اکنون از RiskContext و RiskProjection عبور می‌کند و
        در صورت وجود margin_requests می‌تواند از ProjectionMarginRequest برای
        مارجین محاسبه‌شده به‌شکل instrument-aware استفاده کند.
    - پس از هر modification مهم، projection دوباره محاسبه می‌شود؛
        بنابراین مسیر فعلی صرفاً «حد را cap کن و تمام» نیست.
    - محدودیت‌های فعلی در Engine شامل symbol exposure، total exposure،
        correlation، capital concentration و margin utilization هستند.
    - max_open_positions و max_positions_per_symbol اکنون در مسیر projected وارد شده‌اند.
        برای per-symbol، از projected.symbols[symbol].projected_position_count استفاده می‌شود.
    - در پایان، RiskDecision همراه با خلاصهٔ projected state تولید می‌شود و
        RiskEngineState نیز هر evaluation را ثبت می‌کند.
    - یک نکتهٔ مهم برای مرحلهٔ بعدی این است که risk_engine.py همین حالا
        ترتیب اعمال constraintها را به‌صورت اجرایی دارد؛
        بنابراین مورد «Hard vs Soft constraints + Conflict Resolution» بیشتر از جنس
        formalize کردن همین policy و تست سناریوهای تعارض است،
        نه اینکه ترتیب از صفر ساخته شود. این ترتیب در کد فعلی قابل مشاهده است.
"""
from __future__ import annotations

from dataclasses import replace
from math import isfinite
from typing import Dict, Mapping, Tuple

from f05_agents.contracts import PortfolioDecision

from f06_risk.contracts import (
    RiskDecision,
    RiskDecisionStatus,
    RiskRequest,
    RiskViolation,
)
from f06_risk.limits import RiskLimits
from f06_risk.projection_margin import ProjectionMarginRequest
from f06_risk.risk_context import RiskContext
from f06_risk.risk_projection import (
    ProjectedRiskState,
    RiskProjection,
)
from f06_risk.state import RiskEngineState


# =============================================================================
# Validation
# =============================================================================

def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value

# =============================================================================
# Risk Engine
# =============================================================================

class RiskEngine:
    """
    Deterministic portfolio pre-trade risk engine.

    Responsibilities:
        - hard risk blocks
        - exposure constraints
        - concentration limits
        - correlation limits
        - margin utilization
        - portfolio-risk budget
        - projected-state evaluation

    Non-responsibilities:
        - broker communication
        - order placement
        - fill simulation
        - position accounting
        - price discovery
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        projector: RiskProjection | None = None,
    ) -> None:

        self.limits = limits or RiskLimits()
        self.projector = projector or RiskProjection()
        self.state = RiskEngineState()

    # =========================================================================
    # Public API
    # =========================================================================

    def evaluate(
        self,
        request: RiskRequest,
        *,
        margin_requests: Mapping[str, ProjectionMarginRequest] | None = None,
    ) -> RiskDecision:

        if request is None:
            raise ValueError("request is required.")

        if not isinstance(request, RiskRequest):
            raise TypeError("request must be RiskRequest.")

        if request.risk_context is None:
            if margin_requests is not None:
                raise ValueError("margin_requests requires risk_context.")

            return self._evaluate_legacy(request)

        return self._evaluate_projected(
            request,
            margin_requests=margin_requests,
        )
    
    # =========================================================================
    # Projected-state production path
    # =========================================================================

    def _evaluate_projected(
        self,
        request: RiskRequest,
        *,
        margin_requests: Mapping[str, ProjectionMarginRequest] | None = None,
    ) -> RiskDecision:
        
        decision = request.decision
        context = request.risk_context

        if context is None:
            raise RuntimeError("Projected evaluation requires risk_context.")

        # ---------------------------------------------------------------------
        # Hard blocks that depend on current portfolio state.
        # ---------------------------------------------------------------------

        violations = []

        if request.portfolio.risk_blocked:
            return self._reject(
                request,
                RiskViolation(
                    code="RISK_BLOCKED",
                    message="Portfolio is currently risk blocked.",
                    severity="critical",
                ),
            )

        if (
            request.portfolio.drawdown
            >= self.limits.max_drawdown
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_DRAWDOWN",
                    message="Portfolio drawdown is at or above the configured maximum.",
                    severity="critical",
                ),
            )

        if (
            request.portfolio.daily_drawdown
            >= self.limits.max_daily_drawdown
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_DAILY_DRAWDOWN",
                    message="Portfolio daily drawdown is at or above the configured maximum.",
                    severity="critical",
                ),
            )

        if (
            decision.portfolio_risk
            > self.limits.max_portfolio_risk
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_PORTFOLIO_RISK",
                    message="Requested portfolio risk exceeds the configured portfolio-risk budget.",
                    severity="critical",
                ),
            )

        # ---------------------------------------------------------------------
        # Start from an immutable decision candidate.
        # ---------------------------------------------------------------------

        candidate = decision
        modified = False

        # ---------------------------------------------------------------------
        # Project initial candidate.
        # ---------------------------------------------------------------------

        projected = self._project_candidate(
            context=context,
            decision=candidate,
            margin_requests=margin_requests,
        )

        # ---------------------------------------------------------------------
        # Symbol exposure limits.
        # ---------------------------------------------------------------------

        target_exposure = dict(
            candidate.target_exposure
        )

        for symbol in sorted(target_exposure):

            exposure = float(
                target_exposure[symbol]
            )

            if abs(exposure) > self.limits.max_symbol_exposure:

                target_exposure[symbol] = (
                    self._cap_signed(
                        exposure,
                        self.limits.max_symbol_exposure,
                    )
                )

                violations.append(
                    RiskViolation(
                        code="MAX_SYMBOL_EXPOSURE",
                        message=(
                            f"Projected exposure for {symbol} "
                            "exceeds the symbol exposure limit."
                        ),
                        symbol=symbol,
                        severity="warning",
                    )
                )

                modified = True

        # ---------------------------------------------------------------------
        # Rebuild candidate with symbol exposure modifications.
        # ---------------------------------------------------------------------

        if modified:
            candidate = replace(
                candidate,
                target_exposure=target_exposure,
            )

            projected = self._project_candidate(
                context=context,
                decision=candidate,
                margin_requests=margin_requests,
            )

        # ---------------------------------------------------------------------
        # Total projected exposure.
        # ---------------------------------------------------------------------

        if (
            projected.projected_total_exposure
            > self.limits.max_total_exposure
        ):

            factor = (
                self.limits.max_total_exposure
                / projected.projected_total_exposure
            )

            target_exposure = {
                symbol: float(exposure) * factor
                for symbol, exposure
                in candidate.target_exposure.items()
            }

            candidate = replace(
                candidate,
                target_exposure=target_exposure,
            )

            projected = self._project_candidate(
                context=context,
                decision=candidate,
                margin_requests=margin_requests,
            )

            violations.append(
                RiskViolation(
                    code="MAX_TOTAL_EXPOSURE",
                    message=(
                        "Projected total exposure exceeds the "
                        "portfolio exposure limit and was scaled."
                    ),
                    severity="warning",
                )
            )

            modified = True

        # ---------------------------------------------------------------------
        # Correlation limits.
        # ---------------------------------------------------------------------

        target_exposure = dict(
            candidate.target_exposure
        )

        correlation_modified = False

        for symbol_a in sorted(context.correlation):

            row = context.correlation[symbol_a]
            for symbol_b in sorted(row):
                if symbol_a >= symbol_b:
                    continue

                correlation = float(
                    row[symbol_b]
                )

                if (
                    abs(correlation)
                    < self.limits.high_correlation_threshold
                ):
                    continue

                exposure_a = abs(
                    target_exposure.get(symbol_a, 0.0)
                )

                exposure_b = abs(
                    target_exposure.get(symbol_b, 0.0)
                )

                combined = (
                    exposure_a + exposure_b
                )

                if (
                    combined
                    <= self.limits.max_correlated_exposure
                ):
                    continue

                factor = (
                    self.limits.max_correlated_exposure
                    / combined
                )

                if symbol_a in target_exposure:
                    target_exposure[symbol_a] = (
                        float(
                            target_exposure[symbol_a]
                        )
                        * factor
                    )

                if symbol_b in target_exposure:
                    target_exposure[symbol_b] = (
                        float(
                            target_exposure[symbol_b]
                        )
                        * factor
                    )

                correlation_modified = True

                violations.append(
                    RiskViolation(
                        code="MAX_CORRELATED_EXPOSURE",
                        message=(
                            f"Highly correlated pair {symbol_a}/{symbol_b} "
                            "exceeds the combined exposure limit."
                        ),
                        severity="warning",
                    )
                )

        if correlation_modified:

            candidate = replace(
                candidate,
                target_exposure=target_exposure,
            )

            projected = self._project_candidate(
                context=context,
                decision=candidate,
                margin_requests=margin_requests,
            )

            modified = True

        # ---------------------------------------------------------------------
        # Capital concentration.
        # ---------------------------------------------------------------------

        capital_allocation = dict(
            candidate.capital_allocation
        )

        concentration_modified = False

        for symbol in sorted(capital_allocation):

            allocation = _finite(
                f"capital_allocation[{symbol}]",
                capital_allocation[symbol],
            )

            if abs(allocation) <= self.limits.max_symbol_concentration:
                continue

            capital_allocation[symbol] = (
                self._cap_signed(
                    allocation,
                    self.limits.max_symbol_concentration,
                )
            )

            concentration_modified = True

            violations.append(
                RiskViolation(
                    code="MAX_SYMBOL_CONCENTRATION",
                    message=(
                        f"Capital allocation for {symbol} "
                        "exceeds the symbol concentration limit."
                    ),
                    symbol=symbol,
                    severity="warning",
                )
            )

        if concentration_modified:

            candidate = replace(
                candidate,
                capital_allocation=capital_allocation,
            )

            modified = True

        # ---------------------------------------------------------------------
        # Projected margin utilization.
        #
        # Projection is intentionally based on normalized RiskContext values.
        # Broker-specific margin calculation remains outside this engine.
        # ---------------------------------------------------------------------

        if (
            projected.projected_margin_utilization
            > self.limits.max_margin_utilization
        ):

            current_projected_margin = (
                projected.projected_used_margin
            )

            if current_projected_margin > 0.0:

                factor = (
                    self.limits.max_margin_utilization
                    * context.account.equity
                    / current_projected_margin
                )

                factor = min(
                    max(factor, 0.0),
                    1.0,
                )

                target_exposure = {
                    symbol: float(exposure) * factor
                    for symbol, exposure
                    in candidate.target_exposure.items()
                }

                capital_allocation = {
                    symbol: float(allocation) * factor
                    for symbol, allocation
                    in candidate.capital_allocation.items()
                }

                margin_allocation = {
                    symbol: float(allocation) * factor
                    for symbol, allocation
                    in candidate.margin_allocation.items()
                }

                candidate = replace(
                    candidate,
                    target_exposure=target_exposure,
                    capital_allocation=capital_allocation,
                    margin_allocation=margin_allocation,
                )

                projected = self._project_candidate(
                    context=context,
                    decision=candidate,
                    margin_requests=margin_requests,
                )

                violations.append(
                    RiskViolation(
                        code="MAX_MARGIN_UTILIZATION",
                        message=(
                            "Projected margin utilization exceeds "
                            "the configured maximum and was scaled."
                        ),
                        severity="warning",
                    )
                )

                modified = True

        # ---------------------------------------------------------------------
        # Final projected hard safety checks.
        # ---------------------------------------------------------------------

        if (
            self.limits.max_open_positions is not None
            and projected.projected_position_count
            > self.limits.max_open_positions
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_OPEN_POSITIONS",
                    message=(
                        "Projected open-position count exceeds "
                        "the configured portfolio position limit."
                    ),
                    severity="critical",
                ),
                extra_violations=tuple(violations),
            )

        # ======================= بلوک جدید- شروع
        if self.limits.max_positions_per_symbol is not None:
            for symbol in sorted(projected.symbols):
                symbol_projection = projected.symbols[symbol]

                if (
                    symbol_projection.projected_position_count
                    > self.limits.max_positions_per_symbol
                ):
                    return self._reject(
                        request,
                        RiskViolation(
                            code="MAX_POSITIONS_PER_SYMBOL",
                            message=(
                                f"Projected open-position count for "
                                f"{symbol} exceeds the configured "
                                "per-symbol position limit."
                            ),
                            symbol=symbol,
                            severity="critical",
                        ),
                        extra_violations=tuple(violations),
                    )
        # ======================= بلوک جدید- پایان

        if projected.projected_free_margin <= 0.0:
            return self._reject(
                request,
                RiskViolation(
                    code="NO_FREE_MARGIN",
                    message=(
                        "Projected free margin is zero or negative."
                    ),
                    severity="critical",
                ),
                extra_violations=tuple(violations),
            )
        
        # ---------------------------------------------------------------------
        # Build final RiskDecision.
        # ---------------------------------------------------------------------

        status = (
            RiskDecisionStatus.MODIFIED
            if modified
            else RiskDecisionStatus.APPROVED
        )

        metadata = {
            "evaluation_path": "projected",
            "projected": self._projected_summary(
                projected
            ),
            "risk_context_symbols": tuple(
                sorted(context.symbols)
            ),
        }

        result = RiskDecision(
            status=status,
            timestamp=candidate.timestamp,
            mode=candidate.mode,
            decision_id=candidate.decision_id,
            source_decision_id=candidate.decision_id,
            capital_allocation=dict(
                candidate.capital_allocation
            ),
            margin_allocation=dict(
                candidate.margin_allocation
            ),
            target_exposure=dict(
                candidate.target_exposure
            ),
            target_signals=dict(
                candidate.target_signals
            ),
            portfolio_risk=float(
                candidate.portfolio_risk
            ),
            violations=tuple(
                violations
            ),
            model=candidate.model,
            metadata=metadata,
        )

        self.state.record(
            status=status,
            decision_id=result.decision_id,
            violation_codes=tuple(v.code for v in violations),
            timestamp=result.timestamp,
        )

        return result

    # =========================================================================
    # Projected-state helper
    # =========================================================================

    def _project_candidate(
        self,
        *,
        context: RiskContext,
        decision: PortfolioDecision,
        margin_requests: Mapping[
            str,
            ProjectionMarginRequest,
        ] | None,
    ) -> ProjectedRiskState:
        """
        Project one candidate decision.

        When margin_requests is absent, preserve the original normalized
        RiskProjection behavior.

        When margin_requests is supplied, align each active target exposure
        with its instrument-aware ProjectionMarginRequest before projection.
        """

        if margin_requests is None:
            return self.projector.project(
                context=context,
                decision=decision,
            )

        normalized_requests = {}

        for symbol, request in margin_requests.items():
            normalized_symbol = (
                str(symbol).strip().upper()
            )

            if not normalized_symbol:
                raise ValueError(
                    "margin_requests contains an empty symbol."
                )

            if not isinstance(
                request,
                ProjectionMarginRequest,
            ):
                raise TypeError(
                    f"margin_requests[{normalized_symbol}] must be "
                    "ProjectionMarginRequest."
                )

            if request.symbol != normalized_symbol:
                raise ValueError(
                    f"margin_requests[{normalized_symbol}] does not "
                    f"match request.symbol={request.symbol}."
                )

            if abs(
                float(request.equity)
                - float(context.account.equity)
            ) > 1e-12:
                raise ValueError(
                    f"margin_requests[{normalized_symbol}] equity does not "
                    "match RiskContext account equity."
                )

            normalized_requests[
                normalized_symbol
            ] = request

        aligned_requests = {}

        for symbol, exposure in decision.target_exposure.items():
            normalized_symbol = (
                str(symbol).strip().upper()
            )

            exposure = float(exposure)

            if abs(exposure) <= 1e-12:
                continue

            request = normalized_requests.get(
                normalized_symbol
            )

            if request is None:
                raise ValueError(
                    f"Missing margin request for projected "
                    f"symbol {normalized_symbol!r}."
                )

            aligned_requests[
                normalized_symbol
            ] = replace(
                request,
                target_exposure=exposure,
            )

        return self.projector.project(
            context=context,
            decision=decision,
            margin_requests=aligned_requests,
        )

    # =========================================================================
    # Legacy path
    # =========================================================================

    def _evaluate_legacy(
        self,
        request: RiskRequest,
    ) -> RiskDecision:

        decision = request.decision
        portfolio = request.portfolio

        violations = []
        modified = False

        # ---------------------------------------------------------------------
        # Hard rejects
        # ---------------------------------------------------------------------

        if portfolio.risk_blocked:
            return self._reject(
                request,
                RiskViolation(
                    code="RISK_BLOCKED",
                    message="Portfolio is currently risk blocked.",
                    severity="critical",
                ),
            )

        if (
            portfolio.drawdown
            >= self.limits.max_drawdown
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_DRAWDOWN",
                    message="Portfolio drawdown is at or above the configured maximum.",
                    severity="critical",
                ),
            )

        if (
            portfolio.daily_drawdown
            >= self.limits.max_daily_drawdown
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_DAILY_DRAWDOWN",
                    message="Portfolio daily drawdown is at or above the configured maximum.",
                    severity="critical",
                ),
            )

        if (
            decision.portfolio_risk
            > self.limits.max_portfolio_risk
        ):
            return self._reject(
                request,
                RiskViolation(
                    code="MAX_PORTFOLIO_RISK",
                    message="Requested portfolio risk exceeds the configured portfolio-risk budget.",
                    severity="critical",
                ),
            )

        # ---------------------------------------------------------------------
        # Copy mutable proposal mappings.
        # ---------------------------------------------------------------------

        target_exposure = dict(decision.target_exposure)
        capital_allocation = dict(decision.capital_allocation)
        margin_allocation = dict(decision.margin_allocation)

        # ---------------------------------------------------------------------
        # Symbol exposure.
        # ---------------------------------------------------------------------

        for symbol in sorted(target_exposure):

            exposure = float(
                target_exposure[symbol]
            )

            if abs(exposure) > self.limits.max_symbol_exposure:

                target_exposure[symbol] = (
                    self._cap_signed(
                        exposure,
                        self.limits.max_symbol_exposure,
                    )
                )

                modified = True

                violations.append(
                    RiskViolation(
                        code="MAX_SYMBOL_EXPOSURE",
                        message=(
                            f"Exposure for {symbol} exceeds "
                            "the symbol exposure limit."
                        ),
                        symbol=symbol,
                        severity="warning",
                    )
                )

        # ---------------------------------------------------------------------
        # Capital concentration.
        # ---------------------------------------------------------------------

        for symbol in sorted(capital_allocation):

            allocation = float(
                capital_allocation[symbol]
            )

            if abs(allocation) > self.limits.max_symbol_concentration:

                capital_allocation[symbol] = (
                    self._cap_signed(
                        allocation,
                        self.limits.max_symbol_concentration,
                    )
                )

                modified = True

                violations.append(
                    RiskViolation(
                        code="MAX_SYMBOL_CONCENTRATION",
                        message=(
                            f"Capital allocation for {symbol} "
                            "exceeds the symbol concentration limit."
                        ),
                        symbol=symbol,
                        severity="warning",
                    )
                )

        # ---------------------------------------------------------------------
        # Correlated exposure.
        # ---------------------------------------------------------------------

        symbols = sorted(
            target_exposure
        )

        for i, symbol_a in enumerate(symbols):

            for symbol_b in symbols[i + 1:]:

                # Legacy context carries correlation directly on PortfolioContext
                # in the historical path.
                try:
                    correlation = float(
                        portfolio.correlation.get(
                            symbol_a,
                            {}
                        ).get(
                            symbol_b,
                            0.0,
                        )
                    )
                except AttributeError:
                    correlation = 0.0

                if (
                    abs(correlation)
                    < self.limits.high_correlation_threshold
                ):
                    continue

                combined = (
                    abs(
                        target_exposure[symbol_a]
                    )
                    +
                    abs(
                        target_exposure[symbol_b]
                    )
                )

                if (
                    combined
                    <= self.limits.max_correlated_exposure
                ):
                    continue

                factor = (
                    self.limits.max_correlated_exposure
                    / combined
                )

                target_exposure[symbol_a] *= factor
                target_exposure[symbol_b] *= factor

                modified = True

                violations.append(
                    RiskViolation(
                        code="MAX_CORRELATED_EXPOSURE",
                        message=(
                            f"Highly correlated pair {symbol_a}/{symbol_b} "
                            "exceeds the combined exposure limit."
                        ),
                        severity="warning",
                    )
                )

        # ---------------------------------------------------------------------
        # Total exposure.
        # ---------------------------------------------------------------------

        total_exposure = sum(
            abs(value)
            for value in target_exposure.values()
        )

        if (
            total_exposure
            > self.limits.max_total_exposure
        ):

            factor = (
                self.limits.max_total_exposure
                / total_exposure
            )

            target_exposure = {
                symbol: float(exposure) * factor
                for symbol, exposure
                in target_exposure.items()
            }

            modified = True

            violations.append(
                RiskViolation(
                    code="MAX_TOTAL_EXPOSURE",
                    message=(
                        "Total requested exposure exceeds "
                        "the portfolio exposure limit."
                    ),
                    severity="warning",
                )
            )

        # ---------------------------------------------------------------------
        # Margin allocation.
        # ---------------------------------------------------------------------

        requested_margin = sum(
            abs(float(value))
            for value in margin_allocation.values()
        )

        allowed_margin = (
            portfolio.equity
            * self.limits.max_margin_utilization
        )

        if requested_margin > allowed_margin:

            if requested_margin > 0.0:

                factor = (
                    allowed_margin
                    / requested_margin
                )

                margin_allocation = {
                    symbol: float(value) * factor
                    for symbol, value
                    in margin_allocation.items()
                }

                target_exposure = {
                    symbol: float(value) * factor
                    for symbol, value
                    in target_exposure.items()
                }

                capital_allocation = {
                    symbol: float(value) * factor
                    for symbol, value
                    in capital_allocation.items()
                }

                modified = True

                violations.append(
                    RiskViolation(
                        code="MAX_MARGIN_UTILIZATION",
                        message=(
                            "Requested margin allocation exceeds "
                            "the configured margin utilization limit."
                        ),
                        severity="warning",
                    )
                )

        # ---------------------------------------------------------------------
        # Build legacy result.
        # ---------------------------------------------------------------------

        status = (
            RiskDecisionStatus.MODIFIED
            if modified
            else RiskDecisionStatus.APPROVED
        )

        result = RiskDecision(
            status=status,
            timestamp=decision.timestamp,
            mode=decision.mode,
            decision_id=decision.decision_id,
            source_decision_id=decision.decision_id,
            capital_allocation=capital_allocation,
            margin_allocation=margin_allocation,
            target_exposure=target_exposure,
            target_signals=dict(
                decision.target_signals
            ),
            portfolio_risk=float(
                decision.portfolio_risk
            ),
            violations=tuple(
                violations
            ),
            model=decision.model,
            metadata={
                "evaluation_path": "legacy"
            },
        )

        self.state.record(
            status=status,
            decision_id=result.decision_id,
            violation_codes=tuple(v.code for v in violations),
            timestamp=result.timestamp,
        )

        return result

    # =========================================================================
    # Reject
    # =========================================================================

    def _reject(
        self,
        request: RiskRequest,
        violation: RiskViolation,
        extra_violations: Tuple[RiskViolation, ...] = (),
    ) -> RiskDecision:

        violations = (
            tuple(extra_violations)
            + (violation,)
        )

        decision = request.decision

        result = RiskDecision(
            status=RiskDecisionStatus.REJECTED,
            timestamp=decision.timestamp,
            mode=decision.mode,
            decision_id=decision.decision_id,
            source_decision_id=decision.decision_id,
            capital_allocation={},
            margin_allocation={},
            target_exposure={},
            target_signals={},
            portfolio_risk=float(decision.portfolio_risk),
            violations=violations,
            model=decision.model,
            metadata={
                "evaluation_path": (
                    "projected"
                    if request.risk_context is not None
                    else "legacy"
                ),
            },
        )

        self.state.record(
            status=RiskDecisionStatus.REJECTED,
            decision_id=result.decision_id,
            violation_codes=tuple(v.code for v in violations),
            timestamp=result.timestamp,
        )

        return result

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _cap_signed(
        value: float,
        limit: float,
    ) -> float:

        if value > limit:
            return limit

        if value < -limit:
            return -limit

        return value

    @staticmethod
    def _projected_summary(
        projected: ProjectedRiskState,
    ) -> Mapping[str, object]:

        return {
            "current_total_exposure": projected.current_total_exposure,
            "projected_total_exposure": projected.projected_total_exposure,
            "current_used_margin": projected.current_used_margin,
            "projected_used_margin": projected.projected_used_margin,
            "projected_free_margin": projected.projected_free_margin,
            "projected_margin_utilization": projected.projected_margin_utilization,
            "projected_position_count": projected.projected_position_count,
            "symbols": {
                symbol: {
                    "current_exposure": snapshot.current_exposure,
                    "target_exposure": snapshot.target_exposure,
                    "delta": snapshot.projected_exposure_delta,
                    "current_notional": snapshot.current_notional,
                    "projected_notional": snapshot.projected_notional,
                    "current_used_margin": snapshot.current_used_margin,
                    "projected_used_margin": snapshot.projected_used_margin,
                    "current_position_count": snapshot.current_position_count,
                    "projected_position_count": snapshot.projected_position_count,
                }
                for symbol, snapshot
                in projected.symbols.items()
            },
        }

    # =========================================================================
    # State
    # =========================================================================
    
    def reset(self) -> None:
        self.state.reset()
