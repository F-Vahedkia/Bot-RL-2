# f06_risk/risk_engine_old1.py (4)
#
# Created:
#     1405/06/19
#
# Chapter 4 - Risk Engine
#
# ورودی:
#     PortfolioDecision + PortfolioContext
#
# خروجی:
#     RiskDecision
#
# Risk Engine هیچ execution انجام نمی‌دهد.


from __future__ import annotations

from itertools import combinations

from f06_risk.contracts import (
    RiskDecision,
    RiskDecisionStatus,
    RiskRequest,
    RiskViolation,
)

from f06_risk.limits import (
    RiskLimits,
)

from f06_risk.state import (
    RiskEngineState,
)


class RiskEngine:
    """
    موتور مرکزی Risk & Constraint Layer.

    تصمیم فصل ۳ را بررسی می‌کند و در صورت امکان
    آن را داخل محدوده‌های مجاز اصلاح می‌کند.

    ترتیب:

        Hard Block
            ↓
        Symbol Limits
            ↓
        Concentration
            ↓
        Correlation
            ↓
        Total Exposure
            ↓
        Margin
            ↓
        Final Decision
    """

    def __init__(
        self,
        *,
        limits: RiskLimits,
    ) -> None:

        self.limits = limits

        self.state = RiskEngineState()


    # =================================================================
    # Public API
    # =================================================================

    def evaluate(
        self,
        request: RiskRequest,
    ) -> RiskDecision:

        decision = request.decision
        portfolio = request.portfolio

        violations: list[RiskViolation] = []

        # -------------------------------------------------------------
        # Hard Portfolio Blocks
        # -------------------------------------------------------------

        if portfolio.risk_blocked:

            return self._reject(
                request=request,
                code="PORTFOLIO_RISK_BLOCKED",
                message=(
                    "Portfolio risk state blocks "
                    "new exposure."
                ),
                violations=violations,
            )


        if portfolio.drawdown >= self.limits.max_drawdown:

            return self._reject(
                request=request,
                code="MAX_DRAWDOWN",
                message=(
                    "Portfolio drawdown has reached "
                    "the configured limit."
                ),
                violations=violations,
            )


        if (
            portfolio.daily_drawdown
            >= self.limits.max_daily_drawdown
        ):

            return self._reject(
                request=request,
                code="MAX_DAILY_DRAWDOWN",
                message=(
                    "Daily drawdown has reached "
                    "the configured limit."
                ),
                violations=violations,
            )


        # -------------------------------------------------------------
        # Portfolio risk estimate
        # -------------------------------------------------------------

        if (
            decision.portfolio_risk
            > self.limits.max_portfolio_risk
        ):

            return self._reject(
                request=request,
                code="PORTFOLIO_RISK_LIMIT",
                message=(
                    "Requested portfolio risk exceeds "
                    "configured maximum."
                ),
                violations=violations,
            )


        allocations = {
            symbol: max(
                0.0,
                float(value),
            )
            for symbol, value
            in decision.capital_allocation.items()
        }


        exposures = {
            symbol: float(value)
            for symbol, value
            in decision.target_exposure.items()
        }


        margins = {
            symbol: max(
                0.0,
                float(value),
            )
            for symbol, value
            in decision.margin_allocation.items()
        }


        # -------------------------------------------------------------
        # Symbol exposure cap
        # -------------------------------------------------------------

        changed = False

        for symbol, exposure in list(
            exposures.items()
        ):

            absolute = abs(exposure)

            if absolute <= self.limits.max_symbol_exposure:
                continue

            exposures[symbol] = (
                self.limits.max_symbol_exposure
                if exposure > 0
                else -self.limits.max_symbol_exposure
            )

            violations.append(
                RiskViolation(
                    code="SYMBOL_EXPOSURE_CAP",
                    message=(
                        "Symbol exposure reduced "
                        "to configured maximum."
                    ),
                    symbol=symbol,
                    severity="warning",
                )
            )

            changed = True


        # -------------------------------------------------------------
        # Symbol concentration cap
        # -------------------------------------------------------------

        for symbol, allocation in list(
            allocations.items()
        ):

            if (
                allocation
                <= self.limits.max_symbol_concentration
            ):
                continue

            allocations[symbol] = (
                self.limits.max_symbol_concentration
            )

            violations.append(
                RiskViolation(
                    code="SYMBOL_CONCENTRATION_CAP",
                    message=(
                        "Capital allocation reduced "
                        "to concentration limit."
                    ),
                    symbol=symbol,
                    severity="warning",
                )
            )

            changed = True


        # -------------------------------------------------------------
        # Correlation concentration
        # -------------------------------------------------------------

        correlation_modified = False

        for symbol_a, symbol_b in combinations(
            exposures.keys(),
            2,
        ):

            corr_a = (
                portfolio.correlation
                .get(symbol_a, {})
                .get(symbol_b, 0.0)
            )

            corr_b = (
                portfolio.correlation
                .get(symbol_b, {})
                .get(symbol_a, corr_a)
            )

            correlation = max(
                abs(float(corr_a)),
                abs(float(corr_b)),
            )

            if (
                correlation
                < self.limits.high_correlation_threshold
            ):
                continue


            combined = (
                abs(exposures[symbol_a])
                +
                abs(exposures[symbol_b])
            )


            if (
                combined
                <= self.limits.max_correlated_exposure
            ):
                continue


            factor = (
                self.limits.max_correlated_exposure
                /
                combined
            )


            exposures[symbol_a] *= factor
            exposures[symbol_b] *= factor

            correlation_modified = True

            violations.append(
                RiskViolation(
                    code="CORRELATED_EXPOSURE_CAP",
                    message=(
                        "Combined exposure of highly "
                        "correlated symbols was reduced."
                    ),
                    severity="warning",
                )
            )


        changed = (
            changed
            or correlation_modified
        )


        # -------------------------------------------------------------
        # Total exposure
        # -------------------------------------------------------------

        total_exposure = sum(
            abs(value)
            for value in exposures.values()
        )


        if (
            total_exposure
            > self.limits.max_total_exposure
        ):

            factor = (
                self.limits.max_total_exposure
                /
                total_exposure
            )

            exposures = {
                symbol: value * factor
                for symbol, value
                in exposures.items()
            }

            violations.append(
                RiskViolation(
                    code="TOTAL_EXPOSURE_CAP",
                    message=(
                        "Total portfolio exposure "
                        "was reduced."
                    ),
                    severity="warning",
                )
            )

            changed = True


        # -------------------------------------------------------------
        # Margin utilization
        # -------------------------------------------------------------

        maximum_margin = (
            portfolio.equity
            *
            self.limits.max_margin_utilization
        )


        requested_margin = sum(
            margins.values()
        )


        if requested_margin > maximum_margin:

            if maximum_margin <= 0.0:

                return self._reject(
                    request=request,
                    code="NO_MARGIN_CAPACITY",
                    message=(
                        "No margin capacity remains "
                        "under configured limits."
                    ),
                    violations=violations,
                )


            factor = (
                maximum_margin
                /
                requested_margin
            )


            margins = {
                symbol: value * factor
                for symbol, value
                in margins.items()
            }


            exposures = {
                symbol: value * factor
                for symbol, value
                in exposures.items()
            }


            allocations = {
                symbol: value * factor
                for symbol, value
                in allocations.items()
            }


            violations.append(
                RiskViolation(
                    code="MARGIN_UTILIZATION_CAP",
                    message=(
                        "Allocation, exposure and margin "
                        "were reduced to margin capacity."
                    ),
                    severity="warning",
                )
            )

            changed = True


        # -------------------------------------------------------------
        # If everything was unchanged
        # -------------------------------------------------------------

        status = (
            RiskDecisionStatus.MODIFIED
            if changed
            else RiskDecisionStatus.APPROVED
        )


        reason_code = (
            "risk_modified"
            if changed
            else "risk_approved"
        )


        result = RiskDecision(
            status=status,
            timestamp=decision.timestamp,
            mode=decision.mode,
            decision_id=decision.decision_id,
            source_decision_id=decision.decision_id,
            capital_allocation=allocations,
            margin_allocation=margins,
            target_exposure=exposures,
            target_signals=dict(
                decision.target_signals
            ),
            portfolio_risk=min(
                decision.portfolio_risk,
                self.limits.max_portfolio_risk,
            ),
            violations=tuple(
                violations
            ),
            model=decision.model,
            metadata={
                "risk_modified": (
                    1.0
                    if changed
                    else 0.0
                ),
                "risk_result": (
                    1.0
                    if status
                    == RiskDecisionStatus.APPROVED
                    else 2.0
                ),
                reason_code: 1.0,
            },
        )


        self.state.record(
            status=status,
            decision_id=decision.decision_id,
            timestamp=decision.timestamp,
            violation_codes=tuple(
                violation.code
                for violation in violations
            ),
        )


        return result


    # =================================================================
    # Reject helper
    # =================================================================

    def _reject(
        self,
        *,
        request: RiskRequest,
        code: str,
        message: str,
        violations: list[RiskViolation],
    ) -> RiskDecision:

        violation = RiskViolation(
            code=code,
            message=message,
            severity="critical",
        )

        all_violations = tuple(
            violations
        ) + (violation,)

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
            portfolio_risk=0.0,
            violations=all_violations,
            model=decision.model,
            metadata={
                "risk_rejected": 1.0,
            },
        )


        self.state.record(
            status=RiskDecisionStatus.REJECTED,
            decision_id=decision.decision_id,
            timestamp=decision.timestamp,
            violation_codes=tuple(
                violation.code
                for violation
                in all_violations
            ),
        )


        return result


    # =================================================================
    # Lifecycle
    # =================================================================

    def reset(self) -> None:
        self.state.reset()

