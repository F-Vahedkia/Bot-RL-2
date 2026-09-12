# f05_agents/contracts.py (1)
#
# Created:
#     1405/06/19
#
# فصل 3 - Symbol-Agent + Meta-Agent
#
# این فایل فقط قراردادهای داده‌ای بین اجزای
# لایه تصمیم‌گیری را تعریف می‌کند.
#
# این فایل هیچ وابستگی به:
#     - MT5
#     - Broker
#     - Execution
#     - TradingEnvironment
# ندارد.
#
# هدف:
#     ایجاد قرارداد پایدار برای:
#
#     Observation
#          ↓
#     Symbol-Agent
#          ↓
#     Meta-Agent
#          ↓
#     Portfolio Decision
#
# قراردادها برای:
#     - Multi-Symbol
#     - RL
#     - Train
#     - Optimize
#     - Backtest
#     - Replay
#     - Evaluation
#     - Shadow
#     - Paper
#     - Live
#     - Self-Optimization
#     - Model Versioning
#     طراحی شده‌اند.


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Mapping, Optional, Tuple


# =====================================================================
# Decision Mode
# =====================================================================

class DecisionMode(str, Enum):
    """
    حالت اجرای لایه تصمیم‌گیری.

    همه Agentها باید بتوانند بدون تغییر معماری
    در این modeها کار کنند.
    """

    TRAIN = "train"
    OPTIMIZE = "optimize"
    BACKTEST = "backtest"
    REPLAY = "replay"
    EVAL = "eval"
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


# =====================================================================
# Model Identity
# =====================================================================

@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """
    شناسه مدل مورد استفاده Agent.

    این اطلاعات برای:
        - audit
        - replay
        - evaluation
        - self-optimization
        - model lineage

    نگهداری می‌شود.
    """

    model_name: str
    model_version: str
    policy_version: str
    config_version: str = ""
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.model_name).strip():
            raise ValueError("model_name is required")

        if not str(self.model_version).strip():
            raise ValueError("model_version is required")

        if not str(self.policy_version).strip():
            raise ValueError("policy_version is required")


# =====================================================================
# Agent Observation
# =====================================================================

@dataclass(frozen=True, slots=True)
class AgentObservation:
    """
    Observation متعلق به یک Symbol-Agent.

    هر Observation فقط متعلق به یک symbol است.

    داده اصلی Feature Layer پس از تبدیل به یک
    بردار عددی وارد Agent می‌شود.
    """

    symbol: str
    base_tf: str
    timestamp: datetime

    # بردار ویژگی‌ها
    values: Tuple[float, ...]

    # نام ویژگی‌ها، برای traceability و validation
    feature_names: Tuple[str, ...] = field(
        default_factory=tuple
    )

    # اطلاعات توسعه‌پذیر
    metadata: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()
        base_tf = str(self.base_tf).upper().strip()

        if not symbol:
            raise ValueError("symbol is required")

        if not base_tf:
            raise ValueError("base_tf is required")

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware"
            )

        values = tuple(float(v) for v in self.values)

        if not values:
            raise ValueError(
                "observation values must not be empty"
            )

        if not all(isfinite(v) for v in values):
            raise ValueError(
                "observation contains non-finite values"
            )

        feature_names = tuple(
            str(name) for name in self.feature_names
        )

        if feature_names and (
            len(feature_names) != len(values)
        ):
            raise ValueError(
                "feature_names and values length mismatch"
            )

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )

        object.__setattr__(
            self,
            "base_tf",
            base_tf,
        )

        object.__setattr__(
            self,
            "values",
            values,
        )

        object.__setattr__(
            self,
            "feature_names",
            feature_names,
        )


# =====================================================================
# Local Symbol Context
# =====================================================================

@dataclass(frozen=True, slots=True)
class SymbolContext:
    """
    وضعیت محلی یک Symbol برای Symbol-Agent.

    Symbol-Agent فقط context مربوط به symbol خودش را می‌بیند.

    اطلاعات Portfolio-wide در این کلاس وجود ندارد.
    """

    symbol: str

    current_side: int = 0
    current_lots: float = 0.0

    exposure: float = 0.0
    drawdown: float = 0.0

    volatility: float = 0.0
    local_risk_score: float = 0.0

    metadata: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()

        if not symbol:
            raise ValueError("symbol is required")

        if self.current_side not in (-1, 0, 1):
            raise ValueError(
                "current_side must be -1, 0 or 1"
            )

        if self.current_lots < 0.0:
            raise ValueError(
                "current_lots must be >= 0"
            )

        if self.current_side == 0 and self.current_lots != 0.0:
            raise ValueError(
                "flat context must have current_lots=0"
            )

        if self.exposure < 0.0:
            raise ValueError(
                "exposure must be >= 0"
            )

        if self.drawdown < 0.0:
            raise ValueError(
                "drawdown must be >= 0"
            )

        if self.volatility < 0.0:
            raise ValueError(
                "volatility must be >= 0"
            )

        if not (0.0 <= self.local_risk_score <= 1.0):
            raise ValueError(
                "local_risk_score must be between 0 and 1"
            )

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )


# =====================================================================
# Policy Output
# =====================================================================

@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """
    خروجی خام Policy مربوط به Symbol-Agent.

    Policy فقط تحلیل می‌کند.
    Symbol-Agent این خروجی را validate و ثبت می‌کند.
    """

    signal: int

    confidence: float

    expected_return: float

    risk_score: float

    desired_exposure: float

    metadata: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.signal not in (-1, 0, 1):
            raise ValueError(
                "signal must be -1, 0 or 1"
            )

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        if not isfinite(float(self.expected_return)):
            raise ValueError(
                "expected_return must be finite"
            )

        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(
                "risk_score must be between 0 and 1"
            )

        if not (0.0 <= self.desired_exposure <= 1.0):
            raise ValueError(
                "desired_exposure must be between 0 and 1"
            )


# =====================================================================
# Symbol-Agent Output
# =====================================================================

@dataclass(frozen=True, slots=True)
class SymbolAgentOutput:
    """
    خروجی رسمی Symbol-Agent.

    این خروجی هنوز تصمیم Portfolio نیست.
    """

    symbol: str

    timestamp: datetime

    mode: DecisionMode

    signal: int

    confidence: float

    expected_return: float

    risk_score: float

    desired_exposure: float

    model: ModelIdentity

    decision_id: str

    metadata: Mapping[str, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        symbol = str(self.symbol).upper().strip()

        if not symbol:
            raise ValueError("symbol is required")

        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware"
            )

        if self.signal not in (-1, 0, 1):
            raise ValueError(
                "signal must be -1, 0 or 1"
            )

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        if not isfinite(float(self.expected_return)):
            raise ValueError(
                "expected_return must be finite"
            )

        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(
                "risk_score must be between 0 and 1"
            )

        if not (0.0 <= self.desired_exposure <= 1.0):
            raise ValueError(
                "desired_exposure must be between 0 and 1"
            )

        if not str(self.decision_id).strip():
            raise ValueError(
                "decision_id is required"
            )

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )


# =====================================================================
# Portfolio Context
# =====================================================================

@dataclass(frozen=True, slots=True)
class PortfolioContext:
    """
    وضعیت Portfolio که فقط در اختیار Meta-Agent است.

    Meta-Agent علاوه بر خروجی Symbol-Agentها،
    این وضعیت global را نیز می‌بیند.
    """

    timestamp: datetime

    equity: float
    balance: float

    used_margin: float
    free_margin: float

    margin_level: Optional[float]

    drawdown: float
    daily_drawdown: float

    # exposure فعلی هر symbol
    exposure: Mapping[str, float] = field(
        default_factory=dict
    )

    # concentration فعلی هر symbol
    concentration: Mapping[str, float] = field(
        default_factory=dict
    )

    # correlation ماتریس به شکل:
    # {
    #     "XAUUSD": {"EURUSD": 0.42},
    #     ...
    # }
    correlation: Mapping[
        str,
        Mapping[str, float],
    ] = field(
        default_factory=dict
    )

    # آیا ریسک Portfolio فعلاً تصمیم جدید را block کرده است؟
    risk_blocked: bool = False

    # mode اجرای سیستم
    mode: DecisionMode = DecisionMode.BACKTEST

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware"
            )

        numeric_values = (
            self.equity,
            self.balance,
            self.used_margin,
            self.free_margin,
            self.drawdown,
            self.daily_drawdown,
        )

        if not all(
            isfinite(float(value))
            for value in numeric_values
        ):
            raise ValueError(
                "PortfolioContext contains non-finite values"
            )

        if self.used_margin < 0.0:
            raise ValueError(
                "used_margin must be >= 0"
            )

        if self.free_margin < 0.0:
            raise ValueError(
                "free_margin must be >= 0"
            )

        if self.drawdown < 0.0:
            raise ValueError(
                "drawdown must be >= 0"
            )

        if self.daily_drawdown < 0.0:
            raise ValueError(
                "daily_drawdown must be >= 0"
            )


# =====================================================================
# Meta-Agent Policy Output
# =====================================================================

@dataclass(frozen=True, slots=True)
class MetaPolicyOutput:
    """
    خروجی خام Policy مربوط به Meta-Agent.

    هنوز ممکن است توسط guardrailهای Meta-Agent
    محدود یا اصلاح شود.
    """

    capital_allocation: Mapping[str, float]

    target_signals: Mapping[str, int]

    target_exposure: Mapping[str, float]

    margin_allocation: Mapping[str, float]

    portfolio_risk: float

    reason_codes: Tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        if self.portfolio_risk < 0.0:
            raise ValueError(
                "portfolio_risk must be >= 0"
            )

        for symbol, value in self.capital_allocation.items():
            if float(value) < 0.0:
                raise ValueError(
                    f"negative capital allocation: {symbol}"
                )

        for symbol, value in self.margin_allocation.items():
            if float(value) < 0.0:
                raise ValueError(
                    f"negative margin allocation: {symbol}"
                )

        for symbol, signal in self.target_signals.items():
            if signal not in (-1, 0, 1):
                raise ValueError(
                    f"invalid target signal: {symbol}"
                )


# =====================================================================
# Final Portfolio Decision
# =====================================================================

@dataclass(frozen=True, slots=True)
class PortfolioDecision:
    """
    تصمیم نهایی Meta-Agent.

    این هنوز order نیست.

    این object مرز بین:
        Decision Layer
    و
        Execution Layer
    است.

    بعداً یک Adapter مستقل می‌تواند این تصمیم را
    به PortfolioAction در f04_env تبدیل کند.
    """

    approved: bool

    timestamp: datetime

    mode: DecisionMode

    decision_id: str

    capital_allocation: Mapping[str, float]

    margin_allocation: Mapping[str, float]

    target_exposure: Mapping[str, float]

    target_signals: Mapping[str, int]

    portfolio_risk: float

    reason_codes: Tuple[str, ...]

    model: ModelIdentity

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware"
            )

        if not str(self.decision_id).strip():
            raise ValueError(
                "decision_id is required"
            )

        if self.portfolio_risk < 0.0:
            raise ValueError(
                "portfolio_risk must be >= 0"
            )

        for symbol, value in self.capital_allocation.items():
            if float(value) < 0.0:
                raise ValueError(
                    f"negative capital allocation: {symbol}"
                )

        for symbol, value in self.margin_allocation.items():
            if float(value) < 0.0:
                raise ValueError(
                    f"negative margin allocation: {symbol}"
                )

        for symbol, signal in self.target_signals.items():
            if signal not in (-1, 0, 1):
                raise ValueError(
                    f"invalid target signal: {symbol}"
                )