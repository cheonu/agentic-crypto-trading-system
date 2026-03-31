import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np

from .limits import RiskLimits

logger = logging.getLogger(__name__)


@dataclass
class TradeProposal:
    """A proposed trade to validate."""
    symbol: str
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    agent_id: str = ""


@dataclass
class ValidationResult:
    """Result of risk validation."""
    approved: bool
    reasons: List[str]


class RiskManager:
    """Validates trades and monitors risk."""

    def __init__(self, limits: RiskLimits = None, portfolio_value: float = 100000.0):
        self.limits = limits or RiskLimits()
        self.portfolio_value = portfolio_value
        self.daily_pnl: float = 0.0
        self.open_positions: List[Dict] = []
        self.audit_log: List[Dict] = []

    def validate_trade(self, proposal: TradeProposal) -> ValidationResult:
        """Validate a trade proposal against all risk limits."""
        reasons = []

        # Check position size
        trade_value = proposal.size * proposal.entry_price
        trade_pct = (trade_value / self.portfolio_value) * 100
        if trade_pct > self.limits.max_single_trade_pct:
            reasons.append(
                f"Trade size {trade_pct:.1f}% exceeds max {self.limits.max_single_trade_pct}%"
            )

        # Check max open positions
        if len(self.open_positions) >= self.limits.max_open_positions:
            reasons.append(
                f"Max open positions ({self.limits.max_open_positions}) reached"
            )

        # Check portfolio exposure
        current_exposure = sum(
            p.get("size", 0) * p.get("entry_price", 0) for p in self.open_positions
        )
        new_exposure_pct = ((current_exposure + trade_value) / self.portfolio_value) * 100
        if new_exposure_pct > self.limits.max_portfolio_exposure_pct:
            reasons.append(
                f"Portfolio exposure {new_exposure_pct:.1f}% exceeds max {self.limits.max_portfolio_exposure_pct}%"
            )

        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / self.portfolio_value) * 100 if self.daily_pnl < 0 else 0
        if daily_loss_pct >= self.limits.max_daily_loss_pct:
            reasons.append(
                f"Daily loss {daily_loss_pct:.1f}% exceeds max {self.limits.max_daily_loss_pct}%"
            )

        # Check stop loss
        if proposal.stop_loss:
            sl_distance = abs(proposal.entry_price - proposal.stop_loss) / proposal.entry_price * 100
            if sl_distance > self.limits.stop_loss_pct * 2:
                reasons.append(f"Stop loss distance {sl_distance:.1f}% too wide")

        # Check risk/reward ratio
        if proposal.take_profit and proposal.stop_loss:
            risk = abs(proposal.entry_price - proposal.stop_loss)
            reward = abs(proposal.take_profit - proposal.entry_price)
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.limits.min_risk_reward_ratio:
                    reasons.append(
                        f"Risk/reward {rr_ratio:.2f} below min {self.limits.min_risk_reward_ratio}"
                    )

        approved = len(reasons) == 0
        self._log_validation(proposal, approved, reasons)
        return ValidationResult(approved=approved, reasons=reasons)

    def calculate_position_size(
        self, entry_price: float, stop_loss: float, risk_pct: float = None
    ) -> float:
        """Calculate position size based on risk parameters."""
        if risk_pct is None:
            risk_pct = self.limits.max_single_trade_pct

        risk_amount = self.portfolio_value * (risk_pct / 100)
        price_risk = abs(entry_price - stop_loss)
        if price_risk == 0:
            return 0.0
        return risk_amount / price_risk

    def calculate_portfolio_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Calculate Value at Risk at given confidence level."""
        if not returns:
            return 0.0
        returns_arr = np.array(returns)
        var = np.percentile(returns_arr, (1 - confidence) * 100)
        return float(abs(var) * self.portfolio_value)

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L tracking."""
        self.daily_pnl += pnl

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L (call at start of trading day)."""
        self.daily_pnl = 0.0

    def update_limits(self, **kwargs) -> None:
        """Update risk limits with audit logging."""
        old_limits = {k: v for k, v in self.limits.__dict__.items()}
        for key, value in kwargs.items():
            if hasattr(self.limits, key):
                setattr(self.limits, key, value)
        self.audit_log.append({
            "action": "limits_updated",
            "old": old_limits,
            "new": {k: v for k, v in self.limits.__dict__.items()},
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _log_validation(self, proposal: TradeProposal, approved: bool, reasons: List[str]) -> None:
        """Log validation result."""
        logger.info(
            f"Trade validation: {proposal.symbol} {proposal.direction} "
            f"size={proposal.size} approved={approved} reasons={reasons}"
        )
