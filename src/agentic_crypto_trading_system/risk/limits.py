from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_position_size_pct: float = 5.0       # Max % of portfolio per position
    max_portfolio_exposure_pct: float = 80.0  # Max total exposure %
    max_daily_loss_pct: float = 3.0           # Max daily loss %
    max_leverage: float = 3.0                 # Max leverage ratio
    max_open_positions: int = 10              # Max concurrent positions
    stop_loss_pct: float = 2.0               # Default stop loss %
    max_single_trade_pct: float = 2.0        # Max single trade size %
    min_risk_reward_ratio: float = 1.5       # Minimum risk/reward ratio

    def validate(self) -> bool:
        """Validate risk limit consistency."""
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 100:
            return False
        if self.max_portfolio_exposure_pct <= 0 or self.max_portfolio_exposure_pct > 100:
            return False
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 100:
            return False
        if self.max_leverage < 1.0:
            return False
        if self.max_open_positions < 1:
            return False
        if self.stop_loss_pct <= 0:
            return False
        return True
