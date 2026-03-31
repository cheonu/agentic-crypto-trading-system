"""Configuration for the day trading strategy.

Defines DayTradingConfig with validated parameters for stop-loss,
take-profit, trailing stop, signal weights, and operational settings.
"""

import math
from dataclasses import dataclass


@dataclass
class DayTradingConfig:
    """Day trading strategy configuration with startup validation.

    All percentage fields must be in (0, 1) exclusive.
    Signal weights (news_weight, intraday_weight, regime_weight) must sum to 1.0.
    max_positions must be >= 1.

    Raises ValueError with a descriptive message on any validation failure.
    """

    stop_loss_pct: float = 0.015        # 1.5% stop-loss for protection
    take_profit_pct: float = 0.5         # 50% — effectively disabled, ride the wave
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.008     # 0.8% trailing stop locks in gains
    news_weight: float = 0.1
    intraday_weight: float = 0.6
    regime_weight: float = 0.3
    max_positions: int = 1
    intraday_candle_timeframe: str = "5m"
    intraday_candle_count: int = 50
    news_cache_ttl_minutes: int = 15
    portfolio_value: float = 100.0
    risk_per_trade_pct: float = 1.0      # all-in

    def __post_init__(self) -> None:
        # Validate percentage fields are in (0, 1) exclusive
        pct_fields = {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
        }
        for name, value in pct_fields.items():
            if not (0 < value < 1):
                raise ValueError(
                    f"{name} must be in range (0, 1) exclusive, got {value}"
                )

        # Validate weights sum to 1.0
        weight_sum = self.news_weight + self.intraday_weight + self.regime_weight
        if not math.isclose(weight_sum, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"news_weight + intraday_weight + regime_weight must equal 1.0, "
                f"got {weight_sum} "
                f"(news_weight={self.news_weight}, "
                f"intraday_weight={self.intraday_weight}, "
                f"regime_weight={self.regime_weight})"
            )

        # Validate max_positions >= 1
        if self.max_positions < 1:
            raise ValueError(
                f"max_positions must be >= 1, got {self.max_positions}"
            )
