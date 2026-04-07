"""Configuration for the day trading strategy.

Defines DayTradingConfig with validated parameters for stop-loss,
take-profit, trailing stop, signal weights, and operational settings.
"""

import math
import os
from dataclasses import dataclass, field


@dataclass
class DayTradingConfig:
    """Day trading strategy configuration with startup validation.

    All percentage fields must be in (0, 1) exclusive.
    Signal weights (news_weight, intraday_weight, regime_weight) must sum to 1.0.
    max_positions must be >= 1.

    Raises ValueError with a descriptive message on any validation failure.
    """

    stop_loss_pct: float = 0.008         # 0.8% stop-loss — tight for scalping
    take_profit_pct: float = 0.02        # 2.0% take-profit — 2.5:1 reward/risk
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.012     # 1.2% trailing stop — wider than SL to let winners run
    news_weight: float = 0.2
    intraday_weight: float = 0.5
    regime_weight: float = 0.3
    max_positions: int = 1
    intraday_candle_timeframe: str = "5m"
    intraday_candle_count: int = 50
    news_cache_ttl_minutes: int = 15
    portfolio_value: float = 100.0
    risk_per_trade_pct: float = 1.0      # all-in
    min_buy_confidence: float = 0.45     # lowered from 0.55 to catch more moves
    sentiment_model_name: str = "ElKulako/cryptobert"
    news_api_key: str = ""
    news_source: str = "rss"

    def __post_init__(self) -> None:
        # Override from environment variables
        self.sentiment_model_name = os.environ.get(
            "SENTIMENT_MODEL_NAME", self.sentiment_model_name
        )
        self.news_api_key = os.environ.get("NEWS_API_KEY", self.news_api_key)

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
