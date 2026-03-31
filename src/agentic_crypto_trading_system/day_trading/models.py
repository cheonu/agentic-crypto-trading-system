"""Data models for the day trading strategy.

Defines the core dataclasses used across all day trading components:
OpenPosition, ClosedTrade, TradeSignal, IntradaySignals, NewsSignal, StopLossEvent.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import List, Optional


@dataclass
class OpenPosition:
    """Represents an active trade position.

    Tracks symbol, side, entry details, stop-loss/take-profit levels,
    unrealized P&L, and trailing stop state.

    Validation (in __post_init__):
    - entry_price > 0
    - size > 0
    - stop_loss_price < entry_price for long positions
    - stop_loss_price > entry_price for short positions
    """

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    entry_time: datetime
    stop_loss_price: float
    take_profit_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    highest_price_since_entry: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(
                f"entry_price must be > 0, got {self.entry_price}"
            )
        if self.size <= 0:
            raise ValueError(f"size must be > 0, got {self.size}")
        if self.side not in ("long", "short"):
            raise ValueError(
                f"side must be 'long' or 'short', got '{self.side}'"
            )
        if self.side == "long" and self.stop_loss_price >= self.entry_price:
            raise ValueError(
                f"stop_loss_price ({self.stop_loss_price}) must be < "
                f"entry_price ({self.entry_price}) for long positions"
            )
        if self.side == "short" and self.stop_loss_price <= self.entry_price:
            raise ValueError(
                f"stop_loss_price ({self.stop_loss_price}) must be > "
                f"entry_price ({self.entry_price}) for short positions"
            )


@dataclass
class ClosedTrade:
    """Represents a completed trade with realized P&L.

    exit_reason must be one of: "regime_change", "stop_loss", "take_profit", "manual".
    """

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    exit_price: float
    size: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    realized_pnl_pct: float
    exit_reason: str  # "regime_change", "stop_loss", "take_profit", "manual"

    VALID_EXIT_REASONS = ("regime_change", "stop_loss", "take_profit", "manual")

    def __post_init__(self) -> None:
        if self.exit_reason not in self.VALID_EXIT_REASONS:
            raise ValueError(
                f"exit_reason must be one of {self.VALID_EXIT_REASONS}, "
                f"got '{self.exit_reason}'"
            )


@dataclass
class TradeSignal:
    """Represents a trading decision produced by DayTradingStrategy.

    action is one of "BUY", "SELL", "HOLD".
    confidence is a weighted score from regime, news, and intraday signals.
    """

    action: str  # "BUY", "SELL", "HOLD"
    reason: str
    confidence: float
    stop_loss_pct: float
    take_profit_pct: Optional[float] = None

    VALID_ACTIONS = ("BUY", "SELL", "HOLD")

    def __post_init__(self) -> None:
        if self.action not in self.VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {self.VALID_ACTIONS}, "
                f"got '{self.action}'"
            )


@dataclass
class IntradaySignals:
    """Intraday trend analysis results from short-timeframe candles.

    Produced by IntradayTrendAnalyzer from 5m/15m candle data.
    """

    trend: str  # "up", "down", "sideways"
    momentum: float  # -1.0 to 1.0
    vwap_position: str  # "above", "below", "at"
    ema_cross: Optional[str]  # "golden_cross", "death_cross", or None
    rsi: float  # 0-100
    volume_trend: str  # "increasing", "decreasing", "stable"
    confidence: float


@dataclass
class NewsSignal:
    """News sentiment signal produced by NewsSignalProvider.

    score ranges from -1.0 (very bearish) to 1.0 (very bullish).
    event_flags lists detected market-moving events.
    """

    score: float  # -1.0 to 1.0
    headline_count: int
    top_headlines: List[str] = field(default_factory=list)
    event_flags: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class StopLossEvent:
    """Records a stop-loss or take-profit trigger event.

    Created by StopLossMonitor when a position breaches its threshold.
    exit_reason is "stop_loss" or "take_profit".
    """

    symbol: str
    entry_price: float
    stop_loss_price: float
    trigger_price: float
    loss_pct: float
    timestamp: datetime
    exit_reason: str  # "stop_loss" or "take_profit"

    VALID_EXIT_REASONS = ("stop_loss", "take_profit")

    def __post_init__(self) -> None:
        if self.exit_reason not in self.VALID_EXIT_REASONS:
            raise ValueError(
                f"exit_reason must be one of {self.VALID_EXIT_REASONS}, "
                f"got '{self.exit_reason}'"
            )
