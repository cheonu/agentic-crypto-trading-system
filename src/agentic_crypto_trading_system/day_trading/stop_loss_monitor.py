"""Stop-loss and take-profit monitor for the day trading strategy.

Checks all open positions against current market prices every cycle.
Triggers automatic sell events when stop-loss or take-profit thresholds
are breached. Supports trailing stop-loss that moves upward as price
increases for long positions.
"""

import logging
from datetime import UTC, datetime
from typing import Dict, List

from .models import StopLossEvent
from .position_manager import PositionManager

logger = logging.getLogger(__name__)


class StopLossMonitor:
    """Monitors open positions for stop-loss and take-profit triggers.

    Each cycle, iterates all open positions and compares the current
    market price against the position's stop_loss_price and
    take_profit_price. Returns a list of StopLossEvent objects for
    any positions that breach their thresholds.

    Also supports trailing stop-loss: as the price rises above the
    highest price since entry, the stop-loss is ratcheted upward
    (never downward for longs).
    """

    def __init__(
        self,
        position_manager: PositionManager,
        default_stop_loss_pct: float = 0.03,
        trailing_stop_pct: float = 0.02,
    ) -> None:
        self._position_manager = position_manager
        self._default_stop_loss_pct = default_stop_loss_pct
        self._trailing_stop_pct = trailing_stop_pct
        self._history: List[StopLossEvent] = []

    def check_stop_losses(self, market_data: Dict) -> List[StopLossEvent]:
        """Check all open positions against current prices.

        Args:
            market_data: Dict with a "prices" key mapping symbol -> current_price.

        Returns:
            List of StopLossEvent for positions that breached thresholds.
            Empty list if no thresholds were breached.
        """
        prices: Dict[str, float] = market_data.get("prices", {})
        events: List[StopLossEvent] = []

        for symbol, position in self._position_manager.get_all_positions().items():
            current_price = prices.get(symbol)
            if current_price is None:
                continue

            event = self._check_position(symbol, position, current_price)
            if event is not None:
                events.append(event)
                self._history.append(event)
                logger.info(
                    "Stop-loss event for %s: %s at price %.6f (entry %.6f, stop %.6f)",
                    symbol,
                    event.exit_reason,
                    current_price,
                    position.entry_price,
                    position.stop_loss_price,
                )

        return events

    def update_trailing_stop(self, symbol: str, current_price: float) -> None:
        """Update the trailing stop-loss for a position.

        For long positions:
        - If current_price > highest_price_since_entry, update the high.
        - Recalculate new_stop = highest × (1 - trailing_stop_pct).
        - Only move stop_loss_price upward, never downward.

        For short positions this is a no-op for now (trailing stops
        for shorts would move downward, which is not yet implemented).

        Args:
            symbol: The trading pair symbol.
            current_price: The latest market price.
        """
        position = self._position_manager.get_position(symbol)
        if position is None:
            return

        if position.side == "long":
            if current_price > position.highest_price_since_entry:
                position.highest_price_since_entry = current_price

            new_stop = position.highest_price_since_entry * (
                1 - self._trailing_stop_pct
            )

            # Only move stop-loss upward for longs
            if new_stop > position.stop_loss_price:
                position.stop_loss_price = new_stop

    def get_stop_loss_history(self) -> List[StopLossEvent]:
        """Return a copy of all recorded stop-loss/take-profit events."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_position(self, symbol, position, current_price: float):
        """Check a single position for stop-loss or take-profit breach.

        Returns a StopLossEvent if a threshold is breached, else None.
        """
        if position.side == "long":
            return self._check_long(symbol, position, current_price)
        return self._check_short(symbol, position, current_price)

    @staticmethod
    def _check_long(symbol, position, current_price: float):
        """Check long position thresholds."""
        # Stop-loss: price falls to or below stop_loss_price
        if current_price <= position.stop_loss_price:
            loss_pct = (position.entry_price - current_price) / position.entry_price
            return StopLossEvent(
                symbol=symbol,
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss_price,
                trigger_price=current_price,
                loss_pct=loss_pct,
                timestamp=datetime.now(UTC),
                exit_reason="stop_loss",
            )

        # Take-profit: price rises to or above take_profit_price
        if (
            position.take_profit_price is not None
            and current_price >= position.take_profit_price
        ):
            gain_pct = (current_price - position.entry_price) / position.entry_price
            return StopLossEvent(
                symbol=symbol,
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss_price,
                trigger_price=current_price,
                loss_pct=-gain_pct,  # negative loss = gain
                timestamp=datetime.now(UTC),
                exit_reason="take_profit",
            )

        return None

    @staticmethod
    def _check_short(symbol, position, current_price: float):
        """Check short position thresholds."""
        # Stop-loss: price rises to or above stop_loss_price
        if current_price >= position.stop_loss_price:
            loss_pct = (current_price - position.entry_price) / position.entry_price
            return StopLossEvent(
                symbol=symbol,
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss_price,
                trigger_price=current_price,
                loss_pct=loss_pct,
                timestamp=datetime.now(UTC),
                exit_reason="stop_loss",
            )

        # Take-profit: price falls to or below take_profit_price
        if (
            position.take_profit_price is not None
            and current_price <= position.take_profit_price
        ):
            gain_pct = (position.entry_price - current_price) / position.entry_price
            return StopLossEvent(
                symbol=symbol,
                entry_price=position.entry_price,
                stop_loss_price=position.stop_loss_price,
                trigger_price=current_price,
                loss_pct=-gain_pct,  # negative loss = gain
                timestamp=datetime.now(UTC),
                exit_reason="take_profit",
            )

        return None
