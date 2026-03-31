"""Fee-aware trade filter for the day trading strategy.

Calculates round-trip trading fees (including optional BNB discount) and
blocks trades where expected profit does not exceed 2.5× the fee cost.
"""

import logging

from .models import TradeSignal

logger = logging.getLogger(__name__)


class FeeAwareFilter:
    """Filters trade signals based on fee profitability.

    For BUY signals, calculates the expected profit from take_profit_pct and
    compares it against min_profit_fee_ratio × round-trip fees.  If the trade
    is not profitable enough, the signal is downgraded to HOLD.

    SELL and HOLD signals pass through unchanged.

    Args:
        base_fee_rate: Fee rate per side (default 0.1% for Binance).
        bnb_discount_enabled: Whether to apply the BNB fee discount.
        bnb_discount_rate: Fraction of fee saved when paying with BNB (default 25%).
        min_profit_fee_ratio: Minimum ratio of expected profit to round-trip fee (default 2.5).
    """

    def __init__(
        self,
        base_fee_rate: float = 0.001,
        bnb_discount_enabled: bool = False,
        bnb_discount_rate: float = 0.25,
        min_profit_fee_ratio: float = 2.5,
    ) -> None:
        self._base_fee_rate = base_fee_rate
        self._bnb_discount_enabled = bnb_discount_enabled
        self._bnb_discount_rate = bnb_discount_rate
        self._min_profit_fee_ratio = min_profit_fee_ratio

    def calculate_round_trip_fee(
        self, trade_size: float, entry_price: float
    ) -> float:
        """Calculate the total round-trip fee (entry + exit) for a trade.

        Round-trip fee = 2 × base_fee_rate × trade_size × entry_price.
        If BNB discount is enabled the fee is reduced by bnb_discount_rate.
        """
        fee = 2 * self._base_fee_rate * trade_size * entry_price
        if self._bnb_discount_enabled:
            fee *= 1 - self._bnb_discount_rate
        return fee

    def filter_signal(
        self,
        signal: TradeSignal,
        trade_size: float,
        entry_price: float,
    ) -> TradeSignal:
        """Filter a trade signal based on fee profitability.

        For BUY signals: calculate expected profit from take_profit_pct,
        compare against min_profit_fee_ratio × round-trip fees.
        If unprofitable, return HOLD.

        For SELL/HOLD signals: pass through unchanged.
        """
        if signal.action != "BUY":
            return signal

        round_trip_fee = self.calculate_round_trip_fee(trade_size, entry_price)

        take_profit_pct = signal.take_profit_pct or 0.0
        expected_profit = take_profit_pct * trade_size * entry_price

        if expected_profit < self._min_profit_fee_ratio * round_trip_fee:
            logger.info(
                "FeeAwareFilter blocked trade: expected_profit=%.6f < %.1f × fee=%.6f",
                expected_profit,
                self._min_profit_fee_ratio,
                round_trip_fee,
            )
            return TradeSignal(
                action="HOLD",
                reason="Expected profit below fee threshold",
                confidence=signal.confidence,
                stop_loss_pct=signal.stop_loss_pct,
                take_profit_pct=signal.take_profit_pct,
            )

        return signal
