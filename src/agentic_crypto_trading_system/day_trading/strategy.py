"""Day trading strategy with intraday signal-driven entry/exit logic.

Triggers trades on both regime transitions AND intraday signals (EMA
crossovers, RSI extremes, VWAP position). Designed for scalping small
moves on 5-minute candles with a small account.
"""

import logging
from typing import Dict, Optional

from .config import DayTradingConfig
from .models import IntradaySignals, TradeSignal
from .position_manager import PositionManager
from .session_manager import TradingSessionManager

logger = logging.getLogger(__name__)


class DayTradingStrategy:
    """Evaluates intraday signals and regime transitions for trade decisions.

    BUY triggers:
    - Regime transition to bull (original)
    - Golden cross (EMA9 crosses above EMA21) + RSI < 70
    - RSI bounces off oversold (< 35) with upward trend
    - Price crosses above VWAP with upward momentum

    SELL triggers:
    - Regime transition to bear (original)
    - Death cross (EMA9 crosses below EMA21) with open position
    - RSI enters overbought (> 75) with open position
    - Price drops below VWAP with downward momentum + open position
    """

    def __init__(
        self,
        config: DayTradingConfig,
        session_manager: TradingSessionManager,
    ) -> None:
        self._config = config
        self._session_manager = session_manager

    def evaluate(
        self,
        current_regime: Dict,
        previous_regime: Optional[Dict],
        sentiment: Dict,
        intraday_signals: IntradaySignals,
        position_manager: PositionManager,
        symbol: str,
    ) -> TradeSignal:
        """Produce a TradeSignal for the current cycle."""
        transition = self._detect_regime_transition(current_regime, previous_regime)
        has_position = position_manager.has_open_position(symbol)
        session = self._session_manager.get_current_session()

        # --- Compute confidence ---
        regime_score = 1.0 if transition is not None else 0.0
        raw_news = sentiment.get("score", sentiment.get("news_score", 0.0))
        news_score = max(0.0, min(1.0, (raw_news + 1.0) / 2.0))
        intraday_score = max(0.0, min(1.0, intraday_signals.confidence))

        confidence = (
            self._config.regime_weight * regime_score
            + self._config.news_weight * news_score
            + self._config.intraday_weight * intraday_score
        )

        # --- Quiet-session gate ---
        if session.name == "quiet" and confidence < session.confidence_threshold:
            return TradeSignal(
                action="HOLD",
                reason="Confidence below quiet-session threshold",
                confidence=confidence,
                stop_loss_pct=self._config.stop_loss_pct,
            )

        # --- SELL signals (check first to protect capital) ---
        if has_position:
            sell_reason = self._check_sell_signals(
                transition, intraday_signals
            )
            if sell_reason:
                return TradeSignal(
                    action="SELL",
                    reason=sell_reason,
                    confidence=confidence,
                    stop_loss_pct=self._config.stop_loss_pct,
                )

        # --- BUY signals ---
        if not has_position:
            buy_reason = self._check_buy_signals(
                transition, intraday_signals
            )
            if buy_reason:
                return TradeSignal(
                    action="BUY",
                    reason=buy_reason,
                    confidence=max(confidence, 0.5),
                    stop_loss_pct=self._config.stop_loss_pct,
                    take_profit_pct=self._config.take_profit_pct,
                )

        return TradeSignal(
            action="HOLD",
            reason="No actionable signal",
            confidence=confidence,
            stop_loss_pct=self._config.stop_loss_pct,
        )

    # ------------------------------------------------------------------
    # Buy signal detection
    # ------------------------------------------------------------------

    def _check_buy_signals(
        self,
        transition: Optional[str],
        intraday: IntradaySignals,
    ) -> Optional[str]:
        """Return a reason string if a BUY signal is detected, else None."""

        # 1. Regime transition to bull
        if transition == "bull":
            return "Regime transition to bull"

        # 2. Golden cross + RSI not overbought
        if intraday.ema_cross == "golden_cross" and intraday.rsi < 70:
            return f"Golden cross with RSI={intraday.rsi:.0f}"

        # 3. RSI oversold bounce + upward trend
        if intraday.rsi < 35 and intraday.trend == "up":
            return f"RSI oversold bounce ({intraday.rsi:.0f}) with uptrend"

        # 4. Price above VWAP + positive momentum + upward trend
        if (
            intraday.vwap_position == "above"
            and intraday.momentum > 0.3
            and intraday.trend == "up"
        ):
            return f"Above VWAP with strong momentum ({intraday.momentum:.2f})"

        return None

    # ------------------------------------------------------------------
    # Sell signal detection
    # ------------------------------------------------------------------

    def _check_sell_signals(
        self,
        transition: Optional[str],
        intraday: IntradaySignals,
    ) -> Optional[str]:
        """Return a reason string if a SELL signal is detected, else None."""

        # 1. Regime transition to bear
        if transition == "bear":
            return "Regime transition to bear"

        # 2. Death cross
        if intraday.ema_cross == "death_cross":
            return "Death cross detected"

        # 3. RSI overbought — take profits
        if intraday.rsi > 75:
            return f"RSI overbought ({intraday.rsi:.0f}), taking profits"

        # 4. Price below VWAP + negative momentum
        if (
            intraday.vwap_position == "below"
            and intraday.momentum < -0.3
            and intraday.trend == "down"
        ):
            return f"Below VWAP with negative momentum ({intraday.momentum:.2f})"

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_regime_transition(
        current: Dict, previous: Optional[Dict]
    ) -> Optional[str]:
        """Return the new regime name on a transition, or None."""
        if previous is None:
            return None
        if current["regime"] != previous["regime"]:
            return current["regime"]
        return None
