"""Intraday trend analysis using EMA crossovers, RSI, and VWAP.

Analyzes short-timeframe candles (5m, 15m) to detect intraday momentum
and micro-trends for the day trading strategy.
"""

import logging
from typing import Dict, List, Optional

from .models import IntradaySignals

logger = logging.getLogger(__name__)


class IntradayTrendAnalyzer:
    """Analyzes intraday market data to produce trend signals.

    Uses EMA(9)/EMA(21) crossovers, RSI(14), and VWAP to determine
    trend direction, momentum, and confidence.

    Args:
        short_ema_period: Period for the short EMA (default 9).
        long_ema_period: Period for the long EMA (default 21).
        rsi_period: Period for RSI calculation (default 14).
        vwap_enabled: Whether to calculate VWAP (default True).
    """

    def __init__(
        self,
        short_ema_period: int = 9,
        long_ema_period: int = 21,
        rsi_period: int = 14,
        vwap_enabled: bool = True,
    ) -> None:
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.rsi_period = rsi_period
        self.vwap_enabled = vwap_enabled

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average for a list of prices.

        Uses the standard EMA formula with multiplier = 2 / (period + 1).
        The first EMA value is the simple average of the first `period` prices.

        Args:
            prices: List of closing prices.
            period: EMA period.

        Returns:
            List of EMA values, same length as prices.
            Returns empty list if prices has fewer elements than period.
        """
        if len(prices) < period:
            return []

        multiplier = 2.0 / (period + 1)
        ema_values: List[float] = []

        # Seed with SMA of first `period` prices
        sma = sum(prices[:period]) / period
        ema_values.append(sma)

        # Calculate EMA for remaining prices
        for price in prices[period:]:
            ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)

        return ema_values

    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index.

        Uses the standard RSI formula:
        1. Calculate price changes between consecutive prices.
        2. Separate gains and losses.
        3. Compute average gain and average loss over the period.
        4. RS = average_gain / average_loss
        5. RSI = 100 - (100 / (1 + RS))

        Args:
            prices: List of closing prices (needs at least period + 1 values).
            period: RSI period (typically 14).

        Returns:
            RSI value between 0 and 100.
            Returns 50.0 if insufficient data.
        """
        if len(prices) < period + 1:
            return 50.0

        # Calculate price changes
        changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

        # Initial average gain/loss from first `period` changes
        gains = [max(c, 0.0) for c in changes[:period]]
        losses = [abs(min(c, 0.0)) for c in changes[:period]]

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        # Smooth with remaining changes (Wilder's smoothing)
        for change in changes[period:]:
            gain = max(change, 0.0)
            loss = abs(min(change, 0.0))
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0.0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _calculate_vwap(self, ohlcv: List[Dict]) -> float:
        """Calculate Volume Weighted Average Price from OHLCV data.

        VWAP = sum(typical_price * volume) / sum(volume)
        where typical_price = (high + low + close) / 3

        Args:
            ohlcv: List of candle dicts with keys "high", "low", "close", "volume".

        Returns:
            VWAP value. Returns 0.0 if no data or total volume is zero.
        """
        if not ohlcv:
            return 0.0

        total_tp_volume = 0.0
        total_volume = 0.0

        for candle in ohlcv:
            high = candle["high"]
            low = candle["low"]
            close = candle["close"]
            volume = candle["volume"]

            typical_price = (high + low + close) / 3.0
            total_tp_volume += typical_price * volume
            total_volume += volume

        if total_volume == 0.0:
            return 0.0

        return total_tp_volume / total_volume

    def _detect_ema_cross(
        self, short_ema: List[float], long_ema: List[float]
    ) -> Optional[str]:
        """Detect EMA crossover by comparing the last two values.

        A golden cross occurs when the short EMA crosses above the long EMA.
        A death cross occurs when the short EMA crosses below the long EMA.

        Args:
            short_ema: Short-period EMA values.
            long_ema: Long-period EMA values.

        Returns:
            "golden_cross", "death_cross", or None.
        """
        if len(short_ema) < 2 or len(long_ema) < 2:
            return None

        # Compare the last two aligned values
        prev_short = short_ema[-2]
        curr_short = short_ema[-1]
        prev_long = long_ema[-2]
        curr_long = long_ema[-1]

        # Golden cross: short was below long, now short is above long
        if prev_short <= prev_long and curr_short > curr_long:
            return "golden_cross"

        # Death cross: short was above long, now short is below long
        if prev_short >= prev_long and curr_short < curr_long:
            return "death_cross"

        return None

    def _determine_trend(self, short_ema: List[float], long_ema: List[float]) -> str:
        """Determine trend direction from EMA relationship.

        Args:
            short_ema: Short-period EMA values.
            long_ema: Long-period EMA values.

        Returns:
            "up", "down", or "sideways".
        """
        if not short_ema or not long_ema:
            return "sideways"

        diff = short_ema[-1] - long_ema[-1]
        # Use a small relative threshold to avoid noise
        threshold = long_ema[-1] * 0.001 if long_ema[-1] != 0 else 0.0

        if diff > threshold:
            return "up"
        elif diff < -threshold:
            return "down"
        return "sideways"

    def _calculate_momentum(self, rsi: float, ema_cross: Optional[str]) -> float:
        """Calculate momentum score from RSI and EMA cross.

        Maps RSI to a -1.0 to 1.0 scale and adjusts for EMA cross signals.

        Args:
            rsi: RSI value (0-100).
            ema_cross: EMA cross signal or None.

        Returns:
            Momentum score between -1.0 and 1.0.
        """
        # Map RSI 0-100 to -1.0 to 1.0 (50 = neutral)
        momentum = (rsi - 50.0) / 50.0

        # Boost momentum on cross signals
        if ema_cross == "golden_cross":
            momentum = min(momentum + 0.2, 1.0)
        elif ema_cross == "death_cross":
            momentum = max(momentum - 0.2, -1.0)

        return max(-1.0, min(1.0, momentum))

    def _determine_volume_trend(self, ohlcv: List[Dict]) -> str:
        """Determine volume trend from recent candles.

        Compares average volume of the recent half vs the earlier half.

        Args:
            ohlcv: List of candle dicts with "volume" key.

        Returns:
            "increasing", "decreasing", or "stable".
        """
        if len(ohlcv) < 4:
            return "stable"

        mid = len(ohlcv) // 2
        early_avg = sum(c["volume"] for c in ohlcv[:mid]) / mid
        recent_avg = sum(c["volume"] for c in ohlcv[mid:]) / (len(ohlcv) - mid)

        if early_avg == 0:
            return "stable"

        ratio = recent_avg / early_avg
        if ratio > 1.1:
            return "increasing"
        elif ratio < 0.9:
            return "decreasing"
        return "stable"

    def _calculate_confidence(
        self,
        trend: str,
        ema_cross: Optional[str],
        rsi: float,
        vwap_position: str,
    ) -> float:
        """Calculate confidence score for the intraday signals.

        Higher confidence when multiple indicators align.

        Args:
            trend: Trend direction.
            ema_cross: EMA cross signal.
            rsi: RSI value.
            vwap_position: VWAP position.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.5  # base confidence

        # EMA cross adds confidence
        if ema_cross is not None:
            score += 0.15

        # Strong RSI adds confidence
        if rsi > 70 or rsi < 30:
            score += 0.15

        # Trend alignment with VWAP
        if trend == "up" and vwap_position == "above":
            score += 0.1
        elif trend == "down" and vwap_position == "below":
            score += 0.1

        # Sideways trend reduces confidence
        if trend == "sideways":
            score -= 0.1

        return max(0.0, min(1.0, score))

    def analyze(self, market_data: Dict) -> IntradaySignals:
        """Analyze market data to produce intraday trend signals.

        Expects market_data to contain:
        - "candles": list of OHLCV dicts with keys
          "open", "high", "low", "close", "volume"
        - "current_price": current market price

        Args:
            market_data: Dict with "candles" and "current_price" keys.

        Returns:
            IntradaySignals with trend, momentum, VWAP position,
            EMA cross, RSI, volume trend, and confidence.
        """
        candles = market_data.get("candles", [])
        current_price = market_data.get("current_price", 0.0)

        # Extract closing prices
        close_prices = [c["close"] for c in candles]

        # Calculate EMAs
        short_ema = self._calculate_ema(close_prices, self.short_ema_period)
        long_ema = self._calculate_ema(close_prices, self.long_ema_period)

        # Detect EMA cross
        ema_cross = self._detect_ema_cross(short_ema, long_ema)

        # Calculate RSI
        rsi = self._calculate_rsi(close_prices, self.rsi_period)

        # Calculate VWAP
        vwap = 0.0
        vwap_position = "at"
        if self.vwap_enabled and candles:
            vwap = self._calculate_vwap(candles)
            if vwap > 0:
                price = current_price if current_price > 0 else close_prices[-1]
                relative_diff = (price - vwap) / vwap
                if relative_diff > 0.001:
                    vwap_position = "above"
                elif relative_diff < -0.001:
                    vwap_position = "below"
                else:
                    vwap_position = "at"

        # Determine trend and momentum
        trend = self._determine_trend(short_ema, long_ema)
        momentum = self._calculate_momentum(rsi, ema_cross)

        # Volume trend
        volume_trend = self._determine_volume_trend(candles)

        # Confidence
        confidence = self._calculate_confidence(trend, ema_cross, rsi, vwap_position)

        return IntradaySignals(
            trend=trend,
            momentum=momentum,
            vwap_position=vwap_position,
            ema_cross=ema_cross,
            rsi=rsi,
            volume_trend=volume_trend,
            confidence=confidence,
        )
