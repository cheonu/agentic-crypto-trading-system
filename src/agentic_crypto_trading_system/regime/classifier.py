from enum import Enum
from dataclasses import dataclass
from typing import List

from .indicators import calculate_atr, calculate_adx, calculate_momentum, calculate_volume_profile


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeResult:
    """Result of a regime classification."""
    regime: MarketRegime
    confidence: float
    atr: float
    adx: float
    momentum: float
    volume_ratio: float


class RegimeClassifier:
    """Classifies market regime based on technical indicators."""

    def __init__(
        self,
        atr_high_threshold: float = 2.0,
        adx_trend_threshold: float = 25.0,
        momentum_threshold: float = 1.0,
    ):
        self.atr_high_threshold = atr_high_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.momentum_threshold = momentum_threshold

    def classify(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> RegimeResult:
        """Classify the current market regime."""
        atr = calculate_atr(highs, lows, closes)
        adx = calculate_adx(highs, lows, closes)
        momentum = calculate_momentum(closes)
        volume_ratio = calculate_volume_profile(volumes)

        current_price = closes[-1]
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        regime, confidence = self._apply_rules(atr_pct, adx, momentum, volume_ratio)

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            atr=atr_pct,
            adx=adx,
            momentum=momentum,
            volume_ratio=volume_ratio,
        )

    def _apply_rules(
        self, atr_pct: float, adx: float, momentum: float, volume_ratio: float
    ) -> tuple:
        """Apply classification rules to determine regime."""
        # High volatility takes priority
        if atr_pct > self.atr_high_threshold and volume_ratio > 1.5:
            confidence = min(0.95, 0.6 + (atr_pct - self.atr_high_threshold) * 0.1)
            return MarketRegime.HIGH_VOLATILITY, confidence

        # Strong trend with directional momentum
        if adx > self.adx_trend_threshold:
            if momentum > self.momentum_threshold:
                confidence = min(0.95, 0.5 + adx / 100 + momentum / 20)
                return MarketRegime.BULL, confidence
            elif momentum < -self.momentum_threshold:
                confidence = min(0.95, 0.5 + adx / 100 + abs(momentum) / 20)
                return MarketRegime.BEAR, confidence

        # Weak trend = sideways
        if adx < self.adx_trend_threshold:
            confidence = min(0.95, 0.5 + (self.adx_trend_threshold - adx) / 50)
            return MarketRegime.SIDEWAYS, confidence

        # Default: sideways with low confidence
        return MarketRegime.SIDEWAYS, 0.4
