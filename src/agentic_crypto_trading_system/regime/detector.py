import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .classifier import RegimeClassifier, RegimeResult, MarketRegime


class RegimeDetector:
    """Detects market regime transitions with confirmation logic."""

    def __init__(
        self,
        classifier: RegimeClassifier = None,
        confirmations_required: int = 3,
    ):
        self.classifier = classifier or RegimeClassifier()
        self.confirmations_required = confirmations_required

        self.current_regime: Optional[MarketRegime] = None
        self.pending_regime: Optional[MarketRegime] = None
        self.confirmation_count: int = 0
        self.regime_history: List[Dict] = []
        self.subscribers: List[Callable] = []

    def detect_regime(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> RegimeResult:
        """Detect current regime and check for transitions."""
        result = self.classifier.classify(highs, lows, closes, volumes)

        if self.current_regime is None:
            self.current_regime = result.regime
            self._record_transition(None, result.regime, result.confidence)
        elif result.regime != self.current_regime:
            if result.regime == self.pending_regime:
                self.confirmation_count += 1
            else:
                self.pending_regime = result.regime
                self.confirmation_count = 1

            if self.confirmation_count >= self.confirmations_required:
                old_regime = self.current_regime
                self.current_regime = result.regime
                self.pending_regime = None
                self.confirmation_count = 0
                self._record_transition(old_regime, result.regime, result.confidence)
        else:
            self.pending_regime = None
            self.confirmation_count = 0

        return result

    def _record_transition(
        self, from_regime: Optional[MarketRegime], to_regime: MarketRegime, confidence: float
    ) -> None:
        """Record a regime transition event."""
        event = {
            "from_regime": from_regime.value if from_regime else None,
            "to_regime": to_regime.value,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.regime_history.append(event)

    def get_regime_history(self) -> List[Dict]:
        """Get the history of regime transitions."""
        return self.regime_history

    def subscribe_regime_changes(self, callback: Callable) -> None:
        """Subscribe to regime change notifications."""
        self.subscribers.append(callback)

    async def notify_subscribers(self, event: Dict) -> None:
        """Notify all subscribers of a regime change."""
        for callback in self.subscribers:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
