"""Trading session manager for identifying active market sessions.

Classifies the current UTC time into a trading session (asian, european,
us, eu_us_overlap, quiet) and provides adjusted confidence thresholds
for the DayTradingStrategy.

Session windows (UTC):
- Asian:         00:00–08:00
- European:      07:00–16:00
- US:            13:00–21:00
- EU/US Overlap: 13:00–16:00 (intersection of European and US)
- Quiet:         outside all windows (21:00–00:00)

Priority: eu_us_overlap > us > european > asian > quiet
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SessionInfo:
    """Information about the current trading session.

    Attributes:
        name: Session identifier — one of "asian", "european", "us",
              "eu_us_overlap", or "quiet".
        confidence_threshold: Minimum confidence required for trade signals.
              Defaults to 0.5 for active sessions, higher for quiet hours.
    """

    name: str
    confidence_threshold: float


class TradingSessionManager:
    """Identifies the current trading session and adjusts confidence thresholds.

    During quiet hours the strategy should require higher confidence before
    producing BUY or SELL signals, reducing noise trades in low-liquidity
    periods.

    Args:
        quiet_confidence_threshold: Confidence threshold used during quiet
            sessions. Defaults to 0.7.
        default_confidence_threshold: Confidence threshold used during all
            other sessions. Defaults to 0.5.
    """

    def __init__(
        self,
        quiet_confidence_threshold: float = 0.7,
        default_confidence_threshold: float = 0.5,
    ) -> None:
        self.quiet_confidence_threshold = quiet_confidence_threshold
        self.default_confidence_threshold = default_confidence_threshold

    def get_current_session(
        self, utc_time: Optional[datetime] = None
    ) -> SessionInfo:
        """Classify the trading session for the given UTC time.

        Args:
            utc_time: A datetime to classify. If ``None``, the current
                UTC time is used.

        Returns:
            A :class:`SessionInfo` with the session name and the
            applicable confidence threshold.
        """
        if utc_time is None:
            utc_time = datetime.now(timezone.utc)

        hour = utc_time.hour

        session_name = self._classify_session(hour)

        if session_name == "quiet":
            threshold = self.quiet_confidence_threshold
        else:
            threshold = self.default_confidence_threshold

        return SessionInfo(name=session_name, confidence_threshold=threshold)

    @staticmethod
    def _classify_session(hour: int) -> str:
        """Return the session name for a given UTC hour.

        Priority order: eu_us_overlap > us > european > asian > quiet.
        """
        in_asian = 0 <= hour < 8
        in_european = 7 <= hour < 16
        in_us = 13 <= hour < 21

        # Overlap is the intersection of European and US windows
        if in_european and in_us:
            return "eu_us_overlap"
        if in_us:
            return "us"
        if in_european:
            return "european"
        if in_asian:
            return "asian"
        return "quiet"
