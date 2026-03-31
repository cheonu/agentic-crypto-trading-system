"""Alert system for critical events.

Generates alerts on errors, emergency stops, data quality issues.
Supports configurable alert channels.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """An alert event."""
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False


# Alert handler callback
AlertHandler = Callable[[Alert], None]


class AlertManager:
    """Manages alert generation and routing."""

    def __init__(self):
        self.handlers: List[AlertHandler] = []
        self.alert_history: List[Alert] = []
        self.suppressed_sources: set = set()

    def add_handler(self, handler: AlertHandler) -> None:
        """Register an alert handler (email, Slack, etc.)."""
        self.handlers.append(handler)

    def fire(self, alert: Alert) -> None:
        """Fire an alert to all handlers."""
        if alert.source in self.suppressed_sources:
            return

        self.alert_history.append(alert)
        logger.log(
            logging.CRITICAL if alert.severity == AlertSeverity.EMERGENCY else logging.WARNING,
            f"ALERT [{alert.severity.value}] {alert.title}: {alert.message}"
        )

        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def fire_emergency(self, title: str, message: str, source: str) -> None:
        """Convenience: fire an emergency alert."""
        self.fire(Alert(
            severity=AlertSeverity.EMERGENCY,
            title=title,
            message=message,
            source=source,
        ))

    def suppress(self, source: str) -> None:
        """Suppress alerts from a source."""
        self.suppressed_sources.add(source)

    def unsuppress(self, source: str) -> None:
        """Re-enable alerts from a source."""
        self.suppressed_sources.discard(source)

    def get_history(self, severity: Optional[AlertSeverity] = None, limit: int = 50) -> List[Alert]:
        """Get alert history, optionally filtered."""
        if severity:
            filtered = [a for a in self.alert_history if a.severity == severity]
        else:
            filtered = self.alert_history
        return filtered[-limit:]
