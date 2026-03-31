"""Emergency controls — halt trading, cancel orders, notify operators.

Configurable triggers for automatic emergency stops based on
drawdown thresholds, error rates, or manual intervention.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    MANUAL = "manual"
    DRAWDOWN = "drawdown"
    ERROR_RATE = "error_rate"
    DATA_QUALITY = "data_quality"


@dataclass
class EmergencyTrigger:
    """Configuration for an automatic emergency trigger."""
    trigger_type: TriggerType
    threshold: float  # e.g., 0.10 for 10% drawdown
    enabled: bool = True
    description: str = ""


@dataclass
class EmergencyEvent:
    """Record of an emergency stop event."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trigger_type: TriggerType = TriggerType.MANUAL
    reason: str = ""
    cancelled_orders: int = 0
    resume_approved: bool = False
    approved_by: Optional[str] = None


class EmergencyController:
    """Controls emergency stop logic and resume approval."""

    def __init__(self, triggers: Optional[List[EmergencyTrigger]] = None):
        self.triggers = triggers or [
            EmergencyTrigger(TriggerType.DRAWDOWN, threshold=0.10, description="10% drawdown"),
            EmergencyTrigger(TriggerType.ERROR_RATE, threshold=0.50, description="50% error rate"),
        ]
        self.is_halted = False
        self.events: List[EmergencyEvent] = []
        self._halt_callbacks: List[Callable] = []
        self._notify_callbacks: List[Callable[[EmergencyEvent], None]] = []

    def on_halt(self, callback: Callable) -> None:
        """Register callback for when emergency stop fires."""
        self._halt_callbacks.append(callback)

    def on_notify(self, callback: Callable[[EmergencyEvent], None]) -> None:
        """Register notification callback (email, Slack, etc.)."""
        self._notify_callbacks.append(callback)

    def trigger_emergency(
        self,
        trigger_type: TriggerType = TriggerType.MANUAL,
        reason: str = "Manual emergency stop",
    ) -> EmergencyEvent:
        """Trigger an emergency stop."""
        self.is_halted = True

        event = EmergencyEvent(
            trigger_type=trigger_type,
            reason=reason,
        )
        self.events.append(event)

        logger.critical(f"EMERGENCY STOP [{trigger_type.value}]: {reason}")

        for callback in self._halt_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Halt callback failed: {e}")

        for callback in self._notify_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Notify callback failed: {e}")

        return event

    def check_triggers(
        self,
        current_drawdown: float = 0.0,
        error_rate: float = 0.0,
    ) -> Optional[EmergencyEvent]:
        """Check if any automatic triggers should fire."""
        if self.is_halted:
            return None

        for trigger in self.triggers:
            if not trigger.enabled:
                continue

            if trigger.trigger_type == TriggerType.DRAWDOWN and current_drawdown >= trigger.threshold:
                return self.trigger_emergency(
                    TriggerType.DRAWDOWN,
                    f"Drawdown {current_drawdown:.1%} exceeded threshold {trigger.threshold:.1%}",
                )
            elif trigger.trigger_type == TriggerType.ERROR_RATE and error_rate >= trigger.threshold:
                return self.trigger_emergency(
                    TriggerType.ERROR_RATE,
                    f"Error rate {error_rate:.1%} exceeded threshold {trigger.threshold:.1%}",
                )

        return None

    def approve_resume(self, approved_by: str) -> bool:
        """Approve resuming trading after emergency stop.

        Requires explicit approval — system won't auto-resume.
        """
        if not self.is_halted:
            return True

        if self.events:
            self.events[-1].resume_approved = True
            self.events[-1].approved_by = approved_by

        self.is_halted = False
        logger.info(f"Trading resumed. Approved by: {approved_by}")
        return True

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return not self.is_halted

    def get_events(self, limit: int = 20) -> List[EmergencyEvent]:
        """Get emergency event history."""
        return self.events[-limit:]
