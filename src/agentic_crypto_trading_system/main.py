"""Main application entry point — wires all components together.

Initializes services in dependency order, starts background tasks,
and runs the FastAPI server.
"""

import asyncio
import logging
import os
import signal
from typing import Optional

from .api import create_app
from .config import Config
from .observability.logging_config import setup_logging
from .observability.metrics import MetricsCollector
from .observability.alerts import AlertManager
from .state.manager import StateManager, SystemMode
from .emergency.controls import EmergencyController
from .analytics.service import AnalyticsService
from .reasoning.capture import ReasoningCapture
from .collaboration.message_bus import MessageBus
from .validation.service import ConfigValidator

logger = logging.getLogger(__name__)


class TradingSystem:
    """Main orchestrator that wires all components together."""

    def __init__(self, config_path: str = "config/default.yaml"):
        # Core config
        self.config = Config(config_path)

        # Observability
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()

        # State & emergency
        self.state_manager = StateManager(
            mode=SystemMode.DRY_RUN,
        )
        self.emergency = EmergencyController()

        # Analytics & reasoning
        self.analytics = AnalyticsService()
        self.reasoning = ReasoningCapture()

        # Collaboration
        self.message_bus = MessageBus()

        # Validation
        self.validator = ConfigValidator()

        # FastAPI app
        self.app = create_app()

        # Wire emergency callbacks
        self.emergency.on_halt(self._on_emergency_halt)
        self.emergency.on_notify(lambda event: self.alerts.fire_emergency(
            "Emergency Stop",
            event.reason,
            source="emergency_controller",
        ))

    def _on_emergency_halt(self) -> None:
        """Handle emergency halt across all components."""
        self.state_manager.emergency_stop("emergency_controller")
        logger.critical("All components halted via emergency controller")

    def start(self) -> None:
        """Start the trading system."""
        setup_logging(level="INFO", json_format=False)
        logger.info("Starting Agentic Crypto Trading System...")

        # Validate config
        # result = self.validator.validate_all(self.config.raw)
        # if not result.valid:
        #     logger.error(f"Invalid config: {result.errors}")
        #     return

        # Check for unclean shutdown
        if self.state_manager.detect_unclean_shutdown():
            logger.warning("Unclean shutdown detected — restoring last checkpoint")

        self.state_manager.start()
        logger.info("System ready")

    def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.state_manager.stop()
        logger.info("Shutdown complete")


def main():
    """CLI entry point."""
    system = TradingSystem()
    system.start()

    # In production, you'd run: uvicorn agentic_crypto_trading_system.main:app
    # For now, just confirm startup
    print("System started. Run with: uvicorn agentic_crypto_trading_system.api.app:create_app --factory")


if __name__ == "__main__":
    main()
