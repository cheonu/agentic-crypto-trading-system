"""System state management — shutdown, emergency stop, checkpointing.

Handles graceful shutdown, emergency halt, state persistence,
and recovery from crashes. Supports dry-run and paper trading modes.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"       # Simulated execution
    DRY_RUN = "dry_run"   # No orders at all
    STOPPED = "stopped"
    EMERGENCY = "emergency"


@dataclass
class SystemState:
    """Snapshot of the system state."""
    mode: SystemMode = SystemMode.DRY_RUN
    is_running: bool = False
    positions: Dict[str, Any] = field(default_factory=dict)
    pending_orders: List[Dict[str, Any]] = field(default_factory=list)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    last_checkpoint: Optional[str] = None
    uptime_seconds: float = 0.0


class StateManager:
    """Manages system lifecycle and state persistence."""

    def __init__(
        self,
        checkpoint_dir: str = "data/checkpoints",
        checkpoint_interval_seconds: int = 60,
        mode: SystemMode = SystemMode.DRY_RUN,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval_seconds
        self.state = SystemState(mode=mode)
        self._start_time: Optional[float] = None
        self._emergency_callbacks: List = []

    def start(self) -> None:
        """Start the system."""
        self.state.is_running = True
        self._start_time = time.time()
        logger.info(f"System started in {self.state.mode.value} mode")

    def stop(self) -> None:
        """Graceful shutdown — save state and stop."""
        logger.info("Graceful shutdown initiated")
        self.save_checkpoint("shutdown")
        self.state.is_running = False
        self.state.mode = SystemMode.STOPPED
        logger.info("System stopped gracefully")

    def emergency_stop(self, reason: str = "manual") -> None:
        """Emergency halt — stop immediately."""
        logger.critical(f"EMERGENCY STOP: {reason}")
        self.state.mode = SystemMode.EMERGENCY
        self.state.is_running = False

        # Clear pending orders
        cancelled = len(self.state.pending_orders)
        self.state.pending_orders = []

        # Notify callbacks
        for callback in self._emergency_callbacks:
            try:
                callback(reason)
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")

        self.save_checkpoint(f"emergency_{reason}")
        logger.critical(f"Emergency stop complete. Cancelled {cancelled} pending orders.")

    def on_emergency(self, callback) -> None:
        """Register a callback for emergency stop events."""
        self._emergency_callbacks.append(callback)

    def save_checkpoint(self, label: str = "auto") -> str:
        """Save current state to disk."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self._start_time:
            self.state.uptime_seconds = time.time() - self._start_time

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{label}_{timestamp}.json"
        filepath = os.path.join(self.checkpoint_dir, filename)

        state_dict = asdict(self.state)
        state_dict["mode"] = self.state.mode.value

        with open(filepath, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

        self.state.last_checkpoint = filepath
        logger.info(f"Checkpoint saved: {filepath}")
        return filepath

    def restore_checkpoint(self, filepath: str) -> bool:
        """Restore state from a checkpoint file."""
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint not found: {filepath}")
            return False

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            self.state.mode = SystemMode(data.get("mode", "dry_run"))
            self.state.positions = data.get("positions", {})
            self.state.pending_orders = data.get("pending_orders", [])
            self.state.agent_states = data.get("agent_states", {})
            self.state.last_checkpoint = filepath

            logger.info(f"Restored from checkpoint: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False

    def detect_unclean_shutdown(self) -> bool:
        """Check if the last shutdown was unclean."""
        if not os.path.exists(self.checkpoint_dir):
            return False

        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".json")]
        )
        if not checkpoints:
            return False

        latest = checkpoints[-1]
        # If latest checkpoint isn't a shutdown checkpoint, it was unclean
        return "shutdown" not in latest

    def get_state(self) -> SystemState:
        """Get current system state."""
        if self._start_time:
            self.state.uptime_seconds = time.time() - self._start_time
        return self.state

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        return (
            self.state.is_running
            and self.state.mode in (SystemMode.LIVE, SystemMode.PAPER)
        )
