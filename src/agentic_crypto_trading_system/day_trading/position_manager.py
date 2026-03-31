"""Position manager for the day trading strategy.

Tracks open positions, calculates unrealized/realized P&L, and maintains
trade history across 5-minute trading cycles. Enforces one open position
per symbol at any time. Supports JSON persistence for crash recovery.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import ClosedTrade, OpenPosition

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages open positions and trade history in memory.

    Enforces a single open position per symbol. Calculates realized P&L
    on close and unrealized P&L on demand.
    """

    def __init__(self, state_filepath: Optional[str] = None) -> None:
        self._positions: Dict[str, OpenPosition] = {}
        self._trade_history: List[ClosedTrade] = []
        self._state_filepath: Optional[str] = state_filepath
        if state_filepath:
            self.load_state(state_filepath)

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
    ) -> OpenPosition:
        """Open a new position for the given symbol.

        Raises ValueError if a position is already open for the symbol.
        The OpenPosition dataclass validates entry_price, size, side, and
        stop_loss_price constraints.
        """
        if symbol in self._positions:
            raise ValueError(
                f"Position already open for {symbol}. "
                "Close the existing position before opening a new one."
            )

        position = OpenPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            entry_time=datetime.now(UTC),
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            unrealized_pnl=0.0,
            highest_price_since_entry=entry_price,
        )
        self._positions[symbol] = position
        self._auto_save()
        return position

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "manual",
    ) -> ClosedTrade:
        """Close an open position and record the trade.

        Calculates realized P&L as:
        - Long: (exit_price - entry_price) × size
        - Short: (entry_price - exit_price) × size

        Raises KeyError if no open position exists for the symbol.
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol}")

        pos = self._positions.pop(symbol)
        now = datetime.now(UTC)

        if pos.side == "long":
            realized_pnl = (exit_price - pos.entry_price) * pos.size
        else:
            realized_pnl = (pos.entry_price - exit_price) * pos.size

        realized_pnl_pct = realized_pnl / (pos.entry_price * pos.size)

        trade = ClosedTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=now,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            exit_reason=exit_reason,
        )
        self._trade_history.append(trade)
        self._auto_save()
        return trade

    def get_position(self, symbol: str) -> Optional[OpenPosition]:
        """Return the open position for a symbol, or None if none exists."""
        return self._positions.get(symbol)

    def has_open_position(self, symbol: str) -> bool:
        """Return True if an open position exists for the symbol."""
        return symbol in self._positions

    def update_unrealized_pnl(
        self, symbol: str, current_price: float
    ) -> float:
        """Update and return the unrealized P&L for an open position.

        Long: (current_price - entry_price) × size
        Short: (entry_price - current_price) × size

        Raises KeyError if no open position exists for the symbol.
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol}")

        pos = self._positions[symbol]
        if pos.side == "long":
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
        else:
            pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
        return pos.unrealized_pnl

    def get_all_positions(self) -> Dict[str, OpenPosition]:
        """Return a copy of all open positions keyed by symbol."""
        return dict(self._positions)

    def get_trade_history(self) -> List[ClosedTrade]:
        """Return a copy of the completed trade history."""
        return list(self._trade_history)

    def save_state(self, filepath: str) -> None:
        """Serialize positions and trade history to a JSON file.

        Datetimes are stored in ISO 8601 format for portability.
        """
        state: Dict[str, Any] = {
            "positions": {
                symbol: self._position_to_dict(pos)
                for symbol, pos in self._positions.items()
            },
            "trade_history": [
                self._trade_to_dict(trade) for trade in self._trade_history
            ],
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))

    def load_state(self, filepath: str) -> None:
        """Deserialize positions and trade history from a JSON file.

        If the file is missing or corrupted, initializes with empty state
        and logs a warning.
        """
        path = Path(filepath)
        if not path.exists():
            logger.warning(
                "State file %s not found. Initializing with empty state.",
                filepath,
            )
            return

        try:
            raw = json.loads(path.read_text())
            self._positions = {
                symbol: self._dict_to_position(d)
                for symbol, d in raw.get("positions", {}).items()
            }
            self._trade_history = [
                self._dict_to_trade(d)
                for d in raw.get("trade_history", [])
            ]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Corrupted state file %s: %s. Initializing with empty state.",
                filepath,
                exc,
            )
            self._positions = {}
            self._trade_history = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        """Persist state if a state_filepath was configured."""
        if self._state_filepath:
            self.save_state(self._state_filepath)

    @staticmethod
    def _position_to_dict(pos: OpenPosition) -> Dict[str, Any]:
        return {
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "size": pos.size,
            "entry_time": pos.entry_time.isoformat(),
            "stop_loss_price": pos.stop_loss_price,
            "take_profit_price": pos.take_profit_price,
            "unrealized_pnl": pos.unrealized_pnl,
            "highest_price_since_entry": pos.highest_price_since_entry,
        }

    @staticmethod
    def _dict_to_position(d: Dict[str, Any]) -> OpenPosition:
        return OpenPosition(
            symbol=d["symbol"],
            side=d["side"],
            entry_price=d["entry_price"],
            size=d["size"],
            entry_time=datetime.fromisoformat(d["entry_time"]),
            stop_loss_price=d["stop_loss_price"],
            take_profit_price=d.get("take_profit_price"),
            unrealized_pnl=d.get("unrealized_pnl", 0.0),
            highest_price_since_entry=d.get(
                "highest_price_since_entry", d["entry_price"]
            ),
        )

    @staticmethod
    def _trade_to_dict(trade: ClosedTrade) -> Dict[str, Any]:
        return {
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "size": trade.size,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "realized_pnl": trade.realized_pnl,
            "realized_pnl_pct": trade.realized_pnl_pct,
            "exit_reason": trade.exit_reason,
        }

    @staticmethod
    def _dict_to_trade(d: Dict[str, Any]) -> ClosedTrade:
        return ClosedTrade(
            symbol=d["symbol"],
            side=d["side"],
            entry_price=d["entry_price"],
            exit_price=d["exit_price"],
            size=d["size"],
            entry_time=datetime.fromisoformat(d["entry_time"]),
            exit_time=datetime.fromisoformat(d["exit_time"]),
            realized_pnl=d["realized_pnl"],
            realized_pnl_pct=d["realized_pnl_pct"],
            exit_reason=d["exit_reason"],
        )
