"""Unit tests for PositionManager core logic.

Validates Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import pytest

from agentic_crypto_trading_system.day_trading.position_manager import PositionManager


class TestOpenPosition:
    """Tests for PositionManager.open_position()."""

    def test_open_long_position(self):
        """Req 1.1: records symbol, side, entry price, size, stop-loss, take-profit."""
        pm = PositionManager()
        pos = pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0, 53000.0)

        assert pos.symbol == "BTC/USDT"
        assert pos.side == "long"
        assert pos.entry_price == 50000.0
        assert pos.size == 0.1
        assert pos.stop_loss_price == 48500.0
        assert pos.take_profit_price == 53000.0
        assert pos.unrealized_pnl == 0.0
        assert pos.highest_price_since_entry == 50000.0

    def test_open_short_position(self):
        pm = PositionManager()
        pos = pm.open_position("ETH/USDT", "short", 3000.0, 1.0, 3100.0)

        assert pos.side == "short"
        assert pos.stop_loss_price == 3100.0
        assert pos.take_profit_price is None

    def test_duplicate_position_rejected(self):
        """Req 1.6: only one open position per symbol."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)

        with pytest.raises(ValueError, match="Position already open"):
            pm.open_position("BTC/USDT", "long", 51000.0, 0.2, 49000.0)

    def test_different_symbols_allowed(self):
        """Req 1.6: one position per symbol, but different symbols are fine."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm.open_position("ETH/USDT", "long", 3000.0, 1.0, 2900.0)

        assert pm.has_open_position("BTC/USDT")
        assert pm.has_open_position("ETH/USDT")


class TestClosePosition:
    """Tests for PositionManager.close_position()."""

    def test_close_long_profit(self):
        """Req 1.4: realized P&L = (exit - entry) × size for longs."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        trade = pm.close_position("BTC/USDT", 52000.0)

        assert trade.realized_pnl == pytest.approx((52000.0 - 50000.0) * 0.1)
        assert trade.realized_pnl == pytest.approx(200.0)
        assert trade.exit_reason == "manual"
        assert not pm.has_open_position("BTC/USDT")

    def test_close_long_loss(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        trade = pm.close_position("BTC/USDT", 49000.0)

        assert trade.realized_pnl == pytest.approx(-100.0)

    def test_close_short_profit(self):
        """Short P&L = (entry - exit) × size."""
        pm = PositionManager()
        pm.open_position("ETH/USDT", "short", 3000.0, 1.0, 3100.0)
        trade = pm.close_position("ETH/USDT", 2800.0)

        assert trade.realized_pnl == pytest.approx(200.0)

    def test_close_nonexistent_raises(self):
        pm = PositionManager()
        with pytest.raises(KeyError, match="No open position"):
            pm.close_position("BTC/USDT", 50000.0)

    def test_close_with_exit_reason(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        trade = pm.close_position("BTC/USDT", 52000.0, exit_reason="stop_loss")

        assert trade.exit_reason == "stop_loss"

    def test_realized_pnl_pct(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        trade = pm.close_position("BTC/USDT", 52000.0)

        # pnl_pct = realized_pnl / (entry_price * size) = 200 / 5000 = 0.04
        assert trade.realized_pnl_pct == pytest.approx(0.04)


class TestGetPosition:
    """Tests for get_position() and has_open_position()."""

    def test_get_existing_position(self):
        """Req 1.2: returns OpenPosition for symbol with open position."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pos = pm.get_position("BTC/USDT")

        assert pos is not None
        assert pos.symbol == "BTC/USDT"

    def test_get_nonexistent_returns_none(self):
        """Req 1.3: returns None for symbol with no open position."""
        pm = PositionManager()
        assert pm.get_position("BTC/USDT") is None

    def test_has_open_position_true(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        assert pm.has_open_position("BTC/USDT") is True

    def test_has_open_position_false(self):
        pm = PositionManager()
        assert pm.has_open_position("BTC/USDT") is False


class TestUnrealizedPnl:
    """Tests for update_unrealized_pnl()."""

    def test_long_unrealized_profit(self):
        """Req 1.5: updates unrealized P&L with current price."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pnl = pm.update_unrealized_pnl("BTC/USDT", 51000.0)

        assert pnl == pytest.approx(100.0)
        assert pm.get_position("BTC/USDT").unrealized_pnl == pytest.approx(100.0)

    def test_short_unrealized_profit(self):
        pm = PositionManager()
        pm.open_position("ETH/USDT", "short", 3000.0, 1.0, 3100.0)
        pnl = pm.update_unrealized_pnl("ETH/USDT", 2900.0)

        assert pnl == pytest.approx(100.0)

    def test_unrealized_pnl_nonexistent_raises(self):
        pm = PositionManager()
        with pytest.raises(KeyError):
            pm.update_unrealized_pnl("BTC/USDT", 50000.0)


class TestCollections:
    """Tests for get_all_positions() and get_trade_history()."""

    def test_get_all_positions(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm.open_position("ETH/USDT", "long", 3000.0, 1.0, 2900.0)

        positions = pm.get_all_positions()
        assert len(positions) == 2
        assert "BTC/USDT" in positions
        assert "ETH/USDT" in positions

    def test_get_all_positions_returns_copy(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        positions = pm.get_all_positions()
        positions.pop("BTC/USDT")

        assert pm.has_open_position("BTC/USDT")

    def test_trade_history_after_close(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm.close_position("BTC/USDT", 52000.0)

        history = pm.get_trade_history()
        assert len(history) == 1
        assert history[0].symbol == "BTC/USDT"

    def test_trade_history_returns_copy(self):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm.close_position("BTC/USDT", 52000.0)
        history = pm.get_trade_history()
        history.clear()

        assert len(pm.get_trade_history()) == 1

    def test_empty_state(self):
        pm = PositionManager()
        assert pm.get_all_positions() == {}
        assert pm.get_trade_history() == []


import json
import os
import tempfile

from agentic_crypto_trading_system.day_trading.models import OpenPosition


class TestSaveState:
    """Tests for PositionManager.save_state()."""

    def test_save_creates_file(self, tmp_path):
        """Req 2.1: persists state to JSON on disk."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        filepath = str(tmp_path / "state.json")
        pm.save_state(filepath)

        assert os.path.exists(filepath)
        data = json.loads(open(filepath).read())
        assert "positions" in data
        assert "trade_history" in data
        assert "BTC/USDT" in data["positions"]

    def test_save_includes_trade_history(self, tmp_path):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm.close_position("BTC/USDT", 52000.0)
        filepath = str(tmp_path / "state.json")
        pm.save_state(filepath)

        data = json.loads(open(filepath).read())
        assert len(data["trade_history"]) == 1
        assert data["trade_history"][0]["symbol"] == "BTC/USDT"

    def test_save_creates_parent_dirs(self, tmp_path):
        filepath = str(tmp_path / "nested" / "dir" / "state.json")
        pm = PositionManager()
        pm.save_state(filepath)

        assert os.path.exists(filepath)

    def test_save_datetime_iso_format(self, tmp_path):
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        filepath = str(tmp_path / "state.json")
        pm.save_state(filepath)

        data = json.loads(open(filepath).read())
        entry_time = data["positions"]["BTC/USDT"]["entry_time"]
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(entry_time)


class TestLoadState:
    """Tests for PositionManager.load_state()."""

    def test_load_restores_positions(self, tmp_path):
        """Req 2.2: loads previously persisted state."""
        pm1 = PositionManager()
        pm1.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0, 53000.0)
        filepath = str(tmp_path / "state.json")
        pm1.save_state(filepath)

        pm2 = PositionManager()
        pm2.load_state(filepath)

        assert pm2.has_open_position("BTC/USDT")
        pos = pm2.get_position("BTC/USDT")
        assert pos.entry_price == 50000.0
        assert pos.size == 0.1
        assert pos.stop_loss_price == 48500.0
        assert pos.take_profit_price == 53000.0

    def test_load_restores_trade_history(self, tmp_path):
        pm1 = PositionManager()
        pm1.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm1.close_position("BTC/USDT", 52000.0, exit_reason="stop_loss")
        filepath = str(tmp_path / "state.json")
        pm1.save_state(filepath)

        pm2 = PositionManager()
        pm2.load_state(filepath)

        history = pm2.get_trade_history()
        assert len(history) == 1
        assert history[0].exit_reason == "stop_loss"
        assert history[0].realized_pnl == pytest.approx(200.0)

    def test_load_missing_file_empty_state(self, tmp_path):
        """Req 2.3: missing file initializes empty state."""
        pm = PositionManager()
        pm.load_state(str(tmp_path / "nonexistent.json"))

        assert pm.get_all_positions() == {}
        assert pm.get_trade_history() == []

    def test_load_corrupted_file_empty_state(self, tmp_path):
        """Req 2.3: corrupted file initializes empty state."""
        filepath = str(tmp_path / "bad.json")
        with open(filepath, "w") as f:
            f.write("not valid json {{{")

        pm = PositionManager()
        pm.load_state(filepath)

        assert pm.get_all_positions() == {}
        assert pm.get_trade_history() == []

    def test_load_invalid_data_empty_state(self, tmp_path):
        """Req 2.3: valid JSON but invalid data initializes empty state."""
        filepath = str(tmp_path / "bad_data.json")
        with open(filepath, "w") as f:
            json.dump({"positions": {"BTC": {"bad": "data"}}}, f)

        pm = PositionManager()
        pm.load_state(filepath)

        assert pm.get_all_positions() == {}
        assert pm.get_trade_history() == []


class TestAutoSave:
    """Tests for auto-save on open_position() and close_position()."""

    def test_auto_save_on_open(self, tmp_path):
        """Req 2.1: auto-saves when position is opened."""
        filepath = str(tmp_path / "state.json")
        pm = PositionManager(state_filepath=filepath)
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)

        assert os.path.exists(filepath)
        data = json.loads(open(filepath).read())
        assert "BTC/USDT" in data["positions"]

    def test_auto_save_on_close(self, tmp_path):
        """Req 2.1: auto-saves when position is closed."""
        filepath = str(tmp_path / "state.json")
        pm = PositionManager(state_filepath=filepath)
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)
        pm.close_position("BTC/USDT", 52000.0)

        data = json.loads(open(filepath).read())
        assert len(data["positions"]) == 0
        assert len(data["trade_history"]) == 1

    def test_no_auto_save_without_filepath(self, tmp_path):
        """No file created when state_filepath is not set."""
        pm = PositionManager()
        pm.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)

        # No file should exist in tmp_path
        assert len(list(tmp_path.iterdir())) == 0

    def test_init_loads_existing_state(self, tmp_path):
        """Req 2.2: constructor loads state from filepath if file exists."""
        filepath = str(tmp_path / "state.json")
        pm1 = PositionManager(state_filepath=filepath)
        pm1.open_position("BTC/USDT", "long", 50000.0, 0.1, 48500.0)

        pm2 = PositionManager(state_filepath=filepath)
        assert pm2.has_open_position("BTC/USDT")

    def test_init_missing_file_starts_empty(self, tmp_path):
        """Req 2.3: constructor with missing file starts empty."""
        filepath = str(tmp_path / "nonexistent.json")
        pm = PositionManager(state_filepath=filepath)

        assert pm.get_all_positions() == {}
        assert pm.get_trade_history() == []
