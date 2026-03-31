import uuid
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_crypto_trading_system.database.models import (
    Base,
    TradeModel,
    PositionModel,
    OrderModel,
    RegimeEventModel,
)


def test_trade_model_creation():
    """Test TradeModel can be instantiated."""
    trade = TradeModel(
        id=uuid.uuid4(),
        position_id=uuid.uuid4(),
        symbol="BTC/USDT",
        entry_price=Decimal("50000"),
        exit_price=Decimal("51000"),
        size=Decimal("0.1"),
        pnl=Decimal("100"),
        pnl_pct=Decimal("2.0"),
        commission=Decimal("0.5"),
        duration_seconds=3600,
        agent_id="agent-1",
        regime_at_entry="bull",
        regime_at_exit="bull",
        opened_at=datetime.utcnow(),
        closed_at=datetime.utcnow(),
    )
    assert trade.symbol == "BTC/USDT"
    assert trade.pnl == Decimal("100")


def test_position_model_creation():
    """Test PositionModel can be instantiated."""
    position = PositionModel(
        id=uuid.uuid4(),
        symbol="ETH/USDT",
        direction="long",
        entry_price=Decimal("3000"),
        current_price=Decimal("3100"),
        size=Decimal("1.0"),
        stop_loss=Decimal("2900"),
        unrealized_pnl=Decimal("100"),
        agent_id="agent-2",
        opened_at=datetime.utcnow(),
        exchange="binance",
        status="open",
    )
    assert position.symbol == "ETH/USDT"
    assert position.direction == "long"
    assert position.status == "open"


def test_order_model_creation():
    """Test OrderModel can be instantiated."""
    order = OrderModel(
        id=uuid.uuid4(),
        symbol="BTC/USDT",
        order_type="market",
        side="buy",
        size=Decimal("0.5"),
        status="pending",
    )
    assert order.symbol == "BTC/USDT"
    assert order.order_type == "market"
    assert order.status == "pending"


def test_regime_event_model_creation():
    """Test RegimeEventModel can be instantiated."""
    event = RegimeEventModel(
        id=uuid.uuid4(),
        symbol="BTC/USDT",
        from_regime="sideways",
        to_regime="bull",
        confidence=Decimal("0.85"),
        timestamp=datetime.utcnow(),
    )
    assert event.from_regime == "sideways"
    assert event.to_regime == "bull"


def test_base_metadata_has_tables():
    """Test that all tables are registered in metadata."""
    table_names = Base.metadata.tables.keys()
    assert "trades" in table_names
    assert "positions" in table_names
    assert "orders" in table_names
    assert "regime_events" in table_names
    assert "debate_transcripts" in table_names
    assert "sentiment_scores" in table_names
