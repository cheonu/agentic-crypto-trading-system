import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class TradeModel(Base):
    """Completed trade record."""
    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    position_id = Column(UUID(as_uuid=True), nullable=False)
    symbol = Column(String(20), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    exit_price = Column(Numeric(20, 8), nullable=False)
    size = Column(Numeric(20, 8), nullable=False)
    pnl = Column(Numeric(20, 8), nullable=False)
    pnl_pct = Column(Numeric(10, 4), nullable=False)
    commission = Column(Numeric(20, 8), nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    agent_id = Column(String(100), nullable=False)
    regime_at_entry = Column(String(20), nullable=False)
    regime_at_exit = Column(String(20), nullable=False)
    opened_at = Column(DateTime, nullable=False)
    closed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_trades_symbol", "symbol"),
        Index("idx_trades_agent", "agent_id"),
        Index("idx_trades_opened_at", "opened_at"),
    )


class PositionModel(Base):
    """Open trading position."""
    __tablename__ = "positions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    entry_price = Column(Numeric(20, 8), nullable=False)
    current_price = Column(Numeric(20, 8), nullable=False)
    size = Column(Numeric(20, 8), nullable=False)
    stop_loss = Column(Numeric(20, 8), nullable=False)
    take_profit = Column(Numeric(20, 8), nullable=True)
    unrealized_pnl = Column(Numeric(20, 8), nullable=False)
    agent_id = Column(String(100), nullable=False)
    opened_at = Column(DateTime, nullable=False)
    exchange = Column(String(50), nullable=False)
    status = Column(String(20), nullable=False, default="open")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_positions_status", "status"),
        Index("idx_positions_agent", "agent_id"),
    )


class OrderModel(Base):
    """Exchange order."""
    __tablename__ = "orders"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange_order_id = Column(String(100), nullable=True)
    symbol = Column(String(20), nullable=False)
    order_type = Column(String(10), nullable=False)
    side = Column(String(10), nullable=False)
    price = Column(Numeric(20, 8), nullable=True)
    size = Column(Numeric(20, 8), nullable=False)
    filled_size = Column(Numeric(20, 8), default=0)
    status = Column(String(20), nullable=False, default="pending")
    submitted_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_orders_status", "status"),
        Index("idx_orders_symbol", "symbol"),
    )


class RegimeEventModel(Base):
    """Market regime transition event."""
    __tablename__ = "regime_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    from_regime = Column(String(20), nullable=False)
    to_regime = Column(String(20), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_regime_events_symbol", "symbol"),
        Index("idx_regime_events_timestamp", "timestamp"),
    )


class DebateTranscriptModel(Base):
    """Debate session transcript."""
    __tablename__ = "debate_transcripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    proposal_id = Column(UUID(as_uuid=True), nullable=False)
    transcript = Column(JSONB, nullable=False)
    consensus = Column(JSONB, nullable=False)
    duration_seconds = Column(Numeric(10, 2), nullable=False)
    executed = Column(Boolean, nullable=False)
    trade_id = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_debate_session", "session_id"),
        Index("idx_debate_executed", "executed"),
    )


class SentimentScoreModel(Base):
    """News sentiment score."""
    __tablename__ = "sentiment_scores"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), nullable=False)
    score = Column(Numeric(5, 4), nullable=False)
    confidence = Column(Numeric(5, 4), nullable=False)
    magnitude = Column(Numeric(5, 4), nullable=False)
    source = Column(String(100), nullable=False)
    article_id = Column(String(200), nullable=True)
    keywords = Column(ARRAY(Text), nullable=True)
    timestamp = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_sentiment_symbol", "symbol"),
        Index("idx_sentiment_timestamp", "timestamp"),
    )
