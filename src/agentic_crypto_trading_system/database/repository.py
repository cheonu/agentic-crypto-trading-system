import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    TradeModel,
    PositionModel,
    OrderModel,
    RegimeEventModel,
    DebateTranscriptModel,
    SentimentScoreModel,
)


class BaseRepository:
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def commit(self):
        """Commit the current transaction."""
        await self.session.commit()

    async def rollback(self):
        """Rollback the current transaction."""
        await self.session.rollback()


class TradeRepository(BaseRepository):
    """Repository for trade operations."""

    async def create(self, **kwargs) -> TradeModel:
        """Create a new trade."""
        trade = TradeModel(**kwargs)
        self.session.add(trade)
        await self.session.flush()
        return trade

    async def get_by_id(self, trade_id: uuid.UUID) -> Optional[TradeModel]:
        """Get trade by ID."""
        result = await self.session.execute(
            select(TradeModel).where(TradeModel.id == trade_id)
        )
        return result.scalar_one_or_none()

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> List[TradeModel]:
        """Get trades by symbol."""
        result = await self.session.execute(
            select(TradeModel)
            .where(TradeModel.symbol == symbol)
            .order_by(TradeModel.closed_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class PositionRepository(BaseRepository):
    """Repository for position operations."""

    async def create(self, **kwargs) -> PositionModel:
        """Create a new position."""
        position = PositionModel(**kwargs)
        self.session.add(position)
        await self.session.flush()
        return position

    async def get_by_id(self, position_id: uuid.UUID) -> Optional[PositionModel]:
        """Get position by ID."""
        result = await self.session.execute(
            select(PositionModel).where(PositionModel.id == position_id)
        )
        return result.scalar_one_or_none()

    async def get_open_positions(self) -> List[PositionModel]:
        """Get all open positions."""
        result = await self.session.execute(
            select(PositionModel).where(PositionModel.status == "open")
        )
        return list(result.scalars().all())

    async def update_price(self, position_id: uuid.UUID, current_price: Decimal, unrealized_pnl: Decimal):
        """Update position current price and pnl."""
        await self.session.execute(
            update(PositionModel)
            .where(PositionModel.id == position_id)
            .values(current_price=current_price, unrealized_pnl=unrealized_pnl)
        )

    async def close_position(self, position_id: uuid.UUID):
        """Close a position."""
        await self.session.execute(
            update(PositionModel)
            .where(PositionModel.id == position_id)
            .values(status="closed", closed_at=datetime.utcnow())
        )


class OrderRepository(BaseRepository):
    """Repository for order operations."""

    async def create(self, **kwargs) -> OrderModel:
        """Create a new order."""
        order = OrderModel(**kwargs)
        self.session.add(order)
        await self.session.flush()
        return order

    async def get_by_id(self, order_id: uuid.UUID) -> Optional[OrderModel]:
        """Get order by ID."""
        result = await self.session.execute(
            select(OrderModel).where(OrderModel.id == order_id)
        )
        return result.scalar_one_or_none()

    async def update_status(self, order_id: uuid.UUID, status: str, filled_size: Decimal = None):
        """Update order status."""
        values = {"status": status}
        if filled_size is not None:
            values["filled_size"] = filled_size
        if status == "filled":
            values["filled_at"] = datetime.utcnow()
        await self.session.execute(
            update(OrderModel)
            .where(OrderModel.id == order_id)
            .values(**values)
        )


class RegimeEventRepository(BaseRepository):
    """Repository for regime event operations."""

    async def create(self, **kwargs) -> RegimeEventModel:
        """Create a new regime event."""
        event = RegimeEventModel(**kwargs)
        self.session.add(event)
        await self.session.flush()
        return event

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> List[RegimeEventModel]:
        """Get regime events by symbol."""
        result = await self.session.execute(
            select(RegimeEventModel)
            .where(RegimeEventModel.symbol == symbol)
            .order_by(RegimeEventModel.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


class SentimentRepository(BaseRepository):
    """Repository for sentiment score operations."""

    async def create(self, **kwargs) -> SentimentScoreModel:
        """Create a new sentiment score."""
        score = SentimentScoreModel(**kwargs)
        self.session.add(score)
        await self.session.flush()
        return score

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> List[SentimentScoreModel]:
        """Get sentiment scores by symbol."""
        result = await self.session.execute(
            select(SentimentScoreModel)
            .where(SentimentScoreModel.symbol == symbol)
            .order_by(SentimentScoreModel.timestamp.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
