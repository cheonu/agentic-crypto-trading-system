import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from ..exchange.base import ExchangeConnector

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Request to place an order."""
    symbol: str
    side: str          # "buy" or "sell"
    order_type: str    # "market" or "limit"
    size: float
    price: Optional[float] = None
    agent_id: str = ""


@dataclass
class OrderResult:
    """Result of an order execution."""
    order_id: str
    status: str        # "filled", "partial", "failed", "cancelled"
    filled_size: float = 0.0
    filled_price: float = 0.0
    error: Optional[str] = None


class TradeExecutor:
    """Executes trades with retry logic and order splitting."""

    def __init__(
        self,
        exchange: ExchangeConnector,
        max_retries: int = 3,
        large_order_threshold: float = 10000.0,
        chunk_interval_seconds: float = 5.0,
        max_slippage_pct: float = 0.5,
    ):
        self.exchange = exchange
        self.max_retries = max_retries
        self.large_order_threshold = large_order_threshold
        self.chunk_interval_seconds = chunk_interval_seconds
        self.max_slippage_pct = max_slippage_pct
        self.order_history: List[Dict] = []

    async def execute_trade(self, request: OrderRequest) -> OrderResult:
        """Execute a trade, splitting large orders if needed."""
        trade_value = request.size * (request.price or 0)

        if trade_value > self.large_order_threshold and request.order_type == "market":
            return await self._execute_split_order(request)
        return await self._execute_single_order(request)

    async def _execute_single_order(self, request: OrderRequest) -> OrderResult:
        """Execute a single order with retry logic."""
        last_error = None
        backoff_ms = [100, 200, 400]

        for attempt in range(self.max_retries):
            try:
                order = await self.exchange.create_order(
                    symbol=request.symbol,
                    order_type=request.order_type,
                    side=request.side,
                    amount=request.size,
                    price=request.price,
                )
                result = OrderResult(
                    order_id=order.get("id", str(uuid.uuid4())),
                    status="filled",
                    filled_size=request.size,
                    filled_price=order.get("average", request.price or 0),
                )
                self._record_order(request, result, attempt + 1)
                return result

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = backoff_ms[attempt] / 1000.0
                    await asyncio.sleep(delay)

        result = OrderResult(
            order_id=str(uuid.uuid4()),
            status="failed",
            error=last_error,
        )
        self._record_order(request, result, self.max_retries)
        return result

    async def _execute_split_order(self, request: OrderRequest) -> OrderResult:
        """Split large orders into chunks."""
        chunk_count = max(2, int((request.size * (request.price or 1)) / self.large_order_threshold) + 1)
        chunk_size = request.size / chunk_count
        total_filled = 0.0
        total_cost = 0.0

        for i in range(chunk_count):
            chunk_request = OrderRequest(
                symbol=request.symbol,
                side=request.side,
                order_type=request.order_type,
                size=chunk_size,
                price=request.price,
                agent_id=request.agent_id,
            )
            result = await self._execute_single_order(chunk_request)

            if result.status == "failed":
                return OrderResult(
                    order_id=str(uuid.uuid4()),
                    status="partial" if total_filled > 0 else "failed",
                    filled_size=total_filled,
                    filled_price=total_cost / total_filled if total_filled > 0 else 0,
                    error=result.error,
                )

            total_filled += result.filled_size
            total_cost += result.filled_size * result.filled_price

            if i < chunk_count - 1:
                await asyncio.sleep(self.chunk_interval_seconds)

        return OrderResult(
            order_id=str(uuid.uuid4()),
            status="filled",
            filled_size=total_filled,
            filled_price=total_cost / total_filled if total_filled > 0 else 0,
        )

    def estimate_slippage(self, order_book: Dict, size: float, side: str) -> float:
        """Estimate slippage based on order book depth."""
        key = "asks" if side == "buy" else "bids"
        levels = order_book.get(key, [])
        if not levels:
            return 0.0

        best_price = levels[0][0]
        remaining = size
        weighted_price = 0.0

        for price, qty in levels:
            fill = min(remaining, qty)
            weighted_price += fill * price
            remaining -= fill
            if remaining <= 0:
                break

        if size - remaining == 0:
            return 0.0

        avg_price = weighted_price / (size - remaining)
        return abs(avg_price - best_price) / best_price * 100

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def _record_order(self, request: OrderRequest, result: OrderResult, attempts: int) -> None:
        """Record order execution details."""
        self.order_history.append({
            "symbol": request.symbol,
            "side": request.side,
            "type": request.order_type,
            "size": request.size,
            "status": result.status,
            "filled_size": result.filled_size,
            "filled_price": result.filled_price,
            "attempts": attempts,
            "error": result.error,
            "timestamp": datetime.utcnow().isoformat(),
        })
