import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_crypto_trading_system.execution.executor import TradeExecutor, OrderRequest


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.create_order = AsyncMock(return_value={
        "id": "order-123",
        "average": 50000.0,
        "status": "closed",
    })
    exchange.cancel_order = AsyncMock()
    return exchange


@pytest.fixture
def executor(mock_exchange):
    return TradeExecutor(exchange=mock_exchange)


@pytest.mark.asyncio
async def test_execute_market_order(executor):
    """Test executing a simple market order."""
    request = OrderRequest(
        symbol="BTC/USDT", side="buy",
        order_type="market", size=0.1,
        price=50000.0,
    )
    result = await executor.execute_trade(request)
    assert result.status == "filled"
    assert result.filled_size == 0.1


@pytest.mark.asyncio
async def test_execute_order_retry(mock_exchange):
    """Test retry logic on failure."""
    mock_exchange.create_order = AsyncMock(
        side_effect=[Exception("timeout"), Exception("timeout"), {"id": "ok", "average": 50000}]
    )
    executor = TradeExecutor(exchange=mock_exchange, max_retries=3)
    request = OrderRequest(
        symbol="BTC/USDT", side="buy",
        order_type="market", size=0.1, price=50000.0,
    )
    result = await executor.execute_trade(request)
    assert result.status == "filled"
    assert mock_exchange.create_order.call_count == 3


@pytest.mark.asyncio
async def test_execute_order_all_retries_fail(mock_exchange):
    """Test all retries exhausted."""
    mock_exchange.create_order = AsyncMock(side_effect=Exception("network error"))
    executor = TradeExecutor(exchange=mock_exchange, max_retries=3)
    request = OrderRequest(
        symbol="BTC/USDT", side="buy",
        order_type="market", size=0.1, price=50000.0,
    )
    result = await executor.execute_trade(request)
    assert result.status == "failed"
    assert result.error is not None


def test_estimate_slippage(executor):
    """Test slippage estimation."""
    order_book = {
        "asks": [[50000, 1.0], [50010, 2.0], [50050, 3.0]],
        "bids": [[49990, 1.0], [49980, 2.0]],
    }
    slippage = executor.estimate_slippage(order_book, size=2.0, side="buy")
    assert slippage >= 0


@pytest.mark.asyncio
async def test_cancel_order(executor):
    """Test order cancellation."""
    result = await executor.cancel_order("BTC/USDT", "order-123")
    assert result is True
