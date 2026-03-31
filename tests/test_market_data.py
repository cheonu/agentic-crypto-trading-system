import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from agentic_crypto_trading_system.exchange.models import OHLCV, Ticker, OrderBook
from agentic_crypto_trading_system.market_data import (
    MarketDataLayer,
    DataQualityMonitor,
)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange connector."""
    exchange = AsyncMock()
    return exchange


@pytest.fixture
def market_data(mock_exchange):
    """Create MarketDataLayer without Redis."""
    return MarketDataLayer(exchange=mock_exchange, redis_client=None)


@pytest.fixture
def data_quality():
    """Create DataQualityMonitor."""
    return DataQualityMonitor(max_staleness_seconds=10.0, max_price_change_pct=20.0)


# --- MarketDataLayer tests ---

@pytest.mark.asyncio
async def test_get_ohlcv(market_data, mock_exchange):
    """Test OHLCV retrieval."""
    mock_ohlcv = [
        OHLCV(
            timestamp=datetime.now(),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("100"),
            symbol="BTC/USDT",
            timeframe="1h",
        )
    ]
    mock_exchange.get_ohlcv.return_value = mock_ohlcv

    result = await market_data.get_ohlcv(
        "BTC/USDT", "1h", datetime.now() - timedelta(hours=1), datetime.now()
    )

    assert len(result) == 1
    assert result[0].symbol == "BTC/USDT"
    assert result[0].close == Decimal("50500")


@pytest.mark.asyncio
async def test_get_order_book(market_data, mock_exchange):
    """Test order book retrieval."""
    mock_book = OrderBook(
        symbol="BTC/USDT",
        bids=[(Decimal("50000"), Decimal("1.5"))],
        asks=[(Decimal("50001"), Decimal("2.0"))],
        timestamp=datetime.now(),
    )
    mock_exchange.get_order_book.return_value = mock_book

    result = await market_data.get_order_book("BTC/USDT")

    assert result.symbol == "BTC/USDT"
    assert len(result.bids) == 1


def test_data_freshness(market_data):
    """Test data freshness tracking."""
    assert market_data.get_data_freshness("ticker:BTC/USDT") is None


# --- DataQualityMonitor tests ---

def test_reliable_ticker(data_quality):
    """Test that normal ticker passes quality check."""
    ticker = Ticker(
        symbol="BTC/USDT",
        bid=Decimal("50000"),
        ask=Decimal("50001"),
        last=Decimal("50000"),
        volume_24h=Decimal("1000000"),
        timestamp=datetime.now(),
    )
    assert data_quality.check_ticker(ticker) is True


def test_stale_data_detection(data_quality):
    """Test stale data detection."""
    old_ticker = Ticker(
        symbol="BTC/USDT",
        bid=Decimal("50000"),
        ask=Decimal("50001"),
        last=Decimal("50000"),
        volume_24h=Decimal("1000000"),
        timestamp=datetime.now() - timedelta(seconds=30),
    )
    assert data_quality.check_ticker(old_ticker) is False
    assert not data_quality.is_symbol_reliable("BTC/USDT")


def test_anomalous_price_detection(data_quality):
    """Test anomalous price movement detection."""
    # First ticker sets baseline
    ticker1 = Ticker(
        symbol="ETH/USDT",
        bid=Decimal("3000"),
        ask=Decimal("3001"),
        last=Decimal("3000"),
        volume_24h=Decimal("500000"),
        timestamp=datetime.now(),
    )
    data_quality.check_ticker(ticker1)

    # Second ticker with 25% jump
    ticker2 = Ticker(
        symbol="ETH/USDT",
        bid=Decimal("3750"),
        ask=Decimal("3751"),
        last=Decimal("3750"),
        volume_24h=Decimal("500000"),
        timestamp=datetime.now(),
    )
    assert data_quality.check_ticker(ticker2) is False
    assert not data_quality.is_symbol_reliable("ETH/USDT")
