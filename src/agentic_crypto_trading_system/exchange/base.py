from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, AsyncIterator
from .models import OHLCV, Ticker, OrderBook, Trade, OrderSide, OrderType

class ExchangeConnector(ABC):
    """Abstract base class for exchange connectors."""

    def __init__(self, api_key: str = None, api_secrets: str = None, testnet: bool = True):
        """
        Initialize exchange connector.
        
        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use testnet (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secrets
        self.testnet = testnet
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticket data for a symbol."""
        pass

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000
    ) -> List[OHLCV]:
        """Get historical OHLCV data for a symbol."""
        pass

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Get order book snapbook."""
        pass

    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades."""
        pass

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: Decimal,
        price: Optional[Decimal] = None
    ) -> dict:
        """Place an order."""
        pass

    @abstractmethod
    async def get_balance(self, currency: str) -> Decimal:
        """Get account balance for a currency."""
        pass

    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format (BTC/USDT)."""
        pass

    @abstractmethod
    def normalize_timestamp(self, timestamp: any) -> datetime:
        """Normalize timestamp to datetime object."""
        pass