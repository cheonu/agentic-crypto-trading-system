from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Tuple, Optional
from enum import Enum

class OrderType(str, Enum):
    """ Order type enumeration. """
    LIMIT = "limit"
    MARKET = "market"

class OrderSide(str, Enum):
    """ Order side enumeration. """
    BUY = "buy"
    SELL = "sell"

@dataclass
class OHLCV:
    """Candlestick data."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    symbol: str
    timeframe: str

@dataclass
class Ticker:
    """Real-time ticker data."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime

@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    bids: List[Tuple[Decimal, Decimal]]  # price, amount
    asks: List[Tuple[Decimal, Decimal]]  # price, amount
    timestamp: datetime

@dataclass
class Trade:
    """Recent trade data."""
    symbol: str
    price: Decimal
    size: Decimal
    timestamp: datetime
    side: OrderSide


