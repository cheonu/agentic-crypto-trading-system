import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import AsyncIterator, Dict, List, Optional

import redis

from ..exchange.base import ExchangeConnector
from ..exchange.models import OHLCV, OrderBook, Ticker

logger = logging.getLogger(__name__)

class MarketDataLayer:
    """Service for ingesting, caching, and distributing market data."""

    def __init__(
        self,
        exchange: ExchangeConnector,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 60,
    ):
        self.exchange = exchange
        self.redis = redis_client 
        self.cache_ttl = cache_ttl
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._running = False
        self._data_timestamps: Dict[str, datetime] = {}

    async def subscribe_ticker(self, symbol: str, interval: float = 1.0) -> AsyncIterator[Ticker]:
        """Subscribe to real-time ticker updates for a symbol."""
        queue = asyncio.Queue()
        if symbol not in self._subscribers:
            self._subscribers[symbol] = []
        self._subscribers[symbol].append(queue)

        try:
            self._running = True
            while self._running:
                ticker = await self.exchange.get_ticker(symbol)
                self._data_timestamps[f"ticker:{symbol}"] = datetime.now()

                # Cache in redis
                if self.redis:
                    self._cache_ticker(symbol, ticker)

                # Notify all subscribers
                for q in self._subscribers.get(symbol, []):
                    await q.put(ticker)

                yield ticker
                await asyncio.sleep(interval)

        finally:
            self._subscribers[symbol].remove(queue)

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000
    ) -> List[OHLCV]:
        """Retrieve OHLCV data with caching support."""
    
        # Try to get from cache first
        if self.redis:
            cached = self._get_cached_ohlcv(symbol, timeframe, start, end)
            if cached:
                logger.info(f"Cache hit for {symbol} OHLCV")

        # Fetch from exchange
        ohlcv_data = await self.exchange.get_ohlcv(symbol, timeframe, start, end, limit)
        
        # Cache the result
        if self.redis:
            self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps([item.dict() for item in ohlcv_data])
            )
        return ohlcv_data

    async def get_order_book(self, symbol: str, depth: int = 20) -> OrderBook:
        """Retrieve order book snapshot with caching. """
        # Check cache (very short TTL for order books)
        if self.redis:
            cached = self.get_cached_order_book(symbol)
            if cached:
                return cached
        order_book = await self.exchange.get_order_book(symbol, depth)
        self._data_timestamps[f"orderbook:{symbol}"] = datetime.now()

        if self.redis:
            self._cache_order_book(symbol, order_book)

        return order_book

    def get_data_freshness(self, key: str) -> Optional[float]:
        """Get seconds since last data update for a key."""
        last_update = self._data_timestamps.get(key)
        if last_update is None:
            return None
        return (datetime.now() - last_update).total_seconds()
    
    def stop(self):
        """Stop all subscriptions."""
        self._running = False

    # --- Redis caching methods ----
    def _cache_ticker(self, symbol: str, ticker: Ticker):
        key = f"ticker:{symbol}"
        data = {
            "symbol": ticker.symbol,
            "bid": str(ticker.bid),
            "ask": str(ticker.ask),
            "last": str(ticker.last),
            "volume_24h": str(ticker.volume_24h),
            "timestamp": ticker.timestamp.isoformat(),
        }
        self.redis.setex(key, self.cache_ttl, json.dumps(data))

    def _cache_ohlcv(self, symbol: str, timeframe: str, data: List[OHLCV]):
        """Cache OHLCV data in redis."""
        key = f"ohlcv:{symbol}:{timeframe}"
        serialized_data = [
            {
                "timestamp": c.timestamp.isoformat(),
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": str(c.volume),
                "symbol": c.symbol,
                "timeframe": c.timeframe,
            }
            for c in data
        ]
        self.redis.setex(key, self.cache_ttl, json.dumps(serialized))


    def _get_cached_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Optional[List[OHLCV]]:
        """Retrieve cached OHLCV data."""
        key = f"ohlcv:{symbol}:{timeframe}"
        cached = self.redis.get(key)
        if cached:
            try:
                data = json.loads(cached)
                return [
                    OHLCV(
                        timestamp=datetime.fromisoformat(c["timestamp"]),
                        open=Decimal(c["open"]),
                        high=Decimal(c["high"]),
                        low=Decimal(c["low"]),
                        close=Decimal(c["close"]),
                        volume=Decimal(c["volume"]),
                        symbol=c["symbol"],
                        timeframe=c["timeframe"],
                    )
                    for c in data
                ]
            except Exception as e:
                logger.error(f"Error parsing cached OHLCV data: {e}")
        return None

    def get_cached_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Retrieve cached order book."""
        key = f"orderbook:{symbol}"
        cached = self.redis.get(key)
        if cached:
            try:
                data = json.loads(cached)
                return OrderBook(
                    symbol=data["symbol"],
                    bids=[(Decimal(price), Decimal(amount)) for price, amount in data["bids"]],
                    asks=[(Decimal(price), Decimal(amount)) for price, amount in data["asks"]],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                )
            except Exception as e:
                logger.error(f"Error parsing cached order book: {e}")
        return None