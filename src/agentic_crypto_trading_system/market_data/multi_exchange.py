import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional

from ..exchange.base import ExchangeConnector
from ..exchange.models import OHLCV, Ticker
from .market_data_layer import MarketDataLayer

logger = logging.getLogger(__name__)

class MultiExchangeDataLayer:
    """Aggregates market data from multiple exchanges."""

    def __init__(self):
        self ._exchanges: Dict[str, MarketDataLayer] = {}   

    def add_exchange(self, name: str, data_layer: MarketDataLayer):
        """Register an exchange data layer.""" 
        self._exchanges[name] = data_layer

    async def get_ticker_all_exchanges(self, symbol: str) -> Dict[str, Ticker]:
        """Get ticker for all registered exchanges concurrently."""
        tasks = {
            name: layer.exchange.get_ticker(symbol)
            for name, layer in self._exchanges.items()
        }

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                logger.warning(f"Failed to get ticker from {name}: {e}")
        return results

    async def get_best_price(self, symbol: str) -> Optional[Ticker]:

        """Get the best price across all exchanges."""
        tickers = await self.get_ticker_all_exchanges(symbol)
        if not tickers:
            return None

        # Find the highest bid price (or lowest ask price for sell)
        best = None
        for ticker in tickers.values():
            if best is None:
                best = ticker
            else:
                if ticker.bid > best.bid:
                    best = ticker
        return best
    
    async def get_ohlcv_multi_symbol(
        self,
        symbols: List[str],
        timeframe: str,
        start: datetime,
        end: datetime,
        exchange: str = None,
    ) -> Dict[str, List[OHLCV]]:
        """Get OHLCV data for multiple symbols concurrently."""
        layer = self._get_exchange(exchange)

        tasks = {
            symbol: layer.get_ohlcv(symbol, timeframe, start, end)
            for symbol in symbols
        }

        results = {}
        for symbol, task in tasks.items():
            try:
                results[symbol] = await task
            except Exception as e:
                logger.warning(f"Failed to get OHLCV for {symbol}: {e}")
                results[symbol] = []

        return results

    def _get_exchange(self, name: str = None) -> MarketDataLayer:
        """Get exchange by name or return first available."""
        if name and name in self._exchanges:
            return self._exchanges[name]
        return next(iter(self._exchanges.values()))

