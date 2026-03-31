import ccxt
import asyncio
from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from .base import ExchangeConnector
from .models import OHLCV, Ticker, OrderBook, Trade, OrderType, OrderSide

class BinanceConnector(ExchangeConnector):
    """Binace exchange connector using ccxt."""
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)

        # Initialize ccxt exchange
        if testnet:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': True,
            })
        else:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret
            })
        self.exchange.enableRateLimit = True # Enable built-in rate limiting

    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker data."""
        normalized_symbol = self.normalize_symbol(symbol)
        ticker_data = await self._retry_request(
            self.exchange.fetch_ticker, normalized_symbol
        )

        return Ticker(
            symbol=symbol,
            bid=Decimal(str(ticker_data['bid'])),
            ask=Decimal(str(ticker_data['ask'])),
            last=Decimal(str(ticker_data['last'])),
            volume_24h=Decimal(str(ticker_data['quoteVolume'])),
            timestamp=datetime.fromtimestamp(ticker_data['timestamp']/1000)
        )

    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000
    ) -> List[OHLCV]:
        """Get historical OHLCV data."""
        normalized_symbol = self.normalize_symbol(symbol)
        since = int(start.timestamp() * 1000) # Convert to milliseconds

        ohlcv_data = await self._retry_request(
            self.exchange.fetch_ohlcv,
            normalized_symbol,
            timeframe,
            since,
            limit
        )

        return [
            OHLCV(
                timestamp=self.normalize_timestamp(candle[0]),
                open=Decimal(str(candle[1])),
                high=Decimal(str(candle[2])),
                low=Decimal(str(candle[3])),
                close=Decimal(str(candle[4])),
                volume=Decimal(str(candle[5])),
                symbol=symbol,
                timeframe=timeframe
            )
            for candle in ohlcv_data
            if self.normalize_timestamp(candle[0]) <= end
        ]

    async def get_order_book(
        self, 
        symbol: str, 
        depth: int = 20
        ) -> OrderBook:
        """Get order book snapshot."""
        normalized_symbol = self.normalize_symbol(symbol)
        order_book_data = await self._retry_request(
            self.exchange.fetch_order_book, normalized_symbol, depth
        )

        return OrderBook(
            symbol=symbol,
            bids=[(Decimal(str(bid[0])), Decimal(str(bid[1]))) for bid in order_book_data['bids']],
            asks=[(Decimal(str(ask[0])), Decimal(str(ask[1]))) for ask in order_book_data['asks']],
            timestamp=self.normalize_timestamp(order_book_data['timestamp'])
        )

    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        normalized_symbol = self.normalize_symbol(symbol)
        trades_data = await self._retry_request(
            self.exchange.fetch_trades, normalized_symbol, limit=limit
        
        )

        return [
            Trade(
                symbol=symbol,
                price=Decimal(str(trade['price'])),
                size=Decimal(str(trade['amount'])),
                side=OrderSide.BUY if trade['side'] == 'buy' else OrderSide.SELL,
                timestamp=self.normalize_timestamp(trade['timestamp'])
                )
                for trade in trades_data
        ]

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        size: Decimal,
        price: Optional[Decimal] = None
    ) -> dict:
        """Place an order."""
        normalized_symbol = self.normalize_symbol(symbol)
        order_params = {
            'symbol': normalized_symbol,
            'type': order_type.value,
            'side': side.value,
            'amount': float(size),
        }

        if order_type == OrderType.LIMIT and price:
            order_params['price'] = float(price)

        order_result = await self._retry_request(
            self.exchange.create_order,
            **order_params
        )

        return order_result

    async def get_balance(self, currency: str) -> Decimal:
        """Get account balance."""
        balance_data = await self._retry_request(self.exchange.fetch_balance)
        return Decimal(str(balance_data.get(currency, {}).get('free', 0)))
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol of ccxt form."""
        # Convert BTC/USDT to BTCUSDT for Binance
        return symbol.replace('/', '')

    def normalize_timestamp(self, timestamp: any) -> datetime:
        """Normalize timestamp to datetime."""
        if isinstance(timestamp, datetime):
            return timestamp
        # ccxt returns milliseconds
        return datetime.fromtimestamp(timestamp / 1000)

    async def _retry_request(self, func, *args, max_retries: int = 3, **kwargs):
        """Retry request with exponential backoff."""
        for attempt in range (max_retries):
            try:
                # Run synchronous ccxt methods in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                return result
            except Exception as e:
                if attempt == max_retries -1:
                    raise
                # Exponential backoff: 100ms, 200ms, 400ms
                wait_time = 0.1 * (2 ** attempt)
                await asyncio.sleep(wait_time)