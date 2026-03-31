import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    success: bool
    data: Any = None
    error: Optional[str] = None


class BaseTool:
    """Base class for agent tools."""
    name: str = "base_tool"
    description: str = "Base tool"

    def __init__(self):
        self.invocation_count = 0

    def _log_invocation(self, **kwargs):
        self.invocation_count += 1
        logger.info(f"Tool {self.name} invoked (#{self.invocation_count}): {kwargs}")


class MarketDataTool(BaseTool):
    """Tool for querying market data."""
    name = "market_data"
    description = "Query real-time and historical market data for crypto assets"

    def __init__(self, market_data_layer=None):
        super().__init__()
        self.market_data = market_data_layer

    async def get_ticker(self, symbol: str) -> ToolResult:
        """Get current ticker for a symbol."""
        self._log_invocation(action="get_ticker", symbol=symbol)
        try:
            if self.market_data:
                ticker = await self.market_data.get_ticker(symbol)
                return ToolResult(success=True, data=ticker)
            return ToolResult(success=False, error="Market data layer not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> ToolResult:
        """Get OHLCV candle data."""
        self._log_invocation(action="get_ohlcv", symbol=symbol, timeframe=timeframe)
        try:
            if self.market_data:
                data = await self.market_data.get_ohlcv(symbol, timeframe, limit)
                return ToolResult(success=True, data=data)
            return ToolResult(success=False, error="Market data layer not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class OrderTool(BaseTool):
    """Tool for submitting and managing orders."""
    name = "order"
    description = "Submit buy/sell orders and check order status"

    def __init__(self, trade_executor=None, risk_manager=None):
        super().__init__()
        self.executor = trade_executor
        self.risk_manager = risk_manager

    async def submit_order(self, symbol: str, side: str, size: float, price: float = None) -> ToolResult:
        """Submit a trade order."""
        self._log_invocation(action="submit_order", symbol=symbol, side=side, size=size)
        try:
            if self.executor:
                from ..execution.executor import OrderRequest
                request = OrderRequest(
                    symbol=symbol, side=side,
                    order_type="market" if price is None else "limit",
                    size=size, price=price,
                )
                result = await self.executor.execute_trade(request)
                return ToolResult(success=result.status == "filled", data=vars(result))
            return ToolResult(success=False, error="Trade executor not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class MemoryTool(BaseTool):
    """Tool for querying the memory system."""
    name = "memory"
    description = "Query past trades and discovered patterns from memory"

    def __init__(self, memory_service=None):
        super().__init__()
        self.memory = memory_service

    def query_similar_trades(self, query: str, n_results: int = 5, regime: str = None) -> ToolResult:
        """Find similar past trades."""
        self._log_invocation(action="query_trades", query=query)
        try:
            if self.memory:
                results = self.memory.query_similar_trades(query, n_results=n_results, regime=regime)
                return ToolResult(success=True, data=results)
            return ToolResult(success=False, error="Memory service not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def query_patterns(self, query: str, n_results: int = 5, regime: str = None) -> ToolResult:
        """Find similar patterns."""
        self._log_invocation(action="query_patterns", query=query)
        try:
            if self.memory:
                results = self.memory.query_patterns(query, n_results=n_results, regime=regime)
                return ToolResult(success=True, data=results)
            return ToolResult(success=False, error="Memory service not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class SentimentTool(BaseTool):
    """Tool for querying sentiment data."""
    name = "sentiment"
    description = "Get current news sentiment for crypto assets"

    def __init__(self, sentiment_analyzer=None):
        super().__init__()
        self.analyzer = sentiment_analyzer

    def get_current_sentiment(self) -> ToolResult:
        """Get current aggregated sentiment."""
        self._log_invocation(action="get_sentiment")
        try:
            if self.analyzer:
                score = self.analyzer.get_current_sentiment()
                trend = self.analyzer.get_sentiment_trend()
                return ToolResult(success=True, data={"score": score, "trend": trend})
            return ToolResult(success=False, error="Sentiment analyzer not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class IndicatorTool(BaseTool):
    """Tool for calculating technical indicators."""
    name = "indicators"
    description = "Calculate technical indicators like ATR, ADX, momentum"

    def __init__(self, regime_detector=None):
        super().__init__()
        self.detector = regime_detector

    def get_current_regime(self, highs, lows, closes, volumes) -> ToolResult:
        """Detect current market regime."""
        self._log_invocation(action="get_regime")
        try:
            if self.detector:
                result = self.detector.detect_regime(highs, lows, closes, volumes)
                return ToolResult(success=True, data={
                    "regime": result.regime.value,
                    "confidence": result.confidence,
                    "atr": result.atr,
                    "adx": result.adx,
                    "momentum": result.momentum,
                })
            return ToolResult(success=False, error="Regime detector not configured")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
