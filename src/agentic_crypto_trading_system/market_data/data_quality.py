import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from ..exchange.models import Ticker

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Monitors market data quality and flags issues."""

    def __init__(
        self,
        max_staleness_seconds: float = 10.0,
        max_price_change_pct: float = 20.0,
    ):
        self.max_staleness_seconds = max_staleness_seconds
        self.max_price_change_pct = max_price_change_pct
        self._last_prices: Dict[str, Decimal] = {}
        self._last_timestamps: Dict[str, datetime] = {}
        self._flagged_symbols: Dict[str, str] = {}
        self._issues: List[Dict] = []

    def check_ticker(self, ticker: Ticker) -> bool:
        """Check ticker data quality. Returns true if data is reliable."""

        symbol = ticker.symbol
        is_reliable = True

        # Check for stable data
        if self._is_stale(symbol, ticker.timestamp):
            self._flag_issue(symbol, "stale_data", "Data feed is stale")
            is_reliable = False

        # Check for anomalous price movements
        if self._is_anomalous_price(symbol, ticker.last):
            self._flag_issue(
                symbol,
                "anomalous_price",
                f"Price moved more than {self.max_price_change_pct}%",
            )
            is_reliable = False

        # Update tracking data
        self._last_prices[symbol] = ticker.last
        self._last_timestamps[symbol] = ticker.timestamp

        return is_reliable


    def is_symbol_reliable(self, symbol: str) -> bool:
        """check if a symbol's data is marked as unreliable."""
        return symbol not in self._flagged_symbols

    def get_issues(self) -> List[Dict]:
        """Get all recorded data qaulity issues."""
        return self._issues.copy()

    def get_flagged_symbols(self) -> Dict[str, str]:
        """Get currently flagged symbols and their reasons."""
        return self._flagged_symbols.copy()

    def _is_stale(self, symbol: str, timestamp: datetime) -> bool:
        """Check if data is stale."""
        now = datetime.now()
        age = (now - timestamp).total_seconds()
        return age > self.max_staleness_seconds

    def _is_anomalous_price(self, symbol: str, current_price: Decimal) -> bool:
        """Check for anomalous price movements."""
        last_price = self._last_prices.get(symbol)
        if last_price is None or last_price == 0:
            return False

        change_pct = abs((current_price - last_price)/ last_price * 100)
        return change_pct > self.max_price_change_pct

    def _flag_issue(self, symbol: str, issue_type: str, message: str):
        """Record a data quality issue."""
        issue = {
            "symbol": symbol,
            "type": issue_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        self._issues.append(issue)
        self._flagged_symbols[symbol] = message
        logger.warning(f"Data quality issue: {symbol} - {message}")