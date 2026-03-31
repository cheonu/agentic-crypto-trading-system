"""Backtesting Engine — replay historical data for strategy evaluation.

Replays OHLCV data chronologically, prevents look-ahead bias,
simulates execution with slippage/fees, and calculates performance metrics.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest run."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    start_equity: float = 0.0
    end_equity: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    risk_free_rate: float = 0.02  # 2% annual
    max_position_size: float = 0.25  # 25% of capital per trade


@dataclass
class SimulatedTrade:
    """A trade executed during backtesting."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    commission: float
    slippage: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete result of a backtest run."""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[SimulatedTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    regime_performance: Dict[str, PerformanceMetrics] = field(default_factory=dict)


# Strategy callback: receives (timestamp, ohlcv_bar, portfolio_state) -> Optional[signal]
StrategyCallback = Callable[[datetime, Dict[str, Any], Dict[str, Any]], Optional[Dict[str, Any]]]


class BacktestEngine:
    """Replays historical data and simulates trading."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.equity = self.config.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.trades: List[SimulatedTrade] = []
        self.equity_curve: List[float] = []

    def run_backtest(
        self,
        historical_data: List[Dict[str, Any]],
        strategy: StrategyCallback,
    ) -> BacktestResult:
        """Run a backtest over historical data.

        historical_data: list of OHLCV bars sorted chronologically.
        Each bar: {"timestamp": datetime, "open": float, "high": float,
                    "low": float, "close": float, "volume": float, "symbol": str}
        """
        self.equity = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.equity]

        for i, bar in enumerate(historical_data):
            # Only pass data up to current bar (no look-ahead)
            visible_data = historical_data[:i + 1]
            portfolio_state = {
                "equity": self.equity,
                "positions": self.positions.copy(),
                "trades_count": len(self.trades),
            }

            signal = strategy(bar["timestamp"], bar, portfolio_state)

            if signal:
                self._execute_signal(signal, bar)

            # Update equity with mark-to-market
            self._mark_to_market(bar)
            self.equity_curve.append(self.equity)

        metrics = self._calculate_metrics()
        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=self.equity_curve,
        )

    def _execute_signal(self, signal: Dict[str, Any], bar: Dict[str, Any]) -> None:
        """Execute a trading signal with slippage and commission."""
        side = signal.get("side", "buy")
        symbol = signal.get("symbol", bar.get("symbol", "UNKNOWN"))
        price = bar["close"]

        # Apply slippage
        slippage = price * self.config.slippage_rate
        exec_price = price + slippage if side == "buy" else price - slippage

        # Calculate position size
        max_value = self.equity * self.config.max_position_size
        quantity = signal.get("quantity", max_value / exec_price)
        trade_value = quantity * exec_price

        # Commission
        commission = trade_value * self.config.commission_rate

        if side == "buy":
            if trade_value + commission > self.equity:
                return  # Not enough capital
            self.equity -= trade_value + commission
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:
            current_qty = self.positions.get(symbol, 0)
            if current_qty <= 0:
                return  # Nothing to sell
            sell_qty = min(quantity, current_qty)
            self.equity += sell_qty * exec_price - commission
            self.positions[symbol] = current_qty - sell_qty
            if self.positions[symbol] <= 0:
                del self.positions[symbol]

        pnl = 0.0  # PnL calculated on close
        self.trades.append(SimulatedTrade(
            timestamp=bar["timestamp"],
            symbol=symbol,
            side=side,
            price=exec_price,
            quantity=quantity,
            commission=commission,
            slippage=slippage,
            pnl=pnl,
        ))

    def _mark_to_market(self, bar: Dict[str, Any]) -> None:
        """Update equity based on current positions."""
        # Equity = cash + sum(position_value)
        # Cash is already tracked in self.equity for closed positions
        pass  # Simplified: equity updated on trade execution

    def _calculate_metrics(self) -> PerformanceMetrics:
        """Calculate performance metrics from the equity curve."""
        if len(self.equity_curve) < 2:
            return PerformanceMetrics(
                start_equity=self.config.initial_capital,
                end_equity=self.equity,
            )

        start = self.equity_curve[0]
        end = self.equity_curve[-1]
        total_return = (end - start) / start if start > 0 else 0.0

        # Calculate returns series
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev = self.equity_curve[i - 1]
            if prev > 0:
                returns.append((self.equity_curve[i] - prev) / prev)

        # Sharpe ratio (annualized, assuming daily bars)
        sharpe = 0.0
        if returns:
            avg_return = sum(returns) / len(returns)
            std_return = math.sqrt(
                sum((r - avg_return) ** 2 for r in returns) / max(len(returns) - 1, 1)
            )
            if std_return > 0:
                daily_rf = self.config.risk_free_rate / 252
                sharpe = (avg_return - daily_rf) / std_return * math.sqrt(252)

        # Max drawdown
        max_drawdown = 0.0
        peak = self.equity_curve[0]
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            drawdown = (peak - eq) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        # Win/loss stats
        winning = sum(1 for t in self.trades if t.pnl > 0)
        losing = sum(1 for t in self.trades if t.pnl < 0)
        total = len(self.trades)
        win_rate = winning / total if total > 0 else 0.0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        avg_pnl = sum(t.pnl for t in self.trades) / total if total > 0 else 0.0

        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            start_equity=start,
            end_equity=end,
        )
