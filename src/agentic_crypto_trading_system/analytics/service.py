"""Performance analytics — Sharpe, drawdown, win rate, regime analysis.

Calculates per-agent and per-strategy performance metrics,
segments by market regime, and compares frameworks.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Performance record for a single agent."""
    agent_role: str
    returns: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    regime_returns: Dict[str, List[float]] = field(default_factory=dict)


class AnalyticsService:
    """Calculates and tracks performance analytics."""

    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.agents: Dict[str, AgentPerformance] = {}

    def register_agent(self, agent_role: str) -> None:
        """Register an agent for tracking."""
        self.agents[agent_role] = AgentPerformance(agent_role=agent_role)

    def record_return(self, agent_role: str, ret: float, regime: Optional[str] = None) -> None:
        """Record a return for an agent."""
        perf = self.agents.get(agent_role)
        if not perf:
            return
        perf.returns.append(ret)
        if regime:
            perf.regime_returns.setdefault(regime, []).append(ret)

    def record_trade(self, agent_role: str, trade: Dict[str, Any]) -> None:
        """Record a trade for an agent."""
        perf = self.agents.get(agent_role)
        if not perf:
            return
        perf.trades.append(trade)

    def sharpe_ratio(self, agent_role: str) -> float:
        """Calculate annualized Sharpe ratio for an agent."""
        perf = self.agents.get(agent_role)
        if not perf or len(perf.returns) < 2:
            return 0.0
        return self._calc_sharpe(perf.returns)

    def max_drawdown(self, agent_role: str) -> float:
        """Calculate maximum drawdown for an agent."""
        perf = self.agents.get(agent_role)
        if not perf or not perf.returns:
            return 0.0
        return self._calc_max_drawdown(perf.returns)

    def win_rate(self, agent_role: str) -> float:
        """Calculate win rate for an agent."""
        perf = self.agents.get(agent_role)
        if not perf or not perf.trades:
            return 0.0
        wins = sum(1 for t in perf.trades if t.get("pnl", 0) > 0)
        return wins / len(perf.trades)

    def profit_factor(self, agent_role: str) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        perf = self.agents.get(agent_role)
        if not perf or not perf.trades:
            return 0.0
        gross_profit = sum(t["pnl"] for t in perf.trades if t.get("pnl", 0) > 0)
        gross_loss = abs(sum(t["pnl"] for t in perf.trades if t.get("pnl", 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    def regime_sharpe(self, agent_role: str, regime: str) -> float:
        """Calculate Sharpe ratio for a specific regime."""
        perf = self.agents.get(agent_role)
        if not perf:
            return 0.0
        returns = perf.regime_returns.get(regime, [])
        if len(returns) < 2:
            return 0.0
        return self._calc_sharpe(returns)

    def get_summary(self, agent_role: str) -> Dict[str, Any]:
        """Get full performance summary for an agent."""
        return {
            "agent_role": agent_role,
            "sharpe_ratio": self.sharpe_ratio(agent_role),
            "max_drawdown": self.max_drawdown(agent_role),
            "win_rate": self.win_rate(agent_role),
            "profit_factor": self.profit_factor(agent_role),
            "total_trades": len(self.agents.get(agent_role, AgentPerformance("")).trades),
            "total_returns": len(self.agents.get(agent_role, AgentPerformance("")).returns),
        }

    def compare_agents(self) -> List[Dict[str, Any]]:
        """Compare all agents side by side."""
        return [self.get_summary(role) for role in self.agents]

    def _calc_sharpe(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        avg = sum(returns) / len(returns)
        std = math.sqrt(sum((r - avg) ** 2 for r in returns) / max(len(returns) - 1, 1))
        if std == 0:
            return 0.0
        daily_rf = self.risk_free_rate / 252
        return (avg - daily_rf) / std * math.sqrt(252)

    def _calc_max_drawdown(self, returns: List[float]) -> float:
        """Calculate max drawdown from a returns series."""
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            equity *= (1 + r)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd
