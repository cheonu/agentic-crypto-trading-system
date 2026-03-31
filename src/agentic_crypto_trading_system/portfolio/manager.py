"""Portfolio Manager — allocates capital across trading agents.

Responsibilities:
- Track each agent's capital allocation
- Evaluate agent performance (Sharpe ratio)
- Rebalance allocations based on performance
- Adapt allocations when market regime changes
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentAllocation:
    """Capital allocation for a single agent."""
    agent_role: str
    allocated_capital: float
    min_allocation: float = 0.05   # 5% minimum
    max_allocation: float = 0.50   # 50% maximum
    current_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    last_rebalanced: Optional[datetime] = None

@dataclass
class RebalanceEvent:
    """Record of a rebalancing action."""
    timestamp: datetime
    reason: str  # "scheduled", "regime_change", "performance"
    old_allocations: Dict[str, float] = field(default_factory=dict)
    new_allocations: Dict[str, float] = field(default_factory=dict)

class PortfolioManager:
    """Manages capital allocation across trading agents.

    Key concepts:
    - Each agent gets a share of total capital (0.0 to 1.0)
    - Allocations are adjusted based on Sharpe ratio
    - Rebalancing happens on schedule or regime change
    - Max 10% shift per rebalance to avoid sudden swings
    """

    def __init__(
        self,
        total_capital: float,
        max_shift_per_rebalance: float = 0.10,
        rebalance_interval_hours: int = 24,
    ):
        self.total_capital = total_capital
        self.max_shift = max_shift_per_rebalance
        self.rebalance_interval_hours = rebalance_interval_hours
        self.allocations: Dict[str, AgentAllocation] = {}
        self.rebalance_history: List[RebalanceEvent] = []

    def allocate_capital(
        self,
        agent_roles: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Set initial capital allocation across agents.

        If no weights provided, splits equally.
        Returns dict of role -> allocated dollar amount.
        """
        if initial_weights:
            # Normalize weights to sum to 1.0
            total = sum(initial_weights.values())
            weights = {r: w / total for r, w in initial_weights.items()}
        else:
            equal_weight = 1.0 / len(agent_roles)
            weights = {r: equal_weight for r in agent_roles}

        for role in agent_roles:
            weight = weights.get(role, 0.0)
            self.allocations[role] = AgentAllocation(
                agent_role=role,
                allocated_capital=self.total_capital * weight,
            )
        return {r: a.allocated_capital for r, a in self.allocations.items()}

    def update_performance(
        self,
        agent_role: str,
        pnl: float,
        sharpe,
        trades: int,
        win_rate: float,
    ) -> None:
        """Update an agent's performance metrics."""
        if agent_role not in self.allocations:
            logger.warning(f"Unknown agent role: {agent_role}")
            return

        alloc = self.allocations[agent_role]
        alloc.current_pnl = pnl
        alloc.sharpe_ratio = sharpe
        alloc.trades_count = trades
        alloc.win_rate = win_rate

    def evaluate_agent_performance(self, agent_role: str) -> float:
        """Get the Sharpe ratio for an agent. Higher = better."""
        alloc = self.allocations.get(agent_role)
        if not alloc:
            return 0.0
        return alloc.sharpe_ratio

    def request_capital(self, agent_role: str, amount: float) -> bool:
        """Check if an agent can use the requested capital.

        Returns True if the amount is within their allocation.
        """
        alloc = self.allocations.get(agent_role)
        if not alloc:
            return False
        return amount <= alloc.allocated_capital

    def rebalance(self, reason: str = "scheduled") -> Dict[str, float]:
        """Rebalance allocations based on Sharpe ratios.

        Better-performing agents get more capital.
        Shifts are capped at max_shift per rebalance.
        """
        if not self.allocations:
            return {}

        old_allocs = {
            r: a.allocated_capital for r, a in self.allocations.items()
        }

        # Calculate target weights from Sharpe ratios
        sharpes = {}
        for role, alloc in self.allocations.items():
            # Use max(0, sharpe) so negative Sharpe doesn't get capital
            sharpes[role] = max(0.0, alloc.sharpe_ratio)

        total_sharpe = sum(sharpes.values())

        if total_sharpe > 0:
            target_weights = {r: s / total_sharpe for r, s in sharpes.items()}
        else:
            # All agents have zero or negative Sharpe — equal split
            equal = 1.0 / len(self.allocations)
            target_weights = {r: equal for r in self.allocations}

        # Apply gradual shift (max_shift cap)
        current_weights = {
            r: a.allocated_capital / self.total_capital
            for r, a in self.allocations.items()
        }

        new_weights = {}
        for role in self.allocations:
            current = current_weights[role]
            target = target_weights[role]
            diff = target - current

            # Cap the shift
            if abs(diff) > self.max_shift:
                diff = self.max_shift if diff > 0 else -self.max_shift

            new_weight = current + diff

            # Enforce min/max
            alloc = self.allocations[role]
            new_weight = max(alloc.min_allocation, min(alloc.max_allocation, new_weight))
            new_weights[role] = new_weight

        # Normalize to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {r: w / total for r, w in new_weights.items()}

        # Apply new allocations
        for role, weight in new_weights.items():
            self.allocations[role].allocated_capital = self.total_capital * weight
            self.allocations[role].last_rebalanced = datetime.utcnow()

        new_allocs = {
            r: a.allocated_capital for r, a in self.allocations.items()
        }

        # Record the event
        self.rebalance_history.append(RebalanceEvent(
            timestamp=datetime.utcnow(),
            reason=reason,
            old_allocations=old_allocs,
            new_allocations=new_allocs,
        ))

        logger.info(f"Rebalanced ({reason}): {new_allocs}")
        return new_allocs

    def on_regime_change(self, new_regime: str) -> Dict[str, float]:
        """Trigger rebalance when market regime changes."""
        logger.info(f"Regime changed to {new_regime}, triggering rebalance")
        return self.rebalance(reason=f"regime_change:{new_regime}")

    def get_allocations(self) -> Dict[str, AgentAllocation]:
        """Get current allocations for all agents."""
        return self.allocations.copy()
