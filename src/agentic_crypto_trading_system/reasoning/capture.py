"""Reasoning capture — stores and queries agent decision reasoning.

Every agent decision is recorded with structured reasoning,
data points cited, memory influence, and regime context.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A recorded agent decision with reasoning."""
    agent_role: str
    symbol: str
    decision: str  # BUY, SELL, HOLD
    reasoning: str
    confidence: float = 0.0
    data_points: Dict[str, Any] = field(default_factory=dict)
    memory_influence: str = ""
    regime: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ReasoningCapture:
    """Stores and queries agent decision reasoning."""

    def __init__(self):
        self.records: List[DecisionRecord] = []

    def record(self, decision: DecisionRecord) -> None:
        """Store a decision record."""
        self.records.append(decision)
        logger.info(
            f"Decision recorded: {decision.agent_role} -> "
            f"{decision.decision} {decision.symbol} "
            f"(confidence: {decision.confidence:.2f})"
        )

    def query(
        self,
        agent_role: Optional[str] = None,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        search_text: Optional[str] = None,
        limit: int = 50,
    ) -> List[DecisionRecord]:
        """Query decision records with filters."""
        results = self.records

        if agent_role:
            results = [r for r in results if r.agent_role == agent_role]
        if symbol:
            results = [r for r in results if r.symbol == symbol]
        if regime:
            results = [r for r in results if r.regime == regime]
        if search_text:
            search_lower = search_text.lower()
            results = [
                r for r in results
                if search_lower in r.reasoning.lower()
                or search_lower in r.memory_influence.lower()
            ]

        return results[-limit:]

    def get_by_agent(self, agent_role: str, limit: int = 20) -> List[DecisionRecord]:
        """Get recent decisions for a specific agent."""
        return self.query(agent_role=agent_role, limit=limit)

    def get_by_regime(self, regime: str, limit: int = 20) -> List[DecisionRecord]:
        """Get decisions made during a specific regime."""
        return self.query(regime=regime, limit=limit)
