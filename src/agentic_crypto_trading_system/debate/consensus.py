"""Consensus mechanisms for multi-agent debate.

Three strategies:
- Unanimous: All agents must agree on the same position
- Majority: >50% of agents agree
- Weighted: Agents are weighted by historical performance
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# --- Enums ---

class Position(str, Enum):
    """Trading position an agent can argue for."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class ConsensusMode(str, Enum):
    """Available consensus mechanisms."""
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    WEIGHTED = "weighted"

class DebateStatus(str, Enum):
    """Status of a debate."""
    IN_PROGRESS = "in_progress"
    CONSENSUS_REACHED = "consensus_reached"
    TIMEOUT = "timeout"
    VETOED = "vetoed"

# --- Data Models ---

@dataclass
class DebateArgument:
    """A single argument from one agent in one round."""
    agent_role: str
    position: Position
    confidence: float  # 0.0 to 1.0
    reasoning: str
    counter_arguments: List[str] = field(default_factory=list)
    data_points: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DebateRound:
    """One round of debate — all agents submit arguments."""
    round_number: int
    arguments: List[DebateArgument] = field(default_factory=list)
    consensus_reached: bool = False
    winning_position: Optional[Position] = None

@dataclass
class DebateTranscript:
    """Full record of a debate."""
    debate_id: str
    symbol: str
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    rounds: List[DebateRound] = field(default_factory=list)
    status: DebateStatus = DebateStatus.IN_PROGRESS
    final_position: Optional[Position] = None
    final_confidence: float = 0.0
    final_reasoning: str = ""
    participating_agents: List[str] = field(default_factory=list)

# --- Consensus Interface ---

class ConsensusStrategy(ABC):
    """Abstract base for consensus checking."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    @abstractmethod
    def check(self, arguments: List[DebateArgument]) -> tuple[bool, Optional[Position], float]:
        """Check if consensus is reached.
        
        Returns: (reached, winning_position, confidence)
        """
        pass

# --- Implementations ---

class UnanimousConsensus(ConsensusStrategy):
    """All agents must agree on the same position."""

    def check(self, arguments: List[DebateArgument]) -> tuple[bool, Optional[Position], float]:
        if not arguments:
            return False, None, 0.0

        positions = {arg.position for arg in arguments}
        if len(positions) == 1:
            position = positions.pop()
            avg_confidence = sum(a.confidence for a in arguments) / len(arguments)
            if avg_confidence >= self.min_confidence:
                return True, position, avg_confidence
        return False, None, 0.0


class MajorityConsensus(ConsensusStrategy):
    """More than 50% of agents must agree."""

    def check(self, arguments: List[DebateArgument]) -> tuple[bool, Optional[Position], float]:
        if not arguments:
            return False, None, 0.0

        # Count votes per position
        votes: Dict[Position, List[DebateArgument]] = {}
        for arg in arguments:
            votes.setdefault(arg.position, []).append(arg)

        # Find the position with the most votes
        best_position = max(votes, key=lambda p: len(votes[p]))
        best_args = votes[best_position]
        ratio = len(best_args) / len(arguments)

        if ratio > 0.5:
            avg_confidence = sum(a.confidence for a in best_args) / len(best_args)
            if avg_confidence >= self.min_confidence:
                return True, best_position, avg_confidence
        return False, None, 0.0

class WeightedConsensus(ConsensusStrategy):
    """Agents are weighted by historical performance (Sharpe ratio).

    Each agent's vote is multiplied by their weight.
    The position with the highest weighted score wins
    if it exceeds the min_confidence threshold.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, min_confidence: float = 0.5):
        super().__init__(min_confidence)
        # weights maps agent_role -> weight (e.g. {"technical_analyst": 1.5})
        self.weights = weights or {}

    def check(self, arguments: List[DebateArgument]) -> Tuple[bool, Optional[Position], float]:
        if not arguments:
            return False, None, 0.0

        # Calculate weighted scores per position
        scores: Dict[Position, float] = {}
        weight_totals: Dict[Position, float] = {}

        for arg in arguments:
            weight = self.weights.get(arg.role, 1.0) if hasattr(arg, 'role') else self.weights.get(arg.agent_role, 1.0)
            weighted_conf = arg.confidence * weight
            scores[arg.position] = scores.get(arg.position, 0.0) + weighted_conf
            weight_totals[arg.position] = weight_totals.get(arg.position, 0.0) + weight

        # Find the best position
        best_position = max(scores, key=lambda p: scores[p])
        total_weight = sum(weight_totals.values())

        if total_weight > 0:
            normalized_confidence = scores[best_position] / total_weight
            if normalized_confidence >= self.min_confidence:
                return True, best_position, normalized_confidence

        return False, None, 0.0

# --- Factory ---

def create_consensus_strategy(
    mode: ConsensusMode,
    min_confidence: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
) -> ConsensusStrategy:
    """Factory to create the right consensus strategy."""
    if mode == ConsensusMode.UNANIMOUS:
        return UnanimousConsensus(min_confidence=min_confidence)
    elif mode == ConsensusMode.MAJORITY:
        return MajorityConsensus(min_confidence=min_confidence)
    elif mode == ConsensusMode.WEIGHTED:
        return WeightedConsensus(weights=weights, min_confidence=min_confidence)
    else:
        raise ValueError(f"Unknown consensus mode: {mode}")