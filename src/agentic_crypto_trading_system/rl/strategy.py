"""Reinforcement Learning strategy support.

Provides a base RL strategy interface with reward calculation,
policy updates, and exploration/exploitation controls.
"""

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for an RL strategy."""
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon: float = 0.1          # Exploration rate
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    batch_size: int = 32
    mode: str = "online"  # "online" or "batch"


@dataclass
class RLState:
    """State representation for RL agent."""
    features: Dict[str, float] = field(default_factory=dict)
    regime: str = ""
    position: str = "flat"  # "long", "short", "flat"
    unrealized_pnl: float = 0.0


class RLStrategy(ABC):
    """Abstract base class for RL trading strategies.

    Subclasses implement the policy (select_action) and
    learning logic (update_policy). The base class handles
    reward calculation and exploration/exploitation.
    """

    def __init__(self, config: Optional[RLConfig] = None):
        self.config = config or RLConfig()
        self.epsilon = self.config.epsilon
        self.total_reward: float = 0.0
        self.episode_rewards: List[float] = []
        self.training_steps: int = 0
        self.experience_buffer: List[Dict[str, Any]] = []

    def calculate_reward(self, trade_pnl: float, risk_penalty: float = 0.0) -> float:
        """Calculate reward signal from trade outcome.

        reward = pnl - risk_penalty
        Risk penalty discourages excessive risk-taking.
        """
        reward = trade_pnl - risk_penalty
        self.total_reward += reward
        self.episode_rewards.append(reward)
        return reward

    def should_explore(self) -> bool:
        """Decide whether to explore (random) or exploit (policy)."""
        return random.random() < self.epsilon

    def decay_epsilon(self) -> None:
        """Decay exploration rate over time."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )

    def store_experience(
        self,
        state: RLState,
        action: str,
        reward: float,
        next_state: RLState,
        done: bool = False,
    ) -> None:
        """Store a transition in the experience buffer."""
        self.experience_buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

    @abstractmethod
    def select_action(self, state: RLState) -> str:
        """Select an action given the current state.

        Returns: "buy", "sell", or "hold"
        """
        pass

    @abstractmethod
    def update_policy(self, batch: Optional[List[Dict[str, Any]]] = None) -> float:
        """Update the policy based on experience.

        Returns the training loss.
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_reward": self.total_reward,
            "episodes": len(self.episode_rewards),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "buffer_size": len(self.experience_buffer),
            "avg_reward": (
                sum(self.episode_rewards) / len(self.episode_rewards)
                if self.episode_rewards else 0.0
            ),
        }
