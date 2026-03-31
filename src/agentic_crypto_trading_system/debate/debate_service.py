"""Debate System — orchestrates multi-agent trading debates.

Flow:
1. initiate_debate() — select agents, create transcript
2. conduct_round() — each agent submits an argument
3. check_consensus() — see if agents agree
4. Repeat 2-3 until consensus or max rounds
5. finalize_debate() — produce final decision
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .consensus import (
    ConsensusMode,
    ConsensusStrategy,
    DebateArgument,
    DebateRound,
    DebateStatus,
    DebateTranscript,
    Position,
    create_consensus_strategy,
)

logger = logging.getLogger(__name__)

# Type alias for the function each agent uses to generate arguments
# Signature: (agent_role, task_description, round_number, previous_round) -> DebateArgument

ArgumentGenerator = Callable[
    [str, str, int, Optional[DebateRound]], DebateArgument
]

class DebateService:
    """Orchestrates multi-agent debates for trading decisions.

    The debate service doesn't know about specific agent frameworks
    (CrewAI, LangGraph). Instead, it accepts argument generator
    functions that each framework provides. This keeps it decoupled.
    """

    def __init__(
        self,
        max_rounds: int = 5,
        consensus_mode: ConsensusMode = ConsensusMode.MAJORITY,
        min_confidence: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
        veto_roles : Optional[List[str]] = None
    ):
        self.max_rounds = max_rounds
        self.consensus_strategy: ConsensusStrategy = create_consensus_strategy(
            mode=consensus_mode,
            min_confidence=min_confidence,
            weights=weights,
        )
        self.veto_roles = veto_roles or []
        self.transcripts: Dict[str, DebateTranscript] = {}

    def initiate_debate(
        self,
        symbol: str,
        agent_roles: List[str],
    ) -> DebateTranscript:
        """Start a new debate.

        Creates a transcript and registers participating agents.
        """
        debate_id = str(uuid.uuid4())[:8]
        transcript = DebateTranscript(
            debate_id=debate_id,
            symbol=symbol,
            participating_agents=agent_roles,
        )
        self.transcripts[debate_id] = transcript
        logger.info(
            f"Debate {debate_id} initiated for {symbol} "
            f"with agents: {agent_roles}"
        )
        return transcript

    def conduct_round(
        self,
        transcript: DebateTranscript,
        task_description: str,
        argument_generators: Dict[str, ArgumentGenerator]
        ) -> DebateRound:
        """Run one round of debate.

        Each agent generates an argument based on the task
        and (optionally) the previous round's arguments.
        """ 
        round_number = len(transcript.rounds) + 1
        previous_round = transcript.rounds[-1] if transcript.rounds else None

        debate_round = DebateRound(round_number=round_number)

        for role in transcript.participating_agents:
            generator = argument_generators.get(role)
            if not generator:
                logger.warning(f"No argument for {role}, skipping")
                continue

            try:
                argument = generator(
                    role, task_description, round_number, previous_round
                )
                debate_round.arguments.append(argument)
            except Exception as e:
                logger.error(f"Agent {role} failed to generate argument: {e}")
                # Agent fails to argue - skip them this round
                continue
        transcript.rounds.append(debate_round)
        logger.info(
            f"Debate {transcript.debate_id} round {round_number}: "
            f"{len(debate_round.arguments)} arguments submitted"
        )
        return debate_round

    def check_consensus(
        self, debate_round: DebateRound
    ) -> tuple[bool, Optional[Position], float]:
        """Check if the current round reached consensus.

        Also enforces veto rules — if a veto-role agent disagrees
        with the majority, consensus is blocked.
        """
        reached, position, confidence = self.consensus_strategy.check(
            debate_round.arguments
        )

        # Veto check: if a veto-role agent disagrees, block consensus
        if reached and self.veto_roles:
            for arg in debate_round.arguments:
                if arg.agent_role in self.veto_roles and arg.position != position:
                    logger.info(
                        f"Veto by {arg.agent_role}: "
                        f"argued {arg.position.value} vs consensus {position.value}"
                    )
                    return False, None, 0.0
        if reached:
            debate_round.consensus_reached = True
            debate_round.winning_position = position

        return reached, position, confidence

    def finalize_debate(
        self, transcript: DebateTranscript
    ) -> DebateTranscript:
        """Finalize the debate and produce the decision.

        If consensus was reached, use the winning position.
        If timeout (max rounds exceeded), default to HOLD.
        """
        # Find the last round with consensus
        for debate_round in reversed(transcript.rounds):
            if debate_round.consensus_reached:
                transcript.status = DebateStatus.CONSENSUS_REACHED
                transcript.final_position = debate_round.winning_position
                # Aggregate reasoning from all agents who agreed
                agreeing = [
                    a for a in debate_round.arguments
                    if a.position == debate_round.winning_position
                ]
                transcript.final_confidence = (
                    sum(a.confidence for a in agreeing) / len(agreeing)
                    if agreeing else 0.0
                )
                transcript.final_reasoning = " | ".join(
                    f"{a.agent_role}: {a.reasoning}" for a in agreeing
                )
                return transcript

        # No consensu - timeout
        transcript.status = DebateStatus.TIMEOUT
        transcript.final_position = Position.HOLD
        transcript.final_confidence = 0.0
        transcript.final_reasoning = (
            f"No consensus after {len(transcript.rounds)} rounds. "
            "Defaulting to HOLD."
        )
        return transcript

    def run_debate(
        self,
        symbol: str,
        task_description: str,
        agent_roles: List[str],
        argument_generators: Dict[str, ArgumentGenerator],
    ) -> DebateTranscript:
        """Convenience method: run a full debate from start to finish.

        Loops through rounds until consensus or max_rounds.
        """
        transcript = self.initiate_debate(symbol, agent_roles)

        for _ in range(self.max_rounds):
            debate_round = self.conduct_round(
                transcript, task_description, argument_generators
            )
            reached, position, confidence = self.check_consensus(debate_round)

            if reached:
                break

        return self.finalize_debate(transcript)