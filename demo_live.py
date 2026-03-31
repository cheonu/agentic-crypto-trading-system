"""Live demo — runs LangGraph + CrewAI agents, then a multi-agent debate.

Usage:
    poetry run python demo_live.py

This script:
1. Runs a LangGraph technical analyst (deterministic, no LLM)
2. Runs a CrewAI technical analyst (live OpenAI call)
3. Runs a multi-agent debate using the debate system
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agentic_crypto_trading_system.agents.base import (
    AgentConfig, AgentRole, Task,
)
from agentic_crypto_trading_system.agents.langgraph_framework import (
    LangGraphFramework,
)
from agentic_crypto_trading_system.debate import (
    DebateService, DebateArgument, Position, ConsensusMode,
)


# --- Market scenario ---
SCENARIO = {
    "market_data": {"symbol": "BTC/USDT", "price": 67500, "change_24h": 2.3},
    "sentiment_data": {"score": 0.35, "trend": "improving"},
    "regime_data": {"regime": "bull", "confidence": 0.82},
}


async def demo_langgraph():
    """Part 1: Run LangGraph agent (deterministic, no LLM)."""
    print("\n" + "=" * 60)
    print("PART 1: LangGraph Technical Analyst (No LLM)")
    print("=" * 60)

    lg = LangGraphFramework()

    # Create all three agents
    for role, name, goal in [
        (AgentRole.TECHNICAL_ANALYST, "LG Tech Analyst", "Analyze BTC technicals"),
        (AgentRole.SENTIMENT_ANALYST, "LG Sentiment", "Analyze market sentiment"),
        (AgentRole.RISK_ASSESSOR, "LG Risk", "Assess trade risk"),
    ]:
        lg.create_agent(AgentConfig(name=name, role=role, goal=goal, backstory="Expert"))

    # Execute task
    task = Task(
        description="Should we buy BTC/USDT at $67,500? Bull regime, sentiment improving.",
        agent_role=AgentRole.TECHNICAL_ANALYST,
        context=SCENARIO,
        expected_output="BUY, SELL, or HOLD with reasoning",
    )

    result = await lg.execute_task(task)
    print(f"\nResult: {result.output}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Success: {result.success}")
    return result


async def demo_crewai():
    """Part 2: Run CrewAI agent (live OpenAI call)."""
    print("\n" + "=" * 60)
    print("PART 2: CrewAI Technical Analyst (Live LLM)")
    print("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — skipping CrewAI demo")
        return None

    from agentic_crypto_trading_system.agents.crewai_framework import (
        CrewAIFramework,
    )

    crew = CrewAIFramework(verbose=True)
    crew.create_agent(AgentConfig(
        name="CrewAI Tech Analyst",
        role=AgentRole.TECHNICAL_ANALYST,
        goal="Analyze cryptocurrency markets and provide trading recommendations",
        backstory=(
            "You are a senior crypto technical analyst with 10 years experience. "
            "You analyze price action, market regime, and sentiment to make decisions. "
            "Always respond with BUY, SELL, or HOLD and your reasoning."
        ),
        verbose=True,
    ))

    task = Task(
        description=(
            f"Analyze BTC/USDT. Current price: $67,500. "
            f"24h change: +2.3%. Market regime: BULL (confidence 82%). "
            f"Sentiment score: 0.35 (improving). "
            f"Should we BUY, SELL, or HOLD? Explain your reasoning."
        ),
        agent_role=AgentRole.TECHNICAL_ANALYST,
        expected_output="BUY, SELL, or HOLD with detailed reasoning",
    )

    result = await crew.execute_task(task)
    print(f"\nResult: {result.output[:500]}")
    print(f"Success: {result.success}")
    return result


def demo_debate():
    """Part 3: Multi-agent debate."""
    print("\n" + "=" * 60)
    print("PART 3: Multi-Agent Debate")
    print("=" * 60)

    # Create debate service with majority consensus
    debate = DebateService(
        max_rounds=3,
        consensus_mode=ConsensusMode.MAJORITY,
        min_confidence=0.5,
        veto_roles=["risk_assessor"],
    )

    # Argument generators — simulate what each agent would say
    def tech_analyst_gen(role, task, round_num, prev_round):
        return DebateArgument(
            agent_role=role,
            position=Position.BUY,
            confidence=0.82,
            reasoning="Bull regime confirmed with ADX>25. RSI at 58, not overbought. Momentum positive.",
            data_points={"regime": "bull", "adx": 28, "rsi": 58},
        )

    def sentiment_gen(role, task, round_num, prev_round):
        return DebateArgument(
            agent_role=role,
            position=Position.BUY,
            confidence=0.70,
            reasoning="Sentiment improving, score 0.35. No high-impact negative news.",
            data_points={"sentiment_score": 0.35},
        )

    def risk_gen(role, task, round_num, prev_round):
        # Risk assessor is cautious but agrees
        return DebateArgument(
            agent_role=role,
            position=Position.BUY,
            confidence=0.60,
            reasoning="Risk within limits. Position size OK. Stop-loss at $66,150.",
            data_points={"stop_loss": 66150, "position_pct": 0.15},
        )

    generators = {
        "technical_analyst": tech_analyst_gen,
        "sentiment_analyst": sentiment_gen,
        "risk_assessor": risk_gen,
    }

    roles = ["technical_analyst", "sentiment_analyst", "risk_assessor"]
    transcript = debate.run_debate("BTC/USDT", "Should we buy?", roles, generators)

    print(f"\nStatus: {transcript.status.value}")
    print(f"Final position: {transcript.final_position.value}")
    print(f"Confidence: {transcript.final_confidence:.2f}")
    print(f"Rounds: {len(transcript.rounds)}")
    print(f"Reasoning: {transcript.final_reasoning[:200]}")


async def main():
    # Part 1: LangGraph (fast, deterministic)
    await demo_langgraph()

    # Part 2: CrewAI (live LLM — takes ~10-30 seconds)
    await demo_crewai()

    # Part 3: Debate system
    demo_debate()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
