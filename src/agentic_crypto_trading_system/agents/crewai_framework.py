import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from crewai import Agent as CrewAgent
from crewai import Task as CrewTask
from crewai import Crew, Process
from crewai.tools import BaseTool as CrewBaseTool
from pydantic import Field

from .base import (
    AgentFramework,
    AgentConfig,
    AgentRole,
    AgentState,
    Task,
    TaskResult,
)

logger = logging.getLogger(__name__)


# --- CrewAI Tool Wrappers ---
# CrewAI tools must extend CrewBaseTool and implement _run().
# These wrap our internal tools into CrewAI's expected format.

class CrewMarketDataTool(CrewBaseTool):
    """CrewAI wrapper for market data queries."""
    name: str = "market_data_query"
    description: str = "Query current market data. Input should be a trading pair symbol like 'BTC/USDT'."
    market_data_tool: Any = Field(default=None, exclude=True)

    def _run(self, symbol: str) -> str:
        if not self.market_data_tool:
            return "Market data tool not configured"
        return f"Market data for {symbol}: Use get_ticker or get_ohlcv for detailed data."


class CrewMemoryTool(CrewBaseTool):
    """CrewAI wrapper for memory queries."""
    name: str = "memory_query"
    description: str = "Search past trades and patterns from memory. Input should be a search query describing what you're looking for."
    memory_tool: Any = Field(default=None, exclude=True)

    def _run(self, query: str) -> str:
        if not self.memory_tool:
            return "Memory tool not configured"
        result = self.memory_tool.query_similar_trades(query)
        if result.success:
            return f"Found similar trades: {result.data}"
        return f"Memory query failed: {result.error}"

class CrewSentimentTool(CrewBaseTool):
    """CrewAI wrapper for sentiment queries."""
    name: str = "sentiment_query"
    description: str = "Get current news sentiment for crypto markets. No input required, just call it."
    sentiment_tool: Any = Field(default=None, exclude=True)

    def _run(self, _input: str = "") -> str:
        if not self.sentiment_tool:
            return "Sentiment tool not configured"
        result = self.sentiment_tool.get_current_sentiment()
        if result.success:
            return f"Current sentiment score: {result.data['score']}, trend: {result.data['trend']}"
        return f"Sentiment query failed: {result.error}"

class CrewIndicatorTool(CrewBaseTool):
    """CrewAI wrapper for technical indicators."""
    name: str = "indicator_query"
    description: str = "Get current market regime and technical indicators. No input required."
    indicator_tool: Any = Field(default=None, exclude=True)

    def _run(self, _input: str = "") -> str:
        if not self.indicator_tool:
            return "Indicator tool not configured"
        return "Use get_current_regime with OHLCV data for regime detection."


# --- CrewAI Framework ---

class CrewAIFramework(AgentFramework):
    """CrewAI implementation of the AgentFramework.
    
    Key CrewAI concepts:
    - Agent: An autonomous unit with a role, goal, backstory, and tools
    - Task: A specific job assigned to an agent
    - Crew: A team of agents that work together on tasks
    - Process: How tasks are executed (sequential or hierarchical)
    """

    def __init__(self, name: str = "crewai", verbose: bool = False):
        super().__init__(name)
        self.verbose = verbose
        self.crew_agents: Dict[AgentRole, CrewAgent] = {}
        self.agent_states: Dict[AgentRole, AgentState] = {}
        self.crew_tools: List[CrewBaseTool] = []
        self.llm = self._resolve_llm()

    @staticmethod
    def _resolve_llm() -> Optional[str]:
        """Resolve the LLM model string based on LLM_PROVIDER env var.

        CrewAI accepts model strings like:
        - "openai/gpt-4o-mini" (default)
        - "anthropic/claude-3-5-sonnet-20241022"
        - "gemini/gemini-2.0-flash"
        """
        provider = os.getenv("LLM_PROVIDER", "").lower()
        if provider == "anthropic":
            return "anthropic/claude-3-5-sonnet-20241022"
        elif provider == "google":
            return "gemini/gemini-2.0-flash"
        elif provider == "deepseek":
            return "deepseek/deepseek-chat"
        elif provider == "grok":
            return "xai/grok-3-mini"
        elif provider == "openai":
            return "openai/gpt-4o-mini"
        # No provider set — let CrewAI auto-detect from env vars
        return None

    def create_agent(self, config: AgentConfig) -> CrewAgent:
        """Create a CrewAI agent from our AgentConfig.
        
        This maps our generic config to CrewAI's Agent class.
        CrewAI agents have:
        - role: What the agent does (e.g., "Technical Analyst")
        - goal: What the agent is trying to achieve
        - backstory: Context that shapes the agent's behavior
        - tools: Functions the agent can call
        - memory: Whether the agent remembers past interactions
        - verbose: Whether to log detailed output
        """
        crew_agent = CrewAgent(
            role=config.role.value.replace("_", " ").title(),
            goal=config.goal,
            backstory=config.backstory,
            tools=self.crew_tools,
            memory=config.memory_enabled,
            verbose=config.verbose or self.verbose,
            max_iter=config.max_iterations,
            **({"llm": self.llm} if self.llm else {}),
        )

        self.crew_agents[config.role] = crew_agent
        self.agents[config.role] = crew_agent
        self.agent_states[config.role] = AgentState(
            name=config.name,
            role=config.role,
        )

        logger.info(f"Created CrewAI agent: {config.name} ({config.role.value})")
        return crew_agent

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task using CrewAI.
        
        This creates a CrewAI Task, wraps it in a Crew,
        and kicks off execution. CrewAI handles the LLM
        interaction, tool usage, and reasoning.
        """
        agent = self.crew_agents.get(task.agent_role)
        if not agent:
            return TaskResult(
                output="",
                agent_role=task.agent_role,
                success=False,
                error=f"No agent found for role {task.agent_role.value}",
            )

        # Update state
        state = self.agent_states[task.agent_role]
        state.is_active = True
        state.current_task = task.description

        try:
            # Create CrewAI task
            crew_task = CrewTask(
                description=task.description,
                expected_output=task.expected_output or "Provide a detailed analysis.",
                agent=agent,
            )

            # Create a crew with just this agent and task
            crew = Crew(
                agents=[agent],
                tasks=[crew_task],
                process=Process.sequential,
                verbose=self.verbose,
            )

            # CrewAI's kickoff() is synchronous, so run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, crew.kickoff)

            state.tasks_completed += 1
            state.last_output = str(result)

            return TaskResult(
                output=str(result),
                agent_role=task.agent_role,
                success=True,
                reasoning=str(result),
            )

        except Exception as e:
            logger.error(f"CrewAI task execution failed: {e}")
            return TaskResult(
                output="",
                agent_role=task.agent_role,
                success=False,
                error=str(e),
            )
        finally:
            state.is_active = False
            state.current_task = None

    def get_agent_state(self, role: AgentRole) -> AgentState:
        """Get the current state of a CrewAI agent."""
        if role in self.agent_states:
            return self.agent_states[role]
        return AgentState(name="unknown", role=role)

    def register_crew_tools(
        self,
        market_data_tool=None,
        memory_tool=None,
        sentiment_tool=None,
        indicator_tool=None,
    ) -> None:
        """Register our internal tools as CrewAI tools.
        
        This bridges our tool system with CrewAI's tool system.
        Each internal tool gets wrapped in a CrewAI-compatible class.
        """
        if market_data_tool:
            self.crew_tools.append(
                CrewMarketDataTool(market_data_tool=market_data_tool)
            )
        if memory_tool:
            self.crew_tools.append(
                CrewMemoryTool(memory_tool=memory_tool)
            )
        if sentiment_tool:
            self.crew_tools.append(
                CrewSentimentTool(sentiment_tool=sentiment_tool)
            )
        if indicator_tool:
            self.crew_tools.append(
                CrewIndicatorTool(indicator_tool=indicator_tool)
            )
        logger.info(f"Registered {len(self.crew_tools)} CrewAI tools")

