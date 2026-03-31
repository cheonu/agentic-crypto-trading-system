import asyncio
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .base import (
    AgentFramework,
    AgentConfig,
    AgentRole,
    AgentState,
    Task,
    TaskResult,
)

logger = logging.getLogger(__name__)


# --- LangGraph State ---
# In LangGraph, state is a TypedDict that flows through the graph.
# Every node reads from and writes to this shared state.

class TradingState(TypedDict):
    """Shared state that flows through the agent graph."""
    task_description: str
    expected_output: str
    agent_role: str
    market_data: dict
    sentiment_data: dict
    regime_data: dict
    analysis: str
    recommendation: str
    reasoning: str
    error: str
    completed: bool


# --- Node Functions ---
# Each node is a plain function at module level (NOT inside a class).

def analyze_market_node(state: TradingState) -> TradingState:
    """Node: Analyze market data."""
    task = state.get("task_description", "")
    regime = state.get("regime_data", {})

    analysis = (
        f"Technical analysis for: {task}. "
        f"Market regime: {regime.get('regime', 'unknown')}. "
        f"Confidence: {regime.get('confidence', 0):.2f}."
    )

    return {**state, "analysis": analysis}


def analyze_sentiment_node(state: TradingState) -> TradingState:
    """Node: Analyze sentiment data."""
    sentiment = state.get("sentiment_data", {})
    score = sentiment.get("score", 0)

    label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    analysis = state.get("analysis", "")
    analysis += f" Sentiment: {label} (score: {score:.2f})."

    return {**state, "analysis": analysis}


def assess_risk_node(state: TradingState) -> TradingState:
    """Node: Assess risk and make recommendation."""
    analysis = state.get("analysis", "")
    regime = state.get("regime_data", {})

    regime_name = regime.get("regime", "unknown")
    if regime_name == "high_volatility":
        recommendation = "HOLD - High volatility detected, reduce exposure."
    elif regime_name == "bull":
        recommendation = "BUY - Bullish regime with positive indicators."
    elif regime_name == "bear":
        recommendation = "SELL - Bearish regime, consider reducing positions."
    else:
        recommendation = "HOLD - Sideways market, wait for clearer signals."

    reasoning = f"Based on analysis: {analysis} Recommendation: {recommendation}"

    return {
        **state,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "completed": True,
    }


def error_handler_node(state: TradingState) -> TradingState:
    """Node: Handle errors gracefully."""
    return {
        **state,
        "recommendation": "ERROR - Unable to complete analysis.",
        "reasoning": state.get("error", "Unknown error"),
        "completed": True,
    }


# --- Router ---

def should_assess_risk(state: TradingState) -> str:
    """Decide whether to proceed to risk assessment or error handling."""
    if state.get("error"):
        return "error_handler"
    return "assess_risk"


# --- Graph Builders ---

def build_technical_analyst_graph() -> StateGraph:
    """Build the state graph for technical analysis.
    
    Graph: analyze_market -> analyze_sentiment -> (assess_risk | error_handler) -> END
    """
    graph = StateGraph(TradingState)

    graph.add_node("analyze_market", analyze_market_node)
    graph.add_node("analyze_sentiment", analyze_sentiment_node)
    graph.add_node("assess_risk", assess_risk_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("analyze_market")
    graph.add_edge("analyze_market", "analyze_sentiment")
    graph.add_conditional_edges(
        "analyze_sentiment",
        should_assess_risk,
        {
            "assess_risk": "assess_risk",
            "error_handler": "error_handler",
        },
    )
    graph.add_edge("assess_risk", END)
    graph.add_edge("error_handler", END)

    return graph


def build_sentiment_analyst_graph() -> StateGraph:
    """Build graph for sentiment-focused analysis."""
    graph = StateGraph(TradingState)

    graph.add_node("analyze_sentiment", analyze_sentiment_node)
    graph.add_node("assess_risk", assess_risk_node)

    graph.set_entry_point("analyze_sentiment")
    graph.add_edge("analyze_sentiment", "assess_risk")
    graph.add_edge("assess_risk", END)

    return graph


def build_risk_assessor_graph() -> StateGraph:
    """Build graph for risk assessment."""
    graph = StateGraph(TradingState)

    graph.add_node("assess_risk", assess_risk_node)

    graph.set_entry_point("assess_risk")
    graph.add_edge("assess_risk", END)

    return graph


# --- LangGraph Framework ---
# This maps AgentRole -> graph builder, compiles graphs, and runs them.
# Unlike CrewAI (which uses LLMs directly), LangGraph gives you full
# control over the execution flow via state machines.

# Map each role to its graph builder
GRAPH_BUILDERS = {
    AgentRole.TECHNICAL_ANALYST: build_technical_analyst_graph,
    AgentRole.SENTIMENT_ANALYST: build_sentiment_analyst_graph,
    AgentRole.RISK_ASSESSOR: build_risk_assessor_graph,
}


class LangGraphFramework(AgentFramework):
    """LangGraph implementation of the AgentFramework.

    Key LangGraph concepts:
    - StateGraph: A directed graph where nodes transform shared state
    - Nodes: Functions that read/write to the state dict
    - Edges: Connections between nodes (can be conditional)
    - Compilation: The graph is compiled into a runnable before execution
    - State: A TypedDict that flows through every node

    Unlike CrewAI, LangGraph does NOT call an LLM by default.
    You wire up exactly what happens at each step. This makes it
    deterministic, testable, and easy to reason about.
    """

    def __init__(self, name: str = "langgraph"):
        super().__init__(name)
        self.compiled_graphs: Dict[AgentRole, Any] = {}
        self.agent_states: Dict[AgentRole, AgentState] = {}

    def create_agent(self, config: AgentConfig) -> Any:
        """Create a LangGraph agent by compiling its state graph.

        Steps:
        1. Look up the graph builder for the agent's role
        2. Build the StateGraph (nodes + edges)
        3. Compile it into a runnable
        4. Store it for later execution
        """
        builder = GRAPH_BUILDERS.get(config.role)
        if not builder:
            raise ValueError(f"No graph builder for role: {config.role.value}")

        # Build and compile the graph
        state_graph = builder()
        compiled = state_graph.compile()

        self.compiled_graphs[config.role] = compiled
        self.agents[config.role] = compiled
        self.agent_states[config.role] = AgentState(
            name=config.name,
            role=config.role,
        )

        logger.info(f"Created LangGraph agent: {config.name} ({config.role.value})")
        return compiled

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task by invoking the compiled graph.

        Steps:
        1. Build the initial TradingState from the Task
        2. Invoke the compiled graph (runs all nodes in order)
        3. Extract the final state and return a TaskResult
        """
        compiled = self.compiled_graphs.get(task.agent_role)
        if not compiled:
            return TaskResult(
                output="",
                agent_role=task.agent_role,
                success=False,
                error=f"No compiled graph for role {task.agent_role.value}",
            )

        # Update agent state
        state = self.agent_states[task.agent_role]
        state.is_active = True
        state.current_task = task.description

        try:
            # Build initial state from the task context
            initial_state: TradingState = {
                "task_description": task.description,
                "expected_output": task.expected_output,
                "agent_role": task.agent_role.value,
                "market_data": task.context.get("market_data", {}),
                "sentiment_data": task.context.get("sentiment_data", {}),
                "regime_data": task.context.get("regime_data", {}),
                "analysis": "",
                "recommendation": "",
                "reasoning": "",
                "error": "",
                "completed": False,
            }

            # Invoke the graph — this runs synchronously through all nodes
            # Use run_in_executor since .invoke() is sync
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                None, compiled.invoke, initial_state
            )

            state.tasks_completed += 1
            state.last_output = final_state.get("recommendation", "")

            return TaskResult(
                output=final_state.get("recommendation", ""),
                agent_role=task.agent_role,
                success=final_state.get("completed", False),
                reasoning=final_state.get("reasoning", ""),
                data={
                    "analysis": final_state.get("analysis", ""),
                    "recommendation": final_state.get("recommendation", ""),
                },
            )

        except Exception as e:
            logger.error(f"LangGraph task execution failed: {e}")
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
        """Get the current state of a LangGraph agent."""
        if role in self.agent_states:
            return self.agent_states[role]
        return AgentState(name="unknown", role=role)
