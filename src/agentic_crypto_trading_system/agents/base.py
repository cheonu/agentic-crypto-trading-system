from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentRole(str, Enum):
    TECHNICAL_ANALYST = "technical_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    RISK_ASSESSOR = "risk_assessor"


@dataclass
class AgentConfig:
    """Configuration for creating an agent."""
    name: str
    role: AgentRole
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    memory_enabled: bool = True
    max_iterations: int = 10
    verbose: bool = False


@dataclass
class Task:
    """A task for an agent to execute."""
    description: str
    agent_role: AgentRole
    context: Dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""


@dataclass
class TaskResult:
    """Result of a task execution."""
    output: str
    agent_role: AgentRole
    success: bool = True
    reasoning: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AgentState:
    """Current state of an agent."""
    name: str
    role: AgentRole
    is_active: bool = False
    current_task: Optional[str] = None
    tasks_completed: int = 0
    last_output: Optional[str] = None


class AgentFramework(ABC):
    """Abstract base class for agent frameworks.
    
    Both CrewAI and LangGraph implement this interface,
    allowing the trading system to swap frameworks without
    changing business logic.
    """

    def __init__(self, name: str):
        self.name = name
        self.agents: Dict[AgentRole, Any] = {}
        self.tools: Dict[str, Any] = {}

    @abstractmethod
    def create_agent(self, config: AgentConfig) -> Any:
        """Create an agent with the given configuration.
        
        Returns the framework-specific agent object.
        """
        pass

    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task using the appropriate agent.
        
        Routes the task to the agent matching task.agent_role.
        """
        pass

    @abstractmethod
    def get_agent_state(self, role: AgentRole) -> AgentState:
        """Get the current state of an agent by role."""
        pass

    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool that agents can use."""
        self.tools[name] = tool

    def get_registered_tools(self) -> Dict[str, Any]:
        """Get all registered tools."""
        return self.tools
