####
import pytest
from agentic_crypto_trading_system.agents.crewai_framework import CrewAIFramework
from agentic_crypto_trading_system.agents.base import AgentConfig, AgentRole, Task


# def test_create_agent():
#     framework = CrewAIFramework()

#     config =  AgentConfig (
#         name="test_agent",
#         role=AgentRole.TECHNICAL_ANALYST,
#         goal="Analyze market trends",
#         backstory="You are a senior analyst with 10 years of experience.",
#         memory_enabled=True,
#         verbose=True,
#         max_iterations=5
#     )
    
#     agent = framework.create_agent(config)
    
#     assert agent is not None
#     assert config.role in framework.crew_agents
#     assert config.role in framework.agent_states
#     assert framework.agent_states[config.role].name == "test_agent"

@pytest.mark.asyncio
async def test_execute_task():

    framework = CrewAIFramework()

    config =  AgentConfig (
        name="test_agent",
        role=AgentRole.TECHNICAL_ANALYST,
        goal="Analyze market trends",
        backstory="You are a senior analyst with 10 years of experience.",
        memory_enabled=True,
        verbose=True,
        max_iterations=5
    ) 
    framework.create_agent(config)

    task = Task (

        description="Analyze Bitcoin price movements",
        expected_output="A detailed report on BTC price trends",
        # agent="technical_analyst",
        agent_role=AgentRole.TECHNICAL_ANALYST,
        # output="",
        # success=True,
    )
    
    result_task = await framework.execute_task(task)

    assert result_task is not None
    assert result_task.success is True

    state = framework.agent_states[AgentRole.TECHNICAL_ANALYST]

    assert state.is_active is False   # finished execution
