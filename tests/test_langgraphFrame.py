import pytest
from agentic_crypto_trading_system.agents.langgraph_framework import LangGraphFramework
from agentic_crypto_trading_system.agents.base import AgentConfig, AgentRole, Task

def test_create_agent():
    framework = LangGraphFramework()

    config =  AgentConfig (
        name="test_agent",
        role=AgentRole.TECHNICAL_ANALYST,
        goal="Analyze market trends",
        backstory="You are a senior analyst with 10 years of experience.",
        memory_enabled=True,
        verbose=True,
        max_iterations=5
    )

    agent = framework.create_agent(config)
    print (agent)

    assert agent is not None
    assert config.role in framework.agents
    assert config.role in framework.agent_states
    assert framework.agent_states[config.role].name == "test_agent"

@pytest.mark.asyncio
async def test_execute_task():

    framework = LangGraphFramework()
    config =  AgentConfig (
        name="test_agent",
        role=AgentRole.TECHNICAL_ANALYST,
        goal="Analyze market trends",
        backstory="You are a senior analyst with 10 years of experience.",
        memory_enabled=True,
        verbose=True,
        max_iterations=5
    )

    agent = framework.create_agent(config)

    task = Task (

        description="Analyze Bitcoin price movements",
        expected_output="A detailed report on BTC price trends",
        agent_role=AgentRole.TECHNICAL_ANALYST,
    )

    result = await framework.execute_task(task)
    

    print (result)
    state = framework.agent_states[task.agent_role]
    assert state.tasks_completed == 1
    assert state.current_task is None
    assert state.is_active is False