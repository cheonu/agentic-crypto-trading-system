from agentic_crypto_trading_system.agents.crewai_framework import CrewMemoryTool
from agentic_crypto_trading_system.agents.tools import ToolResult

class FakeMemory():
    def query_similar_trades(self, query):
        return ToolResult(success=True, data=f"Fake {query} here")

def test_crew_memory_tool():
    fake_memory_tool = FakeMemory()
    tool = CrewMemoryTool(memory_tool=fake_memory_tool)
    query = "BTC trades" 
    result = tool._run(query)
    print(result)
    assert "Fake BTC trades here" in result