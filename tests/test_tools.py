### tes fake_market_data():

from agentic_crypto_trading_system.agents.crewai_framework import CrewMarketDataTool
class FakeMarketData():
    def query(self, symbol):
        return f"Fake price for market {symbol}"

def test_crew_market_data_tool():
    fake_tool = FakeMarketData()

    tool = CrewMarketDataTool(market_data_tool=fake_tool)

    result = tool._run("BTC/USDT")
    print(result)
    assert "BTC/USDT" in result
    assert "Fake price for market BTC/USDT" in result