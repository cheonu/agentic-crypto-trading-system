
from agentic_crypto_trading_system.agents.crewai_framework import CrewSentimentTool
from agentic_crypto_trading_system.agents.tools import ToolResult

### test
class fakeSentiment():
    def get_current_sentiment(self):
        return ToolResult(
            success=True,
            data = {
               "score" : 0.5,
               "trend" : "neutral" 
            }
        )

def test_sentiment_tool():
    fake_sentiment = fakeSentiment()
    tool = CrewSentimentTool(sentiment_tool=fake_sentiment)
    query = "test"
    result = tool._run(query)
    print (result)
    assert "Current sentiment score: 0.5" in result
    assert "trend: neutral" in result