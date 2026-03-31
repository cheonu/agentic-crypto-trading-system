from agentic_crypto_trading_system.agents.langgraph_framework import TradingState
from agentic_crypto_trading_system.agents.langgraph_framework import analyze_market_node, analyze_sentiment_node, assess_risk_node, build_technical_analyst_graph, build_sentiment_analyst_graph

# def test_analyze_market_node():

#     state = {
#         "task_description": "BTC/USD trading",
#         "market_data": {"price": 60000, "volume": 1000000},
#         "regime_data": {"regime": "Bullish", "confidence": 0.85}
#     }
    
#     result = analyze_market_node(state)
#     print("Analysis Result:", result["analysis"])

# def test_sentiment_node():
#     state = {
#         "sentiment_data": {"news": "positive"}, 
#         "score": 0.5
#         }
#     result = analyze_sentiment_node(state)
#     print(result["analysis"])

# def test_assess_risk_node():
#     state = {
#         "regime_data": {"regime": "bull", "confidence": 0.85},
#         "analysis": "Market is bullish with high confidence"
#         }
#     result = assess_risk_node(state)
#     regime = state["regime_data"]
#     regime_name = regime.get("regime" "unknown")
#     print ("Debug regime_Name", regime_name)
#     print("Analysis Result:", result["analysis"])
#     assert "BUY" in result["recommendation"]
#     assert "bull" in result["reasoning"]
#     assert result["completed"] is True

# def test_technical_analyst_graph():
#     app = build_technical_analyst_graph().compile()
#     state = {
#         "task_description": "BTC/USD trading",
#         "market_data": {"price": 60000, "volume": 1000000},
#         "regime_data": {"regime": "Bullish", "confidence": 0.85},
#         "regime_data": {"regime": "bull", "confidence": 0.85},
#         "analysis": "Market is bullish with high confidence",
#         "sentiment_data": {"news": "positive"}, 
#         "score": 0.5,
#     }
    
#     result = app.invoke(state)
#     print(result)

def test_build_sentiment_analyst_graph():
    test = build_sentiment_analyst_graph().compile()
    state = {
        "task_description": "BTC/USD trading",
        "market_data": {"price": 60000, "volume": 1000000},
        "regime_data": {"regime": "Bullish", "confidence": 0.85},
        "regime_data": {"regime": "bull", "confidence": 0.85},
        "analysis": "Market is bullish with high confidence",
        "sentiment_data": {"news": "positive"}, 
        "score": 0.5,
    }

    result = test.invoke(state)
    return result

