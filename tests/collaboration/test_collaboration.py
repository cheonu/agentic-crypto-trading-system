from agentic_crypto_trading_system.collaboration.message_bus import (

    Message, 
    MessageBus
)

def test_message_bus():
    framework = MessageBus()

    handler = Message(
        topic="market_update",
        sender="market_data_service",
        payload={
            "symbol": "BTC/USD",
            "price": 60000,
            "volume": 1000000
        }
    )

    result = framework.publish(handler)

    print (result)
