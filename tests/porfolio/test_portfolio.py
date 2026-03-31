from agentic_crypto_trading_system.portfolio.manager import (

    RebalanceEvent, 
    PortfolioManager
)

def test_allocate_capital():

    framework = PortfolioManager(1.0)

    agent_roles = ["technical_analyst", "sentiment", "quant_model"]

    initial_weights = {
        "technical_analyst": 1.5,  # we trust this agent a bit more
        "sentiment": 1.0,          # default weight
        "quant_model": 2.0          # this agent is very reliable
    }

    allocated = framework.allocate_capital(agent_roles, initial_weights)
    print(allocated)


